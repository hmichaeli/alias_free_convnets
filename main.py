# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# convnext_LICENSE.txt file in the root directory of this source tree.


import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import torchvision
import json
import os

from pathlib import Path

from timm.data.mixup import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma
from optim_factory import create_optimizer, LayerDecayValueAssigner

from datasets import build_dataset
from engine import train_one_epoch, evaluate, evaluate_consistency_metrics, evaluate_benchmark

from utils import NativeScalerWithGradNormCount as NativeScaler
import utils
from utils import get_args, get_model


from shift_eval import evaluate_fraction_adversarial_shift_metrics, evaluate_full_adversarial_shift_metrics, \
    evaluate_adversarial_crop_shift_metrics, evaluate_random_shift_metrics
try:
    import wandb

except ImportError:
    wandb = None
    raise ImportError(
        "To use the Weights and Biases Logger please install wandb."
        "Run `pip install wandb` to install it."
    )


def main(args):
    utils.init_distributed_mode(args)
    print(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    print("build datasets:")
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    if args.disable_eval:
        args.dist_eval = False
        dataset_val = None
    else:
        dataset_val, _ = build_dataset(is_train=False, args=args)

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    if global_rank == 0 and args.enable_wandb:
        # added to support wandb sweep
        run = wandb.init(
                project=args.project,
                config=args,
                name=args.wandb_name,
                group=args.wandb_group
            )
        args = wandb.config
        print("wandb.config args: ", args)
        wandb_logger = utils.WandbLogger(args)
    else:
        wandb_logger = None


    print(f"dataset_train: \n{dataset_train}\n dataset_val: \n{dataset_val}\n nb_classes: {args.nb_classes}")

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=args.seed,
    )
    print("Sampler_train = %s" % str(sampler_train))
    if args.dist_eval:
        if len(dataset_val) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None


    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
    else:
        data_loader_val = None

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    total_training_steps = num_training_steps_per_epoch * args.epochs
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training training per epoch = %d" % num_training_steps_per_epoch)
    print("Number of total training steps = %d" % total_training_steps)


    model = get_model(args, total_steps=total_training_steps)

    # sample inputs
    matplotlib.use('Agg')  # turn off gui - use on dgx cluster
    for loader, name in [(data_loader_train, 'train'), (data_loader_val, 'val')]:
        sample = next(iter(loader))[0]
        sample = sample[:12] if sample.shape[0] > 12 else sample
        sample = sample.permute(0, 2, 3, 1).cpu().numpy()
        fig, axs = plt.subplots(4, 3, figsize=(12, 9))
        axs = axs.flatten()
        for img, ax in zip(sample, axs):
            ax.imshow(img)
        fig.suptitle(f'{name} sample')
        if args.output_dir:
            plt.savefig(os.path.join(args.output_dir, f'{name}_sample.png'))
        else:
            plt.show()

    # Profiler analysis
    if args.profiler:
        from utils import profiler_test
        print("run profiler")
        profiler_test(model, args)

    # Wandb watch model
    if global_rank == 0 and args.enable_wandb and wandb is not None:
        wandb.watch(model, criterion=None, log="all", log_freq=1000, idx=None, log_graph=False)

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load ckpt from %s" % args.finetune)
        checkpoint_model = None
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)
    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)



    if args.layer_decay < 1.0 or args.layer_decay > 1.0:
        num_layers = 12 # convnext layers divided into 12 parts, each with a different decayed lr value.
        assert args.model in ['convnext_small', 'convnext_base', 'convnext_large', 'convnext_xlarge'], \
             "Layer Decay impl only supports convnext_small/base/large/xlarge"
        assigner = LayerDecayValueAssigner(list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    optimizer = create_optimizer(
        args, model_without_ddp, skip_list=None,
        get_num_layer=assigner.get_layer_id if assigner is not None else None, 
        get_layer_scale=assigner.get_scale if assigner is not None else None)

    loss_scaler = NativeScaler() # if args.use_amp is False, this won't be used

    lr_schedule_values = utils.get_lr_scheduler(num_training_steps_per_epoch, args)

    if args.weight_decay_end is None:
        # solve wandb bug
        try:
            args.update({'weight_decay_end': args.weight_decay}, allow_val_change=True)
        except:
            args.weight_decay_end = args.weight_decay


    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

    if args.eval:
        print(f"Eval only mode")
        test_stats = evaluate(data_loader_val, model, device, use_amp=args.use_amp)
        consistency_stats = evaluate_consistency_metrics(data_loader_val, model, device, use_amp=args.use_amp, max_shift=args.max_shift)

        print(f"Accuracy of the network on {len(dataset_val)} test images: {test_stats['acc1']:.5f}%")
        if wandb_logger:
            log_stats = {**{f'test_{k}': v for k, v in test_stats.items()},
                         **{f'test_consistency_{k}': v for k, v in consistency_stats.items()},
                         'epoch': 0,
                         'n_parameters': n_parameters}

            wandb_logger.log_epoch_metrics(log_stats)

        return

    if args.eval_imagenet_c:
        # eval imagenet
        test_stats = evaluate(data_loader_val, model, device, use_amp=args.use_amp)
        consistency_stats = evaluate_consistency_metrics(data_loader_val, model, device, use_amp=args.use_amp,
                                                         max_shift=args.max_shift)

        print(f"Accuracy of the network on {len(dataset_val)} test images: {test_stats['acc1']:.5f}%")
        log_stats = {**{f'test_{k}': v for k, v in test_stats.items()},
                     **{f'test_consistency_{k}': v for k, v in consistency_stats.items()},
                     'epoch': 0,
                     'n_parameters': n_parameters}

        # eval imagenet-c
        distortions = [
            'gaussian_noise', 'shot_noise', 'impulse_noise',
            'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
            'snow', 'frost', 'fog', 'brightness',
            'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
            'speckle_noise', 'gaussian_blur', 'spatter', 'saturate'
        ]
        imagenet_c_mean = [0.485, 0.456, 0.406]
        imagenet_c_std = [0.229, 0.224, 0.225]

        error_rates = []
        for distortion_name in distortions:
            distortion_errs = []
            for severity in range(1, 6):
                distortion = distortion_name + '/' + str(severity)
                print("evaluating distortion: ", distortion)
                distorted_dataset = torchvision.datasets.ImageFolder(
                    root=args.imagenet_c_path + distortion_name + '/' + str(severity),
                    transform=torchvision.transforms.Compose([torchvision.transforms.CenterCrop(224),
                                                              torchvision.transforms.ToTensor(),
                                                              torchvision.transforms.Normalize(imagenet_c_mean, imagenet_c_std)]))

                if args.dist_eval:
                    if len(dataset_val) % num_tasks != 0:
                        print(
                            'Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                            'This will slightly alter validation results as extra duplicate entries are added to achieve '
                            'equal num of samples per-process.')
                    sampler_val = torch.utils.data.DistributedSampler(
                        distorted_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False)
                else:
                    sampler_val = torch.utils.data.SequentialSampler(distorted_dataset)

                distorted_dataset_loader = torch.utils.data.DataLoader(
                    distorted_dataset, sampler=sampler_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                    pin_memory=True)

                distorted_consistency_stats = evaluate_consistency_metrics(distorted_dataset_loader, model, device,
                                                                           use_amp=args.use_amp,
                                                                            max_shift=args.max_shift)

                distortion_errs.append(1 - distorted_consistency_stats['acc1'] / 100)
                log_stats.update({
                    **{f'test_{distortion}_consistency_{k}': v for k, v in distorted_consistency_stats.items()}})
                print(json.dumps(log_stats, indent=4))

            print('\n=Average', tuple(distortion_errs))
            distortion_rate = np.mean(distortion_errs)

            error_rates.append(distortion_rate)
            print('Distortion: {:15s}  | CE (unnormalized) (%): {:.2f}'.format(distortion, 100 * distortion_rate))

        if wandb_logger:
            wandb_logger.log_epoch_metrics(log_stats)

        print('mCE (unnormalized by AlexNet errors) (%): {:.2f}'.format(100 * np.mean(error_rates)))
        with open(os.path.join(args.output_dir, f'imagenet_c_results.json'), "w") as outfile:
            json.dump(log_stats, outfile, indent=4)
        return


    if args.eval_adversarial_shift_metrics:

        if args.eval_imagenet_c_adversarial_shift:
            imagenet_c_mean = [0.485, 0.456, 0.406]
            imagenet_c_std = [0.229, 0.224, 0.225]

            distorted_dataset = torchvision.datasets.ImageFolder(
                root=os.path.join(args.imagenet_c_path, args.imagenet_c_distortion + '/' + str(args.imagenet_c_severity)),
                transform=torchvision.transforms.Compose([torchvision.transforms.CenterCrop(224),
                                                          torchvision.transforms.ToTensor(),
                                                          torchvision.transforms.Normalize(imagenet_c_mean,
                                                                                           imagenet_c_std)]))

            # assert not args.dist_eval

            if args.dist_eval:
                if len(distorted_dataset) % num_tasks != 0:
                    print(
                        'Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                        'This will slightly alter validation results as extra duplicate entries are added to achieve '
                        'equal num of samples per-process.')
                sampler_val = torch.utils.data.DistributedSampler(
                    distorted_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False)
            else:
                sampler_val = torch.utils.data.SequentialSampler(distorted_dataset)

            data_loader_val = torch.utils.data.DataLoader(
                distorted_dataset, sampler=sampler_val, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True)

        test_stats = {}
        if args.adversarial_method == 'integer_rand10':
            test_stats = evaluate_full_adversarial_shift_metrics(data_loader_val, model, device, use_amp=False, max_shift=32,
                                                                 half_shift=False, random_shifts=10)
            log_stats = {**{f'test_integer_rand10_adversarial_{k}': v for k, v in test_stats.items()},
                         'epoch': 0,
                         'n_parameters': n_parameters}

        if args.adversarial_method == 'integer':
            test_stats = evaluate_full_adversarial_shift_metrics(data_loader_val, model, device, use_amp=False, max_shift=32,
                                                                 half_shift=False)
            log_stats = {**{f'test_integer_adversarial_{k}': v for k, v in test_stats.items()},
                         'epoch': 0,
                         'n_parameters': n_parameters}

        if args.adversarial_method == 'full':
            test_stats = evaluate_full_adversarial_shift_metrics(data_loader_val, model, device, use_amp=False, max_shift=32,
                                                                 half_shift=True)
            log_stats = {**{f'test_full_adversarial_{k}': v for k, v in test_stats.items()},
                         'epoch': 0,
                         'n_parameters': n_parameters}

        elif args.adversarial_method == 'fraction':
            max_up = args.fraction_adversarial_max_up
            test_stats = evaluate_fraction_adversarial_shift_metrics(data_loader_val, model, device, use_amp=False, max_up=max_up)
            log_stats = {**{f'test_fraction_adversarial_{k}': v for k, v in test_stats.items()},
                         'epoch': 0,
                         'n_parameters': n_parameters}

        if wandb_logger:
            wandb_logger.log_epoch_metrics(log_stats)

        return



    if args.eval_rand_shift:
        up = args.eval_rand_shift_up
        up_method = args.eval_rand_shift_up_method
        res = evaluate_random_shift_metrics(data_loader_val, model, device, use_amp=False, max_shift=32, up=up,
                                            up_method=up_method)

        log_stats = {**{f'test_eval_rand_shift_{up_method}_{up}_{k}': v for k, v in res.items()},
                 'epoch': 0,
                 'n_parameters': n_parameters}

        print(log_stats)
        if wandb_logger:
            wandb_logger.log_epoch_metrics(log_stats)

        return
    if args.eval_crop_shift is not None:
        consistency_stats = evaluate_adversarial_crop_shift_metrics(data_loader_val, model, device, inp_size=args.input_size,
                                                                    max_shift=args.eval_crop_shift, use_amp=False)


        if wandb_logger:
            log_stats = {**{f'test_{k}': v for k, v in consistency_stats.items()},
                         'epoch': 0,
                         'n_parameters': n_parameters}
            print(log_stats)
            wandb_logger.log_epoch_metrics(log_stats)

        return
    if args.eval_benchmark:
        evaluate_benchmark(data_loader_val, model, device, args.use_amp)
        return

    max_accuracy = 0.0
    if args.model_ema and args.model_ema_eval:
        max_accuracy_ema = 0.0

    print("Start training for %d epochs" % args.epochs)
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        if wandb_logger:
            wandb_logger.set_steps()
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer,
            device, epoch, loss_scaler, args.clip_grad, model_ema, mixup_fn,
            log_writer=log_writer, wandb_logger=wandb_logger, start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq,
            use_amp=args.use_amp
        )
        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema)
        if data_loader_val is not None:
            test_stats = evaluate(data_loader_val, model, device, use_amp=args.use_amp)
            print(f"Accuracy of the model on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            if max_accuracy < test_stats["acc1"]:
                max_accuracy = test_stats["acc1"]
                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch="best", model_ema=model_ema)
            print(f'Max accuracy: {max_accuracy:.2f}%')

            if log_writer is not None:
                log_writer.update(test_acc1=test_stats['acc1'], head="perf", step=epoch)
                log_writer.update(test_acc5=test_stats['acc5'], head="perf", step=epoch)
                log_writer.update(test_loss=test_stats['loss'], head="perf", step=epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

            # repeat testing routines for EMA, if ema eval is turned on
            if args.model_ema and args.model_ema_eval:
                test_stats_ema = evaluate(data_loader_val, model_ema.ema, device, use_amp=args.use_amp)
                print(f"Accuracy of the model EMA on {len(dataset_val)} test images: {test_stats_ema['acc1']:.1f}%")
                if max_accuracy_ema < test_stats_ema["acc1"]:
                    max_accuracy_ema = test_stats_ema["acc1"]
                    if args.output_dir and args.save_ckpt:
                        utils.save_model(
                            args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch="best-ema", model_ema=model_ema)
                    print(f'Max EMA accuracy: {max_accuracy_ema:.2f}%')
                if log_writer is not None:
                    log_writer.update(test_acc1_ema=test_stats_ema['acc1'], head="perf", step=epoch)
                log_stats.update({**{f'test_{k}_ema': v for k, v in test_stats_ema.items()}})

            if epoch % 10 == 0:
                # shift consistency metrics
                consistency_stats = evaluate_consistency_metrics(data_loader_val, model, device, use_amp=args.use_amp,
                                                             max_shift=args.max_shift)
                log_stats.update(**{f'test_consistency_{k}': v for k, v in consistency_stats.items()})

        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        if wandb_logger:
            wandb_logger.log_epoch_metrics(log_stats)

    if wandb_logger and args.wandb_ckpt and args.save_ckpt and args.output_dir:
        wandb_logger.log_checkpoints()


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


    print("load and evaluate best model")
    from utils import load_model_state_dict
    output_dir = Path(args.output_dir)
    checkpoint_path = os.path.join(output_dir, 'checkpoint-best.pth')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_without_ddp = load_model_state_dict(model_without_ddp, checkpoint['model'])
    if hasattr(args, 'model_ema') and args.model_ema:
        if 'model_ema' in checkpoint.keys():
            model_ema.ema = load_model_state_dict(model_ema.ema, checkpoint['model_ema'])

    test_stats = evaluate(data_loader_val, model, device, use_amp=args.use_amp)
    consistency_stats = evaluate_consistency_metrics(data_loader_val, model, device, use_amp=args.use_amp,
                                                     max_shift=args.max_shift)

    log_stats = {**{f'best_model_test{k}': v for k, v in test_stats.items()},
                 **{f'best_model_test_consistency_{k}': v for k, v in consistency_stats.items()},
                 'epoch': 0,
                 'n_parameters': n_parameters}
    print(f"Accuracy of the network on {len(dataset_val)} test images: {test_stats['acc1']:.5f}%")

    if hasattr(args, 'model_ema') and args.model_ema:
        checkpoint_path = os.path.join(output_dir, 'checkpoint-best-ema.pth')
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model_ema.ema = load_model_state_dict(model_ema.ema, checkpoint['model_ema'])

        test_stats_ema = evaluate(data_loader_val, model_ema.ema, device, use_amp=args.use_amp)
        log_stats.update({**{f'best_model_test_{k}_ema': v for k, v in test_stats_ema.items()}})
        log_stats = {**{f'best_model_ema_test_{k}': v for k, v in test_stats.items()},
                 **{f'best_model_ema_test_consistency_{k}': v for k, v in consistency_stats.items()},
                 'epoch': 0,
                 'n_parameters': n_parameters}

    print(log_stats)
    if wandb_logger:
        wandb_logger.log_epoch_metrics(log_stats)



if __name__ == '__main__':
    args = get_args()
    main(args)
