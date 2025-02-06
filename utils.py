# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# convnext_LICENSE.txt file in the root directory of this source tree.


import os
import math
import time
from collections import defaultdict, deque
import datetime
import numpy as np
from timm.utils import get_state_dict

from pathlib import Path

import torch
import torch.distributed as dist
from torch._six import inf

from tensorboardX import SummaryWriter
import argparse
import json
import yaml
from types import SimpleNamespace
import random
import string
import secrets


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


class TensorboardLogger(object):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(logdir=log_dir)
        self.step = 0

    def set_step(self, step=None):
        if step is not None:
            self.step = step
        else:
            self.step += 1

    def update(self, head='scalar', step=None, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.writer.add_scalar(head + "/" + k, v, self.step if step is None else step)

    def flush(self):
        self.writer.flush()


class WandbLogger(object):
    def __init__(self, args):
        self.args = args

        try:
            import wandb
            self._wandb = wandb
        except ImportError:
            raise ImportError(
                "To use the Weights and Biases Logger please install wandb."
                "Run `pip install wandb` to install it."
            )

        # Initialize a W&B run 
        if self._wandb.run is None:
            self._wandb.init(
                project=args.project,
                config=args,
                name=args.wandb_name,
                group=args.wandb_group
            )

    def log_epoch_metrics(self, metrics, commit=True):
        """
        Log train/test metrics onto W&B.
        """
        # Log number of model parameters as W&B summary
        self._wandb.summary['n_parameters'] = metrics.get('n_parameters', None)
        metrics.pop('n_parameters', None)

        # Log current epoch
        self._wandb.log({'epoch': metrics.get('epoch')}, commit=False)
        metrics.pop('epoch')

        for k, v in metrics.items():
            if 'train' in k:
                self._wandb.log({f'Global Train/{k}': v}, commit=False)
            elif 'best_model_test' in k:
                self._wandb.log({f'Best Model Test/{k}': v}, commit=False)
            elif 'test' in k:
                self._wandb.log({f'Global Test/{k}': v}, commit=False)

        self._wandb.log({})

    def log_checkpoints(self):
        output_dir = self.args.output_dir
        model_artifact = self._wandb.Artifact(
            self._wandb.run.id + "_model", type="model"
        )

        model_artifact.add_dir(output_dir)
        self._wandb.log_artifact(model_artifact, aliases=["latest", "best"])

    def set_steps(self):
        # Set global training step
        self._wandb.define_metric('Rank-0 Batch Wise/*', step_metric='Rank-0 Batch Wise/global_train_step')
        # Set epoch-wise step
        self._wandb.define_metric('Global Train/*', step_metric='epoch')
        self._wandb.define_metric('Global Test/*', step_metric='epoch')


def gen_rand_str(length=6):
    return ''.join(secrets.choice(string.ascii_letters + string.digits) for x in range(length))


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):

    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()

        os.environ['RANK'] = str(args.rank)
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['WORLD_SIZE'] = str(args.world_size)
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_iters > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule

def lr_steps_scheduler(base_value, lr_steps, step_size, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_iters > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    schedule = np.ones(epochs * niter_per_ep) * base_value
    schedule[:warmup_iters] = warmup_schedule
    for step_epoch in lr_steps:
        step_iter = step_epoch * niter_per_ep
        schedule[step_iter:] *= step_size


    assert len(schedule) == epochs * niter_per_ep
    return schedule


def get_lr_scheduler(num_training_steps_per_epoch, args):
    if args.scheduler == 'cosine':
        print("Use Cosine LR scheduler")
        lr_schedule_values = cosine_scheduler(
            args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
            warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
        )

    elif args.scheduler == 'lr_steps':
        print(f"Use LR Steps scheduler: {args.lr_steps}")
        lr_schedule_values = lr_steps_scheduler(
            args.lr, args.lr_steps, args.lr_step_size, args.epochs, num_training_steps_per_epoch,
            warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
        )

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(lr_schedule_values)
    plt.show()

    return lr_schedule_values


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, model_ema=None):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
    for checkpoint_path in checkpoint_paths:
        try:
            args_dict = args._as_dict()
        except:
            args_dict = args
        to_save = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'scaler': loss_scaler.state_dict(),
            # 'args': args._as_dict() if args.enable_wandb else args,
            'args': args_dict
        }

        if model_ema is not None:
            to_save['model_ema'] = get_state_dict(model_ema)

        save_on_master(to_save, checkpoint_path)
    
    if is_main_process() and isinstance(epoch, int):
        to_del = epoch - args.save_ckpt_num * args.save_ckpt_freq
        old_ckpt = output_dir / ('checkpoint-%s.pth' % to_del)
        if os.path.exists(old_ckpt):
            os.remove(old_ckpt)


def auto_load_model(args, model, model_without_ddp, optimizer, loss_scaler, model_ema=None):
    output_dir = Path(args.output_dir)
    if args.auto_resume and len(args.resume) == 0:
        import glob
        all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*.pth'))
        latest_ckpt = -1
        for ckpt in all_checkpoints:
            t = ckpt.split('-')[-1].split('.')[0]
            if t.isdigit():
                latest_ckpt = max(int(t), latest_ckpt)
        if latest_ckpt >= 0:
            # if args.enable_wandb:
            #     args.update({'resume': os.path.join(output_dir, 'checkpoint-%d.pth' % latest_ckpt)}, allow_val_change=True)
            # else:
            #     args.resume = os.path.join(output_dir, 'checkpoint-%d.pth' % latest_ckpt)
            try:
                args.update({'resume': os.path.join(output_dir, 'checkpoint-%d.pth' % latest_ckpt)}, allow_val_change=True)
            except:
                args.resume = os.path.join(output_dir, 'checkpoint-%d.pth' % latest_ckpt)
        print("Auto resume checkpoint: %s" % args.resume)

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        # model_without_ddp.load_state_dict(checkpoint['model'])
        model_without_ddp = load_model_state_dict(model_without_ddp, checkpoint['model'])
        print("Resume checkpoint %s" % args.resume)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            if not isinstance(checkpoint['epoch'], str): # does not support resuming with 'best', 'best-ema'
                # if args.enable_wandb:
                #     args.update({'start_epoch': checkpoint['epoch'] + 1},
                #                 allow_val_change=True)
                # else:
                #     args.start_epoch = checkpoint['epoch'] + 1
                try:
                    args.update({'start_epoch': checkpoint['epoch'] + 1},
                                allow_val_change=True)
                except:
                    args.start_epoch = checkpoint['epoch'] + 1
            else:
                assert args.eval, 'Does not support resuming with checkpoint-best'
            if hasattr(args, 'model_ema') and args.model_ema:
                if 'model_ema' in checkpoint.keys():
                    # model_ema.ema.load_state_dict(checkpoint['model_ema'])
                    model_ema.ema = load_model_state_dict(model_ema.ema, checkpoint['model_ema'])
                else:
                    # model_ema.ema.load_state_dict(checkpoint['model'])
                    model_ema.ema = load_model_state_dict(model_ema.ema, checkpoint['model'])

            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")

    else:
        print("No model to resume, start training from scratch")


#  load model state dict - allow missing keys for aal models
def load_model_state_dict(model, state_dict):
    print("load model state dict")
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print("missing keys: ", missing_keys)
    print("unexpected keys: ", unexpected_keys)

    # aal model - allow missing rect in state_dict
    # rect params dependent on input size and initialized on forward

    # if len(missing_keys) > 0:
    #     raise ValueError("Missing keys")
    # allow missing 'rect' - because it doesn't appear in EMA model
    for k in missing_keys:
        if 'rect' not in k:
            raise ValueError("Missing keys")

    for k in unexpected_keys:
        if 'rect' not in k:
            raise ValueError("Missing keys")

    return model


def str2bool(v):
    """
    Converts string to bool type; enables command line
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    parser = argparse.ArgumentParser('ConvNeXt training and evaluation script for image classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Per GPU batch size')
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--update_freq', default=1, type=int,
                        help='gradient accumulation steps')

    # Model parameters
    parser.add_argument('--model', default='convnext_tiny', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--drop_path', type=float, default=0, metavar='PCT',
                        help='Drop path rate (default: 0.0)')
    parser.add_argument('--input_size', default=224, type=int,
                        help='image input size')
    parser.add_argument('--layer_scale_init_value', default=1e-6, type=float,
                        help="Layer scale initial values")

    # EMA related parameters
    parser.add_argument('--model_ema', type=str2bool, default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', type=str2bool, default=False, help='')
    parser.add_argument('--model_ema_eval', type=str2bool, default=False, help='Using ema to eval during training.')

    # Optimization parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument('--scheduler', default='cosine', type=str, choices=['cosine', 'lr_steps'],
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--lr', type=float, default=4e-3, metavar='LR',
                        help='learning rate (default: 4e-3), with total batch size 4096')
    parser.add_argument('--layer_decay', type=float, default=1.0)
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')
    parser.add_argument('--warmup_epochs', type=int, default=20, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')
    parser.add_argument("--lr_steps", nargs="+", default=[30, 60, 80], type=int)
    parser.add_argument("--lr_step_size", type=float, default=0.1)

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', type=str2bool, default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    parser.add_argument('--zero_pad_input', type=int, default=0,
                        help='Use zero padding transformation')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--head_init_scale', default=1.0, type=float,
                        help='classifier head initial scale, typically adjusted in fine-tuning')
    parser.add_argument('--model_key', default='model|module', type=str,
                        help='which key to load from saved state dict, usually model or model_ema')
    parser.add_argument('--model_prefix', default='', type=str)

    # Dataset parameters
    parser.add_argument('--data_path', default='/media/ssd/ehoffer/imagenet/', type=str,
                        help='dataset path')
    parser.add_argument('--eval_data_path', default=None, type=str,
                        help='dataset path for evaluation')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')
    parser.add_argument('--imagenet_default_mean_and_std', type=str2bool, default=True)
    parser.add_argument('--data_set', default='IMNET', choices=['CIFAR', 'CIFAR10', 'IMNET', 'IMNET200', 'image_folder'],
                        type=str, help='ImageNet dataset path')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--output_dir_rand', default=False, type=str2bool,
                        help='add random string to output_dir')

    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', type=str2bool, default=True)
    parser.add_argument('--save_ckpt', type=str2bool, default=True)
    parser.add_argument('--save_ckpt_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_num', default=1, type=int)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', type=str2bool, default=False,
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', type=str2bool, default=True,
                        help='Enabling distributed evaluation')
    parser.add_argument('--disable_eval', type=str2bool, default=False,
                        help='Disabling evaluation during training')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', type=str2bool, default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', type=str2bool, default=False)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--use_amp', type=str2bool, default=False,
                        help="Use PyTorch's AMP (Automatic Mixed Precision) or not")

    # Weights and Biases arguments
    parser.add_argument('--enable_wandb', type=str2bool, default=False,
                        help="enable logging to Weights and Biases")
    parser.add_argument('--project', default='convnext', type=str,
                        help="The name of the W&B project where you're sending the new run.")
    parser.add_argument('--wandb_ckpt', type=str2bool, default=False,
                        help="Save model checkpoints as W&B Artifacts.")
    parser.add_argument('--wandb_group', default=None, type=str, help='Group name for W&B')
    parser.add_argument('--wandb_name', default=None, type=str, help='Run name for W&B')

    parser.add_argument('--activation', type=str, default='gelu',
                        choices=['relu', 'lrelu', 'gelu', 'poly', 'poly_per_channel', 'up_gelu', 'up_relu',
                                 'up_poly', 'up_poly_per_channel', 'up_poly_rfft', 'mod_relu',
                                 'filt_poly_per_channel', 'filt_gelu',
                                 'poly_per_channel_rand',
                                 'lpf_poly_per_channel'])

    parser.add_argument('--activation_kwargs', type=str, default='{}')
    parser.add_argument('--normalization_kwargs', type=str, default='{}')

    parser.add_argument('--blurpool_kwargs', type=str, default='{}')

    # parser.add_argument('--profiler', action='store_true',
    #                     help='use torch profiler')
    parser.add_argument('--profiler', type=str, default=None,
                        choices=['forward', 'step'])
    parser.add_argument('--normalization_type', type=str, default='C',
                        choices=['C', 'CHW', 'CHW2', 'CHW3', 'CHW2_act', 'batch', 'instance', 'group', 'layer_std_groups', 'power'])

    # parser.add_argument('--fast_block', type=str2bool, default=False,
    #                     help='use fast block aal implementation')
    parser.add_argument('--max_shift', default=32, type=int)

    # parser.add_argument('--maxpool', type=str2bool, default=False,
    #                     help='Use MaxPool in beginning of ResNet. Default - disable Maxpool, using blurpool instead')

    parser.add_argument('--model_kwargs', type=str, default='{}')
    parser.add_argument('--resnet_default_aug', type=str2bool, default=False,
                        help='Use ResNet Default augmentation - like in aps paper')
    parser.add_argument('--resnet_aug', type=str, default=None,
                        choices=['random_crop', 'random_resize_crop', 'color_jitter'])

    parser.add_argument('--init_weight_std', type=float, default=0.02,
                        help='std of weight initialization')

    # parser.add_argument('--eval_shift_consistency_metrics', type=str2bool, default=False,
    #                     help='use fast block aal implementation')

    parser.add_argument('--eval_shift_consistency_metrics_x', type=str2bool, default=False,
                        help='use fast block aal implementation')

    parser.add_argument('--eval_shift_consistency_metrics_y', type=str2bool, default=False,
                        help='use fast block aal implementation')

    parser.add_argument('--eval_adversarial_shift_metrics', type=str2bool, default=False,
                        help='evaluate adversarial translations')
    parser.add_argument('--adversarial_method', type=str, default=None)
    parser.add_argument('--fraction_adversarial_max_up', type=int, default=4,
                        help='fraction_adversarial_max_up')
    parser.add_argument('--eval_high_precision', type=str2bool, default=False,
                        help='use fp64 for evaluation')

    parser.add_argument('--eval_rand_shift', type=str2bool, default=False,
                        help='evaluate random translations. integer / frac according to following args')
    parser.add_argument('--eval_rand_shift_up', type=int, default=1,
                        help='upsample factor for reval_random_shift fractional shifts')
    parser.add_argument('--eval_rand_shift_up_method', type=str, default='ideal',
                        help='upsample type for reval_random_shift fractional shifts')

    parser.add_argument('--eval_benchmark', type=str2bool, default=False,
                        help='measure evaluation time')

    parser.add_argument('--shift_jump_size', default=8, type=int)
    parser.add_argument('--min_shift', default=32, type=int)

    # stem config
    parser.add_argument('--stem_activation_kwargs', type=str, default='{}')

    # ImageNetC
    parser.add_argument('--eval_imagenet_c', type=str2bool, default=False,
                        help='evaluate model on ImageNet-C')
    parser.add_argument('--eval_imagenet_c_adversarial_shift', type=str2bool, default=False,
                        help='evaluate adversarial translations')
    parser.add_argument('--eval_imagenet_c_random_shift', type=str2bool, default=False,
                        help='evaluate model on ImageNet-C with translations')
    parser.add_argument('--imagenet_c_distortion', type=str, default=None,
                        help='evaluate model on ImageNet-C with translations - specific distortion, default all')
    parser.add_argument('--imagenet_c_severity', type=int, default=None,
                        help='evaluate model on ImageNet-C with translations - specific severity, default all')
    parser.add_argument('--imagenet_c_path', type=str, default='/home/hagaymi/data/imagenet-c')

    parser.add_argument('--eval_crop_shift', type=int, default=None, help='evaluate adversarial crop shift')

    parser.add_argument('--config_yaml', default=None, type=str, help="Path to config yaml file instead of giving all other arguments")

    # # for wandb sweep
    # parser.add_argument('--poly_in_scale', default=None, type=float)
    # parser.add_argument('--poly_out_scale', default=None, type=float)

    args = parser.parse_args()
    print(f"command line args: {args}")
    args.activation_kwargs = json.loads(args.activation_kwargs)
    args.blurpool_kwargs = json.loads(args.blurpool_kwargs)
    args.model_kwargs = json.loads(args.model_kwargs)
    args.normalization_kwargs = json.loads(args.normalization_kwargs)
    # stemconfig
    args.stem_activation_kwargs = json.loads(args.stem_activation_kwargs)

    if (args.config_yaml is not None):
        print(f"loading config from yaml file {args.config_yaml}")
        yaml_path = args.config_yaml
        if (not os.path.exists(yaml_path)):
            print("Given config path is illegal")
            exit(1)
        with open(yaml_path) as YAML:
            config_dict = yaml.load(YAML, Loader=yaml.FullLoader)
        # yaml_args = SimpleNamespace(**config_dict)
        # args.update(yaml_args)

        # yaml_args = argparse.Namespace(**config_dict)
        # args.__dict__.update(yaml_args.__dict__)
        print(f"yaml config dict: {config_dict}")
        args.__dict__.update(config_dict)


    if args.aa == 'None':
        args.aa = None

    if args.output_dir:
        if args.output_dir_rand:
            args.output_dir = os.path.join(args.output_dir, gen_rand_str(6))
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # # added for wandb sweep
    # if args.poly_in_scale is not None:
    #     args.activation_kwargs['in_scale'] = args.poly_in_scale
    # if args.poly_out_scale is not None:
    #     args.activation_kwargs['out_scale'] = args.poly_out_scale


    print(f"args: {args}")
    return args


def get_model(args, pretrained=False, total_steps=0):
    if args.model == 'convnext_tiny':
        from models.convnext import convnext_tiny
        model = convnext_tiny(
                pretrained=pretrained,
                num_classes=args.nb_classes,
                drop_path_rate=args.drop_path,
                layer_scale_init_value=args.layer_scale_init_value,
                head_init_scale=args.head_init_scale,
                )

    elif args.model == 'convnext_poly_tiny':
        from models.convnext_poly import convnext_tiny
        model = convnext_tiny(
                pretrained=pretrained,
                num_classes=args.nb_classes,
                drop_path_rate=args.drop_path,
                layer_scale_init_value=args.layer_scale_init_value,
                head_init_scale=args.head_init_scale,
                activation_kwargs=args.activation_kwargs,
                activation=args.activation
                )


    elif args.model == 'convnext_circ_tiny':
        from models.convnext_circ import convnext_tiny
        model = convnext_tiny(
            pretrained=pretrained,
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            layer_scale_init_value=args.layer_scale_init_value,
            head_init_scale=args.head_init_scale,
        )

    elif args.model == 'convnext_afc_tiny':
        from models.convnext_afc import convnext_afc_tiny
        model = convnext_afc_tiny(
            pretrained=pretrained,
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            layer_scale_init_value=args.layer_scale_init_value,
            head_init_scale=args.head_init_scale,
            activation=args.activation,
            activation_kwargs=args.activation_kwargs,
            blurpool_kwargs=args.blurpool_kwargs,
            normalization_type=args.normalization_type,
            init_weight_std=args.init_weight_std,
            stem_activation_kwargs=args.stem_activation_kwargs,
            normalization_kwargs=args.normalization_kwargs,
            **args.model_kwargs

        )

    elif args.model == 'convnext_aps_tiny':
        from models.convnext_aps import convnext_aps_tiny
        model = convnext_aps_tiny(
            pretrained=pretrained,
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            layer_scale_init_value=args.layer_scale_init_value,
            head_init_scale=args.head_init_scale,
            activation=args.activation,
            activation_kwargs=args.activation_kwargs,
            blurpool_kwargs=args.blurpool_kwargs,
            normalization_type=args.normalization_type,
            init_weight_std=args.init_weight_std,
            **args.model_kwargs
        )
    elif args.model == 'convnext_rep_afc_tiny':
        from models.convnext_rep_afc import convnext_rep_afc_tiny
        model = convnext_rep_afc_tiny(
                        pretrained=pretrained,
                        num_classes=args.nb_classes,
                        drop_path_rate=args.drop_path,
                        layer_scale_init_value=args.layer_scale_init_value,
                        head_init_scale=args.head_init_scale,
                        activation=args.activation,
                        activation_kwargs=args.activation_kwargs,
                        blurpool_kwargs=args.blurpool_kwargs,
                        normalization_type=args.normalization_type,
                        init_weight_std=args.init_weight_std,
                        rep_steps=total_steps,
                        **args.model_kwargs)

    return model

def profiler_test(model, args):
    def get_device():
        # setting device on GPU if available, else CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', device)
        print()

        #Additional Info when using cuda
        cuda_mem(device)
        return device


    def cuda_mem(device):
        print('CUDA Memory Usage:')
        if device.type == 'cuda':
            print(torch.cuda.get_device_name(0))
            print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
        else:
            print("Not cuda device")


    def measure_time(func, args):#, device):
        # time = []
        for i in range(1):
            # inp = torch.rand(*inp_shape).to(device)
            _ = func(*args)
            with torch.autograd.profiler.profile(use_cuda=True) as prof:
                res = func(*args)
            print("sort by cpu_time_total:")
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
            print("sort by cuda_time_total:")
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    device = get_device()
    model.to(device)
    if args.profiler == 'step':
        criterion = torch.nn.CrossEntropyLoss().to(device)
        def func(x, y):
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()

    elif args.profiler == 'forward':
        def func(x, y):
            pred = model(x)

    else:
        assert False, f'Unexpected profiler test type {args.profiler}'
    print("model:", model)
    for bs in [1,2,4,8,16,32,64]:
        print("batch size: {}".format(bs))
        x = torch.randn(bs, 3, args.input_size, args.input_size, device=device)
        y = torch.randint(0, args.nb_classes, (bs,), device=device)

        measure_time(func, (x,y))#, device)
