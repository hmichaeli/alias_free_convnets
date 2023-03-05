# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# convnext_LICENSE.txt file in the root directory of this source tree.


import math
from typing import Iterable, Optional
import torch
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
from tqdm import tqdm
import time
import numpy as np

import utils
from models.ideal_lpf import subpixel_shift


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    wandb_logger=None, start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, use_amp=False):


    torch.cuda.empty_cache()

    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    optimizer.zero_grad()

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(samples)
                loss = criterion(output, targets)
        else: # full precision
            output = model(samples)
            loss = criterion(output, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value): # this could trigger if using AMP
            print("Loss is {}, stopping training".format(loss_value))
            assert math.isfinite(loss_value)

        if use_amp:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
        else: # full precision
            loss /= update_freq
            loss.backward()
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.step()
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)

        if str(device) != 'cpu':
            torch.cuda.synchronize()

        if mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        if use_amp:
            metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            if use_amp:
                log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

        if wandb_logger:
            wandb_logger._wandb.log({
                'Rank-0 Batch Wise/train_loss': loss_value,
                'Rank-0 Batch Wise/train_max_lr': max_lr,
                'Rank-0 Batch Wise/train_min_lr': min_lr
            }, commit=False)
            if class_acc:
                wandb_logger._wandb.log({'Rank-0 Batch Wise/train_class_acc': class_acc}, commit=False)
            if use_amp:
                wandb_logger._wandb.log({'Rank-0 Batch Wise/train_grad_norm': grad_norm}, commit=False)
            wandb_logger._wandb.log({'Rank-0 Batch Wise/global_train_step': it})
            

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, use_amp=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        #
        shift0 = np.random.randint(-32, 32, size=2)
        shifted_inp0 = torch.roll(images, shifts=(shift0[0], shift0[1]), dims=(2, 3))
        # shift_subpix = IdealUpsample(2)(images)[:, :, 1::2, 1::2]

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)
        else:
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

        # #  shift consistency - original vs random shift
        shifted_inp0 = shifted_inp0.to(device, non_blocking=True)
        if use_amp:
            with torch.cuda.amp.autocast():
                output_shifted_inp0 = model(shifted_inp0)

        else:
            output_shifted_inp0 = model(shifted_inp0)

        # measure agreement and record
        cur_agree = agreement(output, output_shifted_inp0).type(torch.FloatTensor).to(output.device)
        metric_logger.meters['consist'].update(cur_agree.item(), n=batch_size)

        # Subpixel shift consistency - removed to reduce evaluation time
        # shift_subpix = shift_subpix.to(device, non_blocking=True)
        # print(batch, shift_subpix.shape)
        # if use_amp:
        #     with torch.cuda.amp.autocast():
        #         output_shifted_inp0 = model(shift_subpix)
        # else:
        #     output_shifted_inp0 = model(shift_subpix)
        #
        # # measure agreement and record
        # cur_agree = agreement(output, output_shifted_inp0).type(torch.FloatTensor).to(output.device)
        # metric_logger.meters['subpix_consist'].update(cur_agree.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_consistency_metrics(data_loader, model, device, use_amp=False, max_shift=32, shift=None):
    '''

    :param data_loader:
    :param model:
    :param device:
    :param use_amp:
    :param max_shift:
    :param shift: tuple (shift_x, shift_y). if None, use random shift in range (-max_shift, max_shift)
    :return:
    '''
    criterion = torch.nn.CrossEntropyLoss()
    num_batches = len(data_loader)

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test_consistency_metrics:'

    def forwrad(images, target):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)
        else:
            output = model(images)
            loss = criterion(output, target)
        # images = images.cpu()

        return output, loss

    # switch to evaluation mode
    model.eval()
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]

        if shift is None:
            shift = np.random.randint(-max_shift, max_shift, 2)

        shifted_images = torch.roll(images, shifts=(shift[0], shift[1]), dims=(2, 3))
        # sub_pix_shifted_images = subpixel_shift(images)  # shift 0.5 pixel
        sub_pix_shifted_images = subpixel_shift(images, up=2,
                                                shift_x=int(2*(shift[0]+0.5)),
                                                shift_y=int(2 * (shift[1] + 0.5)))  # shift + 0.5 pixel

        # regular images
        output_labels, loss = forwrad(images, target)
        target = target.to(output_labels.device)
        acc1, acc5 = accuracy(output_labels, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)


        shifted_output_labels, _ = forwrad(shifted_images, target)

        consistent_labels = (torch.argmax(output_labels, dim=1) == torch.argmax(shifted_output_labels, dim=1)).tolist()
        consistency = np.sum(consistent_labels) / len(consistent_labels)

        pred1 = torch.gather(torch.softmax(output_labels, dim=1), 1,
                             torch.argmax(output_labels, 1, keepdim=True)).detach().cpu().numpy()
        pred2 = torch.gather(torch.softmax(shifted_output_labels, dim=1), 1,
                             torch.argmax(output_labels, 1, keepdim=True)).detach().cpu().numpy()
        mac = np.sum(np.abs(pred1 - pred2)) / pred1.shape[0]  # Mean absolute change

        tmp_accurate_labels = (torch.argmax(shifted_output_labels, dim=1) == target).tolist()
        tmp_accuracy_top1 = 100 * np.sum(tmp_accurate_labels) / len(tmp_accurate_labels)


        metric_logger.meters['shift_consistency'].update(consistency.item(), n=batch_size)
        metric_logger.meters['shift_mac'].update(mac.item(), n=batch_size)
        metric_logger.meters['shift_accuracy'].update(tmp_accuracy_top1.item(), n=batch_size)

        sub_pix_shifted_output_labels, _ = forwrad(sub_pix_shifted_images, target)

        consistent_labels = (
                    torch.argmax(output_labels, dim=1) == torch.argmax(sub_pix_shifted_output_labels, dim=1)).tolist()

        consistency = np.sum(consistent_labels) / len(consistent_labels)

        pred1 = torch.gather(torch.softmax(output_labels, dim=1), 1,
                             torch.argmax(output_labels, 1, keepdim=True)).detach().cpu().numpy()
        pred2 = torch.gather(torch.softmax(sub_pix_shifted_output_labels, dim=1), 1,
                             torch.argmax(output_labels, 1, keepdim=True)).detach().cpu().numpy()
        mac = np.sum(np.abs(pred1 - pred2)) / pred1.shape[0]  # Mean absolute change

        tmp_accurate_labels = (torch.argmax(sub_pix_shifted_output_labels, dim=1) == target).tolist()
        tmp_accuracy_top1 = 100 * np.sum(tmp_accurate_labels) / len(tmp_accurate_labels)


        metric_logger.meters['sub_pix_shift_consistency'].update(consistency.item(), n=batch_size)
        metric_logger.meters['sub_pix_shift_mac'].update(mac.item(), n=batch_size)
        metric_logger.meters['sub_pix_shift_accuracy'].update(tmp_accuracy_top1.item(), n=batch_size)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    res_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    res_dict['shift_accuracy_degredation'] = 100 * (res_dict['acc1'] - res_dict['shift_accuracy']) / res_dict['acc1']
    res_dict['subpix_shift_accuracy_degredation'] = 100 * (res_dict['acc1'] - res_dict['sub_pix_shift_accuracy']) / res_dict['acc1']

    return res_dict



def agreement(output0, output1):
    pred0 = output0.argmax(dim=1, keepdim=False)
    pred1 = output1.argmax(dim=1, keepdim=False)
    agree = pred0.eq(pred1)
    agree = 100. * torch.mean(agree.type(torch.FloatTensor).to(output0.device))
    return agree


@torch.no_grad()
def evaluate_benchmark(data_loader, model, device, use_amp=False):
    num_batches = len(data_loader)
    batch_size = next(iter(data_loader))[0].shape[0]
    print("evaluate benchmark")
    print(f"num_batches: {num_batches}  batch_size: {batch_size}")
    # switch to evaluation mode
    model.eval()
    start = time.time()
    for batch in tqdm(data_loader):
        images = batch[0]
        images = images.to(device, non_blocking=True)
        # compute output
        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(images)

        else:
            output = model(images)

        pred = torch.argmax(output)

    end = time.time()
    elapsed_time = end - start

    print("number of batches: ", num_batches)
    print("batch size: ", batch_size)
    print("total evaluation time: ", elapsed_time)
    print("time per batch: ", elapsed_time / num_batches)
    print("time per sample: ", elapsed_time / (num_batches * batch_size))