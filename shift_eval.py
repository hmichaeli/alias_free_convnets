import math
from typing import Iterable, Optional
import torch
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
from tqdm import tqdm

import utils

import numpy as np

from models.ideal_lpf import subpixel_shift



@torch.no_grad()
def evaluate_fraction_adversarial_shift_metrics(data_loader, model, device, use_amp=False, max_up=4):
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
        images = images.cpu()

        return output, loss


    # find relevant up factors to avoid duplicate
    up_factors = [i for i in range(1, max_up+1)]
    # remove from up_factor all numbers that divide higher number in the list
    up_factors = [i for i in up_factors if not any(j>i and j % i == 0 for j in up_factors)]
    print("[fraction adversarial shift] looking in up factors: ", up_factors)

    # switch to evaluation mode
    model.eval()
    for batch in metric_logger.log_every(data_loader, 2, header):
        images = batch[0]
        target = batch[-1]
        batch_size = images.shape[0]

        # regular images
        output_labels, loss = forwrad(images, target)
        target = target.to(output_labels.device)
        acc1, acc5 = accuracy(output_labels, target, topk=(1, 5))

        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

        correct = torch.argmax(output_labels, dim=1) == target
        robust_sub_shift = correct

        for up in up_factors:
            for shift_x in range(1, up):
                for shift_y in range(1, up):
                    sub_pix_shifted_images = subpixel_shift(images.to(device), up=up, shift_x=shift_x, shift_y=shift_y)
                    sub_shifted_output_labels, _ = forwrad(sub_pix_shifted_images, target)
                    correct_sub_shift = torch.argmax(sub_shifted_output_labels, dim=1) == target
                    robust_sub_shift = robust_sub_shift & correct_sub_shift



        sub_shift_robust_score = torch.sum(robust_sub_shift) / len(robust_sub_shift)


        # metric_logger.meters['shift_robust_score'].update(shift_robust_score.item(), n=batch_size)
        metric_logger.meters['sub_shift_robust_score'].update(sub_shift_robust_score.item(), n=batch_size)
    #
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5))

    print(f'sub_shift_robust_score {metric_logger.meters["sub_shift_robust_score"].global_avg:.3f}')

    res_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return res_dict




def evaluate_full_adversarial_shift_metrics(data_loader, model, device, use_amp=False, max_shift=32, half_shift=False,
                                            random_shifts=None):
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
        images = images.cpu()

        return output, loss

    # switch to evaluation mode
    model.eval()
    for batch in metric_logger.log_every(data_loader, 2, header):
        images = batch[0].to(device)
        target = batch[-1].to(device)
        batch_size = images.shape[0]

        # regular images
        output_labels, loss = forwrad(images, target)
        target = target.to(output_labels.device)
        acc1, acc5 = accuracy(output_labels, target, topk=(1, 5))

        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

        correct = torch.argmax(output_labels, dim=1) == target
        robust_shift = correct
        robust_sub_shift = correct

        # replace nested for loops with itertools.product
        #         for shift_x in range(1, max_shift):
        #             for shift_y in range(1, max_shift):
        if not random_shifts:
            import itertools
            shift_iterator = itertools.product(range(1, max_shift), range(1, max_shift))
        else:
            # create an iterator with 10 random shifts in range(1, max_shift)
            shift_iterator = np.random.randint(1, max_shift, size=(random_shifts, 2))


        for shift_x, shift_y in shift_iterator:
                shifted_images = torch.roll(images, shifts=(shift_x, shift_y), dims=(2, 3))
                shifted_output_labels, _ = forwrad(shifted_images, target)
                correct_shift = torch.argmax(shifted_output_labels, dim=1) == target
                robust_shift = robust_shift & correct_shift

                if half_shift:
                    # sub_pix_shifted_images = subpixel_shift(images, shift_x=shift-1, shift_y=shift-1)  # shift 0.5 pixel
                    sub_pix_shifted_images = subpixel_shift(images.to(device), up=2, shift_x=2 * shift_x - 1, shift_y=2 * shift_y - 1)  # shift 0.5 pixel
                    sub_shifted_output_labels, _ = forwrad(sub_pix_shifted_images, target)
                    correct_sub_shift = torch.argmax(sub_shifted_output_labels, dim=1) == target
                    robust_sub_shift = robust_sub_shift & correct_sub_shift

        shift_robust_score = torch.sum(robust_shift) / len(robust_shift)
        metric_logger.meters['shift_robust_score'].update(shift_robust_score.item(), n=batch_size)
        if half_shift:
            robust_total = robust_shift & robust_sub_shift
            sub_shift_robust_score = torch.sum(robust_sub_shift) / len(robust_sub_shift)
            total_robust_score = torch.sum(robust_total) / len(robust_total)
            metric_logger.meters['sub_shift_robust_score'].update(sub_shift_robust_score.item(), n=batch_size)
            metric_logger.meters['total_robust_score'].update(total_robust_score.item(), n=batch_size)
    #
    #
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5))

    print(f'* shift_robust_score {metric_logger.meters["shift_robust_score"].global_avg:.3f} \n')
    if half_shift:
        print(f'sub_shift_robust_score {metric_logger.meters["sub_shift_robust_score"].global_avg:.3f} \n'
          f'total_robust_score {metric_logger.meters["total_robust_score"].global_avg:.3f}')

    res_dict = {f"full_adversarial_{k}": meter.global_avg for k, meter in metric_logger.meters.items()}
    return res_dict


def evaluate_adversarial_crop_shift_metrics(data_loader, model, device, inp_size, max_shift, use_amp=False):
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
        assert images.shape[2] == images.shape[3], "images should be square"
        assert images.shape[2] == inp_size, f"images.shape[2] = {images.shape[2]} != {inp_size}"
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
        images = images.cpu()

        return output, loss

    # switch to evaluation mode
    model.eval()
    for batch in metric_logger.log_every(data_loader, 2, header):
        expanded_images = batch[0]#.to(device) # size inp + 2 * max_shift
        target = batch[-1].to(device)
        batch_size = expanded_images.shape[0]

        # regular images
        images = expanded_images[:, :, max_shift:inp_size+max_shift, max_shift:inp_size+max_shift]
        output_labels, loss = forwrad(images, target)
        target = target.to(output_labels.device)
        acc1, acc5 = accuracy(output_labels, target, topk=(1, 5))

        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

        correct = torch.argmax(output_labels, dim=1) == target
        robust_shift = correct

        import itertools
        shift_iterator = itertools.product(range(1, 2 * max_shift + 1), range(1, 2 * max_shift + 1))

        for shift_x, shift_y in shift_iterator:
            # crop image at inp_size, starting from shift_x, shift_y
            shifted_images = expanded_images[:, :, shift_x: inp_size+shift_x, shift_y:inp_size+shift_y]
            shifted_output_labels, _ = forwrad(shifted_images, target)
            correct_shift = torch.argmax(shifted_output_labels, dim=1) == target
            robust_shift = robust_shift & correct_shift


        shift_robust_score = 100 * torch.sum(robust_shift) / len(robust_shift)
        metric_logger.meters['shift_robust_score'].update(shift_robust_score.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5))

    print(f'* adversarail_crop_shift_robust_score {metric_logger.meters["shift_robust_score"].global_avg:.3f} \n')

    res_dict = {f"adversarial_crop_shift_{k}": meter.global_avg for k, meter in metric_logger.meters.items()}
    return res_dict


@torch.no_grad()
def evaluate_random_shift_metrics(data_loader, model, device, use_amp=False, max_shift=32, up=1, up_method='ideal'):
    '''
        if up > 1 - perform fractional shifts, multiplication of  (1 / up)
        upsample scheme - according to "up_method"
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
        images = images.cpu()

        return output, loss

    # switch to evaluation mode
    model.eval()
    for batch in metric_logger.log_every(data_loader, 2, header):
        images = batch[0].to(device)
        target = batch[-1].to(device)
        batch_size = images.shape[0]

        # regular images
        output_labels, loss = forwrad(images, target)
        target = target.to(output_labels.device)
        acc1, acc5 = accuracy(output_labels, target, topk=(1, 5))

        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

        # rand shifts
        rand_shift = np.random.randint(1, max_shift, size=2)
        if up == 1:
            # integer shifts
            shifted_images = torch.roll(images, shifts=(rand_shift[0], rand_shift[1]), dims=(2, 3))
            shifted_output_labels, _ = forwrad(shifted_images, target)

        #dev - evaluate integer with upsample
        if up == 0:
            # integer shifts
            # shifted_images = torch.roll(images, shifts=(rand_shift[0], rand_shift[1]), dims=(2, 3))
            # shifted_output_labels, _ = forwrad(shifted_images, target)
            rand_shift = 2 * rand_shift
            sub_pix_shifted_images = subpixel_shift(images, up=2, shift_x=rand_shift[0], shift_y=rand_shift[1],
                                                    up_method=up_method)
            shifted_output_labels, _ = forwrad(sub_pix_shifted_images, target)


        else:
            # translate shift to (1/up) multiplication
            # e.g up=3, s -> (3 * s + 1) / 3 or (3 * s + 2) / 3
            frac_shift = up * rand_shift + np.random.randint(1, up, size=2)
            sub_pix_shifted_images = subpixel_shift(images, up=up, shift_x=frac_shift[0], shift_y=frac_shift[1],
                                                    up_method=up_method)
            shifted_output_labels, _ = forwrad(sub_pix_shifted_images, target)

        correct_shift = (torch.argmax(shifted_output_labels, dim=1) == target).tolist()
        shift_accuracy = 100 * np.sum(correct_shift) / len(correct_shift)

        consistent_labels = (torch.argmax(output_labels, dim=1) == torch.argmax(shifted_output_labels, dim=1)).tolist()
        consistency = 100 * np.sum(consistent_labels) / len(consistent_labels)

        pred1 = torch.gather(torch.softmax(output_labels, dim=1), 1,
                             torch.argmax(output_labels, 1, keepdim=True)).detach().cpu().numpy()
        pred2 = torch.gather(torch.softmax(shifted_output_labels, dim=1), 1,
                             torch.argmax(output_labels, 1, keepdim=True)).detach().cpu().numpy()
        mac = np.sum(np.abs(pred1 - pred2)) / pred1.shape[0]  # Mean absolute change

        metric_logger.meters['shift_consistency'].update(consistency.item(), n=batch_size)
        metric_logger.meters['shift_mac'].update(mac.item(), n=batch_size)
        metric_logger.meters['shift_accuracy'].update(shift_accuracy.item(), n=batch_size)

    #
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    res_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    res_dict['shift_accuracy_degredation'] = 100 * (res_dict['acc1'] - res_dict['shift_accuracy']) / res_dict['acc1']

    print(res_dict)
    return res_dict

