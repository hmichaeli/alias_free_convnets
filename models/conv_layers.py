from doctest import UnexpectedException
import torch
import torch.nn as nn

from .circular_pad_layer import circular_pad


# support other pad types - copied from cifar model
def pad_layer(pad_type, padding):
    if pad_type == 'zero':
        padding = nn.ZeroPad2d(padding)
    elif pad_type == 'reflect':
        padding = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate_pad':
        padding = nn.ReplicationPad2d(padding)
    elif pad_type == 'circular':
        padding = circular_pad(padding)
    else:
        assert False, "pad type {} not supported".format(pad_type)
    return padding


def conv3x3(in_planes, out_planes, stride=1, groups=1, bias=True, conv_pad_type='zeros'):
    """3x3 convolution with padding"""

    padding = pad_layer(conv_pad_type, [1, 1, 1, 1])

    return nn.Sequential(padding,
                         nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                                   padding=0, groups=groups, bias=bias))


def conv2x2(in_planes, out_planes, padding=0, stride=1, groups=1, bias=True, conv_pad_type='zeros'):
    """3x3 convolution with padding"""
    if padding != 0:
        if type(padding) == int:
            padding = pad_layer(conv_pad_type, [padding, padding, padding, padding])
        elif type(padding) == list and len(padding) == 4:
            padding = pad_layer(conv_pad_type, padding)
        else:
            raise UnexpectedException
    else:
        padding = nn.Identity()

    return nn.Sequential(padding,
                         nn.Conv2d(in_planes, out_planes, kernel_size=2, stride=stride,
                                   padding=0, groups=groups, bias=bias))


def conv4x4(in_planes, out_planes, padding=0, stride=1, groups=1, bias=True, conv_pad_type='zeros'):
    """3x3 convolution with padding"""
    if padding != 0:
        if type(padding) == int:
            padding = pad_layer(conv_pad_type, [padding, padding, padding, padding])
        elif type(padding) == list and len(padding) == 4:
            padding = pad_layer(conv_pad_type, padding)
        else:
            raise UnexpectedException
    else:
        padding = nn.Identity()

    return nn.Sequential(padding,
                         nn.Conv2d(in_planes, out_planes, kernel_size=4, stride=stride,
                                   padding=0, groups=groups, bias=bias))


def conv7x7(in_planes, out_planes, padding=0, stride=1, groups=1, bias=True, conv_pad_type='zeros'):
    """3x3 convolution with padding"""
    if padding > 0:
        padding = pad_layer(conv_pad_type, [padding, padding, padding, padding])
    else:
        padding = nn.Identity()

    return nn.Sequential(padding,
                         nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride,
                                   padding=0, groups=groups, bias=bias))


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
