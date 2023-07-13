# Code originally from https://github.com/adobe/antialiased-cnns
# Copyright 2019 Adobe. All rights reserved.
## Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International

import torch
import torch.nn.parallel
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .ideal_lpf import LPF_RFFT as IdealLPF
from .circular_pad_layer import circular_pad

class BlurPool(nn.Module):
    def __init__(self, channels, pad_type='circular', filt_size=1, stride=2, pad_off=0,
                 filter_type='basic', cutoff=0.5, scale_l2=False, eps=1e-6, transform_mode='rfft'
                 ):
        super(BlurPool, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)), int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
        self.pad_sizes = [pad_size+pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride-1)/2.)
        self.channels = channels
        self.pad_type = pad_type
        self.filter_type = filter_type
        self.scale_l2 = scale_l2
        self.eps = eps

        if filter_type == 'ideal':
            self.filt = IdealLPF(cutoff=cutoff, transform_mode=transform_mode)

        elif filter_type == 'basic':
            a = self.get_rect(self.filt_size)


        if filter_type == 'basic':
            filt = torch.Tensor(a[:, None] * a[None, :])
            filt = filt / torch.sum(filt)
            self.filt = Filter(filt, channels, pad_type, self.pad_sizes, scale_l2)
            if self.filt_size == 1 and self.pad_off == 0:
                self.pad = get_pad_layer(pad_type)(self.pad_sizes)


    def forward(self, inp):
        if self.filter_type == 'ideal':
            if self.scale_l2:
                inp_norm = torch.norm(inp, p=2, dim=(-1, -2), keepdim=True)
            out = self.filt(inp)
            if self.scale_l2:
                out_norm = torch.norm(out, p=2, dim=(-1, -2), keepdim=True)
                out = out * (inp_norm / (out_norm + self.eps))
            return out[:, :, ::self.stride, ::self.stride]

        elif self.filt_size == 1:
            if self.pad_off == 0:
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]

        else:
            return self.filt(inp)[:, :, ::self.stride, ::self.stride]

    @staticmethod
    def get_rect(filt_size):
        if filt_size == 1:
            a = np.array([1., ])
        elif filt_size == 2:
            a = np.array([1., 1.])
        elif filt_size == 3:
            a = np.array([1., 2., 1.])
        elif filt_size == 4:
            a = np.array([1., 3., 3., 1.])
        elif filt_size == 5:
            a = np.array([1., 4., 6., 4., 1.])
        elif filt_size == 6:
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif filt_size == 7:
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        return a


    def __repr__(self):
        return f"BlurPool(channels={self.channels}, pad_type={self.pad_type}, " \
               f" stride={self.stride}, filter_type={self.filter_type},  filt_size={self.filt_size}, " \
               f"scale_l2={self.scale_l2})"


class Filter(nn.Module):
    def __init__(self, filt, channels, pad_type=None, pad_sizes=None, scale_l2=False, eps=1e-6):
        super(Filter, self).__init__()
        self.register_buffer('filt', filt[None, None, :, :].repeat((channels, 1, 1, 1)))
        if pad_sizes is not None:
            self.pad = get_pad_layer(pad_type)(pad_sizes)
        else:
            self.pad = None
        self.scale_l2 = scale_l2
        self.eps = eps

    def forward(self, x):
        if self.scale_l2:
            inp_norm = torch.norm(x, p=2, dim=(-1, -2), keepdim=True)
        if self.pad is not None:
            x = self.pad(x)
        out = F.conv2d(x, self.filt, groups=x.shape[1])
        if self.scale_l2:
            out_norm = torch.norm(out, p=2, dim=(-1, -2), keepdim=True)
            out = out * (inp_norm / (out_norm + self.eps))
        return out


def get_pad_layer(pad_type):
    if pad_type in ['refl','reflect']:
        PadLayer = nn.ReflectionPad2d
    elif pad_type in ['repl','replicate']:
        PadLayer = nn.ReplicationPad2d
    elif pad_type=='zero':
        PadLayer = nn.ZeroPad2d
    elif pad_type == 'circular':
        PadLayer = circular_pad
        
    else:
        print('Pad type [%s] not recognized'%pad_type)
    return PadLayer

class BlurPool1D(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=3, stride=2, pad_off=0):
        super(BlurPool1D, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        # print('Filter size [%i]' % filt_size)
        if(self.filt_size == 1):
            a = np.array([1., ])
        elif(self.filt_size == 2):
            a = np.array([1., 1.])
        elif(self.filt_size == 3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size == 4):
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size == 5):
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size == 6):
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size == 7):
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a)
        filt = filt / torch.sum(filt)
        self.register_buffer('filt', filt[None, None, :].repeat((self.channels, 1, 1)))

        self.pad = get_pad_layer_1d(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size == 1):
            if(self.pad_off == 0):
                return inp[:, :, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride]
        else:
            return F.conv1d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])




def get_pad_layer_1d(pad_type):
    if pad_type in ['refl', 'reflect']:
        PadLayer = nn.ReflectionPad1d
    elif pad_type in ['repl', 'replicate']:
        PadLayer = nn.ReplicationPad1d
    elif pad_type == 'zero':
        PadLayer = nn.ZeroPad1d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer

