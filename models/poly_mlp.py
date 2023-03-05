import torch
import torch.nn as nn
from .activation import get_activation, PolyActPerChannel
from .ideal_lpf import UpsampleRFFT, LPF_RFFT

class MLP(nn.Module):
    def __init__(self, dim, expand_ratio, activation, activation_kwargs={}):
        super(MLP, self).__init__()
        self.pwconv1 = nn.Linear(dim, expand_ratio * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = get_activation(activation, channels=expand_ratio * dim, data_format='channels_last', **activation_kwargs)
        self.pwconv2 = nn.Linear(expand_ratio * dim, dim)

    def forward(self, x):
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return x


class AALMLP(nn.Module):
    def __init__(self, dim, expand_ratio,  activation_kwargs={}):
        '''channels last'''
        super(AALMLP, self).__init__()
        transform_mode = activation_kwargs.pop('transform_mode', 'rfft')
        self.upsample = UpsampleRFFT(2, transform_mode=transform_mode)
        self.pwconv1 = nn.Linear(dim, expand_ratio * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = PolyActPerChannel(expand_ratio * dim, data_format='channels_last', **activation_kwargs)
        self.pwconv2 = nn.Linear(expand_ratio * dim, dim)
        self.lpf = LPF_RFFT(cutoff=0.5, transform_mode=transform_mode)


    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = self.upsample(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = self.lpf(x)
        x = x[:,:, ::2,::2]
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        return x
