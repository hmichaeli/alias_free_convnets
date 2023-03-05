import torch
import torch.nn as nn
from .ideal_lpf import UpsampleRFFT, LPF_RFFT

def get_activation(activation, channels=None, data_format="channels_first", **kwargs):
    if data_format not in ["channels_last", "channels_first"]:
        raise NotImplementedError

    if activation == 'relu':
        return nn.ReLU()

    elif activation == 'gelu':
        return nn.GELU()

    elif activation == 'poly':
        return PolyAct(**kwargs)

    elif activation == 'up_poly':
        return UpPolyAct(data_format=data_format,  **kwargs)

    elif activation == 'poly_per_channel':
        return PolyActPerChannel(channels, data_format=data_format, **kwargs)

    elif activation == 'up_poly_per_channel':
        return UpPolyActPerChannel(channels, data_format=data_format,  **kwargs)

    elif activation == 'lpf_poly_per_channel':
        return LPFPolyActPerChannel(channels=channels, data_format=data_format, **kwargs)

    else:
        assert False, "Un implemented activation {}".format(activation)


class UpPolyAct(nn.Module):
    def __init__(self, up=2, data_format="channels_first", **kwargs):
        super(UpPolyAct, self).__init__()
        self.up = up
        # self.lpf = IdealLPF(cutoff=1/up)
        # self.upsample = IdealUpsample(up)
        self.lpf = LPF_RFFT(cutoff=1/up)
        self.upsample = UpsampleRFFT(up)
        self.data_format = data_format

        self.pact = PolyAct(**kwargs)

    def forward(self, x):
        if self.data_format == 'channels_last':
            inp = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        out = self.upsample(inp)
        out = self.pact(out)
        out = self.lpf(out)
        out = out[:,:,::self.up, ::self.up]

        if self.data_format == 'channels_last':
            out = out.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)

        return out


class PolyAct(nn.Module):
    def __init__(self, trainable=False, init_coef=None, in_scale=1,
                 out_scale=1,
                 train_scale=False):
        super(PolyAct, self).__init__()
        if init_coef is None:
            init_coef = [0.0169394634313126, 0.5, 0.3078363963999393, 0.0]
        self.deg = len(init_coef) - 1
        self.trainable = trainable
        coef = torch.Tensor(init_coef)
        
        if trainable:
            self.coef = nn.Parameter(coef, requires_grad=True)
        else:
            self.register_buffer('coef', coef)

        if train_scale:
            self.in_scale = nn.Parameter(torch.tensor([in_scale * 1.0]), requires_grad=True)
            self.out_scale = nn.Parameter(torch.tensor([out_scale * 1.0]), requires_grad=True)

        else:
            if in_scale != 1:
                self.register_buffer('in_scale', torch.tensor([in_scale * 1.0]))
            else:
                self.in_scale = None

            if out_scale != 1:
                self.register_buffer('out_scale', torch.tensor([out_scale * 1.0]))
            else:
                self.out_scale = None

        
    def forward(self, x):
        if self.in_scale is not None:
            x = self.in_scale * x

        x = self.calc_polynomial(x)

        if self.out_scale is not None:
            x = self.out_scale * x

        return x

    def __repr__(self):
        print_coef = self.coef.cpu().detach().numpy()
        return "PolyAct(trainable={}, coef={})".format(
            self.trainable, print_coef)

    def calc_polynomial(self, x):
        res = self.coef[0] + self.coef[1] * x
        for i in range(2, self.deg):
            res = res + self.coef[i] * (x ** i)

        return res


class PolyActPerChannel(nn.Module):
    def __init__(self, channels, init_coef=None, data_format="channels_first", in_scale=1,
                 out_scale=1,
                 train_scale=False):
        super(PolyActPerChannel, self).__init__()
        self.channels = channels
        if init_coef is None:
            init_coef = [0.0169394634313126, 0.5, 0.3078363963999393]
        self.deg = len(init_coef) - 1
        coef = torch.Tensor(init_coef)
        coef = coef.repeat([channels, 1])
        coef = torch.unsqueeze(torch.unsqueeze(coef, -1), -1)
        self.coef = nn.Parameter(coef, requires_grad=True)

        if train_scale:
            self.in_scale = nn.Parameter(torch.tensor([in_scale * 1.0]), requires_grad=True)
            self.out_scale = nn.Parameter(torch.tensor([out_scale * 1.0]), requires_grad=True)

        else:
            if in_scale != 1:
                self.register_buffer('in_scale', torch.tensor([in_scale * 1.0]))
            else:
                self.in_scale = None

            if out_scale != 1:
                self.register_buffer('out_scale', torch.tensor([out_scale * 1.0]))
            else:
                self.out_scale = None

        self.data_format = data_format

    def forward(self, x):
        if self.data_format == 'channels_last':
            x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        if self.in_scale is not None:
            x = self.in_scale * x

        x = self.calc_polynomial(x)

        if self.out_scale is not None:
            x = self.out_scale * x

        if self.data_format == 'channels_last':
            x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)

        return x

    def __repr__(self):
        # print_coef = self.coef.cpu().detach().numpy()
        print_in_scale = self.in_scale.cpu().detach().numpy() if self.in_scale else None
        print_out_scale = self.out_scale.cpu().detach().numpy() if self.out_scale else None

        return "PolyActPerChannel(channels={}, in_scale={}, out_scale={})".format(
            self.channels, print_in_scale, print_out_scale)

    def calc_polynomial(self, x):

        if self.deg == 2:
            # maybe this is faster?
            res = self.coef[:, 0] + self.coef[:, 1] * x + self.coef[:, 2] * (x ** 2)
        else:
            res = self.coef[:, 0] + self.coef[:, 1] * x
            for i in range(2, self.deg):
                res = res + self.coef[:, i] * (x ** i)

        return res



class UpPolyActPerChannel(nn.Module):
    def __init__(self, channels, up=2, data_format="channels_first", transform_mode='rfft', **kwargs):
        super(UpPolyActPerChannel, self).__init__()
        self.up = up
        self.lpf = LPF_RFFT(cutoff=1/up, transform_mode=transform_mode)
        self.upsample = UpsampleRFFT(up, transform_mode=transform_mode)
        self.data_format = data_format

        self.pact = PolyActPerChannel(channels, **kwargs)

    def forward(self, x):
        if self.data_format == 'channels_last':
            x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        out = self.upsample(x)
        # print("[UpPolyActPerChannel] up: ", out)
        out = self.pact(out)
        # print("[UpPolyActPerChannel] pact: ", out)
        out = self.lpf(out)
        out = out[:,:,::self.up, ::self.up]

        if self.data_format == 'channels_last':
            out = out.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)

        return out

class LPFPolyActPerChannel(nn.Module):
    def __init__(self, channels, init_coef=None, data_format="channels_first", in_scale=1,
                 out_scale=1,
                 train_scale=False,
                 cutoff=0.5,
                 fixed_lpf_size=None):
        super(LPFPolyActPerChannel, self).__init__()
        self.fixed_lpf_size = fixed_lpf_size
        self.lpf = LPF_RFFT(cutoff=cutoff, fixed_size=fixed_lpf_size)

        self.channels = channels
        if init_coef is None:
            init_coef = [0.0169394634313126, 0.5, 0.3078363963999393]
        self.deg = len(init_coef) - 1
        coef = torch.Tensor(init_coef)
        coef = coef.repeat([channels, 1])
        coef = torch.unsqueeze(torch.unsqueeze(coef, -1), -1)
        self.coef = nn.Parameter(coef, requires_grad=True)

        if train_scale:
            self.in_scale = nn.Parameter(torch.tensor([in_scale * 1.0]), requires_grad=True)
            self.out_scale = nn.Parameter(torch.tensor([out_scale * 1.0]), requires_grad=True)

        else:
            if in_scale != 1:
                self.register_buffer('in_scale', torch.tensor([in_scale * 1.0]))
            else:
                self.in_scale = None

            if out_scale != 1:
                self.register_buffer('out_scale', torch.tensor([out_scale * 1.0]))
            else:
                self.out_scale = None

        self.data_format = data_format

    def forward(self, x):
        if self.data_format == 'channels_last':
            x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        if self.in_scale is not None:
            x = self.in_scale * x

        x_lpf = self.lpf(x)

        x = self.coef[:, 0] + self.coef[:, 1] * x + self.coef[:, 2] * (x * x_lpf)

        if self.out_scale is not None:
            x = self.out_scale * x

        if self.data_format == 'channels_last':
            x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)

        return x


