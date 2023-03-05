import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from .activation import get_activation
from .layer_norm import LayerNorm
from .apspool import ApsPool
from .conv_layers import conv7x7, conv4x4, conv2x2


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, conv_pad_type='circular',
                 activation='gelu', activation_kwargs={}, normalization_type='C'):
        super().__init__()
        self.dwconv = conv7x7(dim, dim, padding=3, groups=dim, conv_pad_type=conv_pad_type)
        self.norm = LayerNorm(dim, eps=1e-6, normalization_type=normalization_type)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = get_activation(activation, channels=4 * dim, data_format='channels_last', **activation_kwargs)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path_layer = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path_layer(x)
        return x


class ConvNeXtAPS(nn.Module):
    r""" ConvNeXtAPS
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (list): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1., conv_pad_type='circular',
                 blurpool_kwargs={}, activation='gelu', activation_kwargs={}, normalization_type='C',
                 init_weight_std=.02):
        super().__init__()
        layer_block = Block
        self.init_weight_std = init_weight_std

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            conv4x4(in_chans, dims[0], stride=1, conv_pad_type=conv_pad_type, padding=[1, 2, 1, 2]),
            ApsPool(dims[0], conv_pad_type, stride=2, **blurpool_kwargs),
            ApsPool(dims[0], conv_pad_type, stride=2, **blurpool_kwargs),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first", normalization_type=normalization_type)
        )

        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first", normalization_type=normalization_type),
                # add padding because when removing stide output size changes
                conv2x2(dims[i], dims[i + 1], stride=1, conv_pad_type=conv_pad_type, padding=[0, 1, 0, 1]),
                ApsPool(dims[i + 1], conv_pad_type, stride=2, **blurpool_kwargs)
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[layer_block(dim=dims[i], drop_path=dp_rates[cur + j],
                              layer_scale_init_value=layer_scale_init_value,
                              conv_pad_type=conv_pad_type,
                              activation=activation, activation_kwargs=activation_kwargs,
                              normalization_type=normalization_type,
                              ) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=self.init_weight_std)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x




@register_model
def convnext_aps_tiny(pretrained=False, **kwargs):
    print("ConvNext kwargs: ", kwargs)
    model = ConvNeXtAPS(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    assert not pretrained, "pretrained ConvNeXt_APS is unavailable"
    return model


@register_model
def convnext_aps_small(pretrained=False, **kwargs):
    model = ConvNeXtAPS(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    assert not pretrained, "pretrained ConvNeXt_APS is unavailable"
    return model


@register_model
def convnext_aps_base(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXtAPS(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    assert not pretrained, "pretrained ConvNeXt_APS is unavailable"
    return model


@register_model
def convnext_aps_large(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXtAPS(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    assert not pretrained, "pretrained ConvNeXt_APS is unavailable"
    return model


@register_model
def convnext_aps_xlarge(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXtAPS(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    assert not pretrained, "pretrained ConvNeXt_APS is unavailable"
    return model



