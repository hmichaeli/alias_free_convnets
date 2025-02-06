import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

from .blurpool import BlurPool
from .activation import get_activation

from .poly_mlp import MLP, AALMLP
from .layer_norm import get_norm_layer
from .conv_layers import conv7x7, conv4x4, conv2x2


class RepLayer(nn.Module):
    def __init__(self, layer1, layer2, step=300000, r0=1.0):
        super(RepLayer, self).__init__()
        self.register_buffer('iter', torch.tensor(step))
        self.register_buffer('total_step', torch.tensor(step))
        self.r0 = r0
        self.layer1 = layer1
        self.layer2 = layer2

    def forward(self, x):
        lamda = self.r0 * self.iter / self.total_step
        if self.iter > 0 and self.training:
            self.iter.copy_(self.iter - 1)
        x1 = self.layer1(x)
        x2 = self.layer2(x)
        x = lamda * x1 + (1 - lamda) * x2
        return x

class MLPRepAF(nn.Module):
    def __init__(self, dim, expand_ratio, activation, activation_kwargs={}, rep_steps=300000):
        super(MLPRepAF, self).__init__()
        self.pwconv1 = nn.Linear(dim, expand_ratio * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = RepLayer(nn.GELU(),
                            get_activation(activation, channels=expand_ratio * dim, data_format='channels_last', **activation_kwargs),
                            step=rep_steps
                            )
        self.pwconv2 = nn.Linear(expand_ratio * dim, dim)

    def forward(self, x):
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return x

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
                 activation='gelu', activation_kwargs={}, normalization_type='C', normalization_kwargs={},
                 rep_steps=300000):
        super().__init__()
        self.dwconv = conv7x7(dim, dim, padding=3, groups=dim, conv_pad_type=conv_pad_type)

        # self.norm = get_norm_layer(normalization_type, dim, data_format='channels_first', **normalization_kwargs)
        self.norm = RepLayer(get_norm_layer('C', dim, data_format='channels_first', **normalization_kwargs),
                             get_norm_layer(normalization_type, dim, data_format='channels_first', **normalization_kwargs),
                             step=rep_steps)
                
        
        # if activation == 'up_poly_per_channel':
        #     # use AALMLP - faster implementation
        #     self.mlp = AALMLP(dim=dim, expand_ratio=4,  activation_kwargs=activation_kwargs)
        # else:
        #     self.mlp = MLP(dim=dim, expand_ratio=4, activation=activation, activation_kwargs=activation_kwargs)
        self.mlp = MLPRepAF(dim=dim, expand_ratio=4, activation=activation, activation_kwargs=activation_kwargs, rep_steps=rep_steps)

        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path_layer = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)

        x = self.mlp(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path_layer(x)
        return x


class ConvNeXtRepAFC(nn.Module):
    r""" ConvNeXtAFC

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
        activation (str): Model activation function. Default: 'gelu'.
        For Aliasing free implementation use up_poly_per_channel
        activation_kwargs (dict): keyword arguments for activation function
        normalization_type (str): Default: C - original Layernorm. for Alias-free LayerNorm use CHW2
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1., conv_pad_type='circular',
                blurpool_kwargs={}, activation='gelu', activation_kwargs={}, normalization_type='C',
                 init_weight_std=.02, stem_mode=None, stem_activation=None, stem_activation_kwargs={},
                 normalization_kwargs={},
                 rep_steps=300000):
        super().__init__()
        layer_block = Block
        self.init_weight_std = init_weight_std
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        
        stem = nn.Sequential(
                conv4x4(in_chans, dims[0], stride=1, conv_pad_type=conv_pad_type, padding=[1, 2, 1, 2]),
                RepLayer(nn.GELU(),
                        get_activation(stem_activation, dims[0], "channels_first", **stem_activation_kwargs),
                        step=rep_steps),
                RepLayer(BlurPool(dims[0], conv_pad_type, stride=4, filter_type='basic', filt_size=1), # equiv to stride 4
                        BlurPool(dims[0], conv_pad_type, stride=4, cutoff=0.25, **blurpool_kwargs),
                        step=rep_steps),
                RepLayer(get_norm_layer('C', dims[0], data_format='channels_first', **normalization_kwargs),
                        get_norm_layer(normalization_type, dims[0], data_format='channels_first', **normalization_kwargs),
                        step=rep_steps)
            )

        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                                RepLayer(get_norm_layer('C', dims[i], data_format='channels_first'),
                                        get_norm_layer(normalization_type, dims[i], data_format='channels_first', **normalization_kwargs),
                                        step=rep_steps),
                                    
                                conv2x2(dims[i], dims[i + 1], stride=1, conv_pad_type=conv_pad_type, padding=[0,1,0,1]),
                                RepLayer(BlurPool(dims[i+1], conv_pad_type, stride=2, filter_type='basic', filt_size=1), # equiv to stride 4
                                            BlurPool(dims[i + 1], conv_pad_type, stride=2, **blurpool_kwargs),
                                            step=rep_steps)
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[layer_block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value,
                conv_pad_type=conv_pad_type,
                activation=activation, activation_kwargs=activation_kwargs,
                normalization_type=normalization_type,
                normalization_kwargs=normalization_kwargs
                ) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=self.init_weight_std)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x, avgpool=True):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        if avgpool:
            x = self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)
        return x


    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x



class Residual(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.residual = nn.Sequential(*layers)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x + self.gamma * self.residual(x)


class ResidualPerChannel(nn.Module):
    def __init__(self, channels, *layers):
        super().__init__()
        self.residual = nn.Sequential(*layers)
        self.gamma = nn.Parameter(1e-6 * torch.ones((channels, 1, 1)))

    def forward(self, x):
        return x + self.gamma * self.residual(x)


@register_model
def convnext_rep_afc_tiny(pretrained=False, **kwargs):
    print("ConvNext kwargs: ", kwargs)
    model = ConvNeXtRepAFC(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    assert not pretrained, "pretrained ConvNeXt_APS is unavailable"
    return model

# @register_model
# def convnext_afc_small(pretrained=False, **kwargs):
#     model = ConvNeXtAFC(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
#     assert not pretrained, "pretrained ConvNeXt_APS is unavailable"
#     return model

# @register_model
# def convnext_afc_base(pretrained=False, in_22k=False, **kwargs):
#     model = ConvNeXtAFC(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
#     assert not pretrained, "pretrained ConvNeXt_APS is unavailable"
#     return model

# @register_model
# def convnext_afc_large(pretrained=False, in_22k=False, **kwargs):
#     model = ConvNeXtAFC(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
#     assert not pretrained, "pretrained ConvNeXt_APS is unavailable"
#     return model

# @register_model
# def convnext_afc_xlarge(pretrained=False, in_22k=False, **kwargs):
#     model = ConvNeXtAFC(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
#     assert not pretrained, "pretrained ConvNeXt_APS is unavailable"
#     return model


