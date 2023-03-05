import torch
import torch.nn as nn
import torch.nn.functional as F


# Change layer norm implementation
# originaly they normalized only on channels. now normaliazing on all layer [C,H,W] -
# solving shit invariance issue
def LayerNorm(normalized_shape, eps=1e-6, data_format="channels_last", normalization_type="C"):
    # normalizing on channels
    if normalization_type == "C":
        return LayerNorm_C(normalized_shape, eps, data_format)

    # normalize with mean on channels, std on layer
    elif normalization_type == "CHW2":
        return LayerNorm_AF(normalized_shape, eps, data_format, u_dims=1, s_dims=(1, 2, 3))

    else:
        raise NotImplementedError


class LayerNorm_C(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class LayerNorm_AF(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last", u_dims=(1, 2, 3), s_dims=(1, 2, 3)):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)
        self.u_dims = u_dims
        self.s_dims = s_dims

    def forward(self, x):
        if self.data_format == "channels_last":
            x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        u = x.mean(self.u_dims, keepdim=True)
        s = (x - u).pow(2).mean(self.s_dims, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]

        if self.data_format == "channels_last":
            x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)

        return x

def get_norm_layer(normalization_type, dim, data_format='channels_first', **kwargs):
    if normalization_type == 'batch':
        assert data_format == 'channels_first', "BatchNorm2d doesn't support channels last"
        return nn.BatchNorm2d(dim)

    elif normalization_type == 'instance':
        assert data_format == 'channels_first', "InstanceNorm2d doesn't support channels last"
        return nn.InstanceNorm2d(dim)

    if 'num_groups' in kwargs:
        num_groups = kwargs['num_groups']
    elif 'channels_per_group' in kwargs:
        num_groups = int(dim/kwargs['channels_per_group'])
    else:
        num_groups = None

    if normalization_type == 'group':
        assert data_format == 'channels_first', "GroupNorm doesn't support channels last"
        assert num_groups, "missing key word argument for GroupNorm / LayerNormSTDGroups num_groups"
        return nn.GroupNorm(num_groups=num_groups, num_channels=dim)

    else:
        return LayerNorm(dim, eps=1e-6, normalization_type=normalization_type, data_format=data_format)

