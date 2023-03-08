from .resnet import *
from .vgg import *
from .wide_resnet import wide_resnet50
from .alexnet import *

# hagay - add convnext models
# from functools import partial

from .convnext.convnext_aps import convnext_aps_tiny as convnext_aps_tiny_model
from .convnext.convnext_afc import convnext_afc_tiny as convnext_afc_tiny_model


def convnext_aps_tiny(num_classes=1000, pretrained=False):
    return convnext_aps_tiny_model(pretrained=pretrained,
                                    num_classes=num_classes,
                                    drop_path_rate=0,
                                    layer_scale_init_value=1e-6,
                                    head_init_scale=1.0,
                                    activation = 'gelu',
                                    activation_kwargs = {},
                                    blurpool_kwargs = {'filt_size': 1},
                                    normalization_type = 'C',
                                    fast_block = False,
                                    init_weight_std=0.02)

def convnext_afc_tiny(num_classes=1000, pretrained=False):
    model = convnext_afc_tiny_model(pretrained=pretrained,
                                    num_classes=num_classes,
                                    drop_path_rate=0,
                                    layer_scale_init_value=1e-6,
                                    head_init_scale=1.0,
                                    activation='up_poly_per_channel',
                                    activation_kwargs={'in_scale': 7, 'out_scale': 7, 'train_scale': True},
                                    blurpool_kwargs={"filter_type": "ideal", "scale_l2": False},
                                    normalization_type='CHW2',
                                    init_weight_std=0.02,
                                   stem_mode='activation_residual', stem_activation='lpf_poly_per_channel',
                                   stem_activation_kwargs={"in_scale": 7, "out_scale": 7, "train_scale": True, "cutoff": 0.75})

    def forward(self, x, with_latent=False, fake_relu=False, no_relu=False):
        features = self.forward_features(x)
        x = self.head(features)
        if with_latent:
            return x, features
        return x
    # replace model forward function
    model.forward = forward.__get__(model, model.__class__)
    return model
