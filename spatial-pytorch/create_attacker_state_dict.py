import torch
import dill
from copy import deepcopy
# from .convnext.convnext_circ import convnext_tiny
# return convnext_circ_tiny_model(pretrained=pretrained,
#                                 num_classes=num_classes,
#                                 drop_path_rate=0,
#                                 layer_scale_init_value=1e-6,
#                                 head_init_scale=1.0
#                                 )

def create_checkpoint_robustness(ckpt_path, load_key, new_filename):
    print(f"create_checkpoint_robustness({ckpt_path}, {load_key}, {new_filename})")
    checkpoint = torch.load(ckpt_path, pickle_module=dill)
    model_sd = checkpoint[load_key]

    new_sd = {}
    for k,v in model_sd.items():
        new_sd['model.' + k] = deepcopy(v)
        new_sd['attacker.model.'+k] = deepcopy(v)


    new_sd['normalizer.new_mean'] = torch.tensor([[[0.4850]],
                                                    [[0.4560]],
                                                    [[0.4060]]])

    new_sd['normalizer.new_std'] = torch.tensor([[[0.2290]],
                                                [[0.2240]],
                                                [[0.2250]]])

    new_sd['attacker.normalize.new_mean'] = torch.tensor([[[0.4850]],
                                                    [[0.4560]],
                                                    [[0.4060]]])

    new_sd['attacker.normalize.new_std'] = torch.tensor([[[0.2290]],
                                                [[0.2240]],
                                                [[0.2250]]])

    state = {'model': new_sd,
             'epoch': checkpoint['epoch']}
    torch.save(state, new_filename, pickle_module=dill)

# convnext_circ
ckpt_path = '/data/hagaymi/cnn_project_models/convnext_circ_tiny_baseline/checkpoint-best-ema.pth'
load_key = 'model_ema'
new_filename = '/data/hagaymi/cnn_project_models/convnext_circ_tiny_baseline/checkpoint-robustness.pth'
create_checkpoint_robustness(ckpt_path, load_key, new_filename)

# convnext_aps
ckpt_path = '/data/hagaymi/cnn_project_models/convnext_aps_tiny_f1_gelu_c_s2/checkpoint-best-ema.pth'
load_key = 'model_ema'
new_filename = '/data/hagaymi/cnn_project_models/convnext_aps_tiny_f1_gelu_c_s2/checkpoint-robustness.pth'
create_checkpoint_robustness(ckpt_path, load_key, new_filename)

# convnext_aal (afc)
ckpt_path = '/data/hagaymi/cnn_project_models/convnext_aal_tiny_ideal_up_poly_per_channel_scale_7_7_chw2_stem_mode_lpf_poly_cutoff0.75_300_s2/checkpoint-best-ema.pth'
load_key = 'model_ema'
new_filename = '/data/hagaymi/cnn_project_models/convnext_aal_tiny_ideal_up_poly_per_channel_scale_7_7_chw2_stem_mode_lpf_poly_cutoff0.75_300_s2/checkpoint-robustness.pth'
create_checkpoint_robustness(ckpt_path, load_key, new_filename)