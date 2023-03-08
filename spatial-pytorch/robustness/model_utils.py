import torch as ch
import dill
import os
from .tools import helpers, constants
from .attacker import AttackerModel

class FeatureExtractor(ch.nn.Module):
    '''
    Tool for extracting layers from models.

    Args:
        submod (torch.nn.Module): model to extract activations from
        layers (list of functions): list of functions where each function,
            when applied to submod, returns a desired layer. For example, one
            function could be `lambda model: model.layer1`.

    Returns:
        A model whose forward function returns the activations from the layers
            corresponding to the functions in `layers` (in the order that the
            functions were passed in the list).
    '''
    def __init__(self, submod, layers):
        # layers must be in order
        super(FeatureExtractor, self).__init__()
        self.submod = submod
        self.layers = layers
        self.n = 0

        for layer_func in layers:
            layer = layer_func(self.submod)
            def hook(module, _, output):
                module.register_buffer('activations', output)

            layer.register_forward_hook(hook)

    def forward(self, *args, **kwargs):
        # self.layer_outputs = {}
        out = self.submod(*args, **kwargs)
        activs = [layer_fn(self.submod).activations for layer_fn in self.layers]
        return [out] + activs

def make_and_restore_model(*_, arch, dataset, resume_path=None,
         parallel=True, pytorch_pretrained=False, attack_space='spatial'):
    """
    Makes a model and (optionally) restores it from a checkpoint.

    Args:
        arch (str): Model architecture identifier
        dataset (Dataset class [see datasets.py])
        resume_path (str): optional path to checkpoint
        parallel (bool): if True, wrap the model in a DataParallel 
            (default True, recommended)
        pytorch_pretrained (bool): if True, try to load a standard-trained 
            checkpoint from the torchvision library (throw error if failed)
    Returns: 
        A tuple consisting of the model (possibly loaded with checkpoint), and the checkpoint itself
    """
    classifier_model = dataset.get_model(arch, pytorch_pretrained)

    model = AttackerModel(classifier_model, dataset, attack_space=attack_space)

    # optionally resume from a checkpoint
    checkpoint = None
    if resume_path:
        if os.path.isfile(resume_path):
            print("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = ch.load(resume_path, pickle_module=dill)
            
            # Makes us able to load models saved with legacy versions
            state_dict_path = 'model'
            if not ('model' in checkpoint):
                state_dict_path = 'state_dict'

            sd = checkpoint[state_dict_path]

            # sd = {k[len('module.'):]:v for k,v in sd.items()}
            #  in case not saved with module prefix
            new_sd = {}
            for k,v in sd.items():
                if k.startswith('.module'):
                    new_sd[k[len('module.'):]] = v
                else:
                    new_sd[k] = v

            # model.load_state_dict(sd)
            load_model_state_dict(model, sd)
            if parallel:
                model = ch.nn.DataParallel(model)
            model = model.cuda()

            print("=> loaded checkpoint '{}' (epoch {})".format(resume_path, checkpoint['epoch']))
        else:
            error_msg = "=> no checkpoint found at '{}'".format(resume_path)
            raise ValueError(error_msg)

    return model, checkpoint

# [hm] load model state dict - allow missing keys for aal models
# call make_and_restore_model and it has a new argument
def load_model_state_dict(model, state_dict):
    print("load model state dict")
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print("missing keys: ", missing_keys)
    print("unexpected keys: ", unexpected_keys)

    # aal model - allow missing rect in state_dict
    # rect params dependent on input size and initialized on forward

    # if len(missing_keys) > 0:
    #     raise ValueError("Missing keys")
    # allow missing 'rect' - because it doesn't appear in EMA model
    for k in missing_keys:
        if 'rect' not in k:
            raise ValueError("Missing keys")

    for k in unexpected_keys:
        if 'rect' not in k:
            raise ValueError("Missing keys")

    return model


def model_dataset_from_store(s, overwrite_params={}, which='last'):
    '''
    Given a store directory corresponding to a trained model, return the
    original model, dataset object, and args corresponding to the arguments.
    '''
    # which options: {'best', 'last', integer}
    if type(s) is tuple:
        s, e = s
        s = cox.store.Store(s, e, mode='r')

    m = s['metadata']
    df = s['metadata'].df

    args = df.to_dict()
    args = {k:v[0] for k,v in args.items()}
    fns = [lambda x: m.get_object(x), lambda x: m.get_pickle(x)]
    conds = [lambda x: m.schema[x] == s.OBJECT, lambda x: m.schema[x] == s.PICKLE]
    for fn, cond in zip(fns, conds):
        args = {k:(fn(v) if cond(k) else v) for k,v in args.items()}

    args.update(overwrite_params)
    args = Parameters(args)

    data_path = os.path.expandvars(args.data)
    if not data_path:
        data_path = '/tmp/'

    dataset = DATASETS[args.dataset](data_path)

    if which == 'last':
        resume = os.path.join(s.path, constants.CKPT_NAME)
    elif which == 'best':
        resume = os.path.join(s.path, constants.CKPT_NAME_BEST)
    else:
        assert isinstance(which, int), "'which' must be one of {'best', 'last', int}"
        resume = os.path.join(s.path, ckpt_at_epoch(which))

    model, _ = make_and_restore_model(arch=args.arch, dataset=dataset,
                                      resume_path=resume, parallel=False)
    return model, dataset, args
