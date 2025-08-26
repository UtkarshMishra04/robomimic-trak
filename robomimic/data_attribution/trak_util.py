from typing import Any, List, Union, Tuple, Dict, Iterable

import numpy as np
from torch import nn
from copy import deepcopy
from robomimic.algo import register_algo_factory_func, PolicyAlgo
from robomimic.algo.diffusion_policy import DiffusionPolicyUNet

def get_parameter_names(model: PolicyAlgo, keys: List[str]) -> List[str]:
    """Return parameters named specified by keys."""   

    if isinstance(model, DiffusionPolicyUNet):
        nets = model.nets["policy"]["noise_pred_net"]
        obs_nets = model.nets["policy"]["obs_encoder"]
    else:
        nets = model.nets["policy"]

    parameter_names = list(dict(nets.named_parameters()).keys())
    if isinstance(model, DiffusionPolicyUNet):
        parameter_names += list(dict(obs_nets.named_parameters()).keys())
    # Important: Remove dummy parameters to avoid None-type gradients.
    return sorted([k for k in parameter_names if any(_k in k for _k in keys) and "dummy" not in k])