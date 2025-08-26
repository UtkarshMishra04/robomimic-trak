from typing import List, Dict, Iterable

import torch
from einops import reduce
import torch.nn.functional as F
from trak.modelout_functions import AbstractModelOutput

import robomimic.utils.torch_utils as TorchUtils
from robomimic.algo import register_algo_factory_func, PolicyAlgo
from robomimic.algo.diffusion_policy import DiffusionPolicyUNet

class PolicyFunctionalModelOutput(AbstractModelOutput):

    def __init__(self):
        super().__init__()

    def get_output(
        self,
        model: PolicyAlgo,
        weights: Dict[str, torch.Tensor],
        buffers: Dict[str, torch.Tensor],
        action: torch.Tensor,
        obs: torch.Tensor,
        timesteps: torch.Tensor = None,
        goal_obs: torch.Tensor = None
    ) -> torch.Tensor:
        
        print("Using functional forward pass for policy model.")

        # Batchify inputs.
        batch = {
            "action": action.unsqueeze(0).to(torch.float32),
            "obs": TorchUtils.dict_apply(obs, lambda x: x.unsqueeze(0).to(torch.float32))
        }

        if goal_obs is not None:
            batch["goal_obs"] = goal_obs.unsqueeze(0).to(torch.float32) 

        function_input = {}

        if timesteps is not None and isinstance(model, DiffusionPolicyUNet):
            function_input["timesteps"] = timesteps.unsqueeze(0)

        function_input.update({
            "batch": batch,
            "model_weights": weights,
            "model_buffers": buffers
        })

        return model.train_on_batch_functional(**function_input)

    def get_out_to_loss_grad(
        self, 
        model: PolicyAlgo,
        weights: Dict[str, torch.Tensor],
        buffers: Dict[str, torch.Tensor],
        batch: Iterable[torch.Tensor]
    ) -> torch.Tensor:
        """Computes the (reweighting term Q in the paper)."""        
        return torch.ones(batch["actions"].shape[0]).to(batch["actions"].device).unsqueeze(-1)
  


