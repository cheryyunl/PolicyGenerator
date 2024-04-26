from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import copy
import time

from policy_generator.model.common.normalizer import LinearNormalizer
from policy_generator.policy.base_policy import BasePolicy
from policy_generator.model.diffusion.conditional_unet1d import ConditionalUnet1D
from policy_generator.common.pytorch_util import dict_apply
from policy_generator.common.model_util import print_params
from policy_generator.model.diffusion.mask_generator import LowdimMaskGenerator

class PolicyGenerator(BasePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            num_trajectory=1,
            num_inference_steps=None,
            diffusion_step_embed_dim=128,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            condition_type="film",
            use_down_condition=True,
            use_mid_condition=True,
            use_up_condition=True,
            trajectory_feature_dim = 128,

            param_encoder = '',
            traj_embedding = '',
            # parameters passed to step
            **kwargs):
        super().__init__()

        self.condition_type = condition_type

        # parse shape_meta
        parameters_shape = shape_meta['params']['shape']
        trajectory_shape = shape_meta['trajectory']['shape']
        self.parameters_shape = parameters_shape
        trajectory_dim = trajectory_shape[0]
        if len(parameters_shape) == 1:
            parameters_dim = parameters_shape[0]
        elif len(parameters_shape) == 2:
            parameters_dim = parameters_shape[0] * parameters_shape[1]
        else: 
            raise NotImplementedError(f"Unsupported parameter shape {parameters_shape}")

        # create diffusion model
        input_dim = parameters_dim
        cond_dim = trajectory_feature_dim
        if "cross_attention" in self.condition_type:
            global_cond_dim = trajectory_feature_dim
        else:
            global_cond_dim = trajectory_feature_dim * num_trajectory


        model = ConditionalUnet1D(
            input_dim=input_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type=condition_type,
            use_down_condition=use_down_condition,
            use_mid_condition=use_mid_condition,
            use_up_condition=use_up_condition,
        )

        self.model = model
        self.noise_scheduler = noise_scheduler
        
        
        self.noise_scheduler_pc = copy.deepcopy(noise_scheduler)

        self.mask_generator = LowdimMaskGenerator(
            action_dim=parameters_dim,
            obs_dim=0,
            max_n_obs_steps=1,
            fix_obs_steps=True,
            action_visible=False
        )
        
        self.normalizer = LinearNormalizer()
        self.trajectory_feature_dim = trajectory_feature_dim
        self.parameters_dim = parameters_dim
        self.num_trajectory = num_trajectory
        self.kwargs = kwargs
        self.horizon = 2

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps


        print_params(self)
        
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            condition_data_pc=None, condition_mask_pc=None,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler


        parameters = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device)

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)


        for t in scheduler.timesteps:
            parameters[condition_mask] = condition_data[condition_mask]
            model_output = model(sample=parameters,
                                timestep=t, 
                                local_cond=None,
                                global_cond=global_cond)
            
            # compute previous image: x_t -> x_t-1
            parameters = scheduler.step(
                model_output, t, parameters, ).prev_sample
        parameters[condition_mask] = condition_data[condition_mask]   


        return parameters


    def predict_paremeters(self, traj_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        traj_dict: must include "trajectory" key
        result: must include "params" key
        """
        # normalize input
        traj_dict = self.normalizer.normalize(traj_dict)
        
        value = next(iter(traj_dict.values()))
        B, To = value.shape[:2]
        Da = self.parameters_dim
        Do = self.trajectory_feature_dim
        To = self.num_trajectory
        T = 1

        # build input
        device = self.device
        dtype = self.dtype

        # condition through global feature
        # ntraj_dict = dict_apply(traj_dict, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
        ntraj = traj_dict['traj']
        global_cond = None
        if "cross_attention" in self.condition_type:
            # treat as a sequence
            global_cond = ntraj.reshape(B, self.num_trajectory, -1)
        else:
            # reshape back to B, Do
            global_cond = ntraj.reshape(B, -1)
        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        # run sampling
        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        nsample = nsample.reshape(-1, 2, 1024)
        params_pred = self.normalizer['param'].unnormalize(nsample)

        # result = {'param': parameters_pred}
        
        return params_pred.detach()

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        
        batch = self.normalizer.normalize(batch)
        ntraj = batch['traj']
        nparameters = batch['param']
        horizon = 1
        
        batch_size = nparameters.shape[0]
        nparameters = nparameters.reshape(batch_size, horizon, self.parameters_dim)
        global_cond = None
        parameters = nparameters
        cond_data = parameters

        if "cross_attention" in self.condition_type:
            # treat as a sequence
            global_cond = ntraj.reshape(batch_size, self.num_trajectory, -1)
        else:
            # reshape back to B, Do
            global_cond = ntraj.reshape(batch_size, -1)

        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        condition_mask = self.mask_generator(parameters.shape)
        # Sample noise that we'll add to the images
        noise = torch.randn(parameters.shape, device=parameters.device)

        
        batch_size = parameters.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (batch_size,), device=parameters.device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_parameters = self.noise_scheduler.add_noise(
            parameters, noise, timesteps)

        # compute loss mask
        loss_mask = ~condition_mask

        noisy_parameters[condition_mask] = cond_data[condition_mask]
        # Predict the noise residual
        pred = self.model(sample=noisy_parameters, 
                        timestep=timesteps, local_cond = None,
                            global_cond=global_cond)


        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = parameters
        elif pred_type == 'v_prediction':
            self.noise_scheduler.alpha_t = self.noise_scheduler.alpha_t.to(self.device)
            self.noise_scheduler.sigma_t = self.noise_scheduler.sigma_t.to(self.device)
            alpha_t, sigma_t = self.noise_scheduler.alpha_t[timesteps], self.noise_scheduler.sigma_t[timesteps]
            alpha_t = alpha_t.unsqueeze(-1).unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1).unsqueeze(-1)
            v_t = alpha_t * noise - sigma_t * parameters
            target = v_t
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='mean')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        

        loss_dict = {
                'loss': loss.item(),
            }
        
        return loss, loss_dict

