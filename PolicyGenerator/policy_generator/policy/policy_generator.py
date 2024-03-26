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
from policy_generator.model.autoencoder.trajectory_encoder import TrajectoryEncoder

class PolicyGenerator(BasePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            num_trajectory=1,
            num_inference_steps=None,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            condition_type="film",
            use_down_condition=True,
            use_mid_condition=True,
            use_up_condition=True,
            encoder_output_dim=256,
            # parameters passed to step
            **kwargs):
        super().__init__()

        self.condition_type = condition_type

        # parse shape_meta
        parameters_shape = shape_meta['params']['shape']
        trajectory_shape = shape_meta['trajectory']['shape']
        self.parameters_shape = parameters_shape
        parameters_dim = parameters_shape[0]
        trajectory_dim = trajectory_shape[0]
        if len(parameters_shape) != 1:
            raise NotImplementedError(f"Unsupported action shape {parameters_shape}")

        trajectory_encoder = TrajectoryEncoder(input_dim=trajectory_dim, output_dim=encoder_output_dim,)

        # create diffusion model
        trajectory_feature_dim = trajectory_encoder.output_shape()
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

        self.trajectory_encoder = trajectory_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        
        
        self.noise_scheduler_pc = copy.deepcopy(noise_scheduler)
        
        self.normalizer = LinearNormalizer()
        self.trajectory_feature_dim = trajectory_feature_dim
        self.parameters_dim = parameters_dim
        self.num_trajectory = num_trajectory
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps


        print_params(self)
        
    # ========= inference  ============
    def conditional_sample(self, shape, global_cond=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler


        parameters = torch.randn(
            size=shape, 
            dtype=self.dtype,
            device=self.device)

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)


        for t in scheduler.timesteps:
            model_output = model(sample=parameters,
                                timestep=t, 
                                global_cond=global_cond)
            
            # compute previous image: x_t -> x_t-1
            parameters = scheduler.step(
                model_output, t, parameters, ).prev_sample


        return parameters


    def predict_paremeters(self, traj_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        traj_dict: must include "trajectory" key
        result: must include "params" key
        """
        # normalize input
        ntraj = self.normalizer.normalize(traj_dict)
        
        
        value = next(iter(ntraj.values()))
        B, To = value.shape[:2]
        Da = self.parameters_dim
        Do = self.trajectory_feature_dim
        To = self.num_trajectory

        # build input
        device = self.device
        dtype = self.dtype

        # condition through global feature
        this_ntraj = dict_apply(ntraj, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
        ntraj_features = self.trajectory_encoder(this_ntraj)
        global_cond = None
        if "cross_attention" in self.condition_type:
            # treat as a sequence
            global_cond = ntraj_features.reshape(B, self.num_trajectory, -1)
        else:
            # reshape back to B, Do
            global_cond = ntraj_features.reshape(B, -1)

        # run sampling
        nsample = self.conditional_sample(
            (B, 1, Da),
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        parameters_pred = self.normalizer['params'].unnormalize(nsample)


        result = {
            'parameters': parameters_pred
        }
        
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input

        ntraj = self.normalizer.normalize(batch['trajectory'])
        nparameters = self.normalizer['params'].normalize(batch['params'])
        
        batch_size = nparameters.shape[0]

        # handle different ways of passing observation
        global_cond = None
        parameters = nparameters
        cond_data = parameters
        
        ntraj_features = self.trajectory_encoder(ntraj)

        if "cross_attention" in self.condition_type:
            # treat as a sequence
            global_cond = ntraj_features.reshape(batch_size, self.num_trajectory, -1)
        else:
            # reshape back to B, Do
            global_cond = ntraj_features.reshape(batch_size, -1)


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
            noisy_parameters, noise, timesteps)

        # Predict the noise residual
        
        pred = self.model(sample=noisy_parameters, 
                        timestep=timesteps,
                            global_cond=global_cond)


        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = parameters
        elif pred_type == 'v_prediction':
            # https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
            # https://github.com/huggingface/diffusers/blob/v0.11.1-patch/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
            # sigma = self.noise_scheduler.sigmas[timesteps]
            # alpha_t, sigma_t = self.noise_scheduler._sigma_to_alpha_sigma_t(sigma)
            self.noise_scheduler.alpha_t = self.noise_scheduler.alpha_t.to(self.device)
            self.noise_scheduler.sigma_t = self.noise_scheduler.sigma_t.to(self.device)
            alpha_t, sigma_t = self.noise_scheduler.alpha_t[timesteps], self.noise_scheduler.sigma_t[timesteps]
            alpha_t = alpha_t.unsqueeze(-1).unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1).unsqueeze(-1)
            v_t = alpha_t * noise - sigma_t * parameters
            target = v_t
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        

        loss_dict = {
                'bc_loss': loss.item(),
            }
        
        return loss, loss_dict

