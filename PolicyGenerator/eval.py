if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import hydra
import torch
import numpy as np
import copy
import random
import wandb
import tqdm
import time
import dill
from omegaconf import OmegaConf
import datetime
from hydra.core.hydra_config import HydraConfig
from dataset import Dataset
from policy_generator.model.common.normalizer import LinearNormalizer
from policy_generator.model.encoder.param_encoder import EncoderDecoder
from policy_generator.model.encoder.trajectory_embedding import Embedding
from policy_generator.policy.policy_generator import PolicyGenerator
from display.display_policy import display_model


class EvalWorkspace:
    def __init__(self, cfg: OmegaConf, output_dir=None):
        self.cfg = cfg

        self._saving_thread = None

        # set seed
        seed = cfg.train.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: PolicyGenerator  = hydra.utils.instantiate(cfg.policy)

        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        self.normalizer = LinearNormalizer()

        ckpt_path = './pretrain_model/last-model.torch'
        encoder_path = './pretrain_model/param_encoder.ckpt'

        self.param_encoder = EncoderDecoder(1024, 1, 1, 0.0001, 0.001)
        self.load_checkpoint(ckpt_path, evaluate=True)
        self.load_encoder(encoder_path, evaluate=True)

        self.model.set_normalizer(self.normalizer)
        self.model.eval()

    def rollout(self, data, env):
        nparam = data['param']
        ntraj = data['traj']


        eval_dict = {'traj': ntraj}
        pred_param = self.model.predict_paremeters(eval_dict)
        nparam = nparam.reshape(-1, 2, 1024)
        param = self.param_encoder.decode(nparam)

        print("shape of param: ", param.shape)
        avg_reward, avg_success, avg_success_time = display_model(param, env)

        print("After diffusion generation.")

        gen_param = self.param_encoder.decode(pred_param)
        print("shape of generated param: ", gen_param.shape)
        gen_avg_reward, gen_avg_success, gen_avg_success_time = display_model(gen_param, env)


        print("Avg. Reward: {}, Avg. Success: {}, Avg Length: {}".format(round(avg_reward, 2), round(avg_success,2), round(avg_success_time,2)))
        print("After Generated, Avg. Reward: {}, Avg. Success: {}, Avg Length: {}".format(round(gen_avg_reward, 2), round(gen_avg_success,2), round(gen_avg_success_time,2)))


    def load_encoder(self, encoder_path, evaluate=True):
        print("Loading encoders from {}".format(encoder_path))
        encoder_ckpt = torch.load(encoder_path, map_location='cpu')
        weights_dict = {}
        weights = encoder_ckpt['state_dict']
        for k, v in weights.items():
            new_k = k.replace('model.', '') if 'model.' in k else k
            weights_dict[new_k] = v
        self.param_encoder.load_state_dict(weights_dict)
        self.param_encoder.eval()
    
    def load_checkpoint(self, ckpt_path, evaluate=True):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.normalizer.load_state_dict(checkpoint['normalizer'])

            if evaluate:
                self.model.eval()
            else:
                self.model.train()


@hydra.main(config_name="config")  

def main(cfg):
    workspace = EvalWorkspace(cfg)
    data_path = './param_data/process_window_open.pt'
    data = torch.load(data_path)
    workspace.rollout(data, env='window-open')

if __name__ == "__main__":
    main()

