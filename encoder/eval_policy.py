import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

import os
import sys
from envs.make_env import build_environment
from configs.default_config import mw_config
from model_proj import EncoderDecoder

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim = 128, action_space=None):
        super(GaussianPolicy, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.elu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean
    
    def get_log_density(self, state, action):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        log_density = normal.log_prob(action)
        return log_density

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)

# test passed
def save_flattened_params(models):
    pdata = []
    for model in models:
        param = []
        for key in model.keys():
            param.append(model[key].data.reshape(-1))
        param = torch.cat(param, 0)
        pdata.append(param)
    batch = torch.stack(pdata)
    return batch

# test passed
def load_flattened_params(flattened_params, model):
    layer_idx = 0
    for name, param in model.named_parameters():
        param_shape = param.shape
        param_length = param.view(-1).shape[0]
        param.data = flattened_params[layer_idx:layer_idx + param_length].reshape(param_shape)
        param.data.to(flattened_params.device)
        layer_idx += param_length
    return model

def build_env_policy(config=mw_config):
    sys.path.append(os.path.join(os.path.dirname(__file__), 'envs')) 
    env = build_environment(config)
    policy = GaussianPolicy(env.observation_space.shape[0], env.action_space.shape[0], 128, env.action_space).to(config.device)

    return env, policy


def eval_policy(params, env, policy, config=mw_config):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    policy = load_flattened_params(params, policy)

    eval_reward_list = []
    eval_success_list = []
    eval_step_length_list = []

    for _  in range(config.eval_episodes):
        state = env.reset()
        episode_reward = []
        done = False
        first_success_time = 0
        success = False
        while not done:
            _, _, action = policy.sample(state)
            action = action.detach().cpu().numpy()[0]
            next_state, reward, done,  info = env.step(action)
            state = next_state
            episode_reward.append(reward)
            if 'success' in info.keys():       
                success |= bool(info["success"])
                if not success:
                    first_success_time +=1
        eval_reward_list.append(sum(episode_reward))
        eval_step_length_list.append(first_success_time)
        if 'success' in info.keys():
            eval_success_list.append(float(info['success']))

    avg_reward = np.average(eval_reward_list)
    avg_success = np.average(eval_success_list)
    avg_step_length = np.average(eval_step_length_list)
    
    print("----------------------------------------")
    print("Env: {}, Avg. Reward: {}, Avg. Success: {}, Avg. Length: {}".format(config.env_name, round(avg_reward, 2), round(avg_success, 2), round(avg_step_length, 2)))
    print("----------------------------------------")

    return avg_success, avg_reward, avg_step_length


if __name__ == "__main__":
    env, policy = build_env_policy()
    params = 
    eval_policy(params, env, policy)



