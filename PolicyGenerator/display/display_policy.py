import torch
import gym
import numpy as np
from gym import wrappers
import matplotlib.pyplot as plt 
from matplotlib import animation 
from display.config import mw_config
from display.policy import SAC
from copy import copy
import ipdb
import sys
import os
from display.envs.make_env import build_environment
from display.utils import param_to_policy


def display_model(ckpt, config=mw_config):
    config.hidden_size = 128
    config.seed = 620

    env_name = config.env_name
    env = build_environment(config)
    torch.manual_seed(config.seed)

    config.device = ckpt.device

    agent = SAC(env.observation_space.shape[0], env.action_space, config)
    avg_reward = 0.
    avg_success = 0.
    avg_success_time = 0.

    ckpt_num = len(ckpt)
    print("{} parameters evaluate".format(ckpt_num))

    for i in range(ckpt_num):
        state_dict = param_to_policy(ckpt[i], agent.policy.state_dict())
        agent.policy.load_state_dict(state_dict)


        test_reward = 0.
        test_success = 0.
        test_success_time = 0.

        for i in range(config.eval_episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            first_success_time = 0
            success = False
            rewards = []
            while not done:
                action = agent.select_action(state, evaluate=True)
                next_state, reward, done, info = env.step(action)
                
                if 'success' in info.keys():
                    success |= bool(info["success"])

                if not success:
                    first_success_time += 1
                    
                episode_reward += reward
                state = next_state
            
            test_success += float(info['success'])
            test_reward += episode_reward
            test_success_time += first_success_time

        test_reward /= config.eval_episodes
        test_success /= config.eval_episodes
        test_success_time /= config.eval_episodes
        print("----------------------------------------")
        print("Env: {}, Test Episodes: {}, Avg. Reward: {}, Avg. Success: {}".format(config.env_name, config.eval_episodes, round(test_reward, 2), round(test_success,2)))
        print("----------------------------------------")

        avg_reward += test_reward
        avg_success += test_success
        avg_success_time += test_success_time
    
    avg_reward /= ckpt_num
    avg_success /= ckpt_num
    avg_success_time /= ckpt_num
    env.close()
    return avg_reward, avg_success, avg_success_time
    
