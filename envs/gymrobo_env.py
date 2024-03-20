import gym
import gymnasium 
from gymnasium.wrappers import TimeLimit

class GymRoboEnv(object):
	def __init__(self, env_name, seed, max_episode_steps) -> None:
		self.env = gymnasium.make(env_name, max_episode_steps)
		self.env = TimeLimit(self.env, max_episode_steps=max_episode_steps)
		self.seed = seed
		self.max_episode_steps = max_episode_steps
	
	def reset(self):
		obs, info = self.env.reset(seed=self.seed)
		return obs

	def step(self, action):
		obs, reward, terminated, truncated, info = self.env.step(action)
		done = terminated or truncated
		return obs, reward, done, info
	
	@property
	def observation_space(self):
		return self.env.observation_space
	
	@property
	def action_space(self):
		return self.env.action_space
	
	@property
	def _max_episode_steps(self):
		return self.max_episode_steps
	
if __name__ == "__main__":
	env_name = "AdroitHandDoor-v1"
	env = GymRoboEnv(env_name, seed=100, max_episode_steps=400)
	env.reset()
	observation, reward, done, info = env.step(env.action_space.sample())
	print(observation, reward, done, info)