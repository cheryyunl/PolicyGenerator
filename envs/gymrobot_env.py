import gym
import gymnasium 

class gymrobotEnv(object):
	def __init__(self, env_name, seed, max_episode_steps) -> None:
		self.env = gymnasium.make(env_name, max_episode_steps)
        self.max_episode_steps = max_episode_steps
		self.seed = seed
	
	def reset(self):
		obs, info = self.env.reset(seed=self.seed)
		return obs

	def step(self, action):
		obs, reward, done, flag, info = self.env.step(action)
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
	env_name = "Hopper-v2"
	env = gymrobotEnv(env_name, seed=100)
	env.reset()
	observation, reward, terminated, info = env.step(env.action_space.sample())
	print(observation, reward, terminated, info)