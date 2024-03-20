import gym

maniskill_tasks = ["PickCube-v0",
					"StackCube-v0",
					"PickSingleYCB-v0",
					"PickSingleEGAD-v0",
					"PickClutterYCB-v0",
					"PegInsertionSide-v0",
					"PlugCharger-v0",
					"AssemblingKits",
					"PandaAvoidObstacles-v0",
					"TurnFaucet-v0",
					"OpenCabinetDoor-v1",
					"OpenCabinetDrawer-v1",
					"PushChair-v1",
					"MoveBucket-v1",
					"Excavate-v0",
					"Fill-v0",
					"Pour-v0",
					"Hang-v0",
					"Pinch-v0",
					"Write-v0"
				]

dexmv_tasks = ["",]

def build_environment(config):
	# * Meta-World (reward_type = sparse/dense)
	if (config.env_name).endswith("goal-observable"):   # metaworld.envs.ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE.keys():
		from .metaworld_env import metaworld_env
		env = metaworld_env(config.env_name, config.seed, episode_length=500, reward_type=config.reward_type)
	
	# * panda-gym (control_type=joints/ends, reward_type=sparse/dense, render_mode="rgb_array", noise_type=False/True)
	elif (config.env_name).startswith("Panda"):                              # in panda_gym.ENV_IDS:
		from .panda_env import panda_env
		env = panda_env(config.env_name)
	
	# * ROBEL(sparse/dense)
	elif config.env_name.startswith("DKitty") : 
		from .robel_env import RobelEnv, SparseRobelEnv
		if config.reward_type == "sparse":
			env = SparseRobelEnv(config.env_name)
		else:
			env = RobelEnv(config.env_name)

	# * ManiSkill2
	elif config.env_name in maniskill_tasks:
		from .maniskill_env import ManiSkillEnv
		env = ManiSkillEnv(config.env_name, seed=config.seed)

	# * Robohive
	elif config.env_name.endswith("-v1"):
		import robohive
		env = gym.make(config.env_name)

	# * DM Control
	elif config.env_name.endswith("-v0"):
		import dmc_envs
		env = gym.make(config.env_name)

	# * Gymnasium -> Gym interface
	else:       
		from .gymnasium_env import gymnasium2gymEnv   
		specified_kwargs = config.xml_file if 'xml_file' in config else None
		env = gymnasium2gymEnv(env_name=config.env_name, seed=config.seed, xml_file=specified_kwargs)
	return env

if __name__ == "__main__":
	from config_params.default_config import default_config, dmc_config, metaworld_config
	config = dmc_config
	env = build_environment(config)
	state = env.reset()
	next_obs, reward, done, info = env.step(env.action_space.sample())
	print(next_obs, reward, done, info)
