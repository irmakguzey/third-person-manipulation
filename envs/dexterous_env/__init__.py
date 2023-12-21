from gym.envs.registration import register 

register(
	id='CubeFlipping-v1',
	entry_point='dexterous_env.cube_flipping_simulation_env:CubeFlippingEnv',
	max_episode_steps=100,
)

register(
	id='CubeFlippingArm-v1',
	entry_point='dexterous_env.cube_flipping_with_arm_simulation_env:CubeFlippingArmEnv',
	max_episode_steps=100,
)