"""
Demo for assembling furniture with Baxter.
"""

import argparse

import numpy as np

from env import make_env
from env.models import furniture_names, furniture_name2id, background_names
from util import str2bool
from env.controller_planner import ControlBasis

"""
Inputs types of agent, furniture model, and background and simulates the environment.
"""
def main(args):
	print("Baxter Furniture Assembly Environment")

	env_types = ["FurnitureBaxterAssemblyEnv", "FurnitureBaxterAssemblyAutoEnv", "FurnitureBaxterControllerPlannerEnv"]

	# select environment
	print()
	print("Supported environments:\n")
	for i, env_type in enumerate(env_types):
		print('{}: {}'.format(i, env_type))
	print()
	try:
		s = input("Choose an environment (enter a number from 0 to {}): ".format(len(env_types) - 1))
		env_name = env_types[int(s)]
	except:
		print("Input is not valid. Use 0 by default.")
		env_name = env_types[0]

	# set parameters for the environment (furniture_id, background)
	if env_name == 'FurnitureBaxterControllerPlannerEnv':
		# use default furniture and background
		furniture_name = 'block'
		furniture_id = furniture_name2id['block']
		background_name = background_names[3]

		# choose a control arm
		print()
		print("Supported control arms:\n")
		for i, arm in enumerate(["left", "right"]):
			print('{}: {}'.format(i, arm))
		print()
		try:
			s = input("Choose a control arm (enter either 0 or 1): ")
			control_arm = ["left", "right"][int(s)]
		except:
			print("Input is not valid. Use 0 by default.")
			control_arm = ["left", "right"][0]

		# choose a multi-objective action
		cb = ControlBasis()
		print()
		print("Supported multi-objective actions:\n")
		for i, action in enumerate(cb.temporal_decomposition.keys()):
			print('{}: {}'.format(i, action))
		print()
		try:
			s = input("Choose a multi-objective action (enter a number from 0 to {}): ".format(len(cb.temporal_decomposition.keys()) - 1))
			action_idx = int(s)
			action = list(cb.temporal_decomposition.keys())[action_idx]
		except:
			print("Input is not valid. Use 0 by default.")
			action_idx = 0
			action = list(cb.temporal_decomposition.keys())[action_idx]

		# choose a number of samples
		print()
		try:
			s = input("Enter a number of samples: ")
			num_samps = int(s)
			assert num_samps > 0, "Number of samples should be greater than 0."
		except:
			print("Input is not valid. Use 100 by default.")
			num_samps = 100

		# set chosen variables
		args.control_arm = control_arm
		args.action = action
		args.num_samps = num_samps
		print()
		print("Chosen parameters (control arm: %s, action: %s, number of samples: %d)" % (control_arm, action, num_samps))
	else:
		# choose a furniture model
		print()
		print("Supported furniture:\n")
		for i, furniture_name in enumerate(furniture_names):
			print('{}: {}'.format(i, furniture_name))
		print()
		try:
			s = input("Choose a furniture model (enter a number from 0 to {}): ".format(len(furniture_names) - 1))
			furniture_id = int(s)
			furniture_name = furniture_names[furniture_id]
		except:
			print("Input is not valid. Use 0 by default.")
			furniture_id = 0
			furniture_name = furniture_names[0]

		# choose a background scene
		print()
		print("Supported backgrounds:\n")
		for i, background in enumerate(background_names):
			print('{}: {}'.format(i, background))
		print()
		try:
			s = input("Choose a background (enter a number from 0 to {}): ".format(len(background_names) - 1))
			k = int(s)
			background_name = background_names[k]
		except:
			print("Input is not valid. Use 0 by default.")
			background_name = background_names[0]

	# set environment args
	args.env = env_name
	args.furniture_id = furniture_id
	args.furniture_name = furniture_name
	args.background = background_name
	args.live_connect_coppeliasim = False
	args.grasp_pose_json_file = 'default_furniture_grasp_poses.json'

	print()
	print("Creating assembly environment (robot: {}, furniture: {}, background: {})".format(
		env_name, furniture_name, background_name))

	# make environment following arguments
	env = make_env(env_name, args)

	# run environment
	if env_name == "FurnitureBaxterControllerPlannerEnv":
		# compositions = env.plan_controller_compositions(args)
		compositions = env.walkout_controller_compositions(args)
		print("possible compositions: ", compositions)
	else:
		# run assembly of furniture
		env.run_controller(args)
		# env.run_manual(args)

	# close the environment instance
	print("Closing %s" % args.env)
	print("Thank you for trying %s!" % args.env)
	env.close()

"""
Returns argument parser for furniture assembly environment
"""
def argsparser():
	parser = argparse.ArgumentParser("Demo for furniture assembly with Baxter")
	parser.add_argument('--seed', type=int, default=123)
	parser.add_argument('--debug', type=str2bool, default=False)
	parser.add_argument('--record_video', type=str2bool, default=False)
	parser.add_argument('--video_name', type=str, default='FurnitureBaxterAssemblyEnv_test.mp4')
	parser.add_argument('--verbose', type=str2bool, default=True)
	parser.add_argument('--visualize_samples', type=str2bool, default=False)

	import config.furniture as furniture_config
	furniture_config.add_argument(parser)

	args = parser.parse_args()
	return args

if __name__ == "__main__":
	args = argsparser()
	main(args)
