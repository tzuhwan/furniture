"""
Define baxter furniture controller planner environment class FurnitureBaxterControllerPlannerEnv
"""

import os
import time
from collections import OrderedDict

import numpy as np
from itertools import permutations
import random
import math

import env.models
from env.furniture_baxter import FurnitureBaxterEnv
import env.transform_utils as T
from env.controllers import Baxter6DPoseController
from env.controllers import BaxterObject6DPoseController
from env.controllers import Baxter3DPositionController
from env.controllers import BaxterAlignmentController
from env.controllers import BaxterRotationController

from env.controller_planner import ControlBasis

"""
Baxter robot environment for planning controller compositions.
"""
class FurnitureBaxterControllerPlannerEnv(FurnitureBaxterEnv):

	"""
	Constructor.
	@param config, configurations for the environment
	"""
	def __init__(self, config, debug=False):
		config.agent_type = 'Baxter'

		# initialize FurnitureBaxterEnv
		super().__init__(config)

		# debug flag
		self.debug = debug

		# initialize number of samples for estimating
		self.num_samples = 100

		# initialize control basis
		self.cb = ControlBasis()

		# approximate workspace boundaries
		self.workspace_x = (0.31, 0.85)
		self.workspace_y = (-0.70, 0.70)
		self.workspace_z = (-0.11, 0.70)

	"""
	Takes a simulation step with @a and computes reward.
	"""
	def _step(self, a):
		return super(FurnitureBaxterEnv, self)._step(a)

	"""
	Resets simulation and variables to compute reward.
	@param furniture_id, ID of the furniture model to reset
	@param background, name of the background scene to reset
	"""
	def _reset(self, furniture_id=None, background=None):
		super()._reset(furniture_id, background)

	"""
	Sets up simulation.
	"""
	def setup_sim(self, config):
		# sets up environment and robot
		if config.furniture_name is not None:
			config.furniture_id = furniture_name2id[config.furniture_name]
		ob = self.reset(config.furniture_id, config.background)

		if config.render:
			self.render()

		from util.video_recorder import VideoRecorder
		self.vr = VideoRecorder()
		self.vr.add(self.render('rgb_array'))

	"""
	Updates sim to match given velocities.
	"""
	# def update_sim(self, velocities):
	# 	# set up low action
	# 	low_action = np.concatenate([velocities, [-1, -1]])

	# 	# keep trying to reach the target in a closed-loop
	# 	ctrl = self._setup_action(low_action)
	# 	for i in range(self._action_repeat):
	# 			self._do_simulation(ctrl)

	# 			if i + 1 < self._action_repeat:
	# 				low_action = np.concatenate([velocities, [-1, -1]])
	# 				ctrl = self._setup_action(low_action)

	# 	return

	"""
	Plans controller compositions for the given action name
	"""
	def plan_controller_compositions(self, config, control_arm="left", action_name="screw-into"):
		# setup simulator
		self.setup_sim(config)

		# get controllers required to execute given action from temporal decomposition
		action_controllers = self.cb.temporal_decomposition[action_name]

		# initialize controllers
		self.controllers = {}
		for controller in action_controllers:
			if controller == "Baxter6DPoseController":
				self.controllers[controller] = Baxter6DPoseController(
					bullet_data_path=os.path.join(env.models.assets_root, "bullet_data"),
					robot_jpos_getter=self._robot_jpos_getter,
					verbose=False
				)
			elif controller == "Baxter3DPositionController":
				self.controllers[controller] = Baxter3DPositionController(
					bullet_data_path=os.path.join(env.models.assets_root, "bullet_data"),
					robot_jpos_getter=self._robot_jpos_getter,
					verbose=False
				)
			elif controller == "BaxterAlignmentController":
				self.controllers[controller] = BaxterAlignmentController(
					bullet_data_path=os.path.join(env.models.assets_root, "bullet_data"),
					robot_jpos_getter=self._robot_jpos_getter,
					verbose=False
				)
			elif controller == "BaxterRotationController":
				self.controllers[controller] = self._controller = BaxterRotationController(
					bullet_data_path=os.path.join(env.models.assets_root, "bullet_data"),
					robot_jpos_getter=self._robot_jpos_getter,
					verbose=False
				)
			else:
				print("FurnitureBaxterControllerPlannerEnv: controller type %s not recognized" % controller)
				raise NameError

		# get possible compositions
		composition_possibilities = self.get_possible_controller_compositions(action_controllers)

		# initialize list of probabilities
		composition_probabilities = []

		# initialize counter
		count = 0

		# compute probability of achieving combined effects for each composition
		for composition in composition_possibilities:
			count += 1
			print("FurnitureBaxterControllerPlannerEnv: computing computation %d of %d" % (count, len(composition_possibilities)))

			# initialize probability
			prob = 1
			# update probability based on highest order controller
			prob *= self.cb.causal_models[composition[0]]

			# update probability for each lower order controller
			for i in range(1,len(composition)):
				# estimate probability of achieving lower order objective given higher order objectives
				estimated_prob = self.estimate_from_distribution(control_arm, composition[:i+1])
				# update probability based on controller
				prob *= estimated_prob

			# store computed probability in list
			composition_probabilities.append(prob)

		# find max composition probability
		max_prob = max(composition_probabilities)
		compositions_probabilities = [(c, p) for c, p in zip(composition_possibilities, composition_probabilities)]
		print("compositions and probabilities: ", compositions_probabilities)

		# get compositions that achieve max probability
		compositions = [c for c, p in zip(composition_possibilities, composition_probabilities) if p >= max_prob]

		return compositions
	
	"""
	Generates all possible compositions of controllers by permuting the given list of controllers.
	"""
	def get_possible_controller_compositions(self, controllers):
		possible_compositions = list(permutations(controllers))
		return possible_compositions

	"""
	Estimates the probability that the lowest order objective is achieved given the higher
	order objectives from the distribution induced by the composed controller.
	"""
	def estimate_from_distribution(self, control_arm, controllers):
		# get relevant joint info
		joint_info = self.controllers[controllers[0]].get_relevant_joint_info()
		joint_info = list(joint_info)

		# perform sampling
		estimated_prob = 0
		for i in range(self.num_samples):
			print("FurnitureBaxterControllerPlannerEnv: computing sample %d of %d" % (i+1, self.num_samples))

			# compute random initial state
			random_init_state = self.random_joint_state(joint_info)
			# print("random state: ", random_init_state) # TODO
			# velocities = self._robot_jpos_getter() - random_init_state
			# self.update_sim(velocities) # TODO update sim to match state?
			self.sim.data.qpos[self._ref_joint_pos_indexes] = random_init_state
			# print("joint state after update: ", self._robot_jpos_getter()) # TODO
			# render (for visualization purposes)
			# self.vr.add(self.render('rgb_array')) # TODO update sim to match state?

			# compute random controller goals
			for c in self.controllers.keys():
				random_goal_state = self.random_controller_goal(c, self.controllers[c], control_arm)
				# print("controller ", c, " random goal : ", random_goal_state) # TODO

			# compute unprojected command for lowest priority controller
			dq = self.controllers[controllers[-1]].get_control()

			# compute projected command for lowest priority controller
			dq_projected = dq
			for c in controllers[:-1]:
				dq_projected = self.controllers[c].nullspace_projection(dq_projected)

			# compute metric for sample
			dq_norm = np.linalg.norm(dq)
			dq_proj_norm = np.linalg.norm(dq_projected)
			if dq_norm == 0:
				sample_prob = 0
			else:
				sample_prob = dq_proj_norm / dq_norm
			print("sample probability: ", sample_prob)

			# add to running sum
			estimated_prob += sample_prob

			# time.sleep(1)

		# compute average metric across samples
		estimated_prob /= self.num_samples
		print("estimated probability: ", estimated_prob)
		return estimated_prob

	"""
	Computes random joint state.
	"""
	def random_joint_state(self, joint_info):
		# initialize random state
		joint_state = []

		# loop through joints
		for jinfo in joint_info:
			jidx, jname, jlower, jupper, jrange = jinfo
			jstate = random.uniform(jlower, jupper)
			if self.debug:
				print("joint %d %s, lower %f, upper %f, random state %f"
					% (jidx, jname, jlower, jupper, jstate)
				)
			joint_state.append(jstate)

		return joint_state

	"""
	Computes and sets random goal for given controller.
	"""
	def random_controller_goal(self, controller_name, controller, control_arm):
		# compute random goal for controller type
		if controller_name == "Baxter6DPoseController":
			rand_xpos = random.uniform(self.workspace_x[0], self.workspace_x[1])
			rand_ypos = random.uniform(self.workspace_y[0], self.workspace_y[1])
			rand_zpos = random.uniform(self.workspace_z[0], self.workspace_z[1])
			rand_pos = [rand_xpos, rand_ypos, rand_zpos]
			rand_quat = T.random_quat()
			controller.set_goal(control_arm, rand_pos, rand_quat)
			return (rand_pos, rand_quat)
		elif controller_name == "Baxter3DPositionController":
			rand_xpos = random.uniform(self.workspace_x[0], self.workspace_x[1])
			rand_ypos = random.uniform(self.workspace_y[0], self.workspace_y[1])
			rand_zpos = random.uniform(self.workspace_z[0], self.workspace_z[1])
			rand_pos = [rand_xpos, rand_ypos, rand_zpos]
			controller.set_goal(control_arm, rand_pos)
			return rand_pos
		elif controller_name == "BaxterAlignmentController":
			rand_axpos = random.uniform(self.workspace_x[0], self.workspace_x[1])
			rand_aypos = random.uniform(self.workspace_y[0], self.workspace_y[1])
			rand_azpos = random.uniform(self.workspace_z[0], self.workspace_z[1])
			rand_apos = [rand_axpos, rand_aypos, rand_azpos]
			controller.set_goal(control_arm, "+Z", rand_apos)
			return ("+Z", rand_apos)
		elif controller_name == "BaxterRotationController":
			rand_rot = random.uniform(0, 2*math.pi)
			controller.set_goal(control_arm, rand_rot)
			return rand_rot
		else:
			print("FurnitureBaxterControllerPlannerEnv: controller type %s not recognized" % controller)
			raise NameError

		return None

"""
Main function; will not be called when environment is constructed from appropriate demo
"""
def main():
	import argparse
	import config.furniture as furniture_config
	from util import str2bool

	parser = argparse.ArgumentParser()
	furniture_config.add_argument(parser)

	# change default config for Baxter
	parser.add_argument('--seed', type=int, default=123)
	parser.add_argument('--debug', type=str2bool, default=False)

	parser.set_defaults(render=True)

	config, unparsed = parser.parse_known_args()

	# create an environment and run manual control of Baxter environment
	env = FurnitureBaxterAssemblyEnv(config)
	env.plan_controller_compositions(config)

if __name__ == "__main__":
	main()
