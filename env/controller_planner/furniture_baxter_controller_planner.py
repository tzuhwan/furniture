"""
Define baxter furniture controller planner environment class FurnitureBaxterControllerPlannerEnv
"""

import os
import time
from collections import OrderedDict

import numpy as np
import random
import math

import env.models
from env.furniture_baxter import FurnitureBaxterEnv
import env.transform_utils as T
from env.controllers import Baxter6DPoseController
from env.controllers import BaxterObject6DPoseController
from env.controllers import Baxter3DPositionController
from env.controllers import BaxterRotationController
from env.controllers import BaxterAlignmentController
from env.controllers import BaxterScrewController

from env.controller_planner import ControlBasis

"""
Baxter robot environment for planning controller compositions.
"""
class FurnitureBaxterControllerPlannerEnv(FurnitureBaxterEnv):

	"""
	Constructor.
	@param config, configurations for the environment
	"""
	def __init__(self, config):
		config.agent_type = 'Baxter'

		# initialize FurnitureBaxterEnv
		super().__init__(config)

		# verbose and debug flags
		self.verbose = config.verbose
		self.debug = config.debug

		# flag for visualizing random samples (slows down computation while simulator updates)
		self.visualize_samples = config.visualize_samples

		# initialize number of samples for estimating
		self.num_samples = config.num_samps

		# initialize number of walkout iterations
		self.num_walkout_iters = 300

		# initialize control basis
		self.cb = ControlBasis()

		# initialize control arm and action
		self.control_arm = config.control_arm
		self.action_name = config.action

		# approximate workspace boundaries
		self.workspace_x = (0.31, 0.85)
		self.workspace_y = (-0.70, 0.70)
		self.workspace_z = (-0.11, 0.70)

		# approximate rotation bounadries in degrees(keep gripper facing down)
		self.rotation_x = (-70, 70)
		self.rotation_y = (-70, 70)
		self.rotation_z = (-70, 70)

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
	Gets initial rotation of left and right end-effectors.

	Note: random rotation samples will be centered around this initial rotation.
	"""
	def get_init_rotation(self):
		# get initial euler of left and right grippers
		self.left_init_euler = (180/math.pi) * np.array(T.quat_to_euler(self._left_hand_quat))
		self.right_init_euler = (180/math.pi) * np.array(T.quat_to_euler(self._right_hand_quat))
		if self.debug:
			print("initial left euler: ", self.left_init_euler, "initial right euler: ", self.right_init_euler)

	"""
	Sets up simulation.
	"""
	def setup_sim(self, config):
		# sets up environment and robot
		ob = self.reset(config.furniture_id, config.background)

		if config.render:
			self.render()

		from util.video_recorder import VideoRecorder
		self.vr = VideoRecorder()
		self.vr.add(self.render('rgb_array'))

	"""
	Updates sim to match given velocities.
	"""
	def update_sim(self, velocities):
		# set up low action
		low_action = np.concatenate([velocities, [-1, -1]])

		# keep trying to reach the target in a closed-loop
		ctrl = self._setup_action(low_action)
		for i in range(self._action_repeat):
				self._do_simulation(ctrl)

				if i + 1 < self._action_repeat:
					low_action = np.concatenate([velocities, [-1, -1]])
					ctrl = self._setup_action(low_action)

		return

	"""
	Performs the given multi-objective controller command.
	Copied from part of _step_continuous() function in furniture.py

	@param velocities, the change in configuration induced by the multi-objective controller
	@param composition, the list of controllers indicating the order of the composition
	"""
	def perform_multiobjective_command(self, velocities, composition):
		# set up low action
		low_action = np.concatenate([velocities, [-1, -1]])

		# keep trying to reach the target in a closed-loop
		ctrl = self._setup_action(low_action)
		for i in range(self._action_repeat):
				self._do_simulation(ctrl)

				if i + 1 < self._action_repeat:
					velocities, _ = self.cb.compute_multiobjective_controller_update(composition)
					low_action = np.concatenate([velocities, [-1, -1]])
					ctrl = self._setup_action(low_action)

		return

	###########################################################################
	### POINT-BASED SAMPLING FOR DETERMINING CONTROLLER COMPOSITION OFFLINE ###
	###########################################################################

	"""
	Plans controller compositions for the given action name.
	"""
	def plan_controller_compositions(self, config):
		# setup simulator
		self.setup_sim(config)

		# get initial rotation
		self.get_init_rotation()

		# get controllers required to execute given action from temporal decomposition
		action_controllers = self.cb.temporal_decomposition[self.action_name]

		# initialize controllers
		self.cb.initialize_controllers(action_controllers,
			bullet_data_path=os.path.join(env.models.assets_root, "bullet_data"),
			robot_jpos_getter=self._robot_jpos_getter,
			suppress_output=True
		)

		# get possible compositions
		composition_possibilities = self.cb.get_possible_controller_compositions(action_controllers)

		# initialize list of probabilities
		composition_probabilities = []

		# initialize counter
		count = 0

		# compute probability of achieving combined effects for each composition
		for composition in composition_possibilities:
			count += 1
			print("FurnitureBaxterControllerPlanner: computing success probability for composition %s (%d of %d)"
				% (self.cb.multiobjective_controller_string(composition), count, len(composition_possibilities))
			)

			# initialize probability
			prob = 1
			# update probability based on highest order controller
			prob *= self.cb.causal_models[composition[0]]

			# update probability for each lower order controller
			for i in range(1,len(composition)):
				# estimate probability of achieving lower order objective given higher order objectives
				estimated_prob = self.estimate_from_distribution(composition[:i+1])
				# update probability based on controller
				prob *= estimated_prob

			# store computed probability in list
			composition_probabilities.append(prob)

		# find max composition probability
		max_prob = max(composition_probabilities)
		compositions_probabilities = [(c, p) for c, p in zip(composition_possibilities, composition_probabilities)]
		print("***** COMPOSITIONS AND PROBABILITIES *****")
		for c, p in compositions_probabilities:
			print("controller %s, probability of success %f"
				% (self.cb.multiobjective_controller_string(c), p)
			)

		# get compositions that achieve max probability
		compositions = [c for c, p in zip(composition_possibilities, composition_probabilities) if p >= max_prob]

		return compositions

	"""
	Estimates the probability that the lowest order objective is achieved given the higher
	order objectives from the distribution induced by the composed controller.
	"""
	def estimate_from_distribution(self, controllers):
		# get relevant joint info
		joint_info = self.cb.controllers[controllers[0]].get_relevant_joint_info()
		joint_info = list(joint_info)

		# perform sampling
		estimated_prob = 0
		sample_probs = []
		for i in range(self.num_samples):
			if (i % 20) == 0:
				print("FurnitureBaxterControllerPlanner: computing point-wise sample %d of %d" % (i+1, self.num_samples))

			# compute random initial state and set sim state
			random_init_state = self.random_joint_state(joint_info)
			self.sim.data.qpos[self._ref_joint_pos_indexes] = random_init_state

			# if samples visualized, update unity to match
			if self.visualize_samples:
				self._unity_updated = False
				velocities = self._robot_jpos_getter() - random_init_state
				self.update_sim(velocities)
				# render (for visualization purposes)
				self.vr.add(self.render('rgb_array'))

			# compute random controller goals
			for c in self.cb.controllers.keys():
				random_goal_state = self.random_controller_goal(c, self.cb.controllers[c])

			# compute unprojected command for lowest priority controller
			dq = self.cb.controllers[controllers[-1]].get_control()

			# compute projected command for lowest priority controller
			dq_projected = dq
			for c in controllers[:-1]:
				dq_projected = self.cb.controllers[c].nullspace_projection(c, dq_projected)

			# compute metric for sample
			dq_norm = np.linalg.norm(dq)
			dq_proj_norm = np.linalg.norm(dq_projected)
			if dq_norm == 0:
				sample_prob = 0
			else:
				sample_prob = dq_proj_norm / dq_norm

			if self.verbose:
				print("sample probability: %f" % sample_prob)

			# add to running samples
			sample_probs.append(sample_prob)

		# compute average metric across samples
		estimated_prob = np.mean(sample_probs)
		if self.verbose:
			print("estimated probability: %f, spread: %f"
				% (estimated_prob, np.std(sample_probs))
			)
		return estimated_prob

	###############################################################
	### WALKOUTS FOR DETERMINING CONTROLLER COMPOSITION OFFLINE ###
	###############################################################

	"""
	Selects controller compositions for given action name.
	"""
	def walkout_controller_compositions(self, config):
		# setup simulator
		self.setup_sim(config)

		# get initial rotation
		self.get_init_rotation()

		# get controllers required to execute given action from temporal decomposition
		action_controllers = self.cb.temporal_decomposition[self.action_name]

		# initialize controllers
		self.cb.initialize_controllers(action_controllers,
			bullet_data_path=os.path.join(env.models.assets_root, "bullet_data"),
			robot_jpos_getter=self._robot_jpos_getter,
			suppress_output=False
		)

		# get possible compositions
		composition_possibilities = self.cb.get_possible_controller_compositions(action_controllers)

		# initialize dictionary of probabilities
		composition_probabilities = {}
		composition_sample_probs = {}
		composition_iterations = {}
		composition_sample_iters = {}

		# initialize counter
		count = 0

		# compute probability of achieving combined effects for each composition using a walkout
		for composition in composition_possibilities:
			count += 1
			print("FurnitureBaxterControllerPlanner: performing walkout for composition %s (%d of %d)"
				% (self.cb.multiobjective_controller_string(composition), count, len(composition_possibilities))
			)

			# initialize lists of probabilities and iterations for each composition
			composition_probabilities[composition] = 0
			composition_sample_probs[composition] = []
			composition_iterations[composition] = 0
			composition_sample_iters[composition] = []

			# perform sampling by constructing controller composition trajectory
			for i in range(self.num_samples):
				if (i % 20) == 0:
					print("FurnitureBaxterControllerPlanner: computing trajectory sample %d of %d" % (i+1, self.num_samples))

				# compute walkout from initial start state
				prob, num_iters = self.perform_walkout(composition)

				# store computed probability in list
				composition_sample_probs[composition].append(prob)
				composition_sample_iters[composition].append(num_iters)

			# compute average estimated probability across samples
			composition_probabilities[composition] = np.mean(composition_sample_probs[composition])
			composition_iterations[composition] = np.mean(composition_sample_iters[composition])
			if self.verbose:
				print("estimated probability: %f, spread: %f" 
					% (composition_probabilities[composition], np.std(composition_sample_probs[composition]))
				)
				print("average iterations: %f, spread: %f"
					% (composition_iterations[composition], np.std(composition_sample_iters[composition]))
				)

		# find max composition probability and fewest iterations
		max_prob = max(composition_probabilities.values()) # max probability ensures composition is most likely to succeed
		# min_iter = min(composition_iterations.values()) # if min iterations is less than fixed parameter, then probability=1
		compositions_probabilities_iterations = []
		for c in composition_possibilities:
			compositions_probabilities_iterations.append(
				(c, composition_probabilities[c], composition_iterations[c])
			)
		print("***** COMPOSITIONS, PROBABILITIES, AND ITERATIONS *****")
		for c, p, i in compositions_probabilities_iterations:
			print("controller %s, probability of success %f, iterations %d"
				% (self.cb.multiobjective_controller_string(c), p, i)
			)
			if self.debug:
				print("probabilities for each sample: ", composition_sample_probs[c])
				print("iterations for each sample: ", composition_sample_iters[c])

		# get compositions that achieve max probability and fewest iterations
		compositions = [c for c, p, i in compositions_probabilities_iterations if (p >= max_prob)]
		if len(compositions) > 1:
			min_iter = min([composition_iterations[c] for c in compositions])
			compositions = [c for c, p, i in compositions_probabilities_iterations if (p >= max_prob) and (i <= min_iter)]

		return compositions

	"""
	Performs walkout to test different controller compositions for the given action name.
	"""
	def perform_walkout(self, controllers):
		# get relevant joint info
		joint_info = self.cb.controllers[controllers[0]].get_relevant_joint_info()
		joint_info = list(joint_info)

		# initialize flag for objective met
		objective_met = False

		# initialize counter
		num_iters = 0

		# compute random initial state and set sim state
		random_init_state = self.random_joint_state(joint_info)
		self.sim.data.qpos[self._ref_joint_pos_indexes] = random_init_state

		# compute random controller goals
		for c in self.cb.controllers.keys():
			random_goal_state = self.random_controller_goal(c, self.cb.controllers[c])

		# compute initial distance from goal
		init_potential, _ = self.cb.compute_multiobjective_potentials(controllers)

		# compute multi-objective potential and check for controller progress
		pot, progress = self.cb.compute_multiobjective_potentials(controllers)

		# run controller
		while progress and (not objective_met) and (num_iters < self.num_walkout_iters):
			# set flag so unity will update
			# self._unity_updated = False # TODO do not render walkouts for better sim behavior
			# compute controller update
			velocities, objective_met = self.cb.compute_multiobjective_controller_update(controllers)
			num_iters += 1
			# perform controller command
			self.perform_multiobjective_command(velocities, controllers)
			# render
			# self.vr.add(self.render('rgb_array')) # TODO do not render walkouts for better sim behavior

		# compute probability
		if objective_met: # if objective met, probability of reaching goal is 1
			prob = 1
			if self.verbose:
				print("objective met! sample probability: %f" % prob)
		elif not progress: # if progress not being made, probability of reaching goal is 0
			prob = 0
			if self.verbose:
				print("bad progress made, sample probability: %f" % prob)
		else: # num_iters > self.num_walkout_iters
			# compute final distance from goal
			final_potential, _ = self.cb.compute_multiobjective_potentials(controllers)
			# if final potential is greater than initial, composition would not have reached goal
			if final_potential > init_potential:
				prob = 0
			else: # consider progress towards goal
				prob = (init_potential - final_potential) / init_potential
			if self.verbose:
				print("exceeded max iterations, sample probability: %f" % prob)

		return prob, num_iters

	########################################################
	### COMPUTE RANDOM JOINT STATES AND CONTROLLER GOALS ###
	########################################################

	"""
	Computes random joint state.
	"""
	def random_joint_state(self, joint_info):
		# initialize random state
		joint_state = []

		# loop through joints
		for jinfo in joint_info:
			jidx, jname, jlower, jupper, jrange = jinfo
			# sample from uniform distribution over joint range
			jstate = random.uniform(jlower, jupper)
			# sample from normal distribution over joint range, clipping if necessary
			# mu = (jupper + jlower) / 2
			# std = ((jupper - jlower) / 2) / 3
			# jsample = random.gauss(mu, std)
			# if jstate < jlower:
			# 	jstate = jlower
			# elif jstate > jupper:
			# 	jstate = jupper
			# else:
			# 	jstate = jsample
			if self.debug:
				print("joint %d %s, lower %f, upper %f, random state %f"
					% (jidx, jname, jlower, jupper, jstate)
				)
			joint_state.append(jstate)

		return joint_state

	"""
	Computes and sets random goal for given controller.
	"""
	def random_controller_goal(self, controller_name, controller):
		# compute random goal for controller type
		if controller_name == "Baxter6DPoseController":
			rand_xpos = random.uniform(self.workspace_x[0], self.workspace_x[1])
			rand_ypos = random.uniform(self.workspace_y[0], self.workspace_y[1])
			rand_zpos = random.uniform(self.workspace_z[0], self.workspace_z[1])
			rand_pos_unity = [rand_xpos, rand_ypos, rand_zpos]
			rand_xrot = random.uniform(self.left_init_euler[0]-self.rotation_x[0], self.left_init_euler[0]-self.rotation_x[1])
			rand_yrot = random.uniform(self.left_init_euler[1]-self.rotation_y[0], self.left_init_euler[1]-self.rotation_y[1])
			rand_zrot = random.uniform(self.left_init_euler[2]-self.rotation_z[0], self.left_init_euler[2]-self.rotation_z[1])
			rand_quat = T.euler_to_quat([rand_xrot, rand_yrot, rand_zrot])
			rand_pos, _ = T.mat2pose(
				self.pose_in_base_from_pose_in_unity(
					T.pose2mat((rand_pos_unity, T.euler_to_quat([0, 0, 0])))
				)
			)
			controller.set_goal(self.control_arm, rand_pos, rand_quat)
			return (rand_pos, rand_quat)
		elif controller_name == "Baxter3DPositionController":
			rand_xpos = random.uniform(self.workspace_x[0], self.workspace_x[1])
			rand_ypos = random.uniform(self.workspace_y[0], self.workspace_y[1])
			rand_zpos = random.uniform(self.workspace_z[0], self.workspace_z[1])
			rand_pos_unity = [rand_xpos, rand_ypos, rand_zpos]
			rand_pos, _ = T.mat2pose(
				self.pose_in_base_from_pose_in_unity(
					T.pose2mat((rand_pos_unity, T.euler_to_quat([0, 0, 0])))
				)
			)
			controller.set_goal(self.control_arm, rand_pos)
			return rand_pos
		elif controller_name == "BaxterRotationController":
			rand_xrot = random.uniform(self.left_init_euler[0]-self.rotation_x[0], self.left_init_euler[0]-self.rotation_x[1])
			rand_yrot = random.uniform(self.left_init_euler[1]-self.rotation_y[0], self.left_init_euler[1]-self.rotation_y[1])
			rand_zrot = random.uniform(self.left_init_euler[2]-self.rotation_z[0], self.left_init_euler[2]-self.rotation_z[1])
			rand_quat = T.euler_to_quat([rand_xrot, rand_yrot, rand_zrot])
			controller.set_goal(self.control_arm, rand_quat)
		elif controller_name == "BaxterAlignmentController":
			rand_axpos = random.uniform(self.workspace_x[0], self.workspace_x[1])
			rand_aypos = random.uniform(self.workspace_y[0], self.workspace_y[1])
			rand_azpos = random.uniform(self.workspace_z[0], self.workspace_z[1])
			rand_apos_unity = [rand_axpos, rand_aypos, rand_azpos]
			rand_apos, _ = T.mat2pose(
				self.pose_in_base_from_pose_in_unity(
					T.pose2mat((rand_apos_unity, T.euler_to_quat([0, 0, 0])))
				)
			)
			controller.set_goal(self.control_arm, "+Z", rand_apos)
			return ("+Z", rand_apos)
		elif controller_name == "BaxterScrewController":
			rand_rot = random.uniform(0, 2*math.pi)
			controller.set_goal(self.control_arm, rand_rot)
			return rand_rot
		else:
			print("FurnitureBaxterControllerPlanner: controller type %s not recognized" % controller)
			raise NameError

		return None

	"""
	Computes pose in base frame from pose in unity.
	"""
	def pose_in_base_from_pose_in_unity(self, pose_in_unity):
		# get base pose in unity
		base_pos_in_unity = self.sim.data.get_body_xpos("base")
		base_rot_in_unity = self.sim.data.get_body_xmat("base").reshape((3, 3))
		base_pose_in_unity = T.make_pose(base_pos_in_unity, base_rot_in_unity)

		# get pose of unity in base
		unity_pose_in_base = T.pose_inv(base_pose_in_unity)

		# pose in base = pose of unity in base * pose in unity
		pose_in_base = T.pose_in_A_to_pose_in_B(pose_in_unity, unity_pose_in_base)
		return pose_in_base

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
