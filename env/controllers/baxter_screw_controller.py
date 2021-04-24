"""
Screw controller for Baxter

NOTE: requires pybullet module.

Run `pip install pybullet==1.9.5`.
"""

import os
import math
import numpy as np
from pyquaternion import Quaternion

try:
	import pybullet as p
except ImportError:
	raise Exception(
		"Please make sure pybullet is installed. Run `pip install pybullet==1.9.5`"
	)

import env.transform_utils as T
from env.controllers import BaxterAssemblyController

"""
Screw controller for the Baxter robot, using Pybullet and the urdf description
files.
"""
class BaxterScrewController(BaxterAssemblyController):

	"""
	Constructor.
	@param bullet_data_path, a string representing the base path to bullet data
	@param robot_jpos_getter, a function that returns the joint positions of
		the robot to be controlled as a numpy array
	@param verbose, a boolean that indicates how much output gets printed during controller execution
	@param debug, a boolean that indicates whether to print out pose information for end-effectors
	@param suppress_output, a boolean that indicates whether to suppress all output from controller
	@param potential_threshold, a float that indicates when controller is considered converged

	Inherited from Controller base class.
	"""
	def __init__(self, bullet_data_path, robot_jpos_getter, verbose=True, debug=False, suppress_output=False, potential_threshold=0.0005):
		# initialize super class
		super().__init__(bullet_data_path, robot_jpos_getter, verbose, debug, suppress_output, potential_threshold)

		print("BaxterScrewController: Initializing Screw Controller")

		# set move and rotate speed, for scaling motions; these are equivalent to controller gains
		self.default_move_speed = 0.025
		self.move_speed = self.default_move_speed
		self.default_rotate_speed = 0.01
		self.rotate_speed = self.default_rotate_speed

		# set indices of right_w2 and left_w2 joints based on baxter urdf file
		self.right_wrist_idx = 20
		self.left_wrist_idx = 38

		# set joint limits for wrists
		right_info = p.getJointInfo(self.ik_robot, self.right_wrist_idx)
		self.right_lower_limit = right_info[8]
		self.right_upper_limit = right_info[9]
		self.right_range = right_info[9] - right_info[9]
		left_info = p.getJointInfo(self.ik_robot, self.left_wrist_idx)
		self.left_lower_limit = left_info[8]
		self.left_upper_limit = left_info[9]
		self.left_range = left_info[9] - left_info[8]

		# set indices of right_w2 and left_w2 joints based on the mapping of relevant joints
		self.right_wrist_relevant_idx = self.actual.index(self.right_wrist_idx)
		self.left_wrist_relevant_idx = self.actual.index(self.left_wrist_idx)

		# initialize current configuration
		self.set_current_configuration()

	###########################
	### GETTERS AND SETTERS ###
	###########################

	"""
	Sets the goal of the controller in world frame.
	"""
	def set_goal(self, control_arm, relative_rotation):
		# check for valid arm
		if not ((control_arm == "left") or (control_arm == "right")):
			print("BaxterScrewController: Arm %s not recognized" % control_arm)
			raise NameError
			return

		# we now know arm is either "left" or "right"
		self.control_arm = control_arm
		self.total_rotation = relative_rotation
		self.remaining_rotation = relative_rotation
		if self.verbose:
			print("BaxterScrewController: New goal set for %s arm" % self.control_arm)
		return

	"""
	Sets the current configuration of the robot.
	"""
	def set_current_configuration(self):
		# compute the current configuration
		self.curr_q = self.robot_jpos_getter()
		if self.debug:
			print("BaxterScrewController: current configuration: ", self.curr_q)

		# set left and right wrist joint positions
		self.curr_left_wrist_q = self.curr_q[self.left_wrist_relevant_idx]
		self.curr_right_wrist_q = self.curr_q[self.right_wrist_relevant_idx]

		# set pose of control_arm
		if self.control_arm == "left":
			self.curr_wrist_q = self.curr_left_wrist_q
		else: # self.control_arm == "right"
			self.curr_wrist_q = self.curr_right_wrist_q

		return

	"""
	Gets the name of the controller.
	"""
	def controller_name(self):
		return "BaxterScrewController"

	#################################################
	### POTENTIAL FIELD AND CONTROL LAW FUNCTIONS ###
	#################################################

	"""
	Computes the potential of the controller based on the total remaining rotation to achieve.
	Based on attractive potential field.
	"""
	def potential(self):
		# compute difference between current and goal rotation
		diff = -self.remaining_rotation

		# compute distance to goal
		dist = np.linalg.norm(diff)

		# compute potential
		pot = 0.5 * dist * dist
		if self.verbose or ((not self.suppress_output) and ((self.num_iters % self.num_iters_print) == 0)):
			print("BaxterScrewController: potential %f" % pot)
		return min(pot, self.max_potential)

	"""
	Computes the gradient of the controller based on the total remaining rotation to achieve.
	"""
	def gradient(self):
		# compute difference between current and goal rotation
		diff = -self.remaining_rotation

		# compute gradient
		grad = diff
		return grad

	"""
	Compute the change in configuration induced by the controller.

	@param none
	@return the commanded joint positions induced by the controller
	@return the commanded change in the controlled wrist joint
	"""
	def get_dq(self):
		# compute gradient
		grad = -self.gradient()

		# initialize dq
		dq = self.robot_jpos_getter()

		# update dq based on controlled wrist
		if self.control_arm == "left":
			commanded_change = dq[self.left_wrist_relevant_idx] + (self.rotate_speed * grad) # scale down rotation
			if (commanded_change <= self.left_lower_limit) or (commanded_change >= self.left_upper_limit):
				if (not self.suppress_output) and ((self.num_iters % self.num_iters_print) == 0):
					print("BaxterScrewController: too close to joint limits, no update needed")
					print("BaxterScrewController: current wrist position %f, lower joint limit %f, upper joint limit %f"
						% (dq[self.left_wrist_relevant_idx], self.left_lower_limit, self.left_upper_limit))
				self.objective_met = True
			else:
				dq[self.left_wrist_relevant_idx] = commanded_change
		else: # self.control_arm == "right"
			commanded_change = dq[self.right_wrist_relevant_idx] + (self.rotate_speed * grad) # scale down rotation
			if (commanded_change <= self.right_lower_limit) or (commanded_change >= self.right_upper_limit):
				if (not self.suppress_output) and ((self.num_iters % self.num_iters_print) == 0):
					print("BaxterScrewController: too close to joint limits, no update needed")
					print("BaxterScrewController: current wrist position %f, lower joint limit %f, upper joint limit %f"
						% (dq[self.right_wrist_relevant_idx], self.right_lower_limit, self.right_upper_limit))
					self.objective_met = True
			else:
				dq[self.right_wrist_relevant_idx] = commanded_change

		# compute change in rotation to wrist joint
		dq_wrist = abs(self.rotate_speed * grad)

		return dq, dq_wrist

	###################################################
	### JACOBIAN AND NULLSPACE PROJECTION FUNCTIONS ###
	###################################################

	"""
	Computes the Jacobian for this controller.

	In this case, returns 3xn angular manipulator Jacobian.
	"""
	def get_objective_jacobian(self):
		# compute linear and angular Jacobian
		J_lin, J_ang = self.get_jacobians()

		if self.verbose:
			print("BaxterScrewController: computed %dx%d Jacobian" % (len(J_ang), len(J_ang[0])))

		return J_ang
