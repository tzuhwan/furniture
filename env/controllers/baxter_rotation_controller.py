"""
Rotation controller for Baxter

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
Rotation controller for the Baxter robot, using Pybullet and the urdf description
files.
"""
class BaxterRotationController(BaxterAssemblyController):

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

		print("BaxterRotationController: Initializing Rotation Controller")

	###########################
	### GETTERS AND SETTERS ###
	###########################

	"""
	Sets the goal of the controller in world frame.
	"""
	def set_goal(self, control_arm, goal_rotation):
		# check for valid arm
		if not ((control_arm == "left") or (control_arm == "right")):
			print("BaxterRotationController: Arm %s not recognized" % control_arm)
			raise NameError
			return

		# check if rpy or xyzw
		if len(goal_rotation) == 3:
			goal_quat = T.euler_to_quat(goal_rotation)
		elif len(goal_rotation) == 4:
			goal_quat = goal_rotation
		else:
			print("BaxterRotationController: Unexpected rotation length of %d, should be 3 (rpy) or 4 (xyzw)" % len(goal_rotation))
			raise TypeError
			return

		# we now know arm is either "left" or "right"
		self.control_arm = control_arm
		self.goal_quat = np.array(goal_quat)
		if self.verbose:
			print("BaxterRotationController: New goal set for %s arm" % self.control_arm)
		return

	"""
	Gets the name of the controller.
	"""
	def controller_name(self):
		return "BaxterRotationController"

	#################################################
	### POTENTIAL FIELD AND CONTROL LAW FUNCTIONS ###
	#################################################

	"""
	Computes the potential of the controller based on the current and goal poses.
	Based on attractive potential field.
	"""
	def potential(self):
		# compute difference between current and goal pose
		diff_pos = self.curr_pos - self.curr_pos # no difference
		diff_quat = T.quat_multiply(T.quat_inverse(self.curr_quat), self.goal_quat)
		diff = np.hstack([diff_pos, diff_quat])

		# compute distance to goal
		dist_pos = np.linalg.norm(diff[:3]) # no difference
		dist_quat = Quaternion.distance(Quaternion(self.curr_quat), Quaternion(self.goal_quat))
		dist_quat = min(dist_quat , math.pi - dist_quat)
		dist = dist_pos + dist_quat

		# compute potential
		pot = 0.5 * dist * dist
		if self.verbose or ((not self.suppress_output) and ((self.num_iters % self.num_iters_print) == 0)):
			print("BaxterRotationController: potential %f" % pot)
		if self.verbose:
			print("BaxterRotationController: position distance %f, rotation distance %f" % (dist_pos, dist_quat))
		return min(pot, self.max_potential)

	"""
	Computes the gradient of the controller based on the current and goal poses.
	Based on attractive potential field.
	"""
	def gradient(self):
		# compute difference between current and goal pose
		diff_pos = self.curr_pos - self.curr_pos # no difference
		diff_quat = T.quat_multiply(T.quat_inverse(self.curr_quat), self.goal_quat)
		diff = np.hstack([diff_pos, diff_quat])

		# compute gradient
		grad = diff
		return grad

	"""
	Compute the change in pose induced by the controller.

	@param none
	@return the change in 6D pose induced by the controller
	"""
	def get_dx(self):
		# compute gradient
		grad = self.gradient()

		# compute dx
		dx = -grad
		return dx

	"""
	Compute the change in configuration induced by the controller.

	@param none
	@return the commanded joint positions induced by the controller
	"""
	def get_dq(self):
		# compute dx
		dx = self.get_dx()

		# compute targets
		if self.control_arm == "left":
			# target for left arm is to reach goal pose
			target_left_pos = self.curr_pos + (self.move_speed * dx[:3]) # scale down position not necessary, since no position command
			target_left_quat = T.quat_multiply(self.curr_quat, dx[3:7])
			target_left_quat = T.quat_slerp(self.curr_quat, target_left_quat, self.rotate_speed) # scale down rotation
			# target for right arm is to stay in place
			target_right_pos = self.curr_right_pos + np.zeros_like(dx[:3])
			target_right_quat_diff = T.quat_multiply(T.quat_inverse(self.curr_right_quat), self.curr_right_quat)
			target_right_quat = T.quat_multiply(self.curr_right_quat, target_right_quat_diff)
		else: # self.control_arm == "right"
			# target for left arm is to stay in place
			target_left_pos = self.curr_left_pos + np.zeros_like(dx[:3])
			target_left_quat_diff = T.quat_multiply(T.quat_inverse(self.curr_left_quat), self.curr_left_quat)
			target_left_quat = T.quat_multiply(self.curr_left_quat, target_left_quat_diff)
			# target for right arm is to reach goal pose
			target_right_pos = self.curr_pos + (self.move_speed * dx[:3]) # scale down position not necessary, since no position command
			target_right_quat = T.quat_multiply(self.curr_quat, dx[3:7])
			target_right_quat = T.quat_slerp(self.curr_quat, target_right_quat, self.rotate_speed) # scale down rotation

		# use inverse kinematics function to compute dq
		dq = self.inverse_kinematics(
			target_right_pos,
			target_right_quat,
			target_left_pos,
			target_left_quat,
			rest_poses=self.robot_jpos_getter()
		)

		return dq

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
			print("BaxterRotationController: computed %dx%d Jacobian" % (len(J_ang), len(J_ang[0])))

		return J_ang
