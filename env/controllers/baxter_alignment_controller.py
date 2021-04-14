"""
Alignment controller for Baxter

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
Alignment controller for the Baxter robot, using Pybullet and the urdf description
files.
"""
class BaxterAlignmentController(BaxterAssemblyController):

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
	def __init__(self, bullet_data_path, robot_jpos_getter, verbose=True, debug=False, suppress_output=False, potential_threshold=0.004):
		# initialize super class
		super().__init__(bullet_data_path, robot_jpos_getter, verbose, debug, suppress_output, potential_threshold)

		print("BaxterAlignmentController: Initializing Alignment Controller")

		# set move and rotate speed, for scaling motions; these are equivalent to controller gains
		self.default_move_speed = 0.025
		self.move_speed = self.default_move_speed
		self.default_rotate_speed = 0.001
		self.rotate_speed = self.default_rotate_speed

	###########################
	### GETTERS AND SETTERS ###
	###########################

	"""
	Sets the goal of the controller.

	@param control_arm, the arm to be controlled
			(options are "left" or "right")
	@param ee_point_axis, the axis of the end-effector that will point at the target
			(options are "+X", "-X", "+Y", "-Y", "+Z", "-Z")
	@param align_pos, the 3D position that the end-effector will point to in the world frame
	@return none
	@post controller goal updated internally
	"""
	def set_goal(self, control_arm, point_axis, align_pos):
		# check for valid arm
		if not ((control_arm == "left") or (control_arm == "right")):
			print("BaxterAlignmentController: Arm %s not recognized" % control_arm)
			raise NameError
			return

		# get axis
		if point_axis == "+X":
			ee_point_axis = [1, 0, 0]
		elif point_axis == "-X":
			ee_point_axis = [-1, 0, 0]
		elif point_axis == "+Y":
			ee_point_axis = [0, 1, 0]
		elif point_axis == "-Y":
			ee_point_axis = [0, -1, 0]
		elif point_axis == "+Z":
			ee_point_axis = [0, 0, 1]
		elif point_axis == "-Z":
			ee_point_axis = [0, 0, -1]
		else:
			print("BaxterAlignmentController: Axis %s not recognized as +/- XYZ. Using +Z as default.")
			ee_point_axis = [0, 0, 1]

		# we now know arm is either "left" or "right"
		self.control_arm = control_arm
		self.ee_point_axis = ee_point_axis
		self.align_pos = align_pos
		if self.verbose:
			print("BaxterAlignmentController: New goal set for %s arm" % self.control_arm)
		return

	"""
	Gets the axis to align with expressed in end-effector frame.
	"""
	def get_align_axis(self):
		# align pose w.r.t. world
		align_wrt_world = T.pose2mat(
			(self.align_pos, T.euler_to_quat([0, 0, 0]))
		)
		
		# pose of end-effector w.r.t. world
		ee_wrt_world = T.pose2mat(
			(self.curr_pos, self.curr_quat)
		)

		if self.debug:
			print("align pos: ", self.align_pos)
			print("curr pos: ", self.curr_pos)
		
		# pose of world w.r.t. end-effector
		world_wrt_ee = T.pose_inv(ee_wrt_world)

		# align pose w.r.t. end-effector
		# T(align pose w.r.t end-effector) = T(world w.r.t. end-effector) * T(align pose w.r.t. world)
		# (align pose in end-effector frame) = (world pose in end-effector frame) * (align pose in world frame)
		align_wrt_ee = T.pose_in_A_to_pose_in_B(align_wrt_world, world_wrt_ee)

		# get axis from pose
		align_axis, _ = T.mat2pose(align_wrt_ee)
		align_axis = T.norm(align_axis)

		# set align axis
		self.align_axis = align_axis

		return

	"""
	Gets the name of the controller.
	"""
	def controller_name(self):
		return "BaxterAlignmentController"

	#################################################
	### POTENTIAL FIELD AND CONTROL LAW FUNCTIONS ###
	#################################################

	"""
	Computes the potential of the controller based on the point and align axes.
	"""
	def potential(self):
		# compute the difference in angle between the two axes
		s = np.dot(self.ee_point_axis, self.align_axis) / (np.linalg.norm(self.ee_point_axis) * np.linalg.norm(self.align_axis))
		diff = math.acos(s)

		# compute potential
		pot = -((s * diff) - math.sqrt(1 - (s*s)))
		if self.verbose or ((not self.suppress_output) and ((self.num_iters % self.num_iters_print) == 0)):
			print("BaxterAlignmentController: potential %f" % pot)
		if self.verbose:
			print("BaxterAlignmentController: angle difference %f" % diff)
		return min(pot, self.max_potential)

	"""
	Computes the gradient of the controller based on the point and align axes.
	"""
	def gradient(self):
		# compute the difference in angle between the two axes
		s = np.dot(self.ee_point_axis, self.align_axis) / (np.linalg.norm(self.ee_point_axis) * np.linalg.norm(self.align_axis))
		diff = math.acos(s)

		# compute gradient
		grad = diff
		return grad

	"""
	Compute the change in pose induced by the controller.

	@param none
	@return the change in 6D pose induced by the controller
	"""
	def get_dx(self):
		# compute gradient (angle to rotate by)
		grad = -self.gradient()

		# compute the axis to rotate around
		r_axis = np.cross(self.ee_point_axis, self.align_axis)
		r_axis = T.norm(r_axis)
		if self.debug:
			print("point axis: ", self.ee_point_axis, "align axis: ", self.align_axis, "rotate axis: ", r_axis)

		# compute transform based on angle and axis, transform w.r.t. end-effector
		rot_wrt_ee = T.rotation_matrix(grad, r_axis)

		# pose of end-effector w.r.t. world
		ee_wrt_world = T.pose2mat(
			(self.curr_pos, self.curr_quat)
		)

		# compute rotation w.r.t. world
		# T(rot w.r.t. world) = T(end-effector w.r.t. world) * T(rot w.r.t. end-effector)
		# (rotation in world frame) = (end-effector in world frame) * (rotation in end-effector frame)
		rot_wrt_world = T.pose_in_A_to_pose_in_B(rot_wrt_ee, ee_wrt_world)

		# get rotation w.r.t. world
		_, dx_quat = T.mat2pose(rot_wrt_world)

		# we do not want to change linear position
		dx_pos = np.zeros(3)

		# compute dx
		dx = np.hstack([dx_pos, dx_quat])
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
			# target for left arm is to align with align pos
			target_left_pos = self.curr_pos + (self.move_speed * dx[:3]) # dx[:3] is zero vector
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
			# target for right arm is to align with align pos
			target_right_pos = self.curr_pos + (self.move_speed * dx[:3]) # dx[:3] is zero vector
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
			print("BaxterAlignmentController: computed %dx%d Jacobian" % (len(J_ang), len(J_ang[0])))

		return J_ang
