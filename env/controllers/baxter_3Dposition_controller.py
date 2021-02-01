"""
3D position controller for Baxter

NOTE: requires pybullet module.

Run `pip install pybullet==1.9.5`.
"""

import os
import numpy as np
from pyquaternion import Quaternion

try:
    import pybullet as p
except ImportError:
    raise Exception(
        "Please make sure pybullet is installed. Run `pip install pybullet==1.9.5`"
    )

import env.transform_utils as T
from env.controllers import BaxterIKController

"""
3D position controller for the Baxter robot, using Pybullet and the urdf description
files.
"""
class Baxter3DPositionController(BaxterIKController):

	"""
	Constructor.
	@param bullet_data_path, a string representing the base path to bullet data
	@param robot_jpos_getter, a function that returns the joint positions of
		the robot to be controlled as a numpy array
	@param verbose, a boolean that indicates how much output gets printed during controller execution
	@param debug, a boolean that indicates whether to print out pose information for end-effectors

	Inherited from Controller base class.
	"""
	def __init__(self, bullet_data_path, robot_jpos_getter, verbose=True, debug=False):
		print("Baxter3DPositionController: Initializing 3DPosition Controller")

		# initialize super class
		super().__init__(bullet_data_path, robot_jpos_getter)

		# set debug and verbose flags
		self.debug = debug
		self.verbose = verbose

		# max potential
		self.max_potential = 100

		# controller gain
		self.kp = 1

		# potential threshold (potential less than this means no update will be performed)
		self.potential_threshold = 0.0005

		# set move and rotate speed, for scaling motions
		self.move_speed = 0.025
		self.rotate_speed = 0.05

		# initialize control arm
		self.control_arm = ""

		# initialize current pose
		self.set_current_pose()

	"""
    Syncs the internal Pybullet robot state to the joint positions of the
    robot being controlled.

    Inherited from Controller base class.
    """
	def sync_state(self):
		super().sync_state()

	"""
	Returns joint velocities to control the robot after the target end effector
    positions and orientations are updated from arguments @left and @right.
    If no arguments are provided, joint velocities will be computed based
    on the previously recorded target.

    Inherited from Controller base class.
	"""
	def get_control(self, right=None, left=None):
		# sync joint positions for IK
		self.sync_ik_robot(self.robot_jpos_getter())

		# set current pose and compute potential
		self.set_current_pose()
		pot = self.potential()

		# initialize velocities for proportional controller
		velocities = np.zeros(14)

		# if potential is low enough, no update needed
		if pot < self.potential_threshold:
			print("Baxter3DPositionController: Goal met! No update needed.")
			return velocities

		# compute dq and update state
		# this is done in iterations, as in BaxterIKController
		for _ in range(5):
			# get dq
			dq = self.get_dq()

			# sync robot to match joint angles
			self.sync_ik_robot(dq, sync_last=True)

		# compute error between current and commanded joint positions
		deltas = self._get_current_error(self.robot_jpos_getter(), dq)

		# compute velocities based on error
		for i, delta in enumerate(deltas):
			velocities[i] = -2 * delta # TODO what does the 2 do? scaling factor?
		
		# clip velocities
		velocities = self.clip_joint_velocities(velocities)
		return velocities

	###########################
	### GETTERS AND SETTERS ###
	###########################

	"""
	Sets the goal of the controller in world frame.
	"""
	def set_goal(self, control_arm, goal_pos, goal_quat=None):
		# check for valid arm
		if not ((control_arm == "left") or (control_arm == "right")):
			print("Baxter3DPositionController: Arm %s not recognized" % arm)
			raise NameError
			return

        # we now know arm is either "left" or "right"
		self.control_arm = control_arm
		self.goal_pos = goal_pos
		self.goal_quat = goal_quat # rotation will not be used to determine goal
		print("Baxter3DPositionController: New goal set for %s arm" % self.control_arm)
		return

	"""
	Sets the current pose of the robot in the world frame.

	@param curr_left_pos, the current position of the left arm in the base frame of the robot
	@param curr_left_quat, the current rotation of the left arm in the base frame of the robot
	@param curr_right_pos, the current position of the right arm in the base frame of the robot
	@param curr_right_quat, the current rotation of the right arm in the base frame of the robot
	@return none
	@post current pose of robot in world frame is updated internally
	"""
	def set_current_pose(self, curr_right_pos=None, curr_right_quat=None, curr_left_pos=None, curr_left_quat=None):
		# if none, then get pose and orientation of end-effectors in base frame from ik
		if curr_right_pos is None:
			curr_right_pos, curr_right_quat, curr_left_pos, curr_left_quat = self.ik_robot_eef_joint_cartesian_pose()
			if self.debug:
				print("Baxter3DPositionController: left pos from ik in base frame: ", curr_left_pos)
				print("Baxter3DPositionController: left rot from ik in base frame: ", curr_left_quat)
				print("Baxter3DPositionController: right pos from ik in base frame: ", curr_right_pos)
				print("Baxter3DPositionController: right rot from ik in base frame: ", curr_right_quat)
		else:
			if self.debug:	
				print("Baxter3DPositionController: given left pos in base frame: ", curr_left_pos)
				print("Baxter3DPositionController: given left rot in base frame: ", curr_left_quat)
				print("Baxter3DPositionController: given right pos in base frame: ", curr_right_pos)
				print("Baxter3DPositionController: given right rot in base frame: ", curr_right_quat)

		# compute left and right poses in world frame
		curr_left_pos_in_world, curr_left_quat_in_world = self.bullet_base_pose_to_world_pose(
			(curr_left_pos, curr_left_quat)
		)
		curr_right_pos_in_world, curr_right_quat_in_world = self.bullet_base_pose_to_world_pose(
			(curr_right_pos, curr_right_quat)
		)

		if self.debug:
			print("Baxter3DPositionController: left pos in world frame: ", curr_left_pos_in_world)
			print("Baxter3DPositionController: left rot in world frame: ", curr_left_quat_in_world)
			print("Baxter3DPositionController: right pos in world frame: ", curr_right_pos_in_world)
			print("Baxter3DPositionController: right rot in world frame: ", curr_right_quat_in_world)

		# set left and right poses
		self.curr_left_pos = curr_left_pos_in_world
		self.curr_left_quat = curr_left_quat_in_world
		self.curr_right_pos = curr_right_pos_in_world
		self.curr_right_quat = curr_right_quat_in_world

		# set pose of control_arm
		if self.control_arm == "left":
			self.curr_pos = self.curr_left_pos
			self.curr_quat = self.curr_left_quat
		else: # self.control_arm == "right"
			self.curr_pos = self.curr_right_pos
			self.curr_quat = self.curr_right_quat

		return

	"""
	Returns the arm being controlled by the controller.
	"""
	def get_control_arm(self):
		return self.control_arm

	#################################################
	### POTENTIAL FIELD AND CONTROL LAW FUNCTIONS ###
	#################################################

	"""
	Computes the potential of the controller based on the current and goal poses.
	Based on attractive potential field.
	"""
	def potential(self):
		# compute difference between current and goal pose
		diff_pos = self.curr_pos - self.goal_pos
		diff_quat = T.quat_multiply(T.quat_inverse(self.curr_quat), self.curr_quat) # no difference
		diff = np.hstack([diff_pos, diff_quat])

		# compute distance to goal
		dist_pos = np.linalg.norm(diff[:3])
		dist_quat = Quaternion.distance(Quaternion(self.curr_quat), Quaternion(self.curr_quat)) # no difference
		dist = dist_pos + dist_quat

		# compute potential
		pot = 0.5 * dist * dist
		print("Baxter3DPositionController: potential %f" % pot)
		if self.verbose:
			print("Baxter3DPositionController: position distance %f, rotation distance %f" % (dist_pos, dist_quat))
		return min(pot, self.max_potential)

	"""
	Computes the gradient of the controller based on the current and goal poses.
	Based on attractive potential field.
	"""
	def gradient(self):
		# compute difference between current and goal pose
		diff_pos = self.curr_pos - self.goal_pos
		diff_quat = T.quat_multiply(T.quat_inverse(self.curr_quat), self.curr_quat) # no difference
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
			target_left_pos = self.curr_pos + (self.move_speed * dx[:3]) # scale down position
			target_left_quat = T.quat_multiply(self.curr_quat, dx[3:7]) # scale down rotation not necessary, since no rotation command
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
			target_right_pos = self.curr_pos + (self.move_speed * dx[:3]) # scale down position
			target_right_quat = T.quat_multiply(self.curr_quat, dx[3:7]) # scale down rotation not necessary, since no rotation command

		# use inverse kinematics function to compute dq
		dq = self.inverse_kinematics(
			target_right_pos,
			target_right_quat,
			target_left_pos,
			target_left_quat,
			rest_poses=self.robot_jpos_getter()
		)

		return self.kp * dq

	"""
	Computes the nullspace for performing lower-order controller commands subject to this controller.
	"""
	def get_objective_nullspace(self):
		# TODO implement
		# NOTE: needed for nullspace composition later

		# Jlin, Jang = p.calculateJacobian(...)
		raise NotImplementedError
