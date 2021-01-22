"""
6D pose controller for Baxter

NOTE: requires pybullet module.

Run `pip install pybullet==1.9.5`.
"""

import os
import numpy as np

try:
    import pybullet as p
except ImportError:
    raise Exception(
        "Please make sure pybullet is installed. Run `pip install pybullet==1.9.5`"
    )

import env.transform_utils as T
from env.controllers import Controller

"""
6D pose controller for the Baxter robot, using Pybullet and the urdf description
files.
"""
class Baxter6DPoseController(BaxterIKController):

	"""
	Constructor.
	@param bullet_data_path, a string representing the base path to bullet data
	@param robot_jpos_getter, a function that returns the joint positions of
		the robot to be controlled as a numpy array

	Inherited from Controller base class.
	"""
	def __init__(self, bullet_data_path, robot_jpos_getter):
		super().__init__(bullet_data_path, robot_jpos_getter)

	"""
	Returns joint velocities to control the robot after the target end effector
    positions and orientations are updated from arguments @left and @right.
    If no arguments are provided, joint velocities will be computed based
    on the previously recorded target.

    Inherited from Controller base class.
	"""
	def get_control(self, right=None, left=None):
		# TODO implement

		# Jlin, Jang = p.calculateJacobian(...)
		# calls super().joint_positions_for_eef_command(right, left)
		# 	which computes inverse kinematics and sets robot joint configuration
		raise NotImplementedError

	"""
    Syncs the internal Pybullet robot state to the joint positions of the
    robot being controlled.

    Inherited from Controller base class.
    """
	def sync_state(self):
		# TODO implement
		raise NotImplementedError

	def _get_dx(self, current, goal):
		# TODO implement
		raise NotImplementedError


	def _get_dq(self, dx):
		# TODO implement
		# NOTE:
		# 	this is not needed for this controller to work,
		# 	since super().joint_positions_for_eef_command() computes the change in configuration;
		#   however, controllers should have this function implemented for nullspace composition later

		# Jlin, Jang = p.calculateJacobian(...)
		raise NotImplementedError

	def _get_objective_nullspace(self):
		# TODO implement
		# NOTE: needed for nullspace composition later

		# Jlin, Jang = p.calculateJacobian(...)
		raise NotImplementedError