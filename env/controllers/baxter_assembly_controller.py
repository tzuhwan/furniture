"""
Controller parent class for assembly with Baxter

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
from env.controllers import BaxterIKController

"""
Controller parent class for the Baxter robot, using Pybullet and the urdf description
files.
"""
class BaxterAssemblyController(BaxterIKController):

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
		super().__init__(bullet_data_path, robot_jpos_getter)
		
		print("BaxterAssemblyController: Initializing Assembly Controller")

		# set verbose, debug, and suppress_output flags
		self.verbose = verbose
		self.debug = debug
		self.suppress_output = suppress_output

		# max potential
		self.max_potential = 100

		# potential threshold (potential less than this means no update will be performed)
		self.potential_threshold = potential_threshold

		# controller objective met flag
		self.objective_met = False

		# joint limit reached flag
		self.limit_reached = False

		# set move and rotate speed, for scaling motions; these are equivalent to controller gains
		self.default_move_speed = 0.025
		self.move_speed = self.default_move_speed
		self.default_rotate_speed = 0.05
		self.rotate_speed = self.default_rotate_speed

		# set arm speed, which controls how fast the arm performs the commands
		self.arm_step = 2
		self.update_arm_speed = False

		# initialize control arm
		self.control_arm = ""

		# initialize variables for printing potentials
		self.num_iters = 0
		self.num_iters_print = 50

		# initialize variable for tracking convergence of controller
		self.potentials = np.ones(20)

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
		if self.controller_name() == "BaxterAlignmentController":
			self.get_align_axis() # defined in BaxterAlignmentController
		if self.controller_name() == "BaxterScrewController":
			self.set_current_configuration() # defined in BaxterScrewController
		pot = self.potential()
		self.potentials[self.num_iters%len(self.potentials)] = pot
		self.num_iters += 1

		# update motion speed based on convergence of controller
		if self.update_arm_speed:
			if np.std(self.potentials) <= 6e-5:
				self.set_motion_speeds(move_speed=(self.move_speed + 0.01))
		# else: # TODO
		# 	if self.move_speed != self.default_move_speed:
		# 		self.set_motion_speeds(move_speed=self.default_move_speed)

		# check for joint limits
		self.limit_reached = self.check_joint_limits()
		if limit_reached:
			if not self.suppress_output:
				print("%s: Joint limit reached, controller behavior will not work well." % self.controller_name())

		# initialize velocities for proportional controller
		velocities = np.zeros(14)

		# if potential is low enough, no update needed
		if pot < self.potential_threshold:
			if (not self.suppress_output) and ((self.num_iters % self.num_iters_print) == 0):
				print("%s: Goal met! No update needed." % self.controller_name())
			self.objective_met = True
			return velocities
		else:
			self.objective_met = False

		# compute dq and update state
		# this is done in iterations, as in BaxterIKController
		for _ in range(5):
			# get dq
			if self.controller_name() == "BaxterScrewController":
				dq, dq_wrist = self.get_dq()
			else:
				dq = self.get_dq()

			# sync robot to match joint angles
			self.sync_ik_robot(dq, sync_last=True)

		if self.controller_name() == "BaxterScrewController":
			self.remaining_rotation -= dq_wrist

		# compute error between current and commanded joint positions
		deltas = self._get_current_error(self.robot_jpos_getter(), dq)

		# compute velocities based on error
		for i, delta in enumerate(deltas):
			velocities[i] = -self.arm_step * delta
		
		# clip velocities
		velocities = self.clip_joint_velocities(velocities)
		return velocities

	###########################
	### GETTERS AND SETTERS ###
	###########################

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
				print("%s: left pos from ik in base frame: " % self.controller_name(), curr_left_pos)
				print("%s: left rot from ik in base frame: " % self.controller_name(), curr_left_quat)
				print("%s: right pos from ik in base frame: " % self.controller_name(), curr_right_pos)
				print("%s: right rot from ik in base frame: " % self.controller_name(), curr_right_quat)
		else:
			if self.debug:	
				print("%s: given left pos in base frame: " % self.controller_name(), curr_left_pos)
				print("%s: given left rot in base frame: " % self.controller_name(), curr_left_quat)
				print("%s: given right pos in base frame: " % self.controller_name(), curr_right_pos)
				print("%s: given right rot in base frame: " % self.controller_name(), curr_right_quat)

		# compute left and right poses in world frame
		curr_left_pos_in_world, curr_left_quat_in_world = self.bullet_base_pose_to_world_pose(
			(curr_left_pos, curr_left_quat)
		)
		curr_right_pos_in_world, curr_right_quat_in_world = self.bullet_base_pose_to_world_pose(
			(curr_right_pos, curr_right_quat)
		)

		if self.debug:
			print("%s: left pos in world frame: " % self.controller_name(), curr_left_pos_in_world)
			print("%s: left rot in world frame: " % self.controller_name(), curr_left_quat_in_world)
			print("%s: right pos in world frame: " % self.controller_name(), curr_right_pos_in_world)
			print("%s: right rot in world frame: " % self.controller_name(), curr_right_quat_in_world)

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

	"""
	Gets the name of the controller.
	"""
	def controller_name(self):
		return "BaxterAssemblyController"

	"""
	Sets the motion and rotation speeds for the controller.
	"""
	def set_motion_speeds(self, move_speed=None, rotate_speed=None):
		if move_speed is not None:
			self.move_speed = move_speed
		if rotate_speed is not None:
			self.rotate_speed = rotate_speed
		if self.debug:
			print("%s: Set new motion speeds, move_speed=%f, rotate_speed=%f" % (self.controller_name(), self.move_speed, self.rotate_speed))
		return

	"""
	Sets the arm speed for the controller.
	"""
	def set_arm_speed(self, arm_speed):
		self.arm_step = arm_speed
		if self.verbose:
			print("%s: Set new arm speed" % self.controller_name())
		return

	"""
	Gets relevant joint info.
	"""
	def get_relevant_joint_info(self):
		# compute joint info for relevant joints
		joint_infos = [p.getJointInfo(self.ik_robot, i) for i in self.actual]

		# get relevant joint info
		joint_idxs = [i[0] for i in joint_infos]
		joint_names = [i[1] for i in joint_infos]
		joint_lower = [i[8] for i in joint_infos]
		joint_upper = [i[9] for i in joint_infos]
		joint_ranges = [(i[9] - i[8]) for i in joint_infos]

		return zip(joint_idxs, joint_names, joint_lower, joint_upper, joint_ranges)

	"""
	Checks for joint limits.

	@return boolean indicating if a joint limit has been reached
	"""
	def check_joint_limits(self):
		# compute relevant joint info
		joint_infos = list(self.get_relevant_joint_info())

		# get current joint states
		curr_q = self.robot_jpos_getter()

		# set flag for if joints are at limit
		limit_reached = False

		# compare current joints to joint limits
		for i in range(len(curr_q)):
			j_idx, j_name, j_lower, j_upper, j_range = joint_infos[i]
			if self.debug:
				print("joint %s, current position %f, lower limit %f, upper limit %f"
					% (j_name, curr_q[i], j_lower, j_upper)
				)
			if (curr_q[i] <= j_lower) or (abs(curr_q[i] - j_lower) <= 1e-3):
				limit_reached = True
				if self.debug:
					print("joint %s pretty close to lower limit" % j_name)
			if (curr_q[i] >= j_upper) or (abs(curr_q[i] - j_upper) <= 1e-3):
				limit_reached = True
				if self.debug:
					print("joint %s pretty close to upper limit" % j_name)

		return limit_reached

	###################################################
	### JACOBIAN AND NULLSPACE PROJECTION FUNCTIONS ###
	###################################################

	"""
	Multiplies two matrices together.

	@param m1_input, the first (mxp) matrix to multiply
	@param m2_input, the second (pxn) matrix to multiply
	@return the (mxn) matrix as a numpy array
	"""
	def matrixMultiply(self, m1_input, m2_input):
		# make both matrices numpy arrays
		m1 = np.array(m1_input)
		m2 = np.array(m2_input)

		# multiply matrices
		m = np.matmul(m1, m2)

		return m

	"""
	Computes the linear and angular Jacobians for controller.

	Returns 6x(n/2) linear Jacobian and 6x(n/2) angular Jacobian.
	"""
	def get_jacobians(self):
		# compute joint states and joint info
		joint_states = p.getJointStates(self.ik_robot, range(p.getNumJoints(self.ik_robot)))
		joint_infos = [p.getJointInfo(self.ik_robot, i) for i in range(p.getNumJoints(self.ik_robot))]
		
		# get joint states for relevant joints
		joint_states = [j for j, i in zip(joint_states, joint_infos) if i[3] > -1]
		if self.debug:
			relevant_joints = [(i[0], i[1]) for i in joint_infos if i[3] > -1]
			print(relevant_joints)

		# get joint positions from joint states
		joint_positions = [state[0] for state in joint_states]

		# set zero vector (for joint velocities and accelerations)
		zero_vec = [0.0] * len(joint_positions)

		# set local position on relevant joint
		local_pos = [0.0, 0.0, 0.0]

		# get appropriate end-effector
		if self.control_arm == "left":
			ee = self.effector_left
		else: # self.control_arm == "right"
			ee = self.effector_right

		# compute linear and angular Jacobians for all relevant joints
		J_lin_reljoints, J_ang_reljoints = p.calculateJacobian(
			self.ik_robot,
			ee,
			local_pos,
			joint_positions,
			zero_vec,
			zero_vec
		)

		if self.debug:
			print("linear Jacobian row size: ", len(J_lin_reljoints), "col size: ", len(J_lin_reljoints[0]))
			print("angular Jacobian size: ", len(J_ang_reljoints), "col size: ", len(J_ang_reljoints[0]))
		
		# first column corresponds to head_pan, which we do not care about for manipulation
		# initialize tuples for linear and angular Jacobians
		J_lin = ()
		J_ang = ()
		# remove first item in each row, which corresponds to head_pan
		for r in J_lin_reljoints:
			J_lin = J_lin + (r[1:],)
		for r in J_ang_reljoints:
			J_ang = J_ang + (r[1:],)

		if self.debug:
			print("linear Jacobian row size: ", len(J_lin), "col size: ", len(J_lin[0]))
			print("angular Jacobian size: ", len(J_ang), "col size: ", len(J_ang[0]))

		return J_lin, J_ang

	"""
	Computes the nullspace for performing lower-order controller commands subject to this controller.
	"""
	def get_objective_nullspace(self):
		# get (ixn) objective Jacobian
		J = self.get_objective_jacobian() # defined in each subclass

		# compute (nxi) pseudoinverse of Jacobian
		Jinv = np.linalg.pinv(J)

		# get (nxn) identity
		ndof = len(J[0])
		I = np.identity(ndof)

		# compute nullspace (make sure Jinv and J are numpy arrays)
		if self.controller_name() == "BaxterScrewController":
			# lower-order command should not conflict with single joint controlled by screw controller
			N = I
		else:
			N = I - self.matrixMultiply(Jinv, J) # (nxn) = (nxn) - (nxi) * (ixn)

		if self.verbose:
			print("%s: computed %dx%d nullspace" % (self.controller_name(), len(N), len(N[0])))

		return N

	"""
	Projects a lower objective controller command into the nullspace of this controller.

	@param lower_priority_controller_name, the string indicating the name of the lower priority controller being projected
	@param dq_lower_priority, the controller command from the lower priority controller
	@return the change in configuration of the lower priority controller projected into the nullspace of this controller
	"""
	def nullspace_projection(self, lower_priority_controller_name, dq_lower_priority):
		# get objective nullspace
		N = self.get_objective_nullspace()

		# project controller objective into nullspace as numpy array
		if lower_priority_controller_name == "BaxterScrewController":
			dq_projected = dq_lower_priority # screw controller only affects wrist roll, should not conflict, so no projection needed
		else:
			dq_projected = self.matrixMultiply(N, dq_lower_priority)

		# convert numpy array to list
		# dq_projected = list(dq_projected_mat)

		if self.verbose:
			print("%s: computed %dx1 projected controller command" % (self.controller_name(), len(dq_projected)))

		return dq_projected
