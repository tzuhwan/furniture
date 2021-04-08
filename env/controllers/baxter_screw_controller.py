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
from env.controllers import BaxterIKController

"""
Screw controller for the Baxter robot, using Pybullet and the urdf description
files.
"""
class BaxterScrewController(BaxterIKController):

	"""
	Constructor.
	@param bullet_data_path, a string representing the base path to bullet data
	@param robot_jpos_getter, a function that returns the joint positions of
		the robot to be controlled as a numpy array
	@param verbose, a boolean that indicates how much output gets printed during controller execution
	@param debug, a boolean that indicates whether to print out pose information for end-effectors

	Inherited from Controller base class.
	"""
	def __init__(self, bullet_data_path, robot_jpos_getter, verbose=True, debug=False, suppress_output=False, potential_threshold=0.0005):
		print("BaxterScrewController: Initializing Screw Controller")

		# initialize super class
		super().__init__(bullet_data_path, robot_jpos_getter)

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

		# set move and rotate speed, for scaling motions; these are equivalent to controller gains
		self.default_move_speed = 0.025
		self.move_speed = self.default_move_speed
		self.default_rotate_speed = 0.01
		self.rotate_speed = self.default_rotate_speed

		# set arm speed, which controls how fast the arm performs the commands
		self.arm_step = 2

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

		# initialize control arm
		self.control_arm = ""

		# initialize variables for printing potentials
		self.num_iters = 0
		self.num_iters_print = 50

		# initialize variable for tracking convergence of controller
		self.potentials = np.ones(20)

		# initialize current configuration
		self.set_current_configuration()

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

		# set current configuration and compute potential
		self.set_current_configuration()
		pot = self.potential()
		self.potentials[self.num_iters%len(self.potentials)] = pot
		self.num_iters += 1

		# update motion speed based on convergence of controller
		if np.std(self.potentials) <= 6e-5:
			self.set_motion_speeds(move_speed=(self.move_speed + 0.01))
		else:
			if self.move_speed != self.default_move_speed:
				self.set_motion_speeds(move_speed=self.default_move_speed)

		# initialize velocities for proportional controller
		velocities = np.zeros(14)

		# if potential is low enough, no update needed
		if pot < self.potential_threshold:
			if (not self.suppress_output) and ((self.num_iters % self.num_iters_print) == 0):
				print("BaxterScrewController: Goal met! No update needed.")
			self.objective_met = True
			return velocities
		else:
			self.objective_met = False

		# compute dq and update state
		# this is done in iterations, as in BaxterIKController
		for _ in range(5):
			# get dq
			dq, dq_wrist = self.get_dq()

			# sync robot to match joint angles
			self.sync_ik_robot(dq, sync_last=True)

		# update remaining rotation
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
	Returns the arm being controlled by the controller.
	"""
	def get_control_arm(self):
		return self.control_arm

	"""
	Gets the name of the controller.
	"""
	def controller_name(self):
		return "BaxterScrewController"

	"""
	Sets the motion and rotation speeds for the controller.
	"""
	def set_motion_speeds(self, move_speed=None, rotate_speed=None):
		if move_speed is not None:
			self.move_speed = move_speed
		if rotate_speed is not None:
			self.rotate_speed = rotate_speed
		if self.debug:
			print("BaxterScrewController: Set new motion speeds, move_speed=%f, rotate_speed=%f" % (self.move_speed, self.rotate_speed))
		return

	"""
	Sets the arm speed for the controller.
	"""
	def set_arm_speed(self, arm_speed):
		self.arm_step = arm_speed
		if self.verbose:
			print("BaxterScrewController: Set new arm speed")
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
	"""
	def check_joint_limits(self):
		# compute relevant joint info
		joint_infos = list(self.get_relevant_joint_info())

		# get current joint states
		curr_q = self.robot_jpos_getter()

		# compare current joints to joint limits
		for i in range(len(curr_q)):
			j_idx, j_name, j_lower, j_upper, j_range = joint_infos[i]
			print("joint %s, current position %f, lower limit %f, upper limit %f"
				% (j_name, curr_q[i], j_lower, j_upper)
			)
			if (curr_q[i] <= j_lower) or (abs(curr_q[i] - j_lower) <= 1e-3):
				print("joint %s pretty close to lower limit" % j_name)
			if (curr_q[i] >= j_upper) or (abs(curr_q[i] - j_upper) <= 1e-3):
				print("joint %s pretty close to upper limit" % j_name)

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
		if (not self.suppress_output) and ((self.num_iters % self.num_iters_print) == 0):
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
	Computes the Jacobian for this controller.

	In this case, returns 3xn angular manipulator Jacobian.
	"""
	def get_objective_jacobian(self):
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

		if self.verbose:
			print("BaxterScrewController: computed %dx%d Jacobian" % (len(J_ang), len(J_ang[0])))

		return J_ang

	"""
	Computes the nullspace for performing lower-order controller commands subject to this controller.
	"""
	def get_objective_nullspace(self):
		# get (3xn) objective Jacobian
		J = self.get_objective_jacobian()

		# get (nxn) identity
		ndof = len(J[0])
		I = np.identity(ndof)

		# compute nullspace
		# nullspace should be identity
		# lower-order controller command should not conflict with screw command
		N = I # (nxn)

		if self.verbose:
			print("BaxterScrewController: computed %dx%d nullspace" % (len(N), len(N[0])))

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
		dq_projected = self.matrixMultiply(N, dq_lower_priority)

		# convert numpy array to list
		# dq_projected = list(dq_projected_mat)

		if self.verbose:
			print("BaxterScrewController: computed %dx1 projected controller command" % len(dq_projected))

		return dq_projected
