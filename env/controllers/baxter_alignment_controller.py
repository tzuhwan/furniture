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
from env.controllers import BaxterIKController

"""
Alignment controller for the Baxter robot, using Pybullet and the urdf description
files.
"""
class BaxterAlignmentController(BaxterIKController):

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
		print("BaxterAlignmentController: Initializing Alignment Controller")

		# initialize super class
		super().__init__(bullet_data_path, robot_jpos_getter)

		# set debug and verbose flags
		self.verbose = verbose
		self.debug = debug

		# max potential
		self.max_potential = 100

		# potential threshold (potential less than this means no update will be performed)
		self.potential_threshold = 0.004

		# controller objective met flag
		self.objective_met = False

		# set move and rotate speed, for scaling motions; these are equivalent to controller gains
		self.move_speed = 0.025
		self.rotate_speed = 0.001

		# set arm speed, which controls how fast the arm performs the commands
		self.arm_step = 2

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

		# set current pose and align axis and compute potential
		self.set_current_pose()
		self.get_align_axis()
		pot = self.potential()

		# initialize velocities for proportional controller
		velocities = np.zeros(14)

		# if potential is low enough, no update needed
		if pot < self.potential_threshold:
			print("BaxterAlignmentController: Goal met! No update needed.")
			self.objective_met = True
			return velocities
		else:
			self.objective_met = False

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
			velocities[i] = -self.arm_step * delta
		
		# clip velocities
		velocities = self.clip_joint_velocities(velocities)
		return velocities

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
		print("BaxterAlignmentController: New goal set for %s arm" % self.control_arm)
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
				print("BaxterAlignmentController: left pos from ik in base frame: ", curr_left_pos)
				print("BaxterAlignmentController: left rot from ik in base frame: ", curr_left_quat)
				print("BaxterAlignmentController: right pos from ik in base frame: ", curr_right_pos)
				print("BaxterAlignmentController: right rot from ik in base frame: ", curr_right_quat)
		else:
			if self.debug:	
				print("BaxterAlignmentController: given left pos in base frame: ", curr_left_pos)
				print("BaxterAlignmentController: given left rot in base frame: ", curr_left_quat)
				print("BaxterAlignmentController: given right pos in base frame: ", curr_right_pos)
				print("BaxterAlignmentController: given right rot in base frame: ", curr_right_quat)

		# compute left and right poses in world frame
		curr_left_pos_in_world, curr_left_quat_in_world = self.bullet_base_pose_to_world_pose(
			(curr_left_pos, curr_left_quat)
		)
		curr_right_pos_in_world, curr_right_quat_in_world = self.bullet_base_pose_to_world_pose(
			(curr_right_pos, curr_right_quat)
		)

		if self.debug:
			print("BaxterAlignmentController: left pos in world frame: ", curr_left_pos_in_world)
			print("BaxterAlignmentController: left rot in world frame: ", curr_left_quat_in_world)
			print("BaxterAlignmentController: right pos in world frame: ", curr_right_pos_in_world)
			print("BaxterAlignmentController: right rot in world frame: ", curr_right_quat_in_world)

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
	Returns the arm being controlled by the controller.
	"""
	def get_control_arm(self):
		return self.control_arm

	"""
	Sets the motion and rotation speeds for the controller.
	"""
	def set_motion_speeds(self, move_speed=None, rotate_speed=None):
		if move_speed is not None:
			self.move_speed = move_speed
		if rotate_speed is not None:
			self.rotate_speed = rotate_speed
		print("BaxterAlignmentController: Set new motion speeds, move_speed=%f, rotate_speed=%f" % (self.move_speed, self.rotate_speed))
		return

	"""
	Sets the arm speed for the controller.
	"""
	def set_arm_speed(self, arm_speed):
		self.arm_step = arm_speed
		print("BaxterAlignmentController: Set new arm speed")
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
			print("BaxterAlignmentController: computed %dx%d Jacobian" % (len(J_ang), len(J_ang[0])))

		return J_ang

	"""
	Computes the nullspace for performing lower-order controller commands subject to this controller.
	"""
	def get_objective_nullspace(self):
		# get (3xn) objective Jacobian
		J = self.get_objective_jacobian()

		# compute (nx3) pseudoinverse of Jacobian
		Jinv = np.linalg.pinv(J)

		# get (nxn) identity
		ndof = len(J[0])
		I = np.identity(ndof)

		# compute nullspace (make sure Jinv and J are numpy arrays)
		N = I - self.matrixMultiply(Jinv, J) # (nxn) = (nxn) - (nx3) * (3xn)

		if self.verbose:
			print("BaxterAlignmentController: computed %dx%d nullspace" % (len(N), len(N[0])))

		return N

	"""
	Projects a lower objective controller command into the nullspace of this controller.

	@param dq_lower_priority, the controller command from the lower priority controller
	@return the change in configuration of the lower priority controller projected into the nullspace of this controller
	"""
	def nullspace_projection(self, dq_lower_priority):
		# get objective nullspace
		N = self.get_objective_nullspace()

		# project controller objective into nullspace as numpy array
		dq_projected = self.matrixMultiply(N, dq_lower_priority) # TODO

		# convert numpy array to list
		# dq_projected = list(dq_projected_mat)

		if self.verbose:
			print("BaxterAlignmentController: computed %dx1 projected controller command" % len(dq_projected))

		return dq_projected
