"""
6D pose controller for objects being manipulated by Baxter

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
6D pose controller for objects being manipulated by the Baxter robot, using Pybullet
and the urdf description files.
"""
class BaxterObject6DPoseController(BaxterIKController):

	"""
	Constructor.
	@param bullet_data_path, a string representing the base path to bullet data
	@param robot_jpos_getter, a function that returns the joint positions of
		the robot to be controlled as a numpy array
	@param verbose, a boolean that indicates how much output gets printed during controller execution
	@param debug, a boolean that indicates whether to print out pose information for end-effectors

	Inherited from Controller base class.
	"""
	def __init__(self, bullet_data_path, robot_jpos_getter, objects_in_scene, verbose=True, debug=False):
		print("BaxterObject6DPoseController: Initializing Object 6DPose Controller")

		# initialize super class
		super().__init__(bullet_data_path, robot_jpos_getter)

		# set debug and verbose flags
		self.verbose = verbose
		self.debug = debug

		# set list of objects in scene
		self.object_names = objects_in_scene

		# max potential
		self.max_potential = 100

		# potential threshold (potential less than this means no update will be performed)
		self.potential_threshold = 0.003

		# controller objective met flag
		self.objective_met = False

		# set move and rotate speed, for scaling motions; these are equivalent to controller gains
		self.move_speed = 0.025 # TODO
		self.rotate_speed = 0.03 # TODO

		# set arm speed, which controls how fast the arm performs the commands
		self.arm_step = 2

		# initialize control arm and control object
		self.control_arm = ""
		self.object_name = ""

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

		# set current pose and object pose and compute potential
		self.set_current_pose()
		self.get_current_object_pose()
		pot = self.potential()

		# initialize velocities for proportional controller
		velocities = np.zeros(14)

		# if potential is low enough, no update needed
		if pot < self.potential_threshold:
			print("BaxterObject6DPoseController: Goal met! No update needed.")
			self.objective_met = True
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
	def set_goal(self, control_arm, object_name, object_goal_pos, object_goal_quat):
		# check for valid arm
		if not ((control_arm == "left") or (control_arm == "right")):
			print("BaxterObject6DPoseController: Arm %s not recognized" % control_arm)
			raise NameError
			return

		if object_name not in self.object_names:
			print("BaxterObject6DPoseController: Object %s not recognized" % object_name)
			print("BaxterObject6DPoseController: The recognized objects are ", self.object_names)
			raise NameError
			return

        # we now know arm is either "left" or "right" and object is in scene
		self.control_arm = control_arm
		self.object_name = object_name
		self.object_goal_pos = object_goal_pos
		self.object_goal_quat = object_goal_quat
		print("BaxterObject6DPoseController: New goal set for %s arm and object %s" % (self.control_arm, self.object_name))
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
				print("BaxterObject6DPoseController: left pos from ik in base frame: ", curr_left_pos)
				print("BaxterObject6DPoseController: left rot from ik in base frame: ", curr_left_quat)
				print("BaxterObject6DPoseController: right pos from ik in base frame: ", curr_right_pos)
				print("BaxterObject6DPoseController: right rot from ik in base frame: ", curr_right_quat)
		else:
			if self.debug:	
				print("BaxterObject6DPoseController: given left pos in base frame: ", curr_left_pos)
				print("BaxterObject6DPoseController: given left rot in base frame: ", curr_left_quat)
				print("BaxterObject6DPoseController: given right pos in base frame: ", curr_right_pos)
				print("BaxterObject6DPoseController: given right rot in base frame: ", curr_right_quat)

		# compute left and right poses in world frame
		curr_left_pos_in_world, curr_left_quat_in_world = self.bullet_base_pose_to_world_pose(
			(curr_left_pos, curr_left_quat)
		)
		curr_right_pos_in_world, curr_right_quat_in_world = self.bullet_base_pose_to_world_pose(
			(curr_right_pos, curr_right_quat)
		)

		if self.debug:
			print("BaxterObject6DPoseController: left pos in world frame: ", curr_left_pos_in_world)
			print("BaxterObject6DPoseController: left rot in world frame: ", curr_left_quat_in_world)
			print("BaxterObject6DPoseController: right pos in world frame: ", curr_right_pos_in_world)
			print("BaxterObject6DPoseController: right rot in world frame: ", curr_right_quat_in_world)

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
	TODO
	"""
	def set_body_poses(self, body_pose_dict):
		# initialize dictionary of body names to poses in world frame
		self.body_pose_dict_world = {}

		# convert poses to world frame
		for body in body_pose_dict.keys():
			# get pos and quat in base frame from matrix
			body_pos, body_quat = T.mat2pose(body_pose_dict[body])
			# convert to world frame
			body_pos_world, body_quat_world = self.bullet_base_pose_to_world_pose(
				(body_pos, body_quat)
			)
			# set pose in world frame in dictionary
			self.body_pose_dict_world[body] = (body_pos_world, body_quat_world)

		return

	"""
	Sets the pose of the given object name in the base frame.

	@param name, the name of the body whose pose is being set
	@param pose_matrix, the matrix representing the pose of the object in the base frame
	"""
	def set_object_pose(self, name, pose_matrix):
		if name == self.object_name:
			pos, quat = T.mat2pose(pose_matrix)
			self.object_curr_pos_base = pos
			self.object_curr_quat_base = quat
			if self.debug:
				print("BaxterObject6DPoseController:", self.object_name, "pos in base: ", self.object_curr_pos_base)
				print("BaxterObject6DPoseController:", self.object_name, "rot in base: ", self.object_curr_quat_base)
		else:
			print("BaxterObject6DPoseController: object name %s not recognized; not setting pose" % name)
		return

	"""
	Gets the pose of the object in the world frame.
	"""
	def get_current_object_pose(self):
		object_pos_world, object_quat_world = self.bullet_base_pose_to_world_pose(
			(self.object_curr_pos_base, self.object_curr_quat_base)
		)
		self.object_curr_pos = object_pos_world
		self.object_curr_quat = object_quat_world
		if True:#self.debug:
			print("BaxterObject6DPoseController:", self.object_name, "pos in world: ", self.object_curr_pos)
			print("BaxterObject6DPoseController:", self.object_name, "rot in world: ", self.object_curr_quat)
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
		print("BaxterObject6DPoseController: Set new motion speeds, move_speed=%f, rotate_speed=%f" % (self.move_speed, self.rotate_speed))
		return

	"""
	Sets the arm speed for the controller.
	"""
	def set_arm_speed(self, arm_speed):
		self.arm_step = arm_speed
		print("BaxterObject6DPoseController: Set new arm speed")
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
	Returns the object being controlled by the controller.
	"""
	def get_object_name(self):
		return self.object_name

	#################################################
	### POTENTIAL FIELD AND CONTROL LAW FUNCTIONS ###
	#################################################

	"""
	Computes the potential of the controller based on the current and goal poses.
	Based on attractive potential field.
	"""
	def potential(self):
		# compute difference between current and goal pose
		diff_pos = self.object_curr_pos - self.object_goal_pos
		diff_quat = T.quat_multiply(T.quat_inverse(self.object_curr_quat), self.object_goal_quat)
		diff = np.hstack([diff_pos, diff_quat])

		# compute distance to goal
		dist_pos = np.linalg.norm(diff[:3])
		dist_quat = Quaternion.distance(Quaternion(self.object_curr_quat), Quaternion(self.object_goal_quat))
		dist = dist_pos + dist_quat

		# compute potential
		pot = 0.5 * dist * dist
		print("BaxterObject6DPoseController: potential %f" % pot)
		if True:#self.verbose: # TODO
			print("BaxterObject6DPoseController: position distance %f, rotation distance %f" % (dist_pos, dist_quat))
		return min(pot, self.max_potential)

	"""
	Computes the gradient of the controller based on the current and goal poses.
	Based on attractive potential field.
	"""
	def gradient(self):
		# compute difference between current and goal pose
		diff_pos = self.object_curr_pos - self.object_goal_pos
		diff_quat = T.quat_multiply(T.quat_inverse(self.object_curr_quat), self.object_goal_quat)
		diff = np.hstack([diff_pos, diff_quat])

		# compute gradient
		grad = diff
		return grad

	"""
	Compute the change in object pose induced by the controller.

	@param none
	@return the change in object 6D pose induced by the controller
	"""
	def get_dx(self):
		# compute gradient of object
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
		# compute dx of object
		dx_xyzquat = self.get_dx()

		# get euler angles from quaternion
		dx_rpy = T.quat_to_euler(dx_xyzquat[3:7])

		# get (6x1) dx
		dx = np.hstack([dx_xyzquat[:3], dx_rpy])

		# scale down dx
		dx[:3] = self.move_speed * dx[:3]
		dx[3:] = self.rotate_speed * dx[3:]

		# compute (6xn) Jacobian with end-effector as object
		J = self.compute_object_jacobian()

		# compute (nx6) pseudoinverse of Jacobian
		Jinv = np.linalg.pinv(J)

		# compute change in configuration
		dq = self.matrixMultiply(Jinv, dx) # (nx1) = (nx6) * (6x1)

		# compute ik solution by from current joint states and dq
		# note this ik solution does not accurately reflect arm that should stay stationary
		ik_solution = self.robot_jpos_getter() + dq

		# compute arm targets for stationary arms
		target_left_pos = self.curr_left_pos + np.zeros_like(dx[:3])
		target_left_quat_diff = T.quat_multiply(T.quat_inverse(self.curr_left_quat), self.curr_left_quat)
		target_left_quat = T.quat_multiply(self.curr_left_quat, target_left_quat_diff)
		target_right_pos = self.curr_right_pos + np.zeros_like(dx[:3])
		target_right_quat_diff = T.quat_multiply(T.quat_inverse(self.curr_right_quat), self.curr_right_quat)
		target_right_quat = T.quat_multiply(self.curr_right_quat, target_right_quat_diff)

		# use inverse kinematics function to compute ik solution
		ik_solution_stationary = self.inverse_kinematics(
			target_right_pos,
			target_right_quat,
			target_left_pos,
			target_left_quat,
			rest_poses=self.robot_jpos_getter()
		)

		# merge two ik solutions; ik_solution for arm that moves, ik_solution_stationary for arm that does not
		if self.control_arm == "left":
			control_command = np.hstack([ik_solution_stationary[:7], ik_solution[7:]])
		else: # self.control_arm == "right":
			control_command = np.hstack([ik_solution[:7], ik_solution_stationary[7:]])

		return control_command

	"""
	Compute the manipulator Jacobian with the object as the end-effector.
	Replaces the column corresponding to the gripper with the appropriate column
	for the object.
	"""
	def compute_object_jacobian(self):
		# get Jacobian, convert to numpy array
		J = self.get_objective_jacobian()
		J = np.array(J)

		# compute indices for left and right wrists in the Jacobian
		lidx = self.actual.index(38)
		ridx = self.actual.index(20)

		# only columns corresponding to kinematic chain leading to controlled end-effector need to be updated
		if self.control_arm == "left":
			jstart_idx = ridx+1
			jend_idx = lidx+1
		else: # self.control_arm == "right"
			jstart_idx = 0
			jend_idx = ridx+1

		# change appropriate columns of Jacobian
		for i in range(jstart_idx, jend_idx):
			# get the column from the Jacobian
			col = []
			for row in J:
				col.append(row[i])

			# compute new column with object as end-effector
			# get joint axis w.r.t. world
			joint_axis_world = col[3:6]

			# get pose of link controlled by joint in world frame
			info = p.getJointInfo(self.ik_robot, self.actual[i])
			controlled_link = info[12]
			joint_origin_world, _ = self.body_pose_dict_world[controlled_link.decode("utf-8")] # controlled link has same pose as parent joint

			# "end-effector" (object) origin w.r.t. world - joint origin w.r.t. world
			o = self.object_curr_pos - joint_origin_world
			# compute linear component
			lin = np.cross(joint_axis_world, o)
			# compute angular component
			ang = joint_axis_world
			# compute new column
			new_col = np.hstack([lin, ang])

			# set new column in Jacobian
			for r in range(len(J)):
				J[r][i] = new_col[r]

		return J

		# # get columns for left and right wrists from Jacobian
		# lcol = []
		# rcol = []
		# for row in J:
		# 	lcol.append(row[lidx])
		# 	rcol.append(row[ridx])

		# # compute new column with object as end-effector
		# # get joint axis w.r.t. world for control arm
		# if self.control_arm == "left":
		# 	joint_axis_world = lcol[3:6]
		# else: # self.control_arm == "right"
		# 	joint_axis_world = rcol[3:6]

		# # "end-effector" (object) origin w.r.t. world - joint (same as end-effector) origin w.r.t. world
		# o = self.object_curr_pos - self.curr_pos
		# # compute linear component
		# lin = np.cross(joint_axis_world, o)
		# # compute angular component
		# ang = joint_axis_world
		# # compute new column
		# Jcol_object = np.hstack([lin, ang])

		# # get index for new column in Jacobian
		# if self.control_arm == "left":
		# 	idx = lidx
		# else: # self.control_arm == "right"
		# 	idx = ridx

		# # set new column in Jacobian
		# for row in range(len(J)):
		# 	J[row][idx] = Jcol_object[row]

		# return J

	"""
	Compute the target end-effector pose from the given target object pose.
	Assumes the grasp is fixed, and uses the difference between the current end-effector
	and object poses to predict the target end-effector pose.
	"""
	def get_target_gripper_pose(self, target_object_pos, target_object_quat):
		# compute difference between current end-effector and object poses
		diff_pos = self.curr_pos - self.object_curr_pos # points from object to end-effector
		diff_quat = T.quat_multiply(T.quat_inverse(self.curr_quat), self.object_curr_quat)
		diff = np.hstack([diff_pos, diff_quat])

		# compute target for end-effector based on target for object and expected difference between poses
		target_ee_pos = target_object_pos + diff[:3]
		target_ee_quat = T.quat_multiply(target_object_quat, diff[3:7])

		return (target_ee_pos, target_ee_quat)

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

	In this case, returns 6xn manipulator Jacobian.
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

		# set 6xn manipulator Jacobian
		J = J_lin + J_ang

		if self.verbose:
			print("BaxterObject6DPoseController: computed %dx%d Jacobian" % (len(J), len(J[0])))

		return J

	"""
	Computes the nullspace for performing lower-order controller commands subject to this controller.
	"""
	def get_objective_nullspace(self):
		# get (6xn) objective Jacobian
		J = self.get_objective_jacobian()

		# compute (nx6) pseudoinverse of Jacobian
		Jinv = np.linalg.pinv(J)

		# get (nxn) identity
		ndof = len(J[0])
		I = np.identity(ndof)

		# compute nullspace (make sure Jinv and J are numpy arrays)
		N = I - self.matrixMultiply(Jinv, J) # (nxn) = (nxn) - (nx6) * (6xn)

		if self.verbose:
			print("BaxterObject6DPoseController: computed %dx%d nullspace" % (len(N), len(N[0])))

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
		dq_projected_mat = self.matrixMultiply(N, dq_lower_priority)

		# convert numpy array to list
		dq_projected = list(dq_projected_mat)

		if self.verbose:
			print("BaxterObject6DPoseController: computed %dx1 projected controller command" % len(dq_projected))

		return dq_projected
