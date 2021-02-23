"""
6D pose controller for Baxter

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
6D pose controller for the Baxter robot, using Pybullet and the urdf description
files.
"""
class Baxter6DPoseController(BaxterIKController):

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
		print("Baxter6DPoseController: Initializing 6DPose Controller")

		# initialize super class
		super().__init__(bullet_data_path, robot_jpos_getter)

		# set debug and verbose flags
		self.verbose = verbose
		self.debug = debug

		# max potential
		self.max_potential = 100

		# potential threshold (potential less than this means no update will be performed)
		self.potential_threshold = 0.0003 # 0.0005

		# controller objective met flag
		self.objective_met = False

		# set move and rotate speed, for scaling motions; these are equivalent to controller gains
		self.move_speed = 0.025
		self.rotate_speed = 0.05

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

		# set current pose and compute potential
		self.set_current_pose()
		pot = self.potential()

		# initialize velocities for proportional controller
		velocities = np.zeros(14)

		# if potential is low enough, no update needed
		if pot < self.potential_threshold:
			print("Baxter6DPoseController: Goal met! No update needed.")
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
	def set_goal(self, control_arm, goal_pos, goal_quat):
		# check for valid arm
		if not ((control_arm == "left") or (control_arm == "right")):
			print("Baxter6DPoseController: Arm %s not recognized" % control_arm)
			raise NameError
			return

        # we now know arm is either "left" or "right"
		self.control_arm = control_arm
		self.goal_pos = goal_pos
		self.goal_quat = goal_quat
		print("Baxter6DPoseController: New goal set for %s arm" % self.control_arm)
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
				print("Baxter6DPoseController: left pos from ik in base frame: ", curr_left_pos)
				print("Baxter6DPoseController: left rot from ik in base frame: ", curr_left_quat)
				print("Baxter6DPoseController: right pos from ik in base frame: ", curr_right_pos)
				print("Baxter6DPoseController: right rot from ik in base frame: ", curr_right_quat)
		else:
			if self.debug:	
				print("Baxter6DPoseController: given left pos in base frame: ", curr_left_pos)
				print("Baxter6DPoseController: given left rot in base frame: ", curr_left_quat)
				print("Baxter6DPoseController: given right pos in base frame: ", curr_right_pos)
				print("Baxter6DPoseController: given right rot in base frame: ", curr_right_quat)

		# compute left and right poses in world frame
		curr_left_pos_in_world, curr_left_quat_in_world = self.bullet_base_pose_to_world_pose(
			(curr_left_pos, curr_left_quat)
		)
		curr_right_pos_in_world, curr_right_quat_in_world = self.bullet_base_pose_to_world_pose(
			(curr_right_pos, curr_right_quat)
		)

		if self.debug:
			print("Baxter6DPoseController: left pos in world frame: ", curr_left_pos_in_world)
			print("Baxter6DPoseController: left rot in world frame: ", curr_left_quat_in_world)
			print("Baxter6DPoseController: right pos in world frame: ", curr_right_pos_in_world)
			print("Baxter6DPoseController: right rot in world frame: ", curr_right_quat_in_world)

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
	Sets the motion and rotation speeds for the controller.
	"""
	def set_motion_speeds(self, move_speed=None, rotate_speed=None):
		if move_speed is not None:
			self.move_speed = move_speed
		if rotate_speed is not None:
			self.rotate_speed = rotate_speed
		print("Baxter6DPoseController: Set new motion speeds, move_speed=%f, rotate_speed=%f" % (self.move_speed, self.rotate_speed))
		return

	"""
	Sets the arm speed for the controller.
	"""
	def set_arm_speed(self, arm_speed):
		self.arm_step = arm_speed
		print("Baxter6DPoseController: Set new arm speed")
		return

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
		diff_quat = T.quat_multiply(T.quat_inverse(self.curr_quat), self.goal_quat)
		diff = np.hstack([diff_pos, diff_quat])

		# compute distance to goal
		dist_pos = np.linalg.norm(diff[:3])
		dist_quat = Quaternion.distance(Quaternion(self.curr_quat), Quaternion(self.goal_quat))
		dist_quat = np.min([dist_quat, np.pi - dist_quat])
		dist = dist_pos + dist_quat

		# compute potential
		pot = 0.5 * dist * dist
		print("Baxter6DPoseController: potential %f" % pot)
		print("Baxter6DPoseController: position distance %f, rotation distance %f" % (dist_pos, dist_quat))
		return min(pot, self.max_potential)

	"""
	Computes the gradient of the controller based on the current and goal poses.
	Based on attractive potential field.
	"""
	def gradient(self):
		# compute difference between current and goal pose
		diff_pos = self.curr_pos - self.goal_pos
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
			target_left_pos = self.curr_pos + (self.move_speed * dx[:3]) # scale down position
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
			target_right_pos = self.curr_pos + (self.move_speed * dx[:3]) # scale down position
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
			print("Baxter6DPoseController: computed %dx%d Jacobian" % (len(J), len(J[0])))

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
			print("Baxter6DPoseController: computed %dx%d nullspace" % (len(N), len(N[0])))

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
			print("Baxter6DPoseController: computed %dx1 projected controller command" % len(dq_projected))

		return dq_projected
