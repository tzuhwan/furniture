"""
Defines hardcoded sequences of controllers for Baxter assembly tasks.
"""

class BaxterHardcodedAssemblySequences():

	def __init__(self):
		self.init_pose_controller_sequence()
		self.init_position_controller_sequence()
		self.init_rotation_controller_sequence()
		self.init_screw_controller_sequence()
		self.init_alignment_controller_sequence()
		self.init_test_controller_sequence()
		self.init_multiobjective_controller_sequence()
		self.init_swivelchair_sequences()

	"""
	Initialize sequence for testing Baxter6DPoseController.
	"""
	def init_pose_controller_sequence(self):
		# initialize useful poses
		self.goal_pos_left = [0.82516098, 0.3, 0.19841084]
		self.goal_quat_left = [0.68656258, -0.72074515, -0.08759494, 0.03854062]
		self.goal_pos_right = [0.68432551, -0.29451258, 0.21005051]
		self.goal_quat_right = [-0.54690822, 0.48197699, 0.51618452, 0.44960329]
		# initialize sequence
		self.pose_controller_sequence = [("Baxter6DPoseController", ("right", self.goal_pos_right, self.goal_quat_right))]
		return

	"""
	Initialize sequence for testing Baxter3DPositionController.
	"""
	def init_position_controller_sequence(self):
		# no initialization of poses; uses poses from init_pose_controller_sequence()
		# initialize sequence
		self.position_controller_sequence = [("Baxter3DPositionController", ("left", self.goal_pos_left))]
		return

	"""
	Initialize sequence for testing BaxterRotationController.
	"""
	def init_rotation_controller_sequence(self):
		# no initialization of poses; uses poses from init_pose_controller_sequence()
		# initialize sequence
		self.rotation_controller_sequence = [("BaxterRotationController", ("right", self.goal_quat_right))]
		return

	"""
	Initialize sequence for testing BaxterScrewController.
	"""
	def init_screw_controller_sequence(self):
		# initialize useful rotations
		self.rot1 = 1.571
		self.rot2 = 3.141
		self.rot3 = 4.712
		self.rot4 = 6.283
		# initialize sequence
		self.screw_controller_sequence = [("BaxterScrewController", ("left", self.rot3))]
		return

	"""
	Initialize sequence for testing BaxterAlignmentController.
	"""
	def init_alignment_controller_sequence(self):
		# initialize useful positions
		self.align_pos_left = [0.82273044, 0.29672234, 0.00181328]
		self.align_pos_right = [0.82051034, -0.24336258, 0.08321501]
		# initialize sequence
		self.align_controller_sequence = [("BaxterAlignmentController", ("left", "+Z", self.align_pos_left))]
		return

	"""
	Initialize sequence for testing all controller behaviors.
	"""
	def init_test_controller_sequence(self):
		# initialize sequence using predefined poses
		self.test_controller_sequence = [
			("Baxter6DPoseController", ("right", self.goal_pos_right, self.goal_quat_right)),
			("close-gripper", "right"),
			("BaxterAlignmentController", ("left", "+Z", self.align_pos_left)),
			("close-gripper", "left"),
			("open-gripper", "right"),
			("Baxter3DPositionController", ("left", self.goal_pos_left)),
			("BaxterScrewController", ("right", self.rot4))
		]
		return

	"""
	Initialize sequence for testing multi-objective controller composition.
	"""
	def init_multiobjective_controller_sequence(self):
		# initialize useful positions
		self.multiobj_align_pos_left = [0.82273044, 0.2, 0.00181328]
		# initialize sequence
		self.test_multiobjective_controller_sequence = [
			(
				("BaxterAlignmentController", ("left", "+Z", self.multiobj_align_pos_left)),
				("Baxter3DPositionController", ("left", self.multiobj_align_pos_left))
			)
		]
		return

	"""
	Initialize sequences for manipulating swivel chair.
	"""
	def init_swivelchair_sequences(self):
		self.init_swivelchair_cnctpolebase_sequence()
		self.init_swivelchair_pickseat_sequence()
		self.init_swivelchair_assembly_sequence()
		self.init_swivelchair_assembly_planning_sequence()
		return

	"""
	Initialize sequence for connecting swivel chair pole to base.

	Note: alternate controllers are listed underneath working controllers.
	"""
	def init_swivelchair_cnctpolebase_sequence(self):
		# initialize useful poses
		self.swivelchair_poleprep_pos_right = [0.55756265, -0.1, -0.11673727]
		self.swivelchair_poleprep_quat_right = [-0.58808523, 0.53074937, 0.46539465, 0.39480208]
		self.swivelchair_polepick_pos_right = [0.53, -0.00189214, -0.11673727]
		self.swivelchair_polepick_quat_right = [-0.58808523, 0.53074937, 0.46539465, 0.39480208]
		self.swivelchair_polepost_pos_right = [0.65, -0.12, -0.04]
		self.swivelchair_polepost_quat_right = [-0.58846033, 0.52953778, 0.46733307, 0.39357843]
		self.swivelchair_polecnct_pos_right = [0.65, -0.12, -0.118]
		self.swivelchair_polecnct_quat_right = [-0.5465044, 0.48847611, 0.50796767, 0.45242998]
		self.swivelchair_cnctplce_pos_right = [0.65, -0.05, -0.12]
		self.swivelchair_cnctplce_quat_right = [-0.5465044, 0.48847611, 0.50796767, 0.45242998]
		self.swivelchair_cnctpost_pos_right = [0.83195645, -0.45, 0.20856081]
		self.swivelchair_cnctpost_quat_right = [0.70522274, -0.70528875, -0.02417812, 0.06814758]
		# initialize sequence
		self.swivelchair_cnctpolebase_sequence = [
			("Baxter6DPoseController", ("right", self.swivelchair_poleprep_pos_right, self.swivelchair_poleprep_quat_right)),
			("Baxter6DPoseController", ("right", self.swivelchair_polepick_pos_right, self.swivelchair_polepick_quat_right)),
			("close-gripper", "right"),
			("Baxter6DPoseController", ("right", self.swivelchair_polepost_pos_right, self.swivelchair_polepost_quat_right)),
			# ("BaxterObject6DPoseController", ("right", '2_chair_column', self.swivelchair_polepost_pos_right, self.swivelchair_polepost_quat_right)),
			(
				("BaxterRotationController", ("right", self.swivelchair_polecnct_quat_right)),
				# ("BaxterAlignmentController", ("right", "+X", self.swivelchair_polecnct_pos_right)),
				("Baxter3DPositionController", ("right", self.swivelchair_polecnct_pos_right))
			),
			("connect", ""),
			("Baxter6DPoseController", ("right", self.swivelchair_cnctplce_pos_right, self.swivelchair_cnctplce_quat_right)),
			("open-gripper", "right"),
			("Baxter6DPoseController", ("right", self.swivelchair_cnctpost_pos_right, self.swivelchair_cnctpost_quat_right))
		]
		return

	"""
	Initialize sequence for picking up swivel chair seat.

	Note: alternate controllers are listed underneath working controllers.
	"""
	def init_swivelchair_pickseat_sequence(self):
		# initialize useful poses
		self.swivelchair_seatprep_pos_left = [0.45077123, 0.32370803, 0.25]
		self.swivelchair_seatprep_quat_left = [0.68661902, -0.72083896, -0.08676259, 0.03765338]
		self.swivelchair_seatpick_pos_left = [0.45077123, 0.32370803, 0.13]
		self.swivelchair_seatpick_quat_left = [0.68661902, -0.72083896, -0.08676259, 0.03765338]
		self.swivelchair_seatpost_pos_left = [0.68, 0.05, 0.4]
		self.swivelchair_seatpost_quat_left = [0.68661902, -0.72083896, -0.08676259, 0.03765338]
		# initialize sequence
		self.swivelchair_pickseat_sequence = [
			("Baxter6DPoseController", ("left", self.swivelchair_seatprep_pos_left, self.swivelchair_seatprep_quat_left)),
			("Baxter6DPoseController", ("left", self.swivelchair_seatpick_pos_left, self.swivelchair_seatpick_quat_left)),
			("close-gripper", "left"),
			("Baxter6DPoseController", ("left", self.swivelchair_seatpost_pos_left, self.swivelchair_seatpost_quat_left))
			# ("BaxterObject6DPoseController", ("left", "3_chair_seat", self.swivelchair_seatpost_pos_left, self.swivelchair_seatpost_quat_left))
		]
		return

	"""
	Initialize sequence for assembly of full swivel chair.
	"""
	def init_swivelchair_assembly_sequence(self):
		# initialize useful poses
		self.swivelchair_seatcnct_pos_left = [0.68, 0.0, 0.35]
		self.swivelchair_seatcnct_quat_left = [0.6848843787594499, -0.7232040344156025, -0.06457352826104447, 0.06115203826691636] #[0.68661902, -0.72083896, -0.08676259, 0.03765338]
		self.swivelchair_cnctplce_pos_left = [0.68, 0.0, 0.4]
		self.swivelchair_cnctplce_quat_left = [0.6848843787594499, -0.7232040344156025, -0.06457352826104447, 0.06115203826691636]
		self.swivelchair_cnctpost_pos_left = [0.83195645, 0.45, 0.20856081]
		self.swivelchair_cnctpost_quat_left = [0.70522274, -0.70528875, -0.02417812, 0.06814758]
		# initialize sequence
		self.swivelchair_assembly_sequence = [
			# connect pole to base
			("Baxter6DPoseController", ("right", self.swivelchair_poleprep_pos_right, self.swivelchair_poleprep_quat_right)),
			("Baxter6DPoseController", ("right", self.swivelchair_polepick_pos_right, self.swivelchair_polepick_quat_right)),
			("close-gripper", "right"),
			("Baxter6DPoseController", ("right", self.swivelchair_polepost_pos_right, self.swivelchair_polepost_quat_right)),
			(
				("BaxterRotationController", ("right", self.swivelchair_polecnct_quat_right)),
				("Baxter3DPositionController", ("right", self.swivelchair_polecnct_pos_right))
			),
			("connect", ""),
			("Baxter6DPoseController", ("right", self.swivelchair_cnctplce_pos_right, self.swivelchair_cnctplce_quat_right)),
			("open-gripper", "right"),
			("Baxter6DPoseController", ("right", self.swivelchair_cnctpost_pos_right, self.swivelchair_cnctpost_quat_right)),
			# connect seat to pole
			("Baxter6DPoseController", ("left", self.swivelchair_seatprep_pos_left, self.swivelchair_seatprep_quat_left)),
			("Baxter6DPoseController", ("left", self.swivelchair_seatpick_pos_left, self.swivelchair_seatpick_quat_left)),
			("close-gripper", "left"),
			("Baxter6DPoseController", ("left", self.swivelchair_seatpost_pos_left, self.swivelchair_seatpost_quat_left)),
			(
				("BaxterRotationController", ("left", self.swivelchair_seatcnct_quat_left)),
				("Baxter3DPositionController", ("left", self.swivelchair_seatcnct_pos_left))
			),
			("connect", ""),
			("open-gripper", "left"),
			("Baxter6DPoseController", ("left", self.swivelchair_cnctplce_pos_left, self.swivelchair_cnctplce_quat_left)),
			("Baxter6DPoseController", ("left", self.swivelchair_cnctpost_pos_left, self.swivelchair_cnctpost_quat_left))
		]
		return

	"""
	Initialize sequence for assembly of full swivel chair, with composition walkout planning.
	"""
	def init_swivelchair_assembly_planning_sequence(self):
		# initialize useful poses
		self.swivelchair_seatcnct_pos_left = [0.68, 0.0, 0.35]
		self.swivelchair_seatcnct_quat_left = [0.6848843787594499, -0.7232040344156025, -0.06457352826104447, 0.06115203826691636] #[0.68661902, -0.72083896, -0.08676259, 0.03765338]
		self.swivelchair_cnctplce_pos_left = [0.68, 0.0, 0.4]
		self.swivelchair_cnctplce_quat_left = [0.6848843787594499, -0.7232040344156025, -0.06457352826104447, 0.06115203826691636]
		self.swivelchair_cnctpost_pos_left = [0.83195645, 0.45, 0.20856081]
		self.swivelchair_cnctpost_quat_left = [0.70522274, -0.70528875, -0.02417812, 0.06814758]
		# initialize sequence
		self.swivelchair_assembly_planning_sequence = [
			# connect pole to base
			("Baxter6DPoseController", ("right", self.swivelchair_poleprep_pos_right, self.swivelchair_poleprep_quat_right, 8)),
			("Baxter6DPoseController", ("right", self.swivelchair_polepick_pos_right, self.swivelchair_polepick_quat_right, 8)),
			("close-gripper", "right"),
			("Baxter6DPoseController", ("right", self.swivelchair_polepost_pos_right, self.swivelchair_polepost_quat_right, 5)),
			("plan", ("right", "insert-into"),
				(
					("BaxterRotationController", ("right", self.swivelchair_polecnct_quat_right)),
					("Baxter3DPositionController", ("right", self.swivelchair_polecnct_pos_right))
				)
			),
			("connect", ""),
			("Baxter6DPoseController", ("right", self.swivelchair_cnctplce_pos_right, self.swivelchair_cnctplce_quat_right, 8)),
			("open-gripper", "right"),
			("Baxter6DPoseController", ("right", self.swivelchair_cnctpost_pos_right, self.swivelchair_cnctpost_quat_right, 5)),
			# connect seat to pole
			("Baxter6DPoseController", ("left", self.swivelchair_seatprep_pos_left, self.swivelchair_seatprep_quat_left, 8)),
			("Baxter6DPoseController", ("left", self.swivelchair_seatpick_pos_left, self.swivelchair_seatpick_quat_left, 8)),
			("close-gripper", "left"),
			("Baxter6DPoseController", ("left", self.swivelchair_seatpost_pos_left, self.swivelchair_seatpost_quat_left, 5)),
			("plan", ("left", "insert-into"),
				(
					("BaxterRotationController", ("left", self.swivelchair_seatcnct_quat_left)),
					("Baxter3DPositionController", ("left", self.swivelchair_seatcnct_pos_left))
				)
			),
			("connect", ""),
			("open-gripper", "left"),
			("Baxter6DPoseController", ("left", self.swivelchair_cnctplce_pos_left, self.swivelchair_cnctplce_quat_left, 5)),
			("Baxter6DPoseController", ("left", self.swivelchair_cnctpost_pos_left, self.swivelchair_cnctpost_quat_left, 8))
		]
		return
