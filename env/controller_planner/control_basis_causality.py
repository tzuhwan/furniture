"""
Defines causality of controllers in control basis.
"""

from itertools import permutations
import numpy as np

from env.controllers import Baxter6DPoseController
from env.controllers import BaxterObject6DPoseController
from env.controllers import Baxter3DPositionController
from env.controllers import BaxterRotationController
from env.controllers import BaxterAlignmentController
from env.controllers import BaxterScrewController

class ControlBasis:

	def __init__(self):
		self.control_basis_controllers = [
			"Baxter6DPoseController",
			"Baxter3DPositionController",
			"BaxterRotationController",
			"BaxterAlignmentController",
			"BaxterScrewController",
			"BaxterObject6DPoseController"
		]

		self.causal_models = {
			"Baxter6DPoseController": 0.95,
			"Baxter3DPositionController": 0.90,
			"BaxterRotationController": 0.90,
			"BaxterAlignmentController": 0.85,
			"BaxterScrewController": 0.97,
			"BaxterObject6DPoseController": 0.80
		}

		self.temporal_decomposition = {
			"screw-into": ["Baxter3DPositionController", "BaxterRotationController", "BaxterScrewController"],
			"insert-into": ["Baxter3DPositionController", "BaxterRotationController"]
		}

		self.num_iters = 0

	def get_controllers(self):
		return self.control_basis_controllers

	def get_causal_models(self):
		return self.causal_models

	def get_temporal_decomposition(self):
		return self.temporal_decomposition

	"""
	Initializes controllers given list of controllers involved in action.
	"""
	def initialize_controllers(self, action_controllers, bullet_data_path, robot_jpos_getter, suppress_output=False):
		self.controllers = {}
		for controller in action_controllers:
			if controller == "Baxter6DPoseController":
				self.controllers[controller] = Baxter6DPoseController(
					bullet_data_path=bullet_data_path,
					robot_jpos_getter=robot_jpos_getter,
					verbose=False, suppress_output=suppress_output
				)
				self.controllers[controller].set_arm_speed(5)
			elif controller == "Baxter3DPositionController":
				self.controllers[controller] = Baxter3DPositionController(
					bullet_data_path=bullet_data_path,
					robot_jpos_getter=robot_jpos_getter,
					verbose=False, suppress_output=suppress_output
				)
				self.controllers[controller].set_arm_speed(5)
			elif controller == "BaxterRotationController":
				self.controllers[controller] = self._controller = BaxterRotationController(
					bullet_data_path=bullet_data_path,
					robot_jpos_getter=robot_jpos_getter,
					verbose=False, suppress_output=suppress_output
				)
				self.controllers[controller].set_arm_speed(5)
			elif controller == "BaxterAlignmentController":
				self.controllers[controller] = BaxterAlignmentController(
					bullet_data_path=bullet_data_path,
					robot_jpos_getter=robot_jpos_getter,
					verbose=False, suppress_output=suppress_output
				)
				self.controllers[controller].set_arm_speed(5)
			elif controller == "BaxterScrewController":
				self.controllers[controller] = BaxterScrewController(
					bullet_data_path=bullet_data_path,
					robot_jpos_getter=robot_jpos_getter,
					verbose=False, suppress_output=suppress_output
				)
				self.controllers[controller].set_arm_speed(5)
			else:
				print("FurnitureBaxterControllerPlannerEnv: controller type %s not recognized" % controller)
				raise NameError

		self.initialize_potential_dicts()

		return self.controllers

	"""
	Initializes controllers given list of (controller, goal) tuples involved in action.
	"""
	def initialize_controllers_and_goals(self, action_controllers, bullet_data_path, robot_jpos_getter, suppress_output=False):
		self.controllers = {}
		for controller in action_controllers:
			if controller[0] == "Baxter6DPoseController":
				if len(controller[1]) == 3:
					control_arm, goal_pos, goal_quat = controller[1]
					arm_speed = 2
				else:
					control_arm, goal_pos, goal_quat, arm_speed = controller[1]
				self.controllers[controller[0]] = Baxter6DPoseController(
					bullet_data_path=bullet_data_path,
					robot_jpos_getter=robot_jpos_getter,
					verbose=False, suppress_output=suppress_output
				)
				self.controllers[controller[0]].set_goal(control_arm, goal_pos, goal_quat)
				self.controllers[controller[0]].set_arm_speed(arm_speed)
			elif controller[0] == "Baxter3DPositionController":
				if len(controller[1]) == 2:
					control_arm, goal_pos = controller[1]
					arm_speed = 2
				else:
					control_arm, goal_pos, arm_speed = controller[1]
				self.controllers[controller[0]] = Baxter3DPositionController(
					bullet_data_path=bullet_data_path,
					robot_jpos_getter=robot_jpos_getter,
					verbose=False, suppress_output=suppress_output
				)
				self.controllers[controller[0]].set_goal(control_arm, goal_pos)
				self.controllers[controller[0]].set_arm_speed(arm_speed)
			elif controller[0] == "BaxterRotationController":
				if len(controller[1]) == 2:
					control_arm, goal_quat = controller[1]
					arm_speed = 2
				else:
					control_arm, goal_quat, arm_speed = controller[1]
				self.controllers[controller[0]] = self._controller = BaxterRotationController(
					bullet_data_path=bullet_data_path,
					robot_jpos_getter=robot_jpos_getter,
					verbose=False, suppress_output=suppress_output
				)
				self.controllers[controller[0]].set_goal(control_arm, goal_quat)
				self.controllers[controller[0]].set_arm_speed(arm_speed)
			elif controller[0] == "BaxterAlignmentController":
				if len(controller[1]) == 3:
					control_arm, ee_axis, align_pos = controller[1]
					arm_speed = 2
				else:
					control_arm, ee_axis, align_pos, arm_speed = controller[1]
				self.controllers[controller[0]] = BaxterAlignmentController(
					bullet_data_path=bullet_data_path,
					robot_jpos_getter=robot_jpos_getter,
					verbose=False, suppress_output=suppress_output
				)
				self.controllers[controller[0]].set_goal(control_arm, ee_axis, align_pos)
				self.controllers[controller[0]].set_arm_speed(arm_speed)
			elif controller[0] == "BaxterScrewController":
				if len(controller[1]) == 2:
					control_arm, rotation = controller[1]
					arm_speed = 2
				else:
					control_arm, rotation, arm_speed = controller[1]
				self.controllers[controller[0]] = BaxterScrewController(
					bullet_data_path=bullet_data_path,
					robot_jpos_getter=robot_jpos_getter,
					verbose=False, suppress_output=suppress_output
				)
				self.controllers[controller[0]].set_goal(control_arm, rotation)
				self.controllers[controller[0]].set_arm_speed(arm_speed)
			else:
				print("FurnitureBaxterControllerPlannerEnv: controller type %s not recognized" % controller[0])
				raise NameError

		self.initialize_potential_dicts()

		return self.controllers

	"""
	Initializes dictionaries for storing recent potential values to track controller convergence.
	"""
	def initialize_potential_dicts(self):
		# initialize dictionary of potentials
		self.potentials = {}

		# for each controller, initialize list of potentials
		for c in self.controllers.keys():
			self.potentials[c] = np.ones(20)

		return

	"""
	Sets all of the controllers to suppress output the same way.

	@param suppress_output, the boolean indicating whether all controllers should suppress output or not
	@post all controllers suppress output according to given boolean
	"""
	def set_controllers_suppress_output(self, suppress_output=False):
		for c in self.controllers.keys():
			self.controllers[c].suppress_output = suppress_output

		return

	"""
	Generates all possible compositions of controllers by permuting the given list of controllers.
	"""
	def get_possible_controller_compositions(self, controllers):
		possible_compositions = list(permutations(controllers))
		return possible_compositions

	"""
	Computes the potentials of each controller in the composition.

	@param composition, the list of controllers in the given composition
	@return the combined potential of each controller
	@return boolean indicating if controllers are making progress
	"""
	def compute_multiobjective_potentials(self, composition):
		# initialize multi-objective potential
		multiobj_pot = 0
		progress = True

		# for each controller, compute potential
		for c in composition:
			# compute potential and update stored potentials
			pot = self.controllers[c].potential()
			self.potentials[c][self.num_iters%len(self.potentials[c])] = pot
			# check if progress is being made
			if np.std(self.potentials[c]) <= 1e-7:
				print("ControlBasis: %s potentials not changing" % c)
				progress = False
			if np.all(np.diff(self.potentials[c]) > 0):
				print("ControlBasis: %s potentials increasing" % c)
				progress = False
			# compute multi-objective potential
			multiobj_pot += pot

		# increment number of iterations
		self.num_iters += 1

		return multiobj_pot, progress
		
	"""
	Computes the multi-objective controller update by composing controller commands.

	@param composition, the list of controllers in the given composition
	@return the combined controller update from the composition of the controllers
	@return boolean indicating if the combined objective is met
	"""
	def compute_multiobjective_controller_update(self, composition):
		# initialize combined command, controller name, and objective met array
		dq_combined = np.zeros(14)
		lower_priority_controller_name = ""
		objective_met = np.zeros(len(composition))
		# compute combined control command
		for i in range(len(composition)):
			# compute index, from lowest priority to highest
			idx = len(composition) - i - 1
			# get controller name
			c = composition[idx]
			# compute controller command, from lowest priority to highest
			dq = self.controllers[c].get_control()
			# compute combined command using nullspace projection
			dq_combined = dq + self.controllers[c].nullspace_projection(lower_priority_controller_name, dq_combined)
			# check if objective met
			objective_met[idx] = self.controllers[c].objective_met
			# update controller name
			lower_priority_controller_name = self.controllers[c].controller_name()

		return dq_combined, np.all(objective_met)

	"""
	Creates a string representing the name of a multi-objective controller.

	@param controllers, the tuple of (controller, params) tuples involved in the action
						OR the list of controller names
	@return the string indicating the composition (subject-to relations) involved in the multi-objective controller
	"""
	def multiobjective_controller_string(self, controllers):
		# set controller names
		if isinstance(controllers[0], tuple): # (controller, params) tuples
			controllerNames = [n for n, g in controllers]
		else: # list of controller names
			controllerNames = controllers

		# initialize action string
		action = ""

		# construct action string based on controllers
		for i in range(len(controllerNames)):
			action += controllerNames[len(controllerNames)-i-1]
			if i < len(controllerNames) - 1:
				action += " <| "

		return action
