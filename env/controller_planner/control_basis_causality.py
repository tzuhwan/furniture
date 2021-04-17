"""
Defines causality of controllers in control basis.
"""

from itertools import permutations
import numpy as np
from pathlib import Path

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
			# "BaxterObject6DPoseController"
		]

		self.causal_models = {
			"Baxter6DPoseController": 0.95,
			"Baxter3DPositionController": 0.90,
			"BaxterRotationController": 0.90,
			"BaxterAlignmentController": 0.85,
			"BaxterScrewController": 0.97,
			# "BaxterObject6DPoseController": 0.80
		}

		self.temporal_decomposition = {
			"screw-into": ["Baxter3DPositionController", "BaxterRotationController", "BaxterScrewController"],
			"insert-into": ["Baxter3DPositionController", "BaxterRotationController"]
		}

		self.num_iters = 0

		self.composable_causality = {}

		self.file_path = str(Path(__file__).absolute())
		self.file_path = self.file_path[:self.file_path.rfind('/')+1]

	###############
	### GETTERS ###
	###############

	def get_controllers(self):
		return self.control_basis_controllers

	def get_causal_models(self):
		return self.causal_models

	def get_temporal_decomposition(self):
		return self.temporal_decomposition

	#################################
	### CONTROLLER INITIALIZATION ###
	#################################

	"""
	Initializes controllers given list of controllers involved in action.
	"""
	def initialize_controllers(self, action_controllers, bullet_data_path, robot_jpos_getter, verbose=True, debug=False, suppress_output=False):
		self.controllers = {}
		for controller in action_controllers:
			if controller == "Baxter6DPoseController":
				self.controllers[controller] = Baxter6DPoseController(
					bullet_data_path=bullet_data_path,
					robot_jpos_getter=robot_jpos_getter,
					verbose=verbose, debug=debug, suppress_output=suppress_output
				)
				self.controllers[controller].set_arm_speed(5)
			elif controller == "Baxter3DPositionController":
				self.controllers[controller] = Baxter3DPositionController(
					bullet_data_path=bullet_data_path,
					robot_jpos_getter=robot_jpos_getter,
					verbose=verbose, debug=debug, suppress_output=suppress_output
				)
				self.controllers[controller].set_arm_speed(5)
			elif controller == "BaxterRotationController":
				self.controllers[controller] = self._controller = BaxterRotationController(
					bullet_data_path=bullet_data_path,
					robot_jpos_getter=robot_jpos_getter,
					verbose=verbose, debug=debug, suppress_output=suppress_output
				)
				self.controllers[controller].set_arm_speed(5)
			elif controller == "BaxterAlignmentController":
				self.controllers[controller] = BaxterAlignmentController(
					bullet_data_path=bullet_data_path,
					robot_jpos_getter=robot_jpos_getter,
					verbose=verbose, debug=debug, suppress_output=suppress_output
				)
				self.controllers[controller].set_arm_speed(5)
			elif controller == "BaxterScrewController":
				self.controllers[controller] = BaxterScrewController(
					bullet_data_path=bullet_data_path,
					robot_jpos_getter=robot_jpos_getter,
					verbose=verbose, debug=debug, suppress_output=suppress_output
				)
				self.controllers[controller].set_arm_speed(5)
			else:
				print("ControlBasis: controller type %s not recognized" % controller)
				raise NameError

		self.initialize_potential_dicts()

		return self.controllers

	"""
	Initializes controllers given list of (controller, goal) tuples involved in action.
	"""
	def initialize_controllers_and_goals(self, action_controllers, bullet_data_path, robot_jpos_getter, verbose=True, debug=False, suppress_output=False):
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
					verbose=verbose, debug=debug, suppress_output=suppress_output
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
					verbose=verbose, debug=debug, suppress_output=suppress_output
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
					verbose=verbose, debug=debug, suppress_output=suppress_output
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
					verbose=verbose, debug=debug, suppress_output=suppress_output
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
					verbose=verbose, debug=debug, suppress_output=suppress_output
				)
				self.controllers[controller[0]].set_goal(control_arm, rotation)
				self.controllers[controller[0]].set_arm_speed(arm_speed)
			else:
				print("ControlBasis: controller type %s not recognized" % controller[0])
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

		# reset number of iterations
		self.num_iters = 0

		return

	####################
	### COMPOSITIONS ###
	####################

	"""
	Generates all possible compositions of controllers by permuting the given list of controllers.
	"""
	def get_possible_controller_compositions(self, controllers):
		possible_compositions = list(permutations(controllers))
		return possible_compositions

	##########################
	### CONTROLLER UPDATES ###
	##########################

	"""
	Compute the single-objective controller update.

	@param controller_name, the name of the controller being run
		   Note: used for indexing into control basis controller dictionary
	"""
	def compute_singleobjective_controller_update(self, controller_name):
		# check for valid controller name
		if (controller_name is None) or (controller_name not in self.control_basis_controllers):
			print("ControlBasis: unrecognized controller name %s")
			raise NameError

		# compute control command
		velocities = self.controllers[controller_name].get_control()

		return velocities, self.controllers[controller_name].objective_met

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
			if (not self.controllers[c].objective_met) and (np.std(self.potentials[c]) <= 1e-7):
				print("ControlBasis: %s potentials not changing" % c)
				progress = False
			if (not self.controllers[c].objective_met) and (np.all(np.diff(self.potentials[c]) > 0)):
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

	############################
	### COMPOSABLE CAUSALITY ###
	############################

	"""
	Updates the composable causality dictionary based on new samples.

	@param action_name, the name of the action whose composition causality is being predicted
	@param compositions_probabilities_iterations, a list of (composition, probability, iteration) triples
	@param num_samples, the number of samples used to compute the probability and average iterations
	@return none
	@post composable causality dictionary updated with given samples
	"""
	def update_composable_causality(self, action_name, compositions_probabilities_iterations, num_samples):
		# loop through compositions
		for c, p, i in compositions_probabilities_iterations:
			# get composition string
			c_str = self.multiobjective_controller_string(c)
			# get current probability, iterations, and samples
			curr_p = self.composable_causality[action_name][c_str]['probability']
			curr_i = self.composable_causality[action_name][c_str]['iterations']
			curr_s = self.composable_causality[action_name][c_str]['samples']
			# update average probability, average iterations, and samples for composition in dictionary
			self.composable_causality[action_name][c_str]['probability'] = ((curr_p * curr_s) + (p * num_samples)) / (curr_s + num_samples)
			self.composable_causality[action_name][c_str]['iterations'] = ((curr_i * curr_s) + (i * num_samples)) / (curr_s + num_samples)
			self.composable_causality[action_name][c_str]['samples'] = curr_s + num_samples

		return

	"""
	Initializes the composable causality dictionary based on samples.

	@param action_name, the name of the action whose composition causality is being predicted
	@param compositions_probabilities_iterations, a list of (composition, probability, iteration) triples
	@param num_samples, the number of samples used to compute the probability and average iterations
	@return none
	@post composable causality dictionary initialized with given samples
	"""
	def initialize_composable_causality(self, action_name, compositions_probabilities_iterations, num_samples):
		# initialize composable causality dictionary
		self.composable_causality[action_name] = {}

		# loop through compositions
		for c, p, i in compositions_probabilities_iterations:
			# get composition string
			c_str = self.multiobjective_controller_string(c)
			# store probability, iterations, and samples for composition in dictionary
			self.composable_causality[action_name][c_str] = {}
			self.composable_causality[action_name][c_str]['probability'] = p
			self.composable_causality[action_name][c_str]['iterations'] = i
			self.composable_causality[action_name][c_str]['samples'] = num_samples

		return

	###################################
	### CONTROLLER HELPER FUNCTIONS ###
	###################################

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

	#############################
	### FILE INPUT AND OUTPUT ###
	#############################

	"""
	Writes summary information from walkout samples to file.

	@param action_name, the name of the action whose composition causality is being predicted
	@param compositions_probabilities_iterations, a list of (composition, probability, iteration) triples
	@param num_samples, the number of samples used to compute the probability and average iterations
	@param update, a Boolean indicating whether the information in the file should be updated or overwriten
	@param save_previous_info, a Boolean indicating whether to save the existing information in the file, which may not be related to the current action
		   NOTE: used when update is false, but another action in the file should be saved
	@param file_name, the name of the file to write to
	@return none
	@post summary information writen to file
	"""
	def write_walkout_samples_to_file(self, action_name, compositions_probabilities_iterations, num_samples, update=True, save_previous_info=True, file_name="composition_predictions.txt"):
		# check if information is being updated
		if update:
			# read current information from file
			self.read_walkout_samples_from_file(file_name)
			# update composable causality dictionary with new information
			self.update_composable_causality(action_name, compositions_probabilities_iterations, num_samples)
		else:
			# check if other info in file should be saved
			if save_previous_info:
				self.read_walkout_samples_from_file(file_name)
			# initialize composable causality dictionary with new information
			self.initialize_composable_causality(action_name, compositions_probabilities_iterations, num_samples)

		# open file for writing (overwrites anything in file)
		f = open(self.file_path + file_name, 'w')

		# write everything in composable causality dictionary to file
		for a in self.composable_causality.keys():
			f.write("action: %s\n" % a)
			for c in self.composable_causality[a].keys():
				f.write("\tcomposition: %s\n" % c)
				f.write("\t\tprobability: %f\n" % self.composable_causality[a][c]['probability'])
				f.write("\t\titerations: %f\n" % self.composable_causality[a][c]['iterations'])
				f.write("\t\tsamples: %d\n" % self.composable_causality[a][c]['samples'])
			f.write("\n")

		# close file
		f.close()

		return

	"""
	Read summary information from file into composable causality dictionary.

	@param file_name, the name of the file to read from
	@return none
	@post dictionary of compositions and sampling statistics initialized
	"""
	def read_walkout_samples_from_file(self, file_name):
		# open file for reading
		f = open(self.file_path + file_name, 'r')
		# read information from file
		lines = f.readlines()
		# close file
		f.close()

		# initialize line counter and continue flag
		i = 0
		process_text = True

		# process each action in file
		while process_text:
			process_text, i = self.read_action_samples_from_file(lines, i)

		return

	"""
	Reads information for a single action from file.

	@param lines, the list of lines read from the file
	@param idx, the index of the lines list at which to start processing
	@return boolean indicating if there is more text to process in file
	@return index of the lines list to continue processing from
	@post dictionary of compositions and sampling statistics initialized
	"""
	def read_action_samples_from_file(self, lines, idx):
		# initialize line counter
		i = idx

		# get action name and increment line counter
		action_name = lines[i].split()[1]
		i += 1

		# get composition prosibilities
		num_compositions = len(self.get_possible_controller_compositions(self.temporal_decomposition[action_name]))

		# initialize composable causality dictionary
		self.composable_causality[action_name] = {}

		# loop through composition possibilities
		for j in range(num_compositions):
			# get composition string and increment line counter
			composition = lines[i][lines[i].find(' ')+1:-1]
			i += 1
			# get probability and increment line counter
			probability = lines[i].split()[1]
			i += 1
			# get iterations and increment line counter
			iterations = lines[i].split()[1]
			i += 1
			# get samples and increment line counter
			samples = lines[i].split()[1]
			i += 1
			# store probability, iterations, and samples for composition in dictionary
			self.composable_causality[action_name][composition] = {}
			self.composable_causality[action_name][composition]['probability'] = float(probability)
			self.composable_causality[action_name][composition]['iterations'] = float(iterations)
			self.composable_causality[action_name][composition]['samples'] = int(samples)

		# check if end of file reached
		if i >= len(lines)-2: # account for an extra empty line or two
			return False, -1 # end of file reached
		# check if next line is just whitespace
		elif not lines[i].split():
			return True, i+1 # next line contains more information
		else:
			return True, i # current line contains information about next action
