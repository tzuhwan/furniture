"""
Define baxter furniture assembly environment class FurnitureBaxterAssemblyEnv.
"""

import os
import time
from collections import OrderedDict

import numpy as np

import env.models
# from env.furniture_baxter import FurnitureEnv
from env.furniture_baxter import FurnitureBaxterEnv
import env.transform_utils as T
from env.controllers import Baxter6DPoseController
from env.controllers import BaxterObject6DPoseController
from env.controllers import Baxter3DPositionController
from env.controllers import BaxterRotationController
from env.controllers import BaxterAlignmentController
from env.controllers import BaxterScrewController
from env.controllers import BaxterHardcodedAssemblySequences
from env.controller_planner import ControlBasis

"""
Baxter robot environment with furniture assembly task.
"""
class FurnitureBaxterAssemblyEnv(FurnitureBaxterEnv):

    """
    Constructor.
    @param config, configurations for the environment
    """
    def __init__(self, config):
        config.agent_type = 'Baxter'

        # initialize FurnitureBaxterEnv
        super().__init__(config)

        # set vebose and debug flag
        self.verbose = config.verbose
        self.debug = config.debug

        # initialize control basis
        self.cb = ControlBasis()

        # read in hardcoded action sequences
        self.hardcoded_action_sequences = BaxterHardcodedAssemblySequences()

        # initialize sequence of actions, where each action is (action/controller, params) tuple
        self._action_sequence = self.hardcoded_action_sequences.swivelchair_assembly_planning_sequence

    """
    Takes a simulation step with @a and computes reward.
    """
    def _step(self, a):
        return super(FurnitureBaxterEnv, self)._step(a)

    """
    Resets simulation and variables to compute reward.
    @param furniture_id, ID of the furniture model to reset
    @param background, name of the background scene to reset
    """
    def _reset(self, furniture_id=None, background=None):
        super()._reset(furniture_id, background)

    """
    Runs the environment with controllers.
    """
    def run_controller(self, config):
        ### SETUP ###
        ob = self.reset(config.furniture_id, config.background)
        # reset swivel base
        if config.furniture_name == "swivel_chair_0700": # swivel chair
            print("reseting swivel chair base pose")
            self._set_qpos('1_chair_base',
                [-0.1, 0.0, 0.0144],
                [0.996917333733128, 0.07845909572784494, 0.0, 0.0]
            )
            print("reseting swivel chair column pose")
            self._set_qpos('2_chair_column',
                [0.02974198, 0.10812374, 0.06895991],
                [-0.00000004, -0.00000002, 0.03443238, 0.99940703]
            )

        if config.render:
            self.render()

        from util.video_recorder import VideoRecorder
        self.vr = VideoRecorder()
        self.vr.add(self.render('rgb_array'))

        # initialize gripper states (both open)
        self.gripper_grabs = [-1, -1]

        ### ACTION SEQUENCE ###
        # start action sequence
        print("FurnitureBaxterAssemblyEnv: Starting action sequence")

        # perform action sequence
        for i in range(len(self._action_sequence)):
            print("FurnitureBaxterAssemblyEnv: Starting action %d of %d" % (i+1, len(self._action_sequence)))
            
            # set up action
            action = self._action_sequence[i]
            run = ""
            # multi-objective controller
            if isinstance(action[0], tuple):
                controllers_goals = action
                self.cb.initialize_controllers_and_goals(controllers_goals,
                    bullet_data_path=os.path.join(env.models.assets_root, "bullet_data"),
                    robot_jpos_getter=self._robot_jpos_getter,
                    verbose=self.verbose
                )
                self.composition = [controller_name for controller_name, goal_info in controllers_goals]
                run = "multiobjective-controller"
            # perform online walkouts to determine multi-objective controller composition
            elif action[0] == "plan":
                control_arm, action_name = action[1]
                controllers_goals = action[2]
                self.cb.initialize_controllers_and_goals(controllers_goals,
                    bullet_data_path=os.path.join(env.models.assets_root, "bullet_data"),
                    robot_jpos_getter=self._robot_jpos_getter,
                    verbose=self.verbose
                )
                compositions = self.walkout_controller_compositions(control_arm, action_name)
                print("planned compositions: ", compositions)
                self.composition = compositions[0]
                print("executing composition %s" % self.cb.multiobjective_controller_string(self.composition))
                run = "multiobjective-controller"
            # object controller (not working the best)
            elif action[0] == "BaxterObject6DPoseController":
                control_arm, object_name, object_goal_pos, object_goal_quat = action[1]
                self._controller = BaxterObject6DPoseController(
                    bullet_data_path=os.path.join(env.models.assets_root, "bullet_data"),
                    robot_jpos_getter=self._robot_jpos_getter,
                    objects_in_scene=self._object_names,
                    verbose=self.verbose
                )
                self._controller.set_goal(control_arm, object_name, object_goal_pos, object_goal_quat)
                run = "object-controller"
            # single-objective controller
            elif action[0] in self.cb.control_basis_controllers:
                controller_goals = [action]
                self.cb.initialize_controllers_and_goals(controller_goals,
                    bullet_data_path=os.path.join(env.models.assets_root, "bullet_data"),
                    robot_jpos_getter=self._robot_jpos_getter,
                    verbose=self.verbose
                )
                run = action[0]
            # close gripper
            elif action[0] == "close-gripper":
                control_arm = action[1]
                if control_arm == "left":
                    self.gripper_grabs[1] = 1
                elif control_arm == "right":
                    self.gripper_grabs[0] = 1
                else:
                    print("FurnitureBaxterAssemblyEnv: unrecognized arm %s" % control_arm)
                    raise NameError
                run = "gripper"
            # open gripper
            elif action[0] == "open-gripper":
                control_arm = action[1]
                if control_arm == "left":
                    self.gripper_grabs[1] = -1
                elif control_arm == "right":
                    self.gripper_grabs[0] = -1
                else:
                    print("FurnitureBaxterAssemblyEnv: unrecognized arm %s" % control_arm)
                    raise NameError
                run = "gripper"
            # connect
            elif action[0] == "connect":
                run = "connect"
            else:
                print("FurnitureBaxterAssemblyEnv: unrecognized action %s" % action[0])
                raise NameError

            # perform action
            if run == "object-controller":
                # run controller
                while not self._controller.objective_met:
                    # set flag so unity will update
                    self._unity_updated = False
                    # set sim states
                    obj = self._controller.get_object_name()
                    obj_pose_matrix = self.pose_in_base_from_name(obj)
                    self._controller.set_object_pose(obj, obj_pose_matrix)
                    self._controller.set_body_poses(self.get_body_pose_matrices())
                    # compute controller update
                    velocities = self._controller.get_control()
                    # perform controller command
                    self.perform_command(velocities, self.gripper_grabs, True, True)
                    # render
                    self.vr.add(self.render('rgb_array'))
            elif run in self.cb.control_basis_controllers:
                objective_met = False
                # run controller
                while not objective_met:
                    # set flag so unity will update
                    self._unity_updated = False
                    # compute potential
                    pot, progress = self.cb.compute_singleobjective_controller_potential(run)
                    # compute controller update
                    velocities, objective_met = self.cb.compute_singleobjective_controller_update(run)
                    # perform controller command
                    self.perform_command(velocities, self.gripper_grabs, True, False, controller_name=run)
                    # render
                    self.vr.add(self.render('rgb_array'))
                    # check for progress
                    if not progress:
                        success = False
                        print("********** no success because of potentials **********")
                        break
            elif run == "multiobjective-controller":
                objective_met = False
                # run controller
                while not objective_met:
                    # set flag so unity will update
                    self._unity_updated = False
                    # compute potential
                    pot, progress = self.cb.compute_multiobjective_controller_potential(self.composition)
                    # compute controller update
                    velocities, objective_met = self.cb.compute_multiobjective_controller_update(self.composition)
                    # perform controller command
                    self.perform_multiobjective_command(velocities, self.gripper_grabs, self.composition)
                    # render
                    self.vr.add(self.render('rgb_array'))
                    # check for progress
                    if not progress:
                        success = False
                        print("********** no success because of potentials **********")
                        break
            elif run == "gripper":
                # set flag so unity will update
                self._unity_updated = False
                # set velocities to 0 for arm joints
                velocities = np.zeros(14)
                # perform command
                self.perform_command(velocities, self.gripper_grabs, False)
                # render
                self.vr.add(self.render('rgb_array'))
                # sleep
                time.sleep(2)
            else: # run == "connect":
                # set flag so unity will update
                self._unity_updated = False
                # perform connection
                self.perform_connection()
                # render
                self.vr.add(self.render('rgb_array'))

            if isinstance(action[0], tuple):
                print("FurnitureBaxterAssemblyEnv: finished action %s" % self.cb.multiobjective_controller_string(self.composition))
            else:
                print("FurnitureBaxterAssemblyEnv: finished action %s" % action[0])

        print("FurnitureBaxterAssemblyEnv: Goal met!")
        time.sleep(2)
        if config.record_video:
            self.vr.save_video(config.video_name) # FurnitureBaxterAssemblyEnv_test.mp4
        time.sleep(2)
        print("FurnitureBaxterAssemblyEnv returning")

        # from env.controllers import BaxterIKController
        # self._controller = BaxterIKController(
        #     bullet_data_path=os.path.join(env.models.assets_root, "bullet_data"),
        #     robot_jpos_getter=self._robot_jpos_getter,
        # )

        return

    """
    Performs the given controller command.
    Copied from part of _step_continuous() function in furniture.py

    @param velocities, the change in configuration induced by the controller
    @param gripper_grabs, flags indicating whether [right, left] grippers have closed
        where 1 means gripper is closed and -1 means gripper is open
    @param controller_velocities, a boolean indicating whether to update velocities using a controller
    @param object_controller, a boolean indicating whether the controller being run acts on an object
    @param controller_name, the name of the single-objective control basis controller being run
    """
    def perform_command(self, velocities, gripper_grabs, controller_velocities=True, object_controller=False, controller_name=None):
        # set up low action
        low_action = np.concatenate([velocities, gripper_grabs])

        # keep trying to reach the target in a closed-loop
        ctrl = self._setup_action(low_action)
        for i in range(self._action_repeat):
                self._do_simulation(ctrl)

                if i + 1 < self._action_repeat:
                    if controller_velocities:
                        if object_controller:
                            obj = self._controller.get_object_name()
                            obj_pose_matrix = self.pose_in_base_from_name(obj)
                            self._controller.set_object_pose(obj, obj_pose_matrix)
                            self._controller.set_body_poses(self.get_body_pose_matrices())
                            velocities = self._controller.get_control()
                        else:
                            velocities, _ = self.cb.compute_singleobjective_controller_update(controller_name)
                    low_action = np.concatenate([velocities, gripper_grabs])
                    ctrl = self._setup_action(low_action)

        return

    """
    Performs the given multi-objective controller command.
    Copied from part of _step_continuous() function in furniture.py

    @param velocities, the change in configuration induced by the multi-objective controller
    @param gripper_grabs, flags indicating whether [right, left] grippers have closed
        where 1 means gripper is closed and -1 means gripper is open
    @param composition, the list of controllers indicating the order of the composition
    """
    def perform_multiobjective_command(self, velocities, gripper_grabs, composition):
        # set up low action
        low_action = np.concatenate([velocities, gripper_grabs])

        # keep trying to reach the target in a closed-loop
        ctrl = self._setup_action(low_action)
        for i in range(self._action_repeat):
                self._do_simulation(ctrl)

                if i + 1 < self._action_repeat:
                    velocities, _ = self.cb.compute_multiobjective_controller_update(composition)
                    low_action = np.concatenate([velocities, gripper_grabs])
                    ctrl = self._setup_action(low_action)

        return

    """
    Perform connection between two parts.
    Copied from part of _step_continuous() function in furniture.py
    """
    def perform_connection(self):
        num_hands = 2
        for i in range(num_hands):
            touch_left_finger = {}
            touch_right_finger = {}
            for body_id in self._object_body_ids:
                touch_left_finger[body_id] = False
                touch_right_finger[body_id] = False

            for j in range(self.sim.data.ncon):
                c = self.sim.data.contact[j]
                body1 = self.sim.model.geom_bodyid[c.geom1]
                body2 = self.sim.model.geom_bodyid[c.geom2]
                if c.geom1 in self.l_finger_geom_ids[i] and body2 in self._object_body_ids:
                    touch_left_finger[body2] = True
                if body1 in self._object_body_ids and c.geom2 in self.l_finger_geom_ids[i]:
                    touch_left_finger[body1] = True

                if c.geom1 in self.r_finger_geom_ids[i] and body2 in self._object_body_ids:
                    touch_right_finger[body2] = True
                if body1 in self._object_body_ids and c.geom2 in self.r_finger_geom_ids[i]:
                    touch_right_finger[body1] = True

            for body_id in self._object_body_ids:
                if touch_left_finger[body_id] and touch_right_finger[body_id]:
                    if self._debug:
                        print('try connect')
                    result = self._try_connect(self.sim.model.body_id2name(body_id))
                    print("connection result: ", result)
                    if result:
                        return result
                    break

    """
    Get the pose matrices for bodies in the scene.

    @param none
    @return dictionary of body names to pose matrices in the robot base frame
    """
    def get_body_pose_matrices(self):
        # initialize dictionary of bodies to poses
        body_pose_dict = {}

        # get poses of all bodies
        for body in self.sim.model.body_names:
            body_pose_dict[body] = self.pose_in_base_from_name(body)

        return body_pose_dict

    """
    Compute a pose in the Unity frame from the pose in the robot base frame.

    @param pose_in_base, the pose in the robot base frame
    @return the given pose expressed in the Unity frame
    """
    def pose_in_unity_from_pose_in_base(self, pose_in_base):
        # get base pose
        base_pos_in_world = self.sim.data.get_body_xpos("base")
        base_rot_in_world = self.sim.data.get_body_xmat("base").reshape((3, 3))
        base_pose_in_world = T.make_pose(base_pos_in_world, base_rot_in_world)

        # pose in unity world = pose in base * base in unity world
        pose_in_world = T.pose_in_A_to_pose_in_B(pose_in_base, base_pose_in_world)
        return pose_in_world

    ##############################################################
    ### WALKOUTS FOR DETERMINING CONTROLLER COMPOSITION ONLINE ###
    ##############################################################

    """
    Selects controller compositions for given action name.
    """
    def walkout_controller_compositions(self, control_arm="left", action_name="screw"):
        # get initial joint state
        self.walkout_start_state = self.get_current_qpos()

        # get controllers required to execute given action from temporal decomposition
        action_controllers = self.cb.temporal_decomposition[action_name]

        # get possible compositions
        composition_possibilities = self.cb.get_possible_controller_compositions(action_controllers)

        # initialize list of probabilities
        composition_probabilities = []
        composition_iterations = []

        # initialize counter
        count = 0

        print("********** PERFORMING WALKOUT, NOT ACTUAL TASK EXECUTION **********")
        self.cb.set_controllers_suppress_output(True)

        # compute probability of achieving combined effects for each composition using a walkout
        for composition in composition_possibilities:
            count += 1
            print("FurnitureBaxterControllerPlanner: performing walkout for composition %s (%d of %d)"
                % (self.cb.multiobjective_controller_string(composition), count, len(composition_possibilities))
            )

            # compute walkout from initial start state
            prob, num_iters = self.perform_walkout(composition)

            # store computed probability in list
            composition_probabilities.append(prob)
            composition_iterations.append(num_iters)

            # reset sim to initial state
            print("FurnitureBaxterControllerPlanner: reseting to start state")
            self.set_current_qpos(self.walkout_start_state)

        # find max composition probability and fewest iterations
        max_prob = max(composition_probabilities) # max probability ensures composition is most likely to succeed
        min_iter = min(composition_iterations) # if min iterations is less than fixed parameter, then probability=1
        compositions_probabilities_iterations = [(c, p, i) for c, p, i in zip(composition_possibilities, composition_probabilities, composition_iterations)]
        print("***** COMPOSITIONS, PROBABILITIES, AND ITERATIONS *****")
        for c, p, i in compositions_probabilities_iterations:
            print("controller %s, probability of success %d, iterations %d"
                % (self.cb.multiobjective_controller_string(c), p, i)
            )

        # get compositions that achieve max probability and fewest iterations
        compositions = [c for c, p, i in zip(composition_possibilities, composition_probabilities, composition_iterations) if (p >= max_prob) and (i <= min_iter)]

        print("********** RESUMING TASK EXECUTION **********")
        self.cb.set_controllers_suppress_output(False)

        return compositions

    """
    Performs walkout to test different controller compositions for the given action name.
    """
    def perform_walkout(self, controllers):
        # initialize flag for objective met
        objective_met = False

        # initialize counter
        num_iters = 0

        # compute initial distance from goal
        init_potential = 0
        for c in controllers:
            init_potential += self.cb.controllers[c].potential()

        # run controller
        while (not objective_met) and (num_iters <= 200): # TODO initialize parameter somewhere
            # set flag so unity will update
            # self._unity_updated = False # TODO do not render walkouts for better sim behavior
            # compute controller update
            velocities, objective_met = self.cb.compute_multiobjective_controller_update(controllers)
            num_iters += 1
            # perform controller command
            self.perform_multiobjective_command(velocities, self.gripper_grabs, controllers)
            # render
            # self.vr.add(self.render('rgb_array')) # TODO do not render walkouts for better sim behavior

        # initialize probability
        prob = 1

        # if objective met, probability of reaching goal is 1
        if objective_met:
            prob = 1
        else: # num_iters > 700
            # compute final distance from goal
            final_potential = 0
            for c in controllers:
                final_potential += self.cb.controllers[c].potential()
            # if final potential is greater than initial, composition would not have reached goal
            if final_potential > init_potential:
                prob = 0
            else: # consider progress towards goal
                prob = (init_potential - final_potential) / init_potential

        return prob, num_iters

    """
    Gets the current poses of the robot and objects in the simulator.

    @return dictionary of object names to poses
    """
    def get_current_qpos(self):
        # get qpos for robot
        qpos = {
            'r_gripper': self.sim.data.qpos[self._ref_gripper_right_joint_pos_indexes],
            'l_gripper': self.sim.data.qpos[self._ref_gripper_left_joint_pos_indexes],
            'baxter_qpos': self.sim.data.qpos[self._ref_joint_pos_indexes]
        }
        # get qpos for objects
        qpos.update({x: self._get_qpos(x) for x in self._object_names})

        return qpos

    """
    Sets the current poses of the robot and objects in the simulator.

    @param qpos, a dictionary of object names to poses
    @post simulator objects are put at given poses
    """
    def set_current_qpos(self, qpos):
        # set qpos for robot
        self.sim.data.qpos[self._ref_gripper_right_joint_pos_indexes] = qpos['r_gripper']
        self.sim.data.qpos[self._ref_gripper_left_joint_pos_indexes] = qpos['l_gripper']
        self.sim.data.qpos[self._ref_joint_pos_indexes] = qpos['baxter_qpos']
        # set qpos for objects
        for x in self._object_names:
            obj_qpos = qpos[x]
            self._set_qpos(x, obj_qpos[:3], obj_qpos[3:])

        return

"""
Main function; will not be called when environment is constructed from appropriate demo
"""
def main():
    import argparse
    import config.furniture as furniture_config
    from util import str2bool

    parser = argparse.ArgumentParser()
    furniture_config.add_argument(parser)

    # change default config for Baxter
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--debug', type=str2bool, default=False)

    parser.set_defaults(render=True)

    config, unparsed = parser.parse_known_args()

    # create an environment and run manual control of Baxter environment
    env = FurnitureBaxterAssemblyEnv(config)
    env.run_controller(config)

if __name__ == "__main__":
    main()
