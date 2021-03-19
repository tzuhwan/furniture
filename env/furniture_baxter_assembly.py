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

        # read in hardcoded action sequences
        self.hardcoded_action_sequences = BaxterHardcodedAssemblySequences()

        # initialize sequence of actions, where each action is (action/controller, params) tuple
        self._action_sequence = self.hardcoded_action_sequences.swivelchair_assembly_sequence

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
        # sets up environment and robot
        if config.furniture_name is not None:
            config.furniture_id = furniture_name2id[config.furniture_name]
        ob = self.reset(config.furniture_id, config.background)
        # reset swivel base
        if config.furniture_id == 7: # swivel chair
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
        vr = VideoRecorder()
        vr.add(self.render('rgb_array'))

        # initialize gripper states (both open)
        gripper_grabs = [-1, -1]

        ### ACTION SEQUENCE ###
        # start action sequence
        print("FurnitureBaxterAssemblyEnv: Starting action sequence")

        # perform action sequence
        for i in range(len(self._action_sequence)):
            print("FurnitureBaxterAssemblyEnv: Starting action %d of %d" % (i+1, len(self._action_sequence)))
            
            # set up action
            action = self._action_sequence[i]
            run = ""
            if isinstance(action[0], tuple):
                self.initialize_multiobjective_controllers(action)
                run = "multiobjective-controller"
            elif action[0] == "Baxter6DPoseController":
                control_arm, goal_pos, goal_quat = action[1]
                self._controller = Baxter6DPoseController(
                    bullet_data_path=os.path.join(env.models.assets_root, "bullet_data"),
                    robot_jpos_getter=self._robot_jpos_getter,
                    verbose=False
                )
                self._controller.set_goal(control_arm, goal_pos, goal_quat)
                run = "controller"
            elif action[0] == "BaxterObject6DPoseController":
                control_arm, object_name, object_goal_pos, object_goal_quat = action[1]
                self._controller = BaxterObject6DPoseController(
                    bullet_data_path=os.path.join(env.models.assets_root, "bullet_data"),
                    robot_jpos_getter=self._robot_jpos_getter,
                    objects_in_scene=self._object_names,
                    verbose=False
                )
                self._controller.set_goal(control_arm, object_name, object_goal_pos, object_goal_quat)
                run = "object-controller"
            elif action[0] == "Baxter3DPositionController":
                control_arm, goal_pos = action[1]
                self._controller = Baxter3DPositionController(
                    bullet_data_path=os.path.join(env.models.assets_root, "bullet_data"),
                    robot_jpos_getter=self._robot_jpos_getter,
                    verbose=False
                )
                self._controller.set_goal(control_arm, goal_pos)
                run = "controller"
            elif action[0] == "BaxterRotationController":
                control_arm, goal_quat = action[1]
                self._controller = BaxterRotationController(
                    bullet_data_path=os.path.join(env.models.assets_root, "bullet_data"),
                    robot_jpos_getter=self._robot_jpos_getter,
                    verbose=False
                )
                self._controller.set_goal(control_arm, goal_quat)
                run="controller"
            elif action[0] == "BaxterAlignmentController":
                control_arm, ee_axis, align_pos = action[1]
                self._controller = BaxterAlignmentController(
                    bullet_data_path=os.path.join(env.models.assets_root, "bullet_data"),
                    robot_jpos_getter=self._robot_jpos_getter,
                    verbose=False
                )
                self._controller.set_goal(control_arm, ee_axis, align_pos)
                run = "controller"
            elif action[0] == "BaxterScrewController":
                control_arm, rotation = action[1]
                self._controller = BaxterScrewController(
                    bullet_data_path=os.path.join(env.models.assets_root, "bullet_data"),
                    robot_jpos_getter=self._robot_jpos_getter,
                    verbose=False
                )
                self._controller.set_goal(control_arm, rotation)
                run = "controller"
            elif action[0] == "close-gripper":
                control_arm = action[1]
                if control_arm == "left":
                    gripper_grabs[1] = 1
                elif control_arm == "right":
                    gripper_grabs[0] = 1
                else:
                    print("FurnitureBaxterAssemblyEnv: unrecognized arm %s" % control_arm)
                    raise NameError
                run = "gripper"
            elif action[0] == "open-gripper":
                control_arm = action[1]
                if control_arm == "left":
                    gripper_grabs[1] = -1
                elif control_arm == "right":
                    gripper_grabs[0] = -1
                else:
                    print("FurnitureBaxterAssemblyEnv: unrecognized arm %s" % control_arm)
                    raise NameError
                run = "gripper"
            elif action[0] == "connect":
                run = "connect"
            else:
                print("FurnitureBaxterAssemblyEnv: unrecognized action %s" % action[0])
                raise NameError

            # perform action
            if (run == "controller") or (run == "object-controller"):
                # run controller
                while not self._controller.objective_met:
                    # set flag so unity will update
                    self._unity_updated = False
                    # if object controller, set sim states
                    if run == "object-controller":
                        obj = self._controller.get_object_name()
                        obj_pose_matrix = self.pose_in_base_from_name(obj)
                        self._controller.set_object_pose(obj, obj_pose_matrix)
                        self._controller.set_body_poses(self.get_body_pose_matrices())
                    # compute controller update
                    velocities = self._controller.get_control()
                    # perform controller command
                    self.perform_command(velocities, gripper_grabs, True, (run == "object-controller"))
                    # render
                    vr.add(self.render('rgb_array'))
            elif run == "multiobjective-controller":
                objective_met = False
                # run controller
                while not objective_met:
                    # set flag so unity will update
                    self._unity_updated = False
                    # compute controller update
                    velocities, objective_met = self.compute_multiobjective_controller_update()
                    # perform controller command
                    self.perform_multiobjective_command(velocities, gripper_grabs)
                    # render
                    vr.add(self.render('rgb_array'))
            elif run == "gripper":
                # set flag so unity will update
                self._unity_updated = False
                # set velocities to 0 for arm joints
                velocities = np.zeros(14)
                # perform command
                self.perform_command(velocities, gripper_grabs, False)
                # render
                vr.add(self.render('rgb_array'))
                # sleep
                time.sleep(2)
            else: # run == "connect":
                # set flag so unity will update
                self._unity_updated = False
                # perform connection
                self.perform_connection()
                # render
                vr.add(self.render('rgb_array'))

            if isinstance(action[0], tuple):
                print("FurnitureBaxterAssemblyEnv: finished action %s" % self.multiobjective_controller_string(action))
            else:
                print("FurnitureBaxterAssemblyEnv: finished action %s" % action[0])

        print("FurnitureBaxterAssemblyEnv: Goal met!")
        time.sleep(2)
        if config.record_video:
            vr.save_video('FurnitureBaxterAssemblyEnv_test.mp4') # FurnitureBaxterAssemblyEnv_test.mp4
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
    """
    def perform_command(self, velocities, gripper_grabs, controller_velocities=True, object_controller=False):
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

    """
    Creates a string representing the name of a multi-objective controller.

    @param controllers, the tuple of (controller, params) tuples involved in the action
    @return the string indicating the composition (subject-to relations) involved in the multi-objective controller
    """
    def multiobjective_controller_string(self, controllers):
        action = ""
        # construct action string based on controllers
        for i in range(len(controllers)):
            action += controllers[len(controllers)-i-1][0]
            if i < len(controllers) - 1:
                action += " <| "

        return action

    """
    Initializes the list of controllers for multi-objective actions.

    @param action, the tuple of (controller, params) tuples involved in the action
    @post all controllers in the muli-objective behavior are initialized and goals set
    """
    def initialize_multiobjective_controllers(self, action):
        self._controllers = []
        # initialize controllers in action
        for controller in action:
            if controller[0] == "Baxter6DPoseController":
                control_arm, goal_pos, goal_quat = controller[1]
                self._controllers.append(
                    Baxter6DPoseController(
                        bullet_data_path=os.path.join(env.models.assets_root, "bullet_data"),
                        robot_jpos_getter=self._robot_jpos_getter,
                        verbose=False
                    )
                )
                self._controllers[-1].set_goal(control_arm, goal_pos, goal_quat)
            elif controller[0] == "Baxter3DPositionController":
                control_arm, goal_pos = controller[1]
                self._controllers.append(
                    Baxter3DPositionController(
                        bullet_data_path=os.path.join(env.models.assets_root, "bullet_data"),
                        robot_jpos_getter=self._robot_jpos_getter,
                        verbose=False
                    )
                )
                self._controllers[-1].set_goal(control_arm, goal_pos)
            elif controller[0] == "BaxterRotationController":
                control_arm, goal_quat = controller[1]
                self._controllers.append(
                    BaxterRotationController(
                        bullet_data_path=os.path.join(env.models.assets_root, "bullet_data"),
                        robot_jpos_getter=self._robot_jpos_getter,
                        verbose=False
                    )
                )
                self._controllers[-1].set_goal(control_arm, goal_quat)
            elif controller[0] == "BaxterAlignmentController":
                control_arm, ee_axis, align_pos = controller[1]
                self._controllers.append(
                    BaxterAlignmentController(
                        bullet_data_path=os.path.join(env.models.assets_root, "bullet_data"),
                        robot_jpos_getter=self._robot_jpos_getter,
                        verbose=False
                    )
                )
                self._controllers[-1].set_goal(control_arm, ee_axis, align_pos)
            elif controller[0] == "BaxterScrewController":
                control_arm, rotation = controller[1]
                self._controllers.append(
                    BaxterScrewController(
                        bullet_data_path=os.path.join(env.models.assets_root, "bullet_data"),
                        robot_jpos_getter=self._robot_jpos_getter,
                        verbose=False
                    )
                )
                self._controllers[-1].set_goal(control_arm, rotation)
            else:
                print("FurnitureBaxterControllerPlannerEnv: controller type %s not recognized" % controller)
                raise NameError

        return

    """
    Computes the multi-objective controller update by composing controller commands.

    @param none
    @return the combined controller update from the composition of the controllers
    @return boolean indicating if the combined objective is met
    """
    def compute_multiobjective_controller_update(self):
        dq_combined = np.zeros(14)
        lower_priority_controller_name = ""
        objective_met = []
        # compute combined control command
        for i in range(len(self._controllers)):
            # compute index, from lowest priority to highest
            idx = len(self._controllers) - i - 1
            # compute controller command, index from lowest priority to highest
            dq = self._controllers[idx].get_control()
            # compute combined command using nullspace projection
            dq_combined = dq + self._controllers[idx].nullspace_projection(lower_priority_controller_name, dq_combined)
            # check if objective met
            objective_met.insert(0, self._controllers[idx].objective_met)
            # update controller name
            lower_priority_controller_name = self._controllers[idx].controller_name()
            # try connection
            # if self._controllers[idx].objective_met:
            #     self.perform_connection()

        return dq_combined, np.all(objective_met)

    """
    Performs the given multi-objective controller command.
    Copied from part of _step_continuous() function in furniture.py

    @param velocities, the change in configuration induced by the multi-objective controller
    @param gripper_grabs, flags indicating whether [right, left] grippers have closed
        where 1 means gripper is closed and -1 means gripper is open
    """
    def perform_multiobjective_command(self, velocities, gripper_grabs):
        # set up low action
        low_action = np.concatenate([velocities, gripper_grabs])

        # keep trying to reach the target in a closed-loop
        ctrl = self._setup_action(low_action)
        for i in range(self._action_repeat):
                self._do_simulation(ctrl)

                if i + 1 < self._action_repeat:
                    velocities, _ = self.compute_multiobjective_controller_update()
                    low_action = np.concatenate([velocities, gripper_grabs])
                    ctrl = self._setup_action(low_action)

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
