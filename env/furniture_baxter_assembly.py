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
from env.controllers import BaxterAlignmentController
from env.controllers import BaxterRotationController

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

        ### for 6DPose or 3DPosition controllers
        goal_pos_left = np.array([0.82516098, 0.3, 0.19841084])
        goal_quat_left = np.array([0.68656258, -0.72074515, -0.08759494, 0.03854062]) # ("left", goal_pos_left, goal_quat_left)
        goal_pos_right = np.array([0.68432551, -0.29451258, 0.21005051])
        goal_quat_right = np.array([-0.54690822, 0.48197699, 0.51618452, 0.44960329]) # ("right", goal_pos_right, goal_quat_right)
        goal_quat_nonsense = np.array([0.2, -0.3, -0.4, 0.03854062]) # nonsense orientation for 3DPositionController
        ### for Rotation controller
        # 1.571, 3.141, 4.712, 6.283
        ### for Alignment controller
        align_pos_left = [0.82273044, 0.29672234, 0.00181328] # ("left", "+Z", align_pos_left)
        align_pos_right = [0.82051034, -0.24336258, 0.08321501] # ("right", "+Z", align_pos_right)

        ### 6DPoseController
        pose_controller_sequence = [("Baxter6DPoseController", ("right", goal_pos_right, goal_quat_right))]
        
        ### sequence for testing all behaviors
        test_controller_sequence = [
            ("Baxter6DPoseController", ("right", goal_pos_right, goal_quat_right)),
            ("close-gripper", "right"),
            ("BaxterAlignmentController", ("left", "+Z", align_pos_left)),
            ("close-gripper", "left"),
            ("open-gripper", "right"),
            ("Baxter3DPositionController", ("left", goal_pos_left, goal_quat_nonsense)),
            ("BaxterRotationController", ("right", 6.283))
        ]

        ### sequence for assembling swivel chair
        swivelchair_poleprep_pos_right = [0.55756265, -0.1, -0.11673727]
        swivelchair_poleprep_quat_right = [-0.58808523, 0.53074937, 0.46539465, 0.39480208]
        swivelchair_polepick_pos_right = [0.53, -0.00189214, -0.11673727]
        swivelchair_polepick_quat_right = [-0.58808523, 0.53074937, 0.46539465, 0.39480208]
        swivelchair_polepost_pos_right = [0.53, 0.04, -0.04]#[0.65, -0.12, -0.04]
        swivelchair_polepost_quat_right = [-0.05176196, 0.06610491, 0.77816441, 0.62242348]#[-0.58846033, 0.52953778, 0.46733307, 0.39357843]
        swivelchair_polecnct_pos_right = [0.65, -0.12, -0.115]
        swivelchair_polecnct_quat_right = [-0.58846033, 0.52953778, 0.46733307, 0.39357843]

        swivelchair_seatprep_pos_left = [0.45077123, 0.32370803, 0.25]
        swivelchair_seatprep_quat_left = [0.68661902, -0.72083896, -0.08676259, 0.03765338]
        swivelchair_seatpick_pos_left = [0.45077123, 0.32370803, 0.13]
        swivelchair_seatpick_quat_left = [0.68661902, -0.72083896, -0.08676259, 0.03765338]

        # WORKING TO HERE!
        # swivelchair_seatcnct_pos_left = [0.59742394, 0.08848166, 0.64171694]
        # swivelchair_seatcnct_quat_left = [0.71337652, -0.68281197, -0.14455587, 0.06297114]
        swivelchair_seatpost_pos_left = [0.4, 0.1, -0.5]
        swivelchair_seatpost_quat_left = [-0.73133093, -0.00037086, -0.00036539, 0.68202257]
        # swivelchair_cnctprep_pos_left = [0.45077123, 0.32370803, 0.25]
        # swivelchair_cnctprep_quat_left = [0.68661902, -0.72083896, -0.08676259, 0.03765338]
        # swivelchair_seatcnct_pos_left = [0.5916167, -0.00193475, 0.51251066]
        # swivelchair_seatcnct_quat_left = [0.68599818, -0.72016698, -0.09497703, 0.04177788]
        
        swivelchair_cnctpolebase_sequence = [
            ("Baxter6DPoseController", ("right", swivelchair_poleprep_pos_right, swivelchair_poleprep_quat_right)),
            ("Baxter6DPoseController", ("right", swivelchair_polepick_pos_right, swivelchair_polepick_quat_right)),
            ("close-gripper", "right"),
            ("BaxterObject6DPoseController", ("right", '2_chair_column', swivelchair_polepost_pos_right, swivelchair_polepost_quat_right)),
            ("BaxterObject6DPoseController", ("right", '2_chair_column', swivelchair_polecnct_pos_right, swivelchair_polecnct_quat_right)),
            ("connect", "")
        ]
        swivelchair_pickseat_sequence = [
            ("Baxter6DPoseController", ("left", swivelchair_seatprep_pos_left, swivelchair_seatprep_quat_left)),
            ("Baxter6DPoseController", ("left", swivelchair_seatpick_pos_left, swivelchair_seatpick_quat_left)),
            ("close-gripper", "left"),
            # ("Baxter6DPoseController", ("left", swivelchair_seatprep_pos_left, swivelchair_seatprep_quat_left), (0.1, None))
            ("BaxterObject6DPoseController", ("left", "3_chair_seat", swivelchair_seatpost_pos_left, swivelchair_seatpost_quat_left))
        ]

        # initialize sequence of actions, where each action is
        # (action/controller, params) tuple
        self._action_sequence = swivelchair_cnctpolebase_sequence# swivelchair_pickseat_sequence

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
            if action[0] == "Baxter6DPoseController":
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
                    verbose=False # TODO
                )
                self._controller.set_goal(control_arm, object_name, object_goal_pos, object_goal_quat)
                run = "object-controller"
            elif action[0] == "Baxter3DPositionController":
                control_arm, goal_pos, goal_quat = action[1]
                self._controller = Baxter3DPositionController(
                    bullet_data_path=os.path.join(env.models.assets_root, "bullet_data"),
                    robot_jpos_getter=self._robot_jpos_getter,
                    verbose=False
                )
                self._controller.set_goal(control_arm, goal_pos, goal_quat)
                run = "controller"
            elif action[0] == "BaxterAlignmentController":
                control_arm, ee_axis, align_pos = action[1]
                self._controller = BaxterAlignmentController(
                    bullet_data_path=os.path.join(env.models.assets_root, "bullet_data"),
                    robot_jpos_getter=self._robot_jpos_getter,
                    verbose=False
                )
                self._controller.set_goal(control_arm, ee_axis, align_pos)
                run = "controller"
            elif action[0] == "BaxterRotationController":
                control_arm, rotation = action[1]
                self._controller = BaxterRotationController(
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
                    # compute controller update
                    velocities = self._controller.get_control()
                    # perform controller command
                    self.perform_command(velocities, gripper_grabs, True, (run == "object-controller"))
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

            print("FurnitureBaxterAssemblyEnv: finished action %s" % action[0])

        print("FurnitureBaxterAssemblyEnv: Goal met!")
        time.sleep(2)
        if config.record_video:
            vr.save_video('FurnitureBaxterAssemblyEnv_test.mp4')
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
                        return
                    break

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
