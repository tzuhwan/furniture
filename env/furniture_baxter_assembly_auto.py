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
from env.controllers import Baxter3DPositionController
from env.controllers import BaxterAlignmentController
from env.controllers import BaxterRotationController

import env.transform_utils as T
from object_files.scripts.read_local_pose import PoseReader

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
        
        # access to prelabeled poses
        self.pose_reader = PoseReader()
        
        # fixed offsets related to grasping
        self.hold_offset = 0.02  # make sure the gripper covers the object with some distance and object won't slip out
        self.grasp_offset = 0.05 + self.hold_offset
        self.lift_offset = 0.2 # 0.01 is even hard for some cases

        ### sequence for assembling swivel chair
        # swivelchair_poleprep_pos_right = [0.55756265, -0.1, -0.11673727]
        # swivelchair_poleprep_quat_right = [-0.58808523, 0.53074937, 0.46539465, 0.39480208]
        # swivelchair_polepick_pos_right = [0.53, -0.00189214, -0.11673727]
        # swivelchair_polepick_quat_right = [-0.58808523, 0.53074937, 0.46539465, 0.39480208]
        # swivelchair_polepost_pos_right = [0.65, -0.12, -0.04]
        # swivelchair_polepost_quat_right = [-0.58846033, 0.52953778, 0.46733307, 0.39357843]
        # swivelchair_polecnct_pos_right = [0.65, -0.12, -0.115]
        # swivelchair_polecnct_quat_right = [-0.58846033, 0.52953778, 0.46733307, 0.39357843]
        
        # # tested, seems not able to lift the chair seat?
        # swivelchair_seatprep_pos_left = [0.45077123, 0.32370803, 0.25]
        # swivelchair_seatprep_quat_left = [0.68661902, -0.72083896, -0.08676259, 0.03765338]
        # swivelchair_seatpick_pos_left = [0.45077123, 0.32370803, 0.13]
        # swivelchair_seatpick_quat_left = [0.68661902, -0.72083896, -0.08676259, 0.03765338]
        # swivelchair_cnctprep_pos_left = [0.45077123, 0.32370803, 0.25]
        # swivelchair_cnctprep_quat_left = [0.68661902, -0.72083896, -0.08676259, 0.03765338]
        # swivelchair_seatcnct_pos_left = [0.5916167, -0.00193475, 0.51251066]
        # swivelchair_seatcnct_quat_left = [0.68599818, -0.72016698, -0.09497703, 0.04177788]
        
        # swivelchair_cnctpolebase_pickseat_sequence = [
        #     ("Baxter6DPoseController", ("right", swivelchair_poleprep_pos_right, swivelchair_poleprep_quat_right)),
        #     ("Baxter6DPoseController", ("right", swivelchair_polepick_pos_right, swivelchair_polepick_quat_right)),
        #     ("close-gripper", "right"),
        #     ("Baxter6DPoseController", ("right", swivelchair_polepost_pos_right, swivelchair_polepost_quat_right)),
        #     ("Baxter6DPoseController", ("right", swivelchair_polecnct_pos_right, swivelchair_polecnct_quat_right)),
        #     ("connect", ""),
        #     ("open-gripper", "right"),
        #     ("Baxter6DPoseController", ("left", swivelchair_seatprep_pos_left, swivelchair_seatprep_quat_left)),
        #     ("Baxter6DPoseController", ("left", swivelchair_seatpick_pos_left, swivelchair_seatpick_quat_left)),
        #     ("close-gripper", "left"),
        #     ("Baxter6DPoseController", ("left", swivelchair_cnctprep_pos_left, swivelchair_cnctprep_quat_left)),
        #     ("Baxter6DPoseController", ("left", swivelchair_seatcnct_pos_left, swivelchair_seatcnct_quat_left))
        # ]
        # # ADDITIONAL WAYPOINTS NOT TESTED YET
        # #     ("Baxter6DPoseController", ("right", swivelchair_polecnct_pos_right, swivelchair_polecnct_quat_right)),
        # #     ("connect", ""),
        # #     ("open-gripper", "right"),
        # #     ("Baxter6DPoseController", ("right", swivelchair_discnct_pos_right, swivelchair_discnct_quat_right)),
        # #     ("Baxter6DPoseController", ("left", swivelchair_polecnct_pos_left, swivelchair_polecnct_quat_left)),
        # #     ("connect", ""),
        # #     ("Baxter6DPoseController", ("left", swivelchair_success_pos_left, swivelchair_success_quat_left)),
        # #     ("open-gripper", "left")
        # # ]

        # # initialize sequence of actions, where each action is
        # # (action/controller, params) tuple
        # self._action_sequence = swivelchair_cnctpolebase_pickseat_sequence

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

        # initialize gripper states (both open)
        self.gripper_grabs = [-1, -1]

        if config.furniture_name is not None:
            config.furniture_id = furniture_name2id[config.furniture_name]
        ob = self.reset(config.furniture_id, config.background)
        # reset swivel base
        if config.furniture_id == 7: # swivel chair
            print("reseting swivel chair base pose")
            self._set_qpos('1_chair_base',
                [-0.1, 0.1, 0.0144],
                [1.0, 0.0, 0.0, 0.0]
            )
            self._set_qpos('2_chair_column',
                [0, 0, 0.072],
                [1.0, 0.0, 0.0, 0.0]
            )
            self._set_qpos('3_chair_seat',
                           [0.23653631, -0.08022969, 0.14596923],
                           [1.0, 0.0, 0.0, 0.0]
            )
        
        left_gripper_pose = (self.pose_in_base_from_name('l_gripper_l_finger_tip') + self.pose_in_base_from_name('l_gripper_r_finger_tip')) / 2  # orientations are the same
        self.left_obj2eef_mat = T.pose_inv(left_gripper_pose).dot(self.pose_in_base_from_name('left_gripper_base'))
        right_gripper_pose = (self.pose_in_base_from_name('r_gripper_l_finger_tip') + self.pose_in_base_from_name('r_gripper_r_finger_tip')) / 2  # orientations are the same
        self.right_obj2eef_mat = T.pose_inv(right_gripper_pose).dot(self.pose_in_base_from_name('right_gripper_base'))

        base_pos_in_world = self.sim.data.get_body_xpos("base")
        base_rot_in_world = self.sim.data.get_body_xmat("base").reshape((3, 3))
        base_pose_in_world = T.make_pose(base_pos_in_world, base_rot_in_world)
        self.world_pose_in_base = T.pose_inv(base_pose_in_world)

        if config.render:
            self.render()
        
        if config.furniture_id == 7: # swivel_chair:
            self._actions = [
                ('grasp', 'right', '2_chair_column'),
                ('connect', 'right', 'column-base,conn_site1', 'base-column,conn_site1'),  # assumption: has grasped first part
                ('grasp', 'left', '3_chair_seat'),
                ('connect', 'left', 'seat-column,conn_site2', 'column-seat,conn_site2')  # todo: change to screw for rotation operation
            ]
            self.n_grasp_pose = { # should be consistent with CoppeliaSim labels
                'base': 10,
                'column': 2,
                'seat': 14
            }

        from util.video_recorder import VideoRecorder
        self.vr = VideoRecorder()
        self.vr.add(self.render('rgb_array'))

        for action in self._actions:
            action_times = 0
            while 1:
                success = self.high_level_action(action, action_times)
                if success:
                    break
                action_times += 1

        if config.record_video:
            self.vr.save_video('FurnitureBaxterAssemblyEnv_test2.mp4')
        time.sleep(2)
        print("FurnitureBaxterAssemblyEnv returning")

        return

    def high_level_action(self, action, action_times):
        self._action_sequence = []
        if action[0] == 'grasp':
            self.gripper_grabs = [-1, -1]
            # 1. object part grasp pose transformed to workspace coordinate
            part = action[2]
            partname = part[part.rfind('_') + 1:]
            local_grasp_pose = [self.pose_reader.read_object_pose(partname + '_grasp_pose_' + str(i), part) for i in
                                range(self.n_grasp_pose[partname])]
            local_grasp_pose_mat = [T.pose2mat((pose[:3], pose[3:])) for pose in local_grasp_pose]
            part_pose_mat = self.pose_in_base_from_name(part)
            ws_grasp_pose_mat = [T.pose_in_A_to_pose_in_B(grasp_pose_mat0, part_pose_mat) for grasp_pose_mat0 in
                                 local_grasp_pose_mat]
            found_downwards_grasppose = False
            potential_pose = []
            for obj_grasp_mat in ws_grasp_pose_mat:
                # if obj_grasp_mat[2, 3] < 0:  # drop poses that are too low
                #     continue
                if obj_grasp_mat[2, 2] < -0.8:  # select grasp poses that has z axis pointing downwards
                    found_downwards_grasppose = True
                    potential_pose.append(obj_grasp_mat)
            if not found_downwards_grasppose:
                print('cannot find downwards grasppose')

            obj_grasp_mat = potential_pose[action_times]
            # 2. plus offset to gripper_base which is end effector in controller, add other offsets for pre- and post- grasp pose
            obj2eef_mat = self.left_obj2eef_mat if action[1] == 'left' else self.right_obj2eef_mat
            grasp_pose_mat = obj_grasp_mat.dot(obj2eef_mat)
            grasp_pose_mat[:3, -1] += self.hold_offset * grasp_pose_mat[:3, 2]
            grasp_pose = T.mat2pose(grasp_pose_mat)
            pre_eef_grasp_pose_mat = grasp_pose_mat.copy()
            pre_eef_grasp_pose_mat[:3, -1] -= self.grasp_offset * pre_eef_grasp_pose_mat[:3, 2]
            pre_grasp_pose = T.mat2pose(pre_eef_grasp_pose_mat)
            post_grasp_pose_mat = grasp_pose_mat.copy()
            post_grasp_pose_mat[2, 3] += self.lift_offset  # lift in world frame
            post_grasp_pose = T.mat2pose(post_grasp_pose_mat)
            self._action_sequence.append(("Baxter6DPoseController", (action[1], pre_grasp_pose[0], pre_grasp_pose[1])))
            self._action_sequence.append(("Baxter6DPoseController", (action[1], grasp_pose[0], grasp_pose[1])))
            self._action_sequence.append(("close-gripper", action[1]))
            self._action_sequence.append(("Baxter6DPoseController", (action[1], post_grasp_pose[0], post_grasp_pose[1])))
        elif action[0] == 'connect':
            m_site = action[2]
            t_site = action[3]
            target_site_qpos = self._site_xpos_xquat(t_site)
            moving_site_qpos = self._site_xpos_xquat(m_site)

            target_site_qpos_mat = T.pose_in_A_to_pose_in_B(T.pose2mat((target_site_qpos[:3], target_site_qpos[3:])),
                                                            self.world_pose_in_base)
            moving_site_qpos_mat = T.pose_in_A_to_pose_in_B(T.pose2mat((moving_site_qpos[:3], moving_site_qpos[3:])),
                                                            self.world_pose_in_base)

            # translation = target_site_qpos[:3] - moving_site_qpos[:3]
            # m_site_id = self.sim.model.site_name2id(m_site)
            # m_body_id = self.sim.model.site_bodyid[m_site_id]
            # m_body_name = self.sim.model.body_names[m_body_id]
            # new_pos, new_quat = T.transform_to_target_quat(moving_site_qpos, self._get_qpos(m_body_name), target_site_qpos[3:])
            translation = target_site_qpos_mat[:3, 3] - moving_site_qpos_mat[:3, 3]
            rotation = (target_site_qpos_mat[:3, :3]).transpose().dot(moving_site_qpos_mat[:3, :3])
            hand_pose_mat = self.pose_in_base_from_name('left_gripper_base') if action[
                                                                                    1] == 'left' else self.pose_in_base_from_name(
                'right_gripper_base')
            target_hand_pose_mat = hand_pose_mat.copy()
            target_hand_pose_mat[:3, 3] += translation
            # target_hand_pose_mat[:3,:3] = T.pose2mat((new_pos,new_quat))[:3, :3].dot(target_hand_pose_mat[:3,:3])
            target_hand_pose_mat[:3, :3] = rotation.dot(target_hand_pose_mat[:3, :3])
            pre_target_hand_pose_mat = target_hand_pose_mat.copy()
            pre_target_hand_pose_mat[:3, -1] -= self.grasp_offset * pre_target_hand_pose_mat[:3, 2]
            pre_target_hand_pose = T.mat2pose(pre_target_hand_pose_mat)
            target_hand_pose = T.mat2pose(target_hand_pose_mat)
            self._action_sequence.append(
                ("Baxter6DPoseController", (action[1], pre_target_hand_pose[0], pre_target_hand_pose[1])))
            self._action_sequence.append(
                ("Baxter6DPoseController", (action[1], target_hand_pose[0], target_hand_pose[1])))
            self._action_sequence.append(("connect", " "))
        success = self.one_action(self._action_sequence)

        return success

    def one_action(self, _action_sequence):
        ### ACTION SEQUENCE ###
        # start action sequence
        print("FurnitureBaxterAssemblyEnv: Starting action sequence")
        success = True
        # perform action sequence
        for i in range(len(_action_sequence)):
            print("FurnitureBaxterAssemblyEnv: Starting action %d of %d" % (i+1, len(_action_sequence)))

            if not success:
                break
            # set up action
            action = _action_sequence[i]
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
                    self.gripper_grabs[1] = 1
                elif control_arm == "right":
                    self.gripper_grabs[0] = 1
                else:
                    print("FurnitureBaxterAssemblyEnv: unrecognized arm %s" % control_arm)
                    raise NameError
                run = "gripper"
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
            elif action[0] == "connect":
                # TODO TODO TODO IMPLEMENT
                run = "connect"
            else:
                print("FurnitureBaxterAssemblyEnv: unrecognized action %s" % action[0])
                raise NameError

            # perform action
            if run == "controller":
                # run controller
                potentials = np.zeros(10)
                num_moving = 0
                while not self._controller.objective_met:
                    # set flag so unity will update
                    self._unity_updated = False
                    # compute controller update
                    velocities = self._controller.get_control()
                    # perform controller command
                    self.perform_command(velocities, self.gripper_grabs, True)
                    # render
                    self.vr.add(self.render('rgb_array'))
                    potentials[num_moving%10] = self._controller.potential()
                    if np.std(potentials) <= 1e-5:
                        success = False
                        break
                    num_moving += 1
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
                self.perform_connection() # TODO TODO TODO IMPLEMENT
                # render
                self.vr.add(self.render('rgb_array'))

            print("FurnitureBaxterAssemblyEnv: finished action %s" % action[0])
        if success:
            print("FurnitureBaxterAssemblyEnv: Goal met!")
        else:
            print("This step is unfinished.")
        time.sleep(2)
        return success
    """
    Performs the given controller command.
    Copied from part of _step_continuous() function in furniture.py

    @param velocities, the change in configuration induced by the controller
    @param gripper_grabs, flags indicating whether [right, left] grippers have closed
        where 1 means gripper is closed and -1 means gripper is open
    @param controller_velocities, a boolean indicating whether to update velocities using a controller
    """
    def perform_command(self, velocities, gripper_grabs, controller_velocities=True):
        # set up low action
        low_action = np.concatenate([velocities, gripper_grabs])

        # keep trying to reach the target in a closed-loop
        ctrl = self._setup_action(low_action)
        for i in range(self._action_repeat):
                self._do_simulation(ctrl)

                if i + 1 < self._action_repeat:
                    if controller_velocities:
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
    config.furniture_id = 7 # swivel_chair
    config.record_video = True
    # create an environment and run manual control of Baxter environment
    env = FurnitureBaxterAssemblyEnv(config)
    env.run_controller(config)

if __name__ == "__main__":
    main()
