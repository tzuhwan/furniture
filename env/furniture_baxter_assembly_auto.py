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
from task_planner.pyperplan import pyperplan

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
        self.pose_reader = PoseReader(connect=config.live_connect_coppeliasim)
        if not config.live_connect_coppeliasim:
            self.grasp_pose_dict = self.pose_reader.read_json_file(config.grasp_pose_json_file)
        else:
            self.grasp_pose_dict = self.pose_reader.save_dict()
        
        # fixed offsets related to grasping
        self.hold_offset = 0.02  # make sure the gripper covers the object with some distance and object won't slip out
        self.grasp_offset = 0.05 + self.hold_offset
        self.lift_offset = 0.05 # 0.01 is even hard for some cases
    
    """
    Use existing 2D random position with standing random z rotation as pose initialization,
    TODO: add lying down pose
    """
    def random_place_objects(self, x_range=[-0.2, 0.2], y_range=[-0.1, 0.2]):
        self.mujoco_model.initializer.x_range = x_range
        self.mujoco_model.initializer.y_range = y_range
        pos, quat = self.mujoco_model.place_objects() # raise error in id=4 desk_mikael, may need different range for different objects
        for i, (obj_name, obj_mjcf) in enumerate(self.mujoco_objects.items()):
            self._set_qpos(obj_name, pos[i], quat[i])
            print('{} pos: {}, quat {}'.format(obj_name, pos[i], quat[i]))

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
        self.reset(config.furniture_id, config.background)
        # reset swivel base
        if config.furniture_id == 7: # swivel chair
            print("reseting swivel chair base pose")
            self._set_qpos('1_chair_base',
                [0, 0.1, 0.0144],
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
        self.random_place_objects(x_range=[-0.4, 0.4], y_range=[-0.2, 0.4])
        
        
        self.default_eef_mat = {'left': self.pose_in_base_from_name('left_gripper_base'), 'right': self.pose_in_base_from_name('right_gripper_base')}
        left_gripper_pose = (self.pose_in_base_from_name('l_gripper_l_finger_tip') + self.pose_in_base_from_name('l_gripper_r_finger_tip')) / 2  # orientations are the same
        self.left_obj2eef_mat = T.pose_inv(left_gripper_pose).dot(self.default_eef_mat['left'])
        right_gripper_pose = (self.pose_in_base_from_name('r_gripper_l_finger_tip') + self.pose_in_base_from_name('r_gripper_r_finger_tip')) / 2  # orientations are the same
        self.right_obj2eef_mat = T.pose_inv(right_gripper_pose).dot(self.default_eef_mat['right'])
        self.default_eef_mat['left'][1, 3] += 0.25
        self.default_eef_mat['right'][1, 3] -= 0.25  # move arm further in base y direction to clear the space 

        base_pos_in_world = self.sim.data.get_body_xpos("base")
        base_rot_in_world = self.sim.data.get_body_xmat("base").reshape((3, 3))
        base_pose_in_world = T.make_pose(base_pos_in_world, base_rot_in_world)
        self.world_pose_in_base = T.pose_inv(base_pose_in_world)

        if config.render:
            self.render()
        
        if config.furniture_id == 7: # swivel_chair:
            self.object = '7_swivel_chair'
            self._actions = [
                ['side', 'left'],    # move arm to the side to clear the place
                ['grasp', 'right', '2_chair_column'],
                ['connect', 'right', 'column-base,conn_site1', 'base-column,conn_site1'],  # assumption: has grasped first part
                ['side', 'right'],
                ['grasp', 'left', '3_chair_seat'],
                ['connect', 'left', 'seat-column,conn_site2', 'column-seat,conn_site2'],
            ]

        from util.video_recorder import VideoRecorder
        self.vr = VideoRecorder()
        self.vr.add(self.render('rgb_array'))

        for action in self._actions:
            action_times = 0
            while 1:
                success = self.single_action(config, action, action_times)
                if success:
                    break
                action_times += 1

        if config.record_video:
            self.vr.save_video('FurnitureBaxterAssemblyEnv_test2.mp4')
        time.sleep(2)
        print("FurnitureBaxterAssemblyEnv returning")

        return

    def single_action(self, config, action, action_times):
        self._action_sequence = []
        left_hand_pose_mat = self.pose_in_base_from_name('left_gripper_base')
        right_hand_pose_mat = self.pose_in_base_from_name('right_gripper_base')
        if action[0] == 'grasp':
            self.gripper_grabs = [-1, -1]
            # 1. object part grasp pose transformed to workspace coordinate
            part = action[2]
            partname = part[part.rfind('_') + 1:]
            local_grasp_pose = self.grasp_pose_dict[self.object][part]
            part_pose_mat = self.pose_in_base_from_name(part)
            left_distance = np.linalg.norm(np.array(part_pose_mat)[:3, 3]-left_hand_pose_mat[:3, 3], axis=-1)
            right_distance = np.linalg.norm(np.array(part_pose_mat)[:3, 3]-right_hand_pose_mat[:3, 3], axis=-1)
            action[1] = 'left' if left_distance < right_distance else 'right'
            local_grasp_pose_mat = [T.pose2mat((pose[:3], pose[3:])) for pose in local_grasp_pose]
            ws_grasp_pose_mat = [T.pose_in_A_to_pose_in_B(grasp_pose_mat0, part_pose_mat) for grasp_pose_mat0 in local_grasp_pose_mat]
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

            if len(potential_pose)<=action_times:
                # using helper pose
                if config.furniture_id == 7:
                    helper_pose = local_grasp_pose[:2]
                    helper_x_axis = part_pose_mat[:3, 2]
                    if np.abs(np.dot(helper_x_axis,[0, 0, -1]))<0.8:
                        helper_y_axis = np.cross(helper_x_axis, [0, 0, -1])
                        helper_z_axis = np.cross(helper_y_axis, part_pose_mat[:3, 2])
                        if helper_z_axis[2] > 0:
                            helper_y_axis = -1*helper_y_axis
                            helper_z_axis = -1*helper_z_axis
                        hand_pose_mat = left_hand_pose_mat if action[1] == 'left' else right_hand_pose_mat
                        distance = np.linalg.norm(np.array(helper_pose)[:,:3]-hand_pose_mat[:3, 3], axis=-1)
                        helper_pos = np.array(helper_pose[np.argmin(distance)][:3])
                        final_helper_pose_mat = np.zeros((4, 4))
                        final_helper_pose_mat[3, 3] = 1
                        final_helper_pose_mat[:3, 3] = helper_pos
                        final_helper_pose_mat[:3, 0] = helper_x_axis
                        final_helper_pose_mat[:3, 1] = helper_y_axis
                        final_helper_pose_mat[:3, 2] = helper_z_axis
                        potential_pose.append(final_helper_pose_mat)
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
            left_distance = np.linalg.norm(np.array(moving_site_qpos_mat)[:3, 3]-left_hand_pose_mat[:3, 3], axis=-1)
            right_distance = np.linalg.norm(np.array(moving_site_qpos_mat)[:3, 3]-right_hand_pose_mat[:3, 3], axis=-1)
            action[1] = 'left' if left_distance < right_distance else 'right'
            hand_pose_mat = left_hand_pose_mat if action[1] == 'left' else right_hand_pose_mat
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
        elif action[0] == 'side':
            target_pose_mat = self.default_eef_mat[action[1]]
            target_pose = T.mat2pose(target_pose_mat)
            self._action_sequence.append(("open-gripper", action[1]))
            self._action_sequence.append(
                ("Baxter6DPoseController", (action[1], target_pose[0], target_pose[1])))
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
                potentials = np.zeros(20)
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
                    potentials[num_moving%20] = self._controller.potential()
                    if np.std(potentials) <= 1e-7 or np.all(np.diff(potentials) > 0):
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
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--debug', type=str2bool, default=False)

    parser.set_defaults(render=True)

    config, unparsed = parser.parse_known_args()
    config.furniture_id = 7 # swivel_chair
    config.record_video = False
    config.live_connect_coppeliasim = False
    config.grasp_pose_json_file = 'default_furniture_grasp_poses.json'
    # config.seed = np.random.randint(low=0, high=100000)
        
    # create an environment and run manual control of Baxter environment
    env = FurnitureBaxterAssemblyEnv(config)
    env.run_controller(config)

if __name__ == "__main__":
    main()
