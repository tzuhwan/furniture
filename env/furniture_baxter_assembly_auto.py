"""
Define baxter furniture assembly environment class FurnitureBaxterAssemblyEnv.
"""

import os
import time
from collections import OrderedDict

import numpy as np
import copy
import math
from pyquaternion import Quaternion

import env.models
# from env.furniture_baxter import FurnitureEnv
from env.furniture_baxter import FurnitureBaxterEnv
import env.transform_utils as T
from env.controllers import Baxter6DPoseController
from env.controllers import Baxter3DPositionController
from env.controllers import BaxterRotationController
from env.controllers import BaxterAlignmentController
from env.controllers import BaxterScrewController
from itertools import combinations

import env.transform_utils as T
path = os.getcwd()
os.chdir('./object_files/scripts')
from read_local_pose import PoseReader
os.chdir(path)
os.chdir('./task_planner/pyperplan')
import pyperplan
os.chdir(path)
del path

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

        # set furniture name
        self._furniture_name = config.furniture_name
        
        # access to prelabeled poses
        self.pose_reader = PoseReader(connect=config.live_connect_coppeliasim)
        if not config.live_connect_coppeliasim:
            self.grasp_pose_dict = self.pose_reader.read_json_file(config.grasp_pose_json_file)
        else:
            self.grasp_pose_dict = self.pose_reader.save_dict()
            # print("overwriting json file")
            # self.pose_reader.save_json_file()
            # print("saved json file")

        # get number of grasp and helper poses for each part
        self.num_grasp_poses = {}
        for obj in self.pose_reader.obj_list:
            obj_name = obj[obj.find("_")+1:]
            self.num_grasp_poses[obj_name] = {}
            for part in self.pose_reader.obj_list[obj]:
                self.num_grasp_poses[obj_name][part] = {}
                n_pose_list = self.pose_reader.obj_list[obj][part][0]
                if len(n_pose_list) > 1:
                    num_grasps = n_pose_list[0]
                    num_helpers = n_pose_list[1]
                else:
                    num_grasps = n_pose_list[0]
                    num_helpers = 0
                self.num_grasp_poses[obj_name][part]['grasp_poses'] = num_grasps
                self.num_grasp_poses[obj_name][part]['helper_poses'] = num_helpers
        
        # fixed offsets related to grasping
        self.hold_offset = 0.03  # make sure the gripper covers the object with some distance and object won't slip out
        self.grasp_offset = 0.05 + self.hold_offset
        self.lift_offset = 0.05 # 0.01 is even hard for some cases
        self.x_range = {
            'chair_agne_0007': [-0.2, 0.2],
            'chair_bernard_0146': [-0.2, 0.2],
            'desk_mikael_1064': [-0.5, 0.5],
            'shelf_ivar_0678': [-0.2, 0.2],
            'swivel_chair_0700': [-0.2, 0.2],
            'table_klubbo_0743': [-0.5, 0.5],
            'table_lack_0825': [-0.5, 0.5]
        }
        self.y_range = {
            'chair_agne_0007': [-0.2, 0.2],
            'chair_bernard_0146': [-0.1, 0.1],
            'desk_mikael_1064': [-0.3, 0.3],
            'shelf_ivar_0678': [-0.2, 0.2],
            'swivel_chair_0700': [-0.1, 0.1],
            'table_klubbo_0743': [-0.3, 0.3],
            'table_lack_0825': [-0.2, 0.2]
        }
        self.n_iter_random_objects = 1000
    """
    Use existing 2D random position with standing random z rotation as pose initialization,
    TODO: add lying down pose, make table, desk top plane upside-down, close to center, chair_agne_seat upside-down
    """
    def random_place_objects(self):
        self.mujoco_model.initializer.x_range = self.x_range[self._furniture_name]
        self.mujoco_model.initializer.y_range = self.y_range[self._furniture_name]
        part_x_range = self.x_range[self._furniture_name]
        part_y_range = self.y_range[self._furniture_name]
        collision = True  # repeat random generation till parts are not in collision at first
        i_random = 0
        while collision and i_random < self.n_iter_random_objects:
            collision = False
            pos, quat = self.mujoco_model.place_objects()
            for i, (obj_name, obj_mjcf) in enumerate(self.mujoco_objects.items()):
                self._set_qpos(obj_name, pos[i], quat[i])
                print('{} pos: {}, quat {}'.format(obj_name, pos[i], quat[i]))
            self._unity_updated = False
            # velocities = np.zeros(14)
            # self.perform_command(velocities, self.gripper_grabs, True)
            # time.sleep(1)
            self.vr.add(self.render('rgb_array'))
            for obj_name in self.mujoco_objects:
                pos = self._get_qpos(obj_name)
                if not ((pos[0] >= part_x_range[0] and pos[0] <= part_x_range[1]) and (pos[1] >= part_y_range[0] and pos[1] <= part_y_range[1])):
                    collision = True
                    break
            for (part1, part2) in list(combinations(self.mujoco_objects, 2)):
                if self.on_collision(part1, part2):
                    collision = True
                    break
            i_random += 1
        if i_random == self.n_iter_random_objects:
            print('cannot generate feasible init poses for object parts within range: x={}, y={}'.format(self.x_range, self.y_range))
            return False
        else:
            return True
        
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
        if config.furniture_name == 'swivel_chair_0700': # swivel chair
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

        raw_name = env.models.furniture_names[self._furniture_id]
        self.object = raw_name[:raw_name.rfind('_')]
        self._actions = pyperplan.plan_sequence(self.object)
        
        self.default_eef_mat = {'left': self.pose_in_base_from_name('left_gripper_base'), 'right': self.pose_in_base_from_name('right_gripper_base')}
        left_gripper_pose = (self.pose_in_base_from_name('l_gripper_l_finger_tip') + self.pose_in_base_from_name('l_gripper_r_finger_tip')) / 2  # orientations are the same
        self.left_obj2eef_mat = T.pose_inv(left_gripper_pose).dot(self.default_eef_mat['left'])
        right_gripper_pose = (self.pose_in_base_from_name('r_gripper_l_finger_tip') + self.pose_in_base_from_name('r_gripper_r_finger_tip')) / 2  # orientations are the same
        self.right_obj2eef_mat = T.pose_inv(right_gripper_pose).dot(self.default_eef_mat['right'])
        self.side_eef_mat = copy.deepcopy(self.default_eef_mat)
        self.side_eef_mat['left'][1, 3] += 0.15
        self.side_eef_mat['right'][1, 3] -= 0.15  # move arm further in base y direction to clear the space

        base_pos_in_world = self.sim.data.get_body_xpos("base")
        base_rot_in_world = self.sim.data.get_body_xmat("base").reshape((3, 3))
        base_pose_in_world = T.make_pose(base_pos_in_world, base_rot_in_world)
        self.world_pose_in_base = T.pose_inv(base_pose_in_world)

        if config.render:
            self.render()
        
        if config.furniture_name == 'swivel_chair_0700': # swivel_chair:
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
        
        res = self.random_place_objects()
        if not res:
            return

        for action in self._actions:
            self.action_times = 0
            self.tried_pose = []
            while self.action_times <5 :
                print("trying action ", action)
                success = self.single_action(config, action, self.action_times)
                if success or self.action_times >= 5:
                    break
                self.action_times += 1
            if self.action_times >= 5:
                print('This task cannot finish')
                break

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
            num_grasps = self.num_grasp_poses[self.object][part]['grasp_poses']
            local_grasp_pose_mat = [T.pose2mat((pose[:3], pose[3:])) for pose in local_grasp_pose[:num_grasps]]
            ws_grasp_pose_mat = [T.pose_in_A_to_pose_in_B(grasp_pose_mat0, part_pose_mat) for grasp_pose_mat0 in local_grasp_pose_mat]
            found_downwards_grasppose = False

            # TODO TODO TODO check computation of pregarsp poses; Emily copied this from down below
            # initialize list of pregrasp, grasp pose tuples
            pregrasp_grasp_pose_mats = []
            # compute pregrasp pose for each possible grasp pose
            for obj_grasp_mat in ws_grasp_pose_mat:
                # copied from down below; check this?
                obj2eef_mat = self.left_obj2eef_mat if action[1] == 'left' else self.right_obj2eef_mat
                grasp_pose_mat = obj_grasp_mat.dot(obj2eef_mat) # TODO isn't grasp pose already in base frame? what is this transformation doing?
                grasp_pose_mat[:3, -1] += self.hold_offset * grasp_pose_mat[:3, 2] # TODO does this account for not-downward grasps?  how do we determine what dimension to apply offset to?
                grasp_pose = T.mat2pose(grasp_pose_mat)
                pre_eef_grasp_pose_mat = grasp_pose_mat.copy()
                pre_eef_grasp_pose_mat[:3, -1] -= self.grasp_offset * pre_eef_grasp_pose_mat[:3, 2]
                pregrasp_grasp_pose_mats.append((obj_grasp_mat, pre_eef_grasp_pose_mat))

            dists_poses = self.compute_pregrasp_pose_distances(pregrasp_grasp_pose_mats, action[1])
            print("poses and grasps: ")
            for i in range(len(dists_poses)):
                print("pose %d, distance %f" % (i, dists_poses[i][0]))

            # TODO TODO TODO check selection of grasp poses by distance
            for dist, pregrasp_pose, obj_pose in dists_poses:
                if obj_pose not in self.tried_pose: # TODO TODO TODO this is a bug, this check will not work
                    print("trying grasp pose closest to %s arm, distance: %f" % (action[1], dist))
                    obj_grasp_mat = obj_pose
                    break

            # TODO TODO TODO uncomment later; Xiaotong's original grasp selection
            # for obj_grasp_mat in ws_grasp_pose_mat:
            #     # if obj_grasp_mat[2, 3] < 0:  # drop poses that are too low
            #     #     continue
            #     if obj_grasp_mat[2, 2] < -0.8:  # select grasp poses that has z axis pointing downwards
            #         if len(self.tried_pose) != 0:
            #             for t_p in self.tried_pose:
            #                 if np.linalg.norm(obj_grasp_mat[:3, 3] - t_p[:3, 3]) < 1e-1:
            #                     continue
            #         found_downwards_grasppose = True
            #         break
            self.hold_offset = 0.03
            # TODO TODO TODO uncomment later; Xiaotong's code for using helper poses
            # if not found_downwards_grasppose:
            #     # using helper pose
            #     if config.furniture_name == 'swivel_chair_0700': # swivel chair
            #         local_helper_pose = local_grasp_pose[num_grasps:]
            #         local_helper_pos = []
            #         for h in local_helper_pose:
            #             local_helper_pos.append(T.pose_in_A_to_pose_in_B(T.pose2mat((h[:3], h[3:])), part_pose_mat)[:3, 3])
            #         helper_x_axis = part_pose_mat[:3, 2]
            #         if np.abs(np.dot(helper_x_axis,[0, 0, -1]))<0.8:
            #             helper_y_axis = np.cross([0, 0, -1], helper_x_axis)
            #             helper_z_axis = np.cross(helper_x_axis, helper_y_axis)
            #             if helper_z_axis[2] > 0:
            #                 helper_y_axis = -1*helper_y_axis
            #                 helper_z_axis = -1*helper_z_axis
            #             hand_pose_mat = left_hand_pose_mat if action[1] == 'left' else right_hand_pose_mat
            #             distance = np.linalg.norm(np.array(local_helper_pos)-hand_pose_mat[:3, 3], axis=-1)
            #             helper_pos = np.array(local_helper_pos[np.argmin(distance)][:3])
            #             final_helper_pose_mat = np.zeros((4, 4))
            #             final_helper_pose_mat[3, 3] = 1
            #             final_helper_pose_mat[:3, 3] = helper_pos
            #             final_helper_pose_mat[:3, 0] = helper_x_axis
            #             final_helper_pose_mat[:3, 1] = helper_y_axis
            #             final_helper_pose_mat[:3, 2] = helper_z_axis
            #             obj_grasp_mat = final_helper_pose_mat
            #             self.hold_offset = 0.01
            #             found_downwards_grasppose = True
            # if not found_downwards_grasppose:
            #     print('cannot find downwards grasppose')
            self.tried_pose.append(obj_grasp_mat)
            # 2. plus offset to gripper_base which is end effector in controller, add other offsets for pre- and post- grasp pose
            obj2eef_mat = self.left_obj2eef_mat if action[1] == 'left' else self.right_obj2eef_mat
            grasp_pose_mat = obj_grasp_mat.dot(obj2eef_mat)
            grasp_pose_mat[:3, -1] += self.hold_offset * grasp_pose_mat[:3, 2]
            grasp_pose = T.mat2pose(grasp_pose_mat)
            pre_eef_grasp_pose_mat = grasp_pose_mat.copy()
            pre_eef_grasp_pose_mat[:3, -1] -= self.grasp_offset * pre_eef_grasp_pose_mat[:3, 2]
            pre_grasp_pose = T.mat2pose(pre_eef_grasp_pose_mat)
            post_grasp_pose_mat = grasp_pose_mat.copy()
            post_grasp_pose_mat[:3, -1] -= self.lift_offset * post_grasp_pose_mat[:3, 2] # lift in world frame
            post_grasp_pose = T.mat2pose(post_grasp_pose_mat)
            target_pose_mat = self.default_eef_mat[action[1]]
            target_pose = T.mat2pose(target_pose_mat)
            self._action_sequence.append(("Baxter6DPoseController", (action[1], target_pose[0], target_pose[1], 1e-3)))
            self._action_sequence.append(("Baxter6DPoseController", (action[1], pre_grasp_pose[0], pre_grasp_pose[1])))
            self._action_sequence.append(("Baxter6DPoseController", (action[1], grasp_pose[0], grasp_pose[1])))
            self._action_sequence.append(("close-gripper", action[1]))
            self._action_sequence.append(("Baxter6DPoseController", (action[1], post_grasp_pose[0], post_grasp_pose[1])))
        elif action[0] in ['connect', 'screw']:  # todo: develop rotation controller for screw motion
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
            target_pose_mat = self.side_eef_mat[action[1]]
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
                if len(action[1])==3:
                    control_arm, goal_pos, goal_quat = action[1]
                    self._controller = Baxter6DPoseController(
                        bullet_data_path=os.path.join(env.models.assets_root, "bullet_data"),
                        robot_jpos_getter=self._robot_jpos_getter,
                        verbose=False
                    )
                else:
                    control_arm, goal_pos, goal_quat, potential = action[1]
                    self._controller = Baxter6DPoseController(
                        bullet_data_path=os.path.join(env.models.assets_root, "bullet_data"),
                        robot_jpos_getter=self._robot_jpos_getter,
                        verbose=False, potential_threshold = potential
                    )
                self._controller.set_goal(control_arm, goal_pos, goal_quat)
                self._controller.set_arm_speed(8)
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
                run = "connect"
            else:
                print("FurnitureBaxterAssemblyEnv: unrecognized action %s" % action[0])
                raise NameError

            # perform action
            if run == "controller":
                # run controller
                potentials = np.ones(20)
                num_moving = 0
                while not self._controller.objective_met:
                    # set flag so unity will update
                    self._unity_updated = False
                    # compute controller update
                    velocities = self._controller.get_control()
                    # perform controller command
                    self.perform_command(velocities, self.gripper_grabs, True)
                    if (np.array(self.gripper_grabs)==1).any():
                        l_finger_tip = self.pose_in_base_from_name('l_gripper_l_finger_tip') if control_arm == "left" else self.pose_in_base_from_name('r_gripper_l_finger_tip')
                        r_finger_tip = self.pose_in_base_from_name('l_gripper_r_finger_tip') if control_arm == "left" else self.pose_in_base_from_name('r_gripper_r_finger_tip')
                        if np.linalg.norm(l_finger_tip[:3, 3] - r_finger_tip[:3, 3])<2e-2:
                            success = False
                            print("********** no success because of finger tips **********")
                            self.action_times -= 1
                            break
                    # render
                    self.vr.add(self.render('rgb_array'))
                    potentials[num_moving%20] = self._controller.potential()
                    if np.std(potentials) <= 1e-7 or np.all(np.diff(potentials) > 0):
                        success = False
                        print("********** no success because of potentials **********")
                        if np.std(potentials) <= 1e-7:
                            print("no movement")
                        else:
                            print("potentials increasing")
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
                self.perform_connection()
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
    Computes the distances between current end-effector pose and possible part grasp poses

    @param grasp_pose_mats, a list of part grasp poses in base frame
    @param control_arm, the arm being controlled for this action
    @return a list of (dist, pose) tuples, sorted by distance
    """
    def compute_grasp_pose_distances(self, grasp_pose_mats, control_arm):
        # initialize list of (dist, pose) tuples
        dists_poses = []

        # get current pose of end-effector on given control arm
        if control_arm == "left":
            ee_pose_mat = self.pose_in_base_from_name('left_gripper_base')
        else: # control_arm == "right"
            ee_pose_mat = self.pose_in_base_from_name('right_gripper_base')
        
        # get end-effector position and quaternion
        ee_pos, ee_quat = T.mat2pose(ee_pose_mat)

        # compute distance to each grasp pose
        for grasp_pose in grasp_pose_mats:
            # get object position and quaternion
            obj_pos, obj_quat = T.mat2pose(grasp_pose)

            # compute difference between current and goal pose
            dist_pos = np.linalg.norm(ee_pos - obj_pos)
            dist_quat = Quaternion.distance(Quaternion(ee_quat), Quaternion(obj_quat))
            dist_quat = min(dist_quat, math.pi - dist_quat)
            dist = dist_pos + dist_quat
            print("total distance: %f, position distance: %f, rotation distance: %f"
                % (dist, dist_pos, dist_quat)
            )

            # add distance and pose to list of tuples
            dists_poses.append((dist, grasp_pose))

        # sort grasp poses by shortest distance
        dists_poses.sort()

        return dists_poses

    """
    Computes the distances between current end-effector pose and possible part pre-grasp poses

    @param pregrasp_grasp_pose_mats, a list of part (pregrasp, grasp) pose tuples in base frame
    @param control_arm, the arm being controlled for this action
    @return a list of (dist, pregrasp pose, grasp pose) tuples, sorted by distance
    """
    def compute_pregrasp_pose_distances(self, pregrasp_grasp_pose_mats, control_arm):
        # initialize list of (dist, pregrasp pose, grasp pose) tuples
        dists_poses = []

        # get current pose of end-effector on given control arm
        if control_arm == "left":
            ee_pose_mat = self.pose_in_base_from_name('left_gripper_base')
        else: # control_arm == "right"
            ee_pose_mat = self.pose_in_base_from_name('right_gripper_base')
        
        # get end-effector position and quaternion
        ee_pos, ee_quat = T.mat2pose(ee_pose_mat)

        # compute distance to each grasp pose
        for pregrasp_pose, grasp_pose in pregrasp_grasp_pose_mats:
            # get object position and quaternion
            pregrasp_obj_pos, pregrasp_obj_quat = T.mat2pose(pregrasp_pose)
            grasp_obj_pos, grasp_obj_quat = T.mat2pose(grasp_pose)

            # compute difference between current and pregrasp pose
            dist_pos = np.linalg.norm(ee_pos - pregrasp_obj_pos)
            dist_quat = Quaternion.distance(Quaternion(ee_quat), Quaternion(pregrasp_obj_quat))
            dist_quat = min(dist_quat, math.pi - dist_quat)
            dist = dist_pos + dist_quat
            print("total distance: %f, position distance: %f, rotation distance: %f"
                % (dist, dist_pos, dist_quat)
            )

            # add distance and pose to list of tuples
            dists_poses.append((dist, pregrasp_pose, grasp_pose))

        # sort grasp poses by shortest distance
        dists_poses.sort()

        return dists_poses

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
    config.furniture_id = 9 # swivel_chair
    config.record_video = False
    config.live_connect_coppeliasim = False
    config.grasp_pose_json_file = 'default_furniture_grasp_poses.json'
    # config.seed = np.random.randint(low=0, high=100000)
        
    # create an environment and run manual control of Baxter environment
    env = FurnitureBaxterAssemblyEnv(config)
    env.run_controller(config)

if __name__ == "__main__":
    main()
