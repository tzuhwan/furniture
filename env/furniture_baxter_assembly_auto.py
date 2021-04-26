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
from env.controller_planner import ControlBasis
from itertools import combinations
from scipy.spatial.transform import Rotation as R
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

        # set vebose and debug flag
        self.verbose = config.verbose
        self.debug = config.debug

        # initialize control basis
        self.cb = ControlBasis()

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
        self.grasp_offset = 0.08
        self.lift_offset = 0.1 # 0.01 is even hard for some cases
        self.x_range = { # left-right shift from workspace center (- robot right, viewer left)
            'chair_agne_0007': [-0.2, 0.2],
            'chair_bernhard_0146': [-0.2, 0.2],
            'desk_mikael_1064': [-0.5, 0.5],
            'shelf_ivar_0678': [-0.2, 0.2],
            'swivel_chair_0700': [-0.2, 0.2],
            'table_klubbo_0743': [-0.5, 0.5],
            'table_lack_0825': [-0.5, 0.5]
        }
        self.y_range = { # forwards-backwards shift from workspace center (+ towards robot)
            'chair_agne_0007': [-0.2, 0.2],
            'chair_bernhard_0146': [-0.1, 0.1],
            'desk_mikael_1064': [-0.3, 0.3],
            'shelf_ivar_0678': [-0.2, 0.2],
            'swivel_chair_0700': [-0.15, 0.05], # [0.0, 0.2]
            'table_klubbo_0743': [-0.3, 0.3],
            'table_lack_0825': [-0.2, 0.2]
        }
        self.n_iter_random_objects = 1000

        # dictionary of furniture names to list of parts that need to be flipped
        self.flip_parts = {
            'chair_agne_0007': ['3_seat'],
            'table_klubbo_0743': ['5_table_top'],
            'table_lack_0825': ['1_table_leg1', '2_table_leg2', '3_table_leg3', '4_table_leg4', '5_table_top']
        }
        # dictionary of furniture names to list of parts that have bounded positions for reachability (usually towards center of workspace)
        # TODO currently not used since x_range and y_range (set above) should keep table tops within reachable workspace even if not in center
        self.position_bounds = {
            'swivel_chair_0700': ['3_chair_seat'],
            'table_klubbo_0743': ['5_table_top'],
            'table_lack_0825': ['5_table_top']
        }
        # dictionary of furniture names to dictionary of parts to bounded rotations for reachability
        self.rotation_bounds = {
            'table_lack_0825': {
                '5_table_top': [(np.pi / 180) * -10, (np.pi / 180) * 10]
            }
        }
        # dictionary of furniture names to list of parts that can be grasped anywhere along a specified plane
        self.allow_grasp_interpolation = {
            'swivel_chair_0700': ['2_chair_column']
        }

    """
    Use existing 2D random position with standing random z rotation as pose initialization,
    TODO: add lying down pose, make table, desk top plane upside-down, close to center
    """
    def random_place_objects(self):
        self.mujoco_model.initializer.object_name = self._furniture_name
        self.mujoco_model.initializer.x_range = self.x_range[self._furniture_name]
        self.mujoco_model.initializer.y_range = self.y_range[self._furniture_name]
        part_x_range = self.x_range[self._furniture_name]
        part_y_range = self.y_range[self._furniture_name]
        # for particular objects, set initializer to flip parts so connection sites are oriented properly
        if self._furniture_name in self.flip_parts.keys():
            self.mujoco_model.initializer.flip_parts = self.flip_parts[self._furniture_name]
        # for particular objects, set initializer to limit position of parts for reachability
        if self._furniture_name in self.position_bounds.keys(): # TODO this affects nothing
            self.mujoco_model.initializer.part_position_bounds = self.position_bounds[self._furniture_name]
        # for particular objects, set initializer to limit rotation of parts for reachability
        if self._furniture_name in self.rotation_bounds.keys():
            self.mujoco_model.initializer.part_rotation_bounds = self.rotation_bounds[self._furniture_name]
        collision = True  # repeat random generation till parts are not in collision at first
        i_random = 0
        while collision and i_random < self.n_iter_random_objects:
            collision = False
            pos, quat = self.mujoco_model.place_objects()
            for i, (obj_name, obj_mjcf) in enumerate(self.mujoco_objects.items()):
                self._set_qpos(obj_name, pos[i], quat[i])
                print('{} pos: {}, quat: {}'.format(obj_name, pos[i], quat[i]))
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
    Append the potential with Rotating along the X-axis
    """
    def Rotation_interpolation(self, pose_mat, axis = 'x', angle = 360, num_pts = 12):
        d_angle = angle/num_pts
        if pose_mat.shape == (4,4):
            r = np.eye(4)
            r[:3, :3] = R.from_euler(axis, d_angle, degrees=True).as_dcm()
        else:
            r = R.from_euler(axis, d_angle, degrees=True).as_dcm()
        inter_pose = []
        pose = pose_mat.copy()
        for _ in range(num_pts):
            pose = pose.dot(r)
            inter_pose.append(pose)
        return inter_pose
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
        self._furniture_id = env.models.furniture_name2id[self._furniture_name]
        self.reset(self._furniture_id, config.background)
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

        # raw_name = env.models.furniture_names[self._furniture_id]
        self.object = self._furniture_name[:self._furniture_name.rfind('_')]
        self._actions = pyperplan.plan_sequence(self.object)
        
        self.default_eef_mat = {'left': self.pose_in_base_from_name('left_gripper_base'), 'right': self.pose_in_base_from_name('right_gripper_base')}
        left_gripper_pose = (self.pose_in_base_from_name('l_gripper_l_finger_tip') + self.pose_in_base_from_name('l_gripper_r_finger_tip')) / 2  # orientations are the same
        self.left_gripper_base2eef_mat = T.pose_inv(left_gripper_pose).dot(self.default_eef_mat['left'])
        right_gripper_pose = (self.pose_in_base_from_name('r_gripper_l_finger_tip') + self.pose_in_base_from_name('r_gripper_r_finger_tip')) / 2  # orientations are the same
        self.right_gripper_base2eef_mat = T.pose_inv(right_gripper_pose).dot(self.default_eef_mat['right'])
        self.side_eef_mat = copy.deepcopy(self.default_eef_mat)
        self.side_eef_mat['left'][1, 3] += 0.4#0.15
        self.side_eef_mat['left'][0, 3] -= 0.15
        self.side_eef_mat['right'][1, 3] -= 0.4#0.15 # move arm further in base y direction to clear the space
        self.side_eef_mat['right'][0, 3] -= 0.15

        base_pos_in_world = self.sim.data.get_body_xpos("base")
        base_rot_in_world = self.sim.data.get_body_xmat("base").reshape((3, 3))
        base_pose_in_world = T.make_pose(base_pos_in_world, base_rot_in_world)
        self.world_pose_in_base = T.pose_inv(base_pose_in_world)

        if config.render:
            self.render()
        
        if config.furniture_name == 'swivel_chair_0700': # swivel_chair:
            self._actions = [
                ['side', 'right'],    # move arm to the side to clear the place
                ['grasp', 'left', '2_chair_column'],
                ['connect', 'left', 'column-base,conn_site1', 'base-column,conn_site1'],  # assumption: has grasped first part
                ['side', 'left'],
                ['grasp', 'right', '3_chair_seat'],
                ['connect', 'right', 'seat-column,conn_site2', 'column-seat,conn_site2'],
            ]

        from util.video_recorder import VideoRecorder
        self.vr = VideoRecorder()
        self.vr.add(self.render('rgb_array'))
        
        res = self.random_place_objects()
        if not res:
            return

        while True:
            self.need_regrasp = False
            action = self._actions[0]
            print('trying action ', action)
            success = self.single_action(config, action)
            if success:
                previous_action = action.copy()
                self._actions.remove(action)
            else:
                if self.need_regrasp:
                    self._actions.insert(0, previous_action)
                else:
                    print('The system is finished')
                    break
            if len(self._actions) == 0:
                break
        # for i, action in enumerate(self._actions):
        #     success = self.single_action(config, action)
            # self.action_times = 0
            # while self.action_times < 100:
            #     print("trying action ", action)
            #     success = self.single_action(config, action, self.action_times)
            #     if success:
            #         break
            #     self.action_times += 1
            # if self.action_times >= 12:
            #     print('This task cannot finish')
            #     break

        if config.record_video:
            self.vr.save_video('FurnitureBaxterAssemblyEnv_test2.mp4')
        time.sleep(2)
        print("FurnitureBaxterAssemblyEnv returning")

        return

    def single_action(self, config, action):
        self.tried_pose = []
        self.action_times = 0
        success = 0
        self.tried_pose_threshold = 0.01
        while not success:
            left_hand_pose_mat = self.pose_in_base_from_name('left_gripper_base')
            right_hand_pose_mat = self.pose_in_base_from_name('right_gripper_base')
            self._action_sequence = []
            if action[0] == 'grasp':
                self.gripper_grabs = [-1, -1]
                # 1. object part grasp pose transformed to workspace coordinate
                part = action[2]
                partname = part[part.rfind('_') + 1:]
                local_grasp_pose = self.grasp_pose_dict[self.object][part]
                part_pose_mat = self.pose_in_base_from_name(part)
                left_distance = np.linalg.norm(np.array(part_pose_mat)[:3, 3]-left_hand_pose_mat[:3, 3], axis=-1)
                right_distance = np.linalg.norm(np.array(part_pose_mat)[:3, 3]-right_hand_pose_mat[:3, 3], axis=-1)
                # if action_times % 2 == 0:
                #     action[1] = 'left' if left_distance < right_distance else 'right'
                # else:
                #     action[1] = 'right' if left_distance < right_distance else 'left'
                action[1] = 'left' if left_distance < right_distance else 'right'
                num_grasps = self.num_grasp_poses[self.object][part]['grasp_poses']
                local_grasp_pose_mat = [T.pose2mat((pose[:3], pose[3:])) for pose in local_grasp_pose[:num_grasps]]
                final_local_grasp_pose_mats = []
                # if part supports rotation interpolation of grasp poses, then get additional grasp poses
                if (self._furniture_name in self.allow_grasp_interpolation.keys()) and (part in self.allow_grasp_interpolation[self._furniture_name]):
                    for l_p in local_grasp_pose_mat:
                        final_local_grasp_pose_mats += self.Rotation_interpolation(l_p)
                else:
                    final_local_grasp_pose_mats = local_grasp_pose_mat
                ws_grasp_pose_mat = [T.pose_in_A_to_pose_in_B(grasp_pose_mat0, part_pose_mat) for grasp_pose_mat0 in final_local_grasp_pose_mats]
                found_downwards_grasppose = False

                obj2eef_mat = self.left_gripper_base2eef_mat if action[1] == 'left' else self.right_gripper_base2eef_mat
                # initialize list of pregrasp, grasp pose tuples
                pregrasp_grasp_pose_mats = []
                # compute pregrasp pose for each possible grasp pose
                for obj_grasp_mat in ws_grasp_pose_mat:
                    grasp_pose_mat = obj_grasp_mat.dot(obj2eef_mat)
                    pre_eef_grasp_pose_mat = grasp_pose_mat.copy()
                    grasp_pose_mat[:3, -1] += self.hold_offset * grasp_pose_mat[:3, 2]
                    # grasp_pose = T.mat2pose(grasp_pose_mat)
                    pre_eef_grasp_pose_mat[:3, -1] -= self.grasp_offset * pre_eef_grasp_pose_mat[:3, 2]
                    pregrasp_grasp_pose_mats.append((pre_eef_grasp_pose_mat, grasp_pose_mat))

                dists_poses = self.compute_pregrasp_pose_distances(pregrasp_grasp_pose_mats, action[1])
                # print("poses and dists: ")
                # for i in range(len(dists_poses)):
                #     print("pose %d, distance %f" % (i, dists_poses[i][0]))
                for dist, pregrasp_pose_mat, grasp_pose_mat in dists_poses:
                    find_pose = True
                    # check whether the current grasp pose is tried or not
                    grasp_pose = T.mat2pose(grasp_pose_mat)
                    for t_p in self.tried_pose:
                        dist_pos = np.linalg.norm(grasp_pose[0] - t_p[0])
                        dist_quat = Quaternion.distance(Quaternion(grasp_pose[1]), Quaternion(t_p[1]))
                        dist_quat = min(dist_quat, math.pi - dist_quat)
                        tried_pose_dist = dist_pos + dist_quat
                        if tried_pose_dist < self.tried_pose_threshold:
                            find_pose = False
                    if find_pose:
                        break
                self.tried_pose.append(grasp_pose)
                default_pose_mat = self.default_eef_mat[action[1]]
                default_pose = T.mat2pose(default_pose_mat)
                pre_grasp_pose = T.mat2pose(pregrasp_pose_mat)
                post_grasp_pose_mat = grasp_pose_mat.copy()
                post_grasp_pose_mat[2, -1] += self.lift_offset  # lift in world frame
                post_grasp_pose = T.mat2pose(post_grasp_pose_mat)
                if self.action_times == 0:
                    self._action_sequence.append(("Baxter6DPoseController", (action[1], default_pose[0], default_pose[1], 1e-3)))
                self._action_sequence.append(("open-gripper", action[1]))
                self._action_sequence.append(("Baxter6DPoseController", (action[1], pre_grasp_pose[0] + [0, 0, 0.15], pre_grasp_pose[1])))
                self._action_sequence.append(("Baxter6DPoseController", (action[1], pre_grasp_pose[0], pre_grasp_pose[1])))
                self._action_sequence.append(("Baxter6DPoseController", (action[1], grasp_pose[0], grasp_pose[1])))
                self._action_sequence.append(("close-gripper", action[1]))
                self._action_sequence.append( ("Baxter6DPoseController", (action[1], post_grasp_pose[0], post_grasp_pose[1], 1e-3)))

                # TODO uncomment later; Xiaotong's original grasp selection
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
                # TODO uncomment later; Xiaotong's code for using helper poses
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

                # self.tried_pose.append(obj_grasp_mat)
                # # 2. plus offset to gripper_base which is end effector in controller, add other offsets for pre- and post- grasp pose
                # obj2eef_mat = self.left_obj2eef_mat if action[1] == 'left' else self.right_obj2eef_mat
                # grasp_pose_mat = obj_grasp_mat.dot(obj2eef_mat)
                # grasp_pose_mat[:3, -1] += self.hold_offset * grasp_pose_mat[:3, 2]
                # grasp_pose = T.mat2pose(grasp_pose_mat)
                # pre_eef_grasp_pose_mat = grasp_pose_mat.copy()
                # pre_eef_grasp_pose_mat[:3, -1] -= self.grasp_offset * pre_eef_grasp_pose_mat[:3, 2]
                # pre_grasp_pose = T.mat2pose(pre_eef_grasp_pose_mat)
                # post_grasp_pose_mat = grasp_pose_mat.copy()
                # post_grasp_pose_mat[:3, -1] -= self.lift_offset * post_grasp_pose_mat[:3, 2] # lift in world frame
                # post_grasp_pose = T.mat2pose(post_grasp_pose_mat)
                # target_pose_mat = self.default_eef_mat[action[1]]
                # target_pose = T.mat2pose(target_pose_mat)
                # self._action_sequence.append(("Baxter6DPoseController", (action[1], target_pose[0], target_pose[1], 1e-3)))
                # self._action_sequence.append(("Baxter6DPoseController", (action[1], pre_grasp_pose[0], pre_grasp_pose[1])))
                # self._action_sequence.append(("Baxter6DPoseController", (action[1], grasp_pose[0], grasp_pose[1])))
                # self._action_sequence.append(("close-gripper", action[1]))
                # self._action_sequence.append(("Baxter6DPoseController", (action[1], post_grasp_pose[0], post_grasp_pose[1])))
            elif action[0] in ['connect', 'screw']:  # todo: develop rotation controller for screw motion
                control_arm, pre_pre_target_hand_pose, pre_target_hand_pose, target_hand_pose = self.compute_connect_target_pose(action[2], action[3])
                self.tried_pose.append(target_hand_pose)
                
                default_pose_mat = self.default_eef_mat[control_arm]
                default_pose = T.mat2pose(default_pose_mat)
                target_pose_mat = self.side_eef_mat[control_arm]
                target_pose = T.mat2pose(target_pose_mat)
                # if self.action_times == 0:
                #     self._action_sequence.append(("Baxter6DPoseController", (control_arm, default_pose[0], default_pose[1], 1e-3)))
                # self._action_sequence.append(
                #     ("pre-pre-connect-pose", action[2], action[3], "Baxter6DPoseController", (control_arm, pre_pre_target_hand_pose[0], pre_pre_target_hand_pose[1], 0.005)))
                self._action_sequence.append(
                    ("pre-connect-pose", action[2], action[3], "Baxter6DPoseController", (control_arm, pre_target_hand_pose[0], pre_target_hand_pose[1], 0.005)))
                self._action_sequence.append(
                    ("connect-pose", action[2], action[3], "Baxter6DPoseController", (control_arm, target_hand_pose[0], target_hand_pose[1])))
                self._action_sequence.append(("connect", " "))
                self._action_sequence.append(("open-gripper", control_arm))
                self._action_sequence.append(
                    ("Baxter6DPoseController", (control_arm, target_pose[0], target_pose[1])))
            elif action[0] == 'side':
                target_pose_mat = self.side_eef_mat[action[1]]
                target_pose = T.mat2pose(target_pose_mat)
                self._action_sequence.append(("open-gripper", action[1]))
                self._action_sequence.append(
                    ("Baxter6DPoseController", (action[1], target_pose[0], target_pose[1])))
            previous_action_times = self.action_times
            success = self.one_action(self._action_sequence)
            self.action_times+=1
            print('{}: {} times'.format(action[0], self.action_times))
            if action[0] in ['connect', 'screw'] and previous_action_times == self.action_times:
                self.need_regrasp = True
                print('The object fall down from the gripper. We need to regrasp it.')
                break
            if success or self.action_times>=20:
                if success:
                    print('Finish the step:',action[0])
                else:
                    print(action[0],' cannot finish')
                break

        return success

    def one_action(self, _action_sequence):
        ### ACTION SEQUENCE ###
        # start action sequence
        print("FurnitureBaxterAssemblyEnv: Starting action sequence")
        success = True
        i = 0
        # perform action sequence
        while i < len(_action_sequence): #for i in range(len(_action_sequence)):
            print("FurnitureBaxterAssemblyEnv: Starting action %d of %d" % (i+1, len(_action_sequence)))

            if not success:
                break
            # set up action
            action = _action_sequence[i]
            i += 1
            run = ""

            # check if action needs periodic updates based on object poses
            if action[0] in ['pre-pre-connect-pose', 'pre-connect-pose', 'connect-pose']:
                periodic_update = [action[0], action[1], action[2]]
                action = action[3:]
            else:
                periodic_update = None

            # multi-objective controller
            if isinstance(action[0], tuple):
                controllers_goals = action
                self.cb.initialize_controllers_and_goals(controllers_goals,
                    bullet_data_path=os.path.join(env.models.assets_root, "bullet_data"),
                    robot_jpos_getter=self._robot_jpos_getter
                )
                self.composition = [controller_name for controller_name, goal_info in controllers_goals]
                run = "multiobjective-controller"
            # single-objective controller
            elif action[0] in self.cb.control_basis_controllers:
                controller_goals = [action]
                self.cb.initialize_controllers_and_goals(controller_goals,
                    bullet_data_path=os.path.join(env.models.assets_root, "bullet_data"),
                    robot_jpos_getter=self._robot_jpos_getter,
                    verbose=self.verbose, update_arm_speed=True
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
            if run in self.cb.control_basis_controllers:
                objective_met = False
                control_arm = self.cb.get_control_arm()
                num_iters = 0
                # run controller
                while not objective_met:
                    # set flag so unity will update
                    self._unity_updated = False
                    # compute potential
                    pot, progress = self.cb.compute_singleobjective_controller_potential(run)
                    # compute controller update
                    velocities, objective_met = self.cb.compute_singleobjective_controller_update(run)
                    num_iters += 1
                    # perform controller command
                    self.perform_command(velocities, self.gripper_grabs, True, controller_name=run)
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
                    # check for progress
                    if not progress:
                        success = False
                        print("********** no success because of potentials **********")
                        # try connection anyway
                        if periodic_update and periodic_update[0] == 'connect-pose':
                            print("trying connection anyway")
                            # set flag so unity will update
                            self._unity_updated = False
                            # perform connection
                            success = self.perform_connection()
                            # render
                            self.vr.add(self.render('rgb_array'))
                            # increment counter to pass connect pose and connect action
                            if success:
                                objective_met = True
                                i += 1
                        break
                    # check if update needed
                    if periodic_update and (num_iters%50 == 0):
                        print("Updating controller goal for %s pose" % periodic_update[0])
                        if periodic_update[0] == 'pre-pre-connect-pose':
                            control_arm, pre_pre_connect_pose, _, _ = self.compute_connect_target_pose(periodic_update[1], periodic_update[2])
                            self.cb.update_singleobjective_controller_goal(run, control_arm, pre_pre_connect_pose)
                        elif periodic_update[0] == 'pre-connect-pose':
                            control_arm, _, pre_connect_pose, _ = self.compute_connect_target_pose(periodic_update[1], periodic_update[2])
                            self.cb.update_singleobjective_controller_goal(run, control_arm, pre_connect_pose)
                        elif periodic_update[0] == 'connect-pose':
                            control_arm, _, _, connect_pose = self.compute_connect_target_pose(periodic_update[1], periodic_update[2])
                            self.cb.update_singleobjective_controller_goal(run, control_arm, connect_pose)
                        else:
                            print("FurnitureBaxterAssemblyEnv: unrecognized update %s" % periodic_update)
                            raise NameError
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
                success = self.perform_connection()
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
    @param controller_name, the name of the single-objective control basis controller being run
    """
    def perform_command(self, velocities, gripper_grabs, controller_velocities=True, controller_name=None):
        # set up low action
        low_action = np.concatenate([velocities, gripper_grabs])

        # keep trying to reach the target in a closed-loop
        ctrl = self._setup_action(low_action)
        for i in range(self._action_repeat):
                self._do_simulation(ctrl)
                
                if i + 1 < self._action_repeat:
                    if controller_velocities:
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
                    return result
                    # if result:
                    #     return
                    # break

    """
    Computes the distances between current end-effector pose and possible grasp/connect poses.
    Note: distance is computed based on position and rotation, since positions of the grasp poses will
          be the same for any poses defined on the same local point on the object.

    @param pose_mats, a list of part grasp/connect poses in base frame
    @param control_arm, the arm being controlled for this action
    @return a list of (dist, pose) tuples, sorted by distance
    """
    def compute_pose_distances(self, pose_mats, control_arm):
        # initialize list of (dist, pose) tuples
        dists_poses = []

        # get current pose of end-effector on given control arm
        if control_arm == "left":
            ee_pose_mat = self.pose_in_base_from_name('left_gripper_base')
        else: # control_arm == "right"
            ee_pose_mat = self.pose_in_base_from_name('right_gripper_base')
        
        # get end-effector position and quaternion
        ee_pos, ee_quat = T.mat2pose(ee_pose_mat)
        ee_euler = (180/np.pi) * np.array(T.quat_to_euler(ee_quat))
        if self.debug:
            print("ee euler angles: ", ee_euler)

        # compute distance to each grasp pose
        for pose in pose_mats:
            # get object position and quaternion
            pos, quat = T.mat2pose(pose)
            euler = (180/np.pi) * np.array(T.quat_to_euler(quat))
            if self.debug:
                print("possible pos: ", pos, "quat: ", quat, "euler: ", euler)

            # compute difference between current and goal pose
            dist_pos = np.linalg.norm(ee_pos - pos)
            dist_rot = np.linalg.norm(ee_euler - euler)
            # alternative distance option: angle difference between quaternions
            # dist_quat = Quaternion.distance(Quaternion(ee_quat), Quaternion(quat))
            # dist_quat = min(dist_quat, math.pi - dist_quat)
            # alternative distance option: angle difference between z-axis
            # dist_quat = np.arccos(np.dot(ee_pose_mat[2, :2], pose[2, :2]))
            dist = dist_pos + dist_rot #dist_quat
            if self.debug:
                print("total distance: %f, position distance: %f, rotation distance: %f"
                    % (dist, dist_pos, dist_rot) # dist_quat
                )

            # add distance and pose to list of tuples
            dists_poses.append((dist, pose))

        # sort grasp poses by shortest distance
        dists_poses.sort()

        return dists_poses

    """
    Computes the distances between current end-effector pose and possible part pre-grasp poses.
    Note: distance is computed based on position only, since pre-grasp poses are defined off the object
          along the approach axis.

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
            # dist_quat = Quaternion.distance(Quaternion(ee_quat), Quaternion(pregrasp_obj_quat))
            # dist_quat = min(dist_quat, math.pi - dist_quat)
            dist_quat = np.arccos(np.dot(ee_pose_mat[2, :2], pregrasp_pose[2, :2])) # change to z-axis angle difference
            dist = dist_pos + dist_quat
            if self.debug:
                print("total distance: %f, position distance: %f, rotation distance: %f"
                    % (dist, dist_pos, dist_quat)
                )

            # add distance and pose to list of tuples
            dists_poses.append((dist, pregrasp_pose, grasp_pose))

        # sort grasp poses by shortest distance
        dists_poses.sort()

        return dists_poses

    """
    Computes the target hand pose for a connection between two sites.

    @param moving_site_name, the name of the site to be connected on the object being acted on by the gripper
    @param target_site_name, the name of the site to be connected on the stationary object (usually on the floor)
    @return the arm being controlled
    @return the pre-pre-target pose
    @return the pre-target pose
    @return the target pose for the gripper based on the transformation between the moving and target sites
    """
    def compute_connect_target_pose(self, moving_site_name, target_site_name):
        # get manipulated part name based on moving connection site
        partial_part_name = moving_site_name[:moving_site_name.find('-')]

        # determine if connection sites should be interpolated for part
        allow_interpolation = False
        if self._furniture_name in self.allow_grasp_interpolation.keys():
            for part in self.allow_grasp_interpolation[self._furniture_name]:
                if partial_part_name in part:
                    allow_interpolation = True

        # get the poses of the connection sites in unity
        target_site_qpos = self._site_xpos_xquat(target_site_name)
        moving_site_qpos = self._site_xpos_xquat(moving_site_name)

        # get the poses of the connection sites in base
        target_site_qpos_mat = T.pose_in_A_to_pose_in_B(T.pose2mat((target_site_qpos[:3], target_site_qpos[3:])),
                                                        self.world_pose_in_base)
        moving_site_qpos_mat = T.pose_in_A_to_pose_in_B(T.pose2mat((moving_site_qpos[:3], moving_site_qpos[3:])),
                                                        self.world_pose_in_base)
        target_site_pose = T.mat2pose(target_site_qpos_mat)
        moving_site_pose = T.mat2pose(moving_site_qpos_mat)
        target_site_euler = (180/np.pi) * np.array(T.quat_to_euler(target_site_pose[1]))
        moving_site_euler = (180/np.pi) * np.array(T.quat_to_euler(moving_site_pose[1]))
        if self.debug:
            print("target site pos: ", target_site_pose[0], "target site quat: ", target_site_pose[1], "target site euler: ", target_site_euler)
            print("moving site pos: ", moving_site_pose[0], "moving site quat: ", moving_site_pose[1], "moving site euler: ", moving_site_euler)

        # get the poses of the grippers
        left_hand_pose_mat = self.pose_in_base_from_name('left_gripper_base')
        right_hand_pose_mat = self.pose_in_base_from_name('right_gripper_base')

        # get control arm and initialize target pose
        left_distance = np.linalg.norm(np.array(moving_site_qpos_mat)[:3, 3]-left_hand_pose_mat[:3, 3], axis=-1)
        right_distance = np.linalg.norm(np.array(moving_site_qpos_mat)[:3, 3]-right_hand_pose_mat[:3, 3], axis=-1)
        # TODO: A potential bug if the moving site is closer to the ungrasped hand
        control_arm = 'left' if left_distance < right_distance else 'right'
        hand_pose_mat = left_hand_pose_mat if control_arm == 'left' else right_hand_pose_mat
        target_hand_pose_mat = hand_pose_mat.copy()

        ##### Emily's connection computation; this seems to work better for Emily in practice
        # # compute change in position and rotation (6D)
        # dx_pos = np.array(target_site_pose[0]) - np.array(moving_site_pose[0])
        # dx_rot = np.array(target_site_euler) - np.array(moving_site_euler)
        # # fix rotation about z-axis to be 0 # TODO, may not always work with z-axis
        # dx_rot[2] = 0
        # if self.debug:
        #     print("dx pos: ", dx_pos, "dx rot: ", dx_rot)

        # # compute initial pose
        # init_hand_pose = T.mat2pose(target_hand_pose_mat)
        # init_hand_euler = (180/np.pi) * np.array(T.quat_to_euler(init_hand_pose[1]))

        # # compute goal pose base on change in position and rotation
        # goal_hand_pos = np.array(init_hand_pose[0]) + dx_pos
        # goal_hand_rot = np.array(init_hand_euler) + dx_rot
        # if self.debug:
        #     print("init hand pos: ", init_hand_pose[0], "init hand rot: ", init_hand_euler)
        #     print("goal hand pos: ", goal_hand_pos, "goal hand rot: ", goal_hand_rot)

        # # compute target hand pose
        # target_hand_pose = (goal_hand_pos, T.euler_to_quat(goal_hand_rot))
        # target_hand_pose_mat = T.pose2mat(target_hand_pose)

        # # compute possible rotations and rank by distance
        # possible_connection_targets = self.Rotation_interpolation(target_hand_pose_mat, axis='x')
        # dists_targets = self.compute_pose_distances(possible_connection_targets, control_arm)
        # # print("poses and dists: ")
        # # for i in range(len(dists_targets)):
        # #     print("pose %d, distance %f" % (i, dists_targets[i][0]))

        # # get closest connection pose that has not been tried yet
        # for dist, p in dists_targets:
        #     find_pose = True
        #     target_hand_pose_mat = p
        #     target_hand_pose = T.mat2pose(target_hand_pose_mat)
        #     for t_p in self.tried_pose:
        #         dist_pos = np.linalg.norm(target_hand_pose[0] - t_p[0])
        #         dist_quat = Quaternion.distance(Quaternion(target_hand_pose[1]), Quaternion(t_p[1]))
        #         dist_quat = min(dist_quat, math.pi - dist_quat)
        #         tried_pose_dist = dist_pos + dist_quat
        #         if tried_pose_dist < self.tried_pose_threshold:
        #             find_pose = False
        #     if find_pose:
        #         print("trying pose closest to gripper with dist %f" % dist)
        #         break
        #####

        ##### Kaizhi's connection computation
        # compute difference in translation and rotation
        # translation = target_site_qpos[:3] - moving_site_qpos[:3]
        # m_site_id = self.sim.model.site_name2id(m_site)
        # m_body_id = self.sim.model.site_bodyid[m_site_id]
        # m_body_name = self.sim.model.body_names[m_body_id]
        # new_pos, new_quat = T.transform_to_target_quat(moving_site_qpos, self._get_qpos(m_body_name), target_site_qpos[3:])
        translation = target_site_qpos_mat[:3, 3] - moving_site_qpos_mat[:3, 3]
        rotation = (moving_site_qpos_mat[:3, :3]).transpose().dot(target_site_qpos_mat[:3, :3])
        # rotation = (target_site_qpos_mat[:3, :3]).transpose().dot(moving_site_qpos_mat[:3, :3])
        appended_rotation = self.Rotation_interpolation(rotation)
        # compute target translation
        target_hand_pose_mat[:3, 3] += translation
        # target_hand_pose_mat[:3,:3] = T.pose2mat((new_pos,new_quat))[:3, :3].dot(target_hand_pose_mat[:3,:3])
        # compute target rotation
        for rotation in appended_rotation:
            find_pose = True
            target_hand_pose_mat[:3, :3] = rotation.dot(hand_pose_mat[:3, :3])
            target_hand_pose = T.mat2pose(target_hand_pose_mat)
            for t_p in self.tried_pose:
                dist_pos = np.linalg.norm(target_hand_pose[0] - t_p[0])
                dist_quat = Quaternion.distance(Quaternion(target_hand_pose[1]), Quaternion(t_p[1]))
                dist_quat = min(dist_quat, math.pi - dist_quat)
                tried_pose_dist = dist_pos + dist_quat
                if tried_pose_dist < tried_pose_threshold:
                    find_pose = False
            if find_pose:
                break
        #####

        # compute the pre-target pose
        # pre_target_hand_pose_mat = target_hand_pose_mat.copy()
        # pre_target_hand_pose_mat[:3, -1] -= self.grasp_offset * pre_target_hand_pose_mat[:3, 2]
        # pre_target_hand_pose = T.mat2pose(pre_target_hand_pose_mat)
        pre_target_hand_pose = (target_hand_pose[0]+[0,0,self.lift_offset], target_hand_pose[1])
        # compute the pre-pre-target pose
        pre_pre_target_hand_pose = (pre_target_hand_pose[0]+[0,0,self.lift_offset], pre_target_hand_pose[1])

        return control_arm, pre_pre_target_hand_pose, pre_target_hand_pose, target_hand_pose

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
    parser.add_argument('--verbose', type=str2bool, default=False)

    parser.set_defaults(render=True)

    config, unparsed = parser.parse_known_args()
    config.furniture_name = 'swivel_chair_0700'
    config.record_video = False
    config.live_connect_coppeliasim = False
    config.grasp_pose_json_file = 'default_furniture_grasp_poses.json'
    # config.seed = np.random.randint(low=0, high=100000)
        
    # create an environment and run manual control of Baxter environment
    env = FurnitureBaxterAssemblyEnv(config)
    env.run_controller(config)

if __name__ == "__main__":
    main()
