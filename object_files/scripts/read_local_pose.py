from .vrep_newlib import sim # need relative path to call from furniture_baxter_assembly_auto
# from vrep_newlib import sim
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import math
import numpy as np
import os
import json

# pose: [x, y, z, qx, qy, qz, qw]

def euler2rotm(euler):
    return R.from_euler('zyx', [euler[2], euler[1], euler[0]]).as_matrix()


def rotm2euler(rotm):
    return R.from_matrix(rotm).as_euler('zyx')[::-1]


def euler2quat(euler):
    return R.from_euler('zyx', [euler[2], euler[1], euler[0]]).as_quat()


def interpolate_pos(pose1, pose2, n=10):
    # assume same quaternion of two poses
    return np.linspace(pose1, pose2, num=n)


def interpolate_rotate_axis(pose, axis='z', n=10):
    pose_array = pose
    pos = pose[:3]
    ct, st = math.cos(2*math.pi/n), math.sin(2*math.pi/n)
    if axis == 'z':
        rot_delta = np.array([[ct, -st, 0], [st, ct, 0], [0, 0, 1]])
    elif axis == 'x':
        rot_delta = np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]])
    elif axis == 'y':
        rot_delta = np.array([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])
    else:
        print('wrong axis input')
        return np.array([])
    rotm = R.from_quat(pose_array[3:]).as_dcm()
    for i in range(1, n):
        rotm = np.matmul(rotm, rot_delta)
        pose_array = np.vstack((pose_array, np.concatenate([pos, R.from_dcm(rotm).as_quat()])))
        # print(R.from_quat(pose_array[-1][3:]).as_dcm())
    return pose_array
    

'''
live connection interface class to CoppeliaSim
'''
class PoseReader(object):
    def __init__(self, connect=True):
        if connect:
            self.connect()
        else:
            self.sim_client = -1
        self.obj_list = {
            '1_chair_agne': {
                '1_leg1': [[6], ['chair_leg1_grasp_pose']],
                '2_leg2': [[6], ['chair_leg2_grasp_pose']],
                '3_seat': [[4], ['chair_seat_grasp_pose']]
            },
            '2_chair_bernhard': {
                '1_chair_leg_left': [[6], ['chair_bernhard_left_leg_grasp_pose']],
                '2_chair_leg_right': [[6], ['chair_bernhard_right_leg_grasp_pose']],
                '3_chair_seat': [[12], ['chair_bernhard_seat_grasp_pose']]
            },
            '4_desk_mikael': {
                '1_leftplane': [[8], ['desk_mikael_left_plane_grasp_pose']],
                '2_rightplane': [[8], ['desk_mikael_left_plane_grasp_pose'], '1_leftplane'],
                '3_drawer': [[6], ['desk_mikael_drawer_grasp_pose']],
                '4_topplane': [[8], ['desk_mikael_top_plane_grasp_pose']],
            },
            '5_shelf_ivar': {
                '2_box1': [[16], ['shelf_ivar_box_grasp_pose'], 'box'], # coppeliasim object name is different
                '3_box2': [[16], ['shelf_ivar_box_grasp_pose'], 'box'],
                '4_box3': [[16], ['shelf_ivar_box_grasp_pose'], 'box'],
                '5_box4': [[16], ['shelf_ivar_box_grasp_pose'], 'box'],
                '6_box5': [[16], ['shelf_ivar_box_grasp_pose'], 'box'],
                '1_column': [[9], ['shelf_ivar_column_grasp_pose']]
            },
            '7_swivel_chair': {
                '1_chair_base': [[10], ['base_grasp_pose']], # n_grasp_pose
                '2_chair_column': [[2, 2], ['column_grasp_pose', 'column_helper_pose']], # for objects with helper_poses, [n_grasp_pose, n_helper_pose]
                '3_chair_seat': [[14], ['seat_grasp_pose']]
            },
            '8_table_klubbo': {
                '1_table_leg1': [[8], ['klubbo_table_leg1_grasp_pose']],
                '2_table_leg2': [[8], ['klubbo_table_leg2_grasp_pose']],
                '3_table_leg3': [[8], ['klubbo_table_leg3_grasp_pose']],
                '4_table_leg4': [[8], ['klubbo_table_leg4_grasp_pose']],
                '5_table_top': [[8], ['klubbo_table_top_grasp_pose']]
            },
            '9_table_lack': {
                '1_table_leg1': [[2, 4], ['table_lack_leg_grasp_pose', 'table_lack_leg_helper_pose'], 'table_leg'],
                '2_table_leg2': [[2, 4], ['table_lack_leg_grasp_pose', 'table_lack_leg_helper_pose'], 'table_leg'],
                '3_table_leg3': [[2, 4], ['table_lack_leg_grasp_pose', 'table_lack_leg_helper_pose'], 'table_leg'],
                '4_table_leg4': [[2, 4], ['table_lack_leg_grasp_pose', 'table_lack_leg_helper_pose'], 'table_leg'],
                '5_table_top': [[6], 'table_lack_top_grasp_pose']
        }
}
    
    def connect(self):
        self.sim_client = sim.simxStart('127.0.0.1', 19997, True, True, -5000, 5)  # Connect to V-REP on port 19997
        if self.sim_client == -1:
            print('Failed to connect to simulation (V-REP remote API server). Exiting.')
            return False
        else:
            print('Connected to simulation.')
            return True
    
    def read_obj_size(self, obj):
        if self.sim_client == -1:
            connected = self.connect()
            if not connected:
                return []
        _, object_handle = sim.simxGetObjectHandle(self.sim_client, obj, sim.simx_opmode_blocking)
        _, xmin = sim.simxGetObjectFloatParameter(self.sim_client, object_handle, 15, sim.simx_opmode_blocking)
        _, ymin = sim.simxGetObjectFloatParameter(self.sim_client, object_handle, 16, sim.simx_opmode_blocking)
        _, zmin = sim.simxGetObjectFloatParameter(self.sim_client, object_handle, 17, sim.simx_opmode_blocking)
        _, xmax = sim.simxGetObjectFloatParameter(self.sim_client, object_handle, 18, sim.simx_opmode_blocking)
        _, ymax = sim.simxGetObjectFloatParameter(self.sim_client, object_handle, 19, sim.simx_opmode_blocking)
        _, zmax = sim.simxGetObjectFloatParameter(self.sim_client, object_handle, 20, sim.simx_opmode_blocking)
        # print(xmax - xmin, ymax - ymin, zmax - zmin)
        return [xmin, ymin, zmin, xmax, ymax, zmax]

    def read_object_pose(self, obj1, obj2):
        if self.sim_client == -1:
            connected = self.connect()
            if not connected:
                return []
        sim_ret, obj_handle1 = sim.simxGetObjectHandle(self.sim_client, obj1, sim.simx_opmode_blocking)
        sim_ret, obj_handle2 = sim.simxGetObjectHandle(self.sim_client, obj2, sim.simx_opmode_blocking)
        sim_ret, pos = sim.simxGetObjectPosition(self.sim_client, obj_handle1, obj_handle2, sim.simx_opmode_blocking)
        sim_ret, rot = sim.simxGetObjectOrientation(self.sim_client, obj_handle1, obj_handle2, sim.simx_opmode_blocking)
        return np.concatenate([pos, euler2quat(rot)])
    
    def save_dict(self):
        grasp_pose_dict = {}
        for obj in self.obj_list:
            print(obj)
            grasp_pose_dict[obj] = {}
            for part in self.obj_list[obj]:
                print(part)
                grasp_pose_dict[obj][part] = []
                grasp_n, posename = self.obj_list[obj][part][0], self.obj_list[obj][part][1]
                if len(self.obj_list[obj][part]) == 3: # name in coppeliasim is different from objectname__partname because many parts share a model, etc.
                    partname = '{}__{}'.format(obj, self.obj_list[obj][part][2])
                else:
                    partname = '{}__{}'.format(obj, part)
                for i in range(grasp_n[0]):
                    pose_name = '{}_{}'.format(posename[0], i)
                    pose = self.read_object_pose(pose_name, partname).tolist()
                    grasp_pose_dict[obj][part].append(pose)
                if len(grasp_n) == 2: # has helper grasp pose, append to the end
                    for j in range(grasp_n[1]):
                        pose_name = '{}_{}'.format(posename[1], j)
                        pose = self.read_object_pose(pose_name, partname).tolist()
                        grasp_pose_dict[obj][part].append(pose)
        return grasp_pose_dict
    
    def save_json_file(self, filename='default_furniture_grasp_poses.json'):
        grasp_pose_dict = self.save_dict()
        with open(filename, 'w') as f:
            json.dump(grasp_pose_dict, f, indent=4)
            
    def read_json_file(self, filename='default_furniture_grasp_poses.json'):
        grasp_pose_dict = {}
        with open(filename, 'r') as f:
            grasp_pose_dict = json.load(f)
        return grasp_pose_dict
        

if __name__ == '__main__':    
    p = PoseReader()
    p.save_json_file()
    p.read_json_file()