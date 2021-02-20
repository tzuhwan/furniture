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
    

obj_list_global = {
    '1_chair_base': [10], # n_grasp_pose
    '2_chair_column': [2, 2],  # for objects with helper_poses, [n_grasp_pose, n_helper_pose]
    '3_chair_seat': [14]
}
'''
live connection interface class to CoppeliaSim
'''
class PoseReader(object):
    def __init__(self, connect=True):
        if connect:
            self.connect()
        else:
            self.sim_client = -1
        self.obj_list = obj_list_global
    
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
    
    def save_json_file(self, filename='default_furniture_grasp_poses.json'):
        grasp_pose_dict = {}
        for obj in self.obj_list:
            grasp_n = self.obj_list[obj]
            grasp_pose_dict[obj] = []
            for i in range(grasp_n[0]):
                pose_name = '{}_grasp_pose_{}'.format(obj[obj.rfind('_')+1:], i)
                pose = self.read_object_pose(pose_name, obj).tolist()
                grasp_pose_dict[obj].append(pose)
            if len(grasp_n) == 2: # has helper grasp pose, append to the end
                for j in range(grasp_n[1]):
                    pose_name = '{}_helper_pose_{}'.format(obj[obj.rfind('_')+1:], j)
                    pose = self.read_object_pose(pose_name, obj).tolist()
                    grasp_pose_dict[obj].append(pose)
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