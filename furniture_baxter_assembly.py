""" Define baxter furniture assembly environment class FurnitureBaxterAssemblyEnv. """

import os
from collections import OrderedDict

import numpy as np

import env.models
from env.models import furniture_names, background_names
# from env.furniture_baxter import FurnitureEnv
from env.furniture_baxter import FurnitureBaxterEnv
import env.transform_utils as T
from env.controllers import Baxter6DPoseController

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
        super(FurnitureBaxterEnv, self).__init__(config)
        # TODO not recognizing sim unless reset is explicitly called; this is causing the robot/furniture initialization to be very weird
        super(FurnitureBaxterEnv, self)._reset(config.furniture_id, config.background)

        print("FurnitureBaxterAssemblyEnv: Starting Baxter6DPoseController")
        self._controller = Baxter6DPoseController(
            bullet_data_path=os.path.join(env.models.assets_root, "bullet_data"),
            robot_jpos_getter=self._robot_jpos_getter,
        )

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
    Returns the fixed initial positions and rotations of furniture parts.

    Returns:
        xpos((float * 3) * n_obj): x,y,z position of the objects in world frame
        xquat((float * 4) * n_obj): quaternion of the objects
    """
    # def _place_objects(self):
    #     pos_init = [[-0.3, -0.2, 0.05], [0.1, -0.2, 0.05]]
    #     quat_init = [[1, 0, 0, 0], [1, 0, 0, 0]]
    #     return pos_init, quat_init

    """
    Computes reward for the block picking up task.
    """
    def _compute_reward(self, a):
        return super()._compute_reward()

    """
    Gets the current pose of the end-effector

    @param arm, the arm to get the pose of
    @return r_pos, r_quat, l_pos, l_quat the position and quaternion of the end-effectors
    """
    def _get_current_pose(self):
        return self._right_hand_pos, self._right_hand_quat, self._left_hand_pos, self._left_hand_quat

    def run_controller(self):
        goal_pos = np.array([0.25, 0.25, 0.25])
        goal_rot = np.array([0, 0, 0])
        goal_quat = T.euler_to_quat(goal_rot)
        control_arm = "left"

        # set goal in controller
        self._controller.set_goal(control_arm, goal_pos, goal_quat)

        # get initial current poses of arms
        # curr_left_pos, curr_left_quat = self._get_current_pose("left")
        # curr_right_pos, curr_right_quat = self._get_current_pose("right")
        # set current poses of arms in controller
        # self._controller.set_current_pose(
        #     curr_left_pos, curr_left_quat,
        #     curr_right_pos, curr_right_quat
        # )

        while abs(self._controller.potential()) > 0.05:
            # set current poses of arms
            self._controller.set_current_pose(self._get_current_pose())
            
            # compute controller update and perform command
            self._controller.get_control()

        print("FurnitureBaxterAssemblyEnv: Goal met!")

        return

"""
Inputs types of agent, furniture model, and background and simulates the environment.
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
    parser.set_defaults(debug=True)
    # fix the initialization of furniture parts
    parser.set_defaults(fix_init=True)

    config, unparsed = parser.parse_known_args()

    print("Baxter Furniture Assembly Environment")

    # choose a furniture model
    print()
    print("Supported furniture:\n")
    for i, furniture_name in enumerate(furniture_names):
        print('{}: {}'.format(i, furniture_name))
    print()
    try:
        s = input("Choose a furniture model (enter a number from 0 to {}): ".format(len(furniture_names) - 1))
        furniture_id = int(s)
        furniture_name = furniture_names[furniture_id]
    except:
        print("Input is not valid. Use 0 by default.")
        furniture_id = 0
        furniture_name = furniture_names[0]


    # choose a background scene
    print()
    print("Supported backgrounds:\n")
    for i, background in enumerate(background_names):
        print('{}: {}'.format(i, background))
    print()
    try:
        s = input("Choose a background (enter a number from 0 to {}): ".format(len(background_names) - 1))
        k = int(s)
        background_name = background_names[k]
    except:
        print("Input is not valid. Use 0 by default.")
        background_name = background_names[0]

    env_name = 'FurnitureBaxterAssemblyEnv'
    config.env = env_name
    config.furniture_id = furniture_id
    config.background = background_name

    print()
    print("Creating assembly environment (robot: {}, furniture: {}, background: {})".format(
        env_name, furniture_name, background_name))

    # make environment following arguments
    env = FurnitureBaxterAssemblyEnv(config)

    env.run_controller()

if __name__ == "__main__":
    main()
