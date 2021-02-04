"""
Demo for assembling furniture with Baxter.
"""

import argparse

import numpy as np

from env import make_env
from env.models import furniture_names, background_names
from util import str2bool

"""
Inputs types of agent, furniture model, and background and simulates the environment.
"""
def main(args):
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

    # set parameters for the environment (furniture_id, background)
    env_name = 'FurnitureBaxterAssemblyEnv'
    args.env = env_name
    args.furniture_id = furniture_id
    args.background = background_name

    print()
    print("Creating assembly environment (robot: {}, furniture: {}, background: {})".format(
        env_name, furniture_name, background_name))

    # make environment following arguments
    env = make_env(env_name, args)

    # run assembly of furniture
    env.run_controller(args)

    # close the environment instance
    print("Closing FurnitureBaxterAssemblyEnv")
    print("Thank you for trying FurnitureBaxterAssemblyEnv!")
    env.close()

"""
Returns argument parser for furniture assembly environment
"""
def argsparser():
    parser = argparse.ArgumentParser("Demo for furniture assembly with Baxter")
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--debug', type=str2bool, default=False)
    parser.add_argument('--record_video', type=str2bool, default=False)

    import config.furniture as furniture_config
    furniture_config.add_argument(parser)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = argsparser()
    main(args)
