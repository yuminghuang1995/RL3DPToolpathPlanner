import numpy as np
import os
import sys
import argparse
from main_gamma import process_block

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.setrecursionlimit(20000)

np.set_printoptions(suppress=True, threshold=100)


def determine_printing_mode(env_name_):
    if '_' in env_name_:
        mode = env_name_.split('_')[1]
        if mode in ['CCF', 'wireframe', 'metal']:
            return mode
    return 'wireframe'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process 3D printing parameters.')
    parser.add_argument('--model', type=str, required=True,
                        help="Choose from Bearing_CCF, Gear_CCF, Shell_CCF, "
                             "Bunny_wireframe, Coral_wireframe, Duck_wireframe, "
                             "Cat_wireframe, Femur_wireframe, Molar_wireframe, Lounge_wireframe, "
                             "Bracket_metal, Hook_metal, Femur_metal")

    args = parser.parse_args()
    env_name = args.model
    printing_mode = determine_printing_mode(env_name)
    process_block(env_name, printing_mode)

