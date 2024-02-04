import numpy as np
import os
import sys
from main_gamma import process_block

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.setrecursionlimit(20000)

np.set_printoptions(suppress=True, threshold=100)


if __name__ == '__main__':
    # choose from  'Bearing_CCF' 'Gear_CCF' 'Shell_CCF'  'Bunny_wireframe' 'Coral_wireframe' 'Pavilion-0-2-1-3-5-6-4-7_wireframe' 'Femur_metal' 'Hook_metal' 'Bracket_metal'
    env_name = 'Bunny_wireframe'
    LSG_range = 6  # choose from 1-6
    printing_mode = env_name.split('_')[-1]  # choose from 'CCF', 'wireframe' and 'metal'
    process_block(env_name, printing_mode, LSG_range)
