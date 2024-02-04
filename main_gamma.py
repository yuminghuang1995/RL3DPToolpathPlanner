import random
# import gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import rl_utils
import math
import networkx as nx
import functions as func
import environment as env
import dqn as dqn
from torch_geometric.data import Data
# torch.backends.cudnn.benchmark = True
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from calculate_module import calculate
import concurrent.futures
import pickle

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.setrecursionlimit(20000)

np.set_printoptions(suppress=True, threshold=100)


def process_block(env_name, printing_mode, LSG_range):

    if printing_mode == 'CCF':
        calc_mode = 'Euler'
        material = 'CCF'
    elif printing_mode == 'wireframe':
        calc_mode = 'Euler'
        material = 'PLA3D'
    else:
        calc_mode = 'Tsp'
        material = 'Metal'

    max_edge_pass = LSG_range

    block = 0
    interval = 0.05

    if calc_mode == 'Tsp':
        env_o = env.Environ(env_name, 'data/Node' + env_name + '.txt', None, plot_finished_path_=False, mode=calc_mode)
    else:
        env_o = env.Environ(env_name, 'data/Node' + env_name + '.txt', 'data/Edge' + env_name + '.txt', plot_finished_path_=False, mode=calc_mode)
    coordinates, init_adjacency_matrix, edges, G, max_edge_length, boundary_nodes_array = env_o.build_graph(interval, material)

    print('Calculating: ', env_name)
    print('LSG range: ', LSG_range)

    figure_index = 1
    angle_limit = 120
    state_dim = 3  # fix 3
    beam_num = 1
    random_beam = 0
    heat_radius = 3 * interval
    rl = True
    train_mode = True
    dataset_mode = False
    savept = False
    dataset_path = 'dataset/' + env_name
    if calc_mode == 'Tsp':
        max_edge_pass = 3

    # # fix random_seed
    # random.seed(0)
    # np.random.seed(0)
    # torch.manual_seed(0)

    ori_center = np.array([0, 0, 0])
    rays = func.generate_rays(16)

    start_node = 0
    global_beam_seq = calculate(start_node, state_dim, init_adjacency_matrix, G, coordinates, edges, max_edge_pass, env_name, angle_limit,
                                beam_num, boundary_nodes_array, heat_radius, calc_mode, material, block, rl, random_beam, ori_center, dataset_path, rays, train_mode, dataset_mode, max_edge_length, savept)

    best_ave_angle = 100000
    best_limit_angle = 100000
    best_lifting = 100000
    best_reward = 0
    best_path = []
    best_dis = 0
    location = 0
    if len(global_beam_seq) == 0:
        print('Cannot find path!')
    else:
        for i in range(len(global_beam_seq)):
            temp_ave_angle = 0
            temp_limit_angle = 0
            temp_best_path = global_beam_seq[i][0]
            temp_lifting = 0
            temp_distance = 0
            for ii in range(len(temp_best_path) - 1):
                if ii == 0:
                    temp_ave_angle += 0
                else:
                    angle_ = 0
                    if material == 'Metal' or material == 'CCF' or material == 'PLA3D' or material == 'Tsp':
                        angle_ = func.calculate_angle(coordinates[temp_best_path[ii - 1]],
                                                      coordinates[temp_best_path[ii]],
                                                      coordinates[temp_best_path[ii + 1]])
                    if material == 'Clay':
                        angle_ = func.calculate_angle_along_stress(coordinates[temp_best_path[ii]],
                                                                   coordinates[temp_best_path[ii + 1]],
                                                                   np.array([10, 0, 0]))

                    temp_distance += func.calculate_distance(coordinates[temp_best_path[ii - 1]], coordinates[temp_best_path[ii]])
                    if ii >= len(temp_best_path) - 2:
                        temp_distance += func.calculate_distance(coordinates[temp_best_path[ii]], coordinates[temp_best_path[ii + 1]])
                    temp_ave_angle += angle_
                    if angle_limit < angle_:
                        temp_limit_angle += 1
                    if calc_mode == 'Tsp' and material == 'Metal':
                        if init_adjacency_matrix[temp_best_path[ii - 1]][temp_best_path[ii]] == 0:
                            temp_lifting += 1
                    else:
                        if angle_ >= 179:
                            temp_lifting += 1
            temp_ave_angle = temp_ave_angle / len(temp_best_path)
            if temp_limit_angle < best_limit_angle or (
                    temp_limit_angle == best_limit_angle and temp_lifting < best_lifting):

                best_limit_angle = temp_limit_angle
                best_ave_angle = temp_ave_angle
                best_path = temp_best_path
                best_lifting = temp_lifting
                best_reward = global_beam_seq[i][1]
                best_dis = temp_distance
                location = i

        if material == 'CCF':
            print('Best limit angle:', best_limit_angle)
            best_dis = best_dis * 0.001
            print('Best distance', '%.3f' % best_dis)
            # print('Best edge pass:', len(best_path) - 1)

        # print('Best path', best_path)


        adjacency_matrix_final = func.create_adj_matrix(init_adjacency_matrix.copy(), global_beam_seq[location][0])
        G_orig = func.create_heat_field(best_path, G.copy(), heat_radius)

        if calc_mode == 'Euler':
            func.draw_graph(env_name, G_orig, coordinates, best_path, adjacency_matrix_final, block=block, i=figure_index, show=material)
        else:
            func.draw_graph(env_name, G_orig, coordinates, best_path, adjacency_matrix_final, block=block, i=figure_index, show=material, mode='Tsp')

        result_path = "results"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        file_path = os.path.join(result_path, f"{env_name}_results.txt")

        with open(file_path, "w") as f:
            f.write(f"{env_name}\n")
            # f.write(f"Best limit angle: {best_limit_angle}\n")
            # f.write(f"Best distance: {best_dis}\n")
            f.write(f"Best path: {best_path}\n")

        coordinates_path = [G.nodes[node]['pos'] for node in best_path]
        total_length = len(coordinates_path)

        if material == 'PLA3D':
            print('Calculating maximum deformation...')
            total_max_def, max_ind = func.calculate_total_max_deformation(f"data/Node{env_name}.txt", f"results/{env_name}_results.txt")
            print('Max Deform:', '%.4f' % total_max_def)
            # print('Max Ind:', max_ind)

        if calc_mode == 'Euler':
            if material == 'CCF':
                print('Post processing start...')
                func.update_progress_post(0, total_length)

                base_exists = os.path.exists(f"data/{env_name}Base.txt")
                cross_exists = os.path.exists(f"data/{env_name}BaseCross.txt")

                new_coordinates_path = coordinates_path

                num_layers = 25
                layer_height = 0.75

                if base_exists and cross_exists:

                    folder_path = f"outputs/{env_name}_withBase"
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)
                    for layer in range(num_layers):

                        output_path = os.path.join(folder_path, f"{layer}.txt")

                        if layer % 2 == 0:
                            if layer % 4 == 0:
                                input_path = f"data/{env_name}Base.txt"
                            else:
                                input_path = f"data/{env_name}BaseCross.txt"

                            with open(input_path, "r") as ff:
                                lines = ff.readlines()

                            processed_lines = []
                            prev_coords = None
                            extrusion = 0
                            pre_extrusion = 0
                            ll = 0
                            for line in lines:
                                values = list(map(float, line.strip().split()))

                                values[0] = values[0] - 19.9
                                values[2] = values[2] + 0.4 + 0.5 * layer_height * layer

                                current_coords = np.array(values[:3])

                                if prev_coords is None:
                                    extrusion += 0.0
                                else:
                                    extrusion += np.linalg.norm(current_coords - prev_coords)

                                lift = 1 if extrusion - pre_extrusion > 20 or ll == 0 else 0
                                if extrusion - pre_extrusion > 20:
                                    extrusion = pre_extrusion

                                nr = np.array([0, 0, 1])
                                printer_y_rot = np.pi / 90
                                rotation_matrix = np.array([
                                    [np.cos(printer_y_rot), 0, np.sin(printer_y_rot)],
                                    [0, 1, 0],
                                    [-np.sin(printer_y_rot), 0, np.cos(printer_y_rot)]
                                ])
                                rotated_nr = rotation_matrix.dot(nr)

                                values[3:6] = rotated_nr

                                formatted_values = [f"{val:.3f}" for val in values]
                                formatted_values.extend([f"{extrusion:.3f}", str(lift), "0", "0"])
                                processed_lines.append(' '.join(formatted_values))

                                prev_coords = current_coords
                                pre_extrusion = extrusion
                                ll += 1

                            with open(output_path, "w") as fff:
                                fff.write('\n'.join(processed_lines))

                        else:
                            with open(output_path, "w") as fff:
                                extrusion = 0
                                index = 0
                                printer = 1
                                for coord in new_coordinates_path:
                                    if index == 0:
                                        lift = 1
                                    else:
                                        lift = 0
                                    fff.write(f'{coord[0]:.3f} {coord[1]:.3f} {coord[2] + 0.5 * (layer - 1) * layer_height:.3f} 0 0 1 {extrusion:.3f} {lift} {printer} 0\n')
                                    if index < len(new_coordinates_path) - 1:
                                        extrusion += func.calculate_distance(new_coordinates_path[index], new_coordinates_path[index + 1])
                                        index += 1

                    func.final_progress()
                    print('Generate CCF layer and base layer')
                else:
                    folder_path = "outputs"
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)
                    output_path = os.path.join(folder_path, f"{env_name}_output.txt")

                    layer = 1
                    with open(output_path, "w") as fff:
                        extrusion = 0
                        index = 0
                        printer = 1
                        for coord in new_coordinates_path:
                            if index == 0:
                                lift = 1
                            else:
                                lift = 0

                            fff.write(
                                f'{coord[0]:.3f} {coord[1]:.3f} {coord[2] + 0.5 * (layer - 1) * layer_height:.3f} 0 0 1\n')

                            if index < len(new_coordinates_path) - 1:
                                extrusion += func.calculate_distance(new_coordinates_path[index], new_coordinates_path[index + 1])
                                index += 1

                    func.final_progress()
                    print('Generate CCF layer')

            if material == 'PLA3D':

                folder_path = "outputs"
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                print('Post processing start...')
                output_path = os.path.join(folder_path, f"{env_name}_output.txt")
                adjacency_matrix = init_adjacency_matrix.copy()

                with open(output_path, 'w') as f:
                    extrusion = 0
                    pre_extrusion = extrusion
                    rise = 0
                    ii = 0
                    stop_printer = 0
                    pre_coord = coordinates_path[0]
                    pre_normal = [0, 0, 1]
                    for coord in coordinates_path:

                        lift = 0
                        normal = [0, 0, 1]
                        angle = 0
                        if ii == 0:
                            lift = 1
                        else:
                            lines = func.get_merged_diff(init_adjacency_matrix, adjacency_matrix)

                            if adjacency_matrix[best_path[ii - 1]][best_path[ii]] == 0:
                                lift = 1
                                if coordinates[best_path[ii-1]][2] > coordinates[best_path[ii]][2]:
                                    lines.append((best_path[ii-1], best_path[ii]))
                                _, normal, angle = func.collision_check_simulation(coordinates, best_path[ii], best_path[ii + 1], rays, lines, norm_output=True, specified_vector=np.array(pre_normal))

                            if adjacency_matrix[best_path[ii - 1]][best_path[ii]] != 0:
                                lift = 0
                                adjacency_matrix[best_path[ii - 1]][best_path[ii]] = 0.0
                                adjacency_matrix[best_path[ii]][best_path[ii - 1]] = 0.0

                                if coordinates[best_path[ii - 1]][2] > coordinates[best_path[ii]][2]:
                                    lines.append((best_path[ii - 1], best_path[ii]))

                                _, normal, angle = func.collision_check_simulation(coordinates, best_path[ii - 1], best_path[ii], rays, lines, norm_output=True, specified_vector=np.array(pre_normal))

                            extrusion += func.calculate_distance(coordinates_path[ii - 1], coordinates_path[ii])

                        if normal is not None:
                            f.write(f'{pre_coord[0]:0.2f} {pre_coord[1]:0.2f} {pre_coord[2]:0.2f} {normal[0]:0.2f} {normal[1]:0.2f} {normal[2]:0.2f}\n')
                            f.write(f'{coord[0]:0.2f} {coord[1]:0.2f} {coord[2]:0.2f} {normal[0]:0.2f} {normal[1]:0.2f} {normal[2]:0.2f}\n')
                        else:
                            f.write(f'{pre_coord[0]:0.2f} {pre_coord[1]:0.2f} {pre_coord[2]:0.2f} 0 0 1\n')
                            f.write(f'{coord[0]:0.2f} {coord[1]:0.2f} {coord[2]:0.2f} 0 0 1\n')
                            if pre_normal is not None:
                                normal = pre_normal
                            else:
                                normal = [0, 0, 1]

                        pre_coord = coord
                        pre_extrusion = extrusion
                        pre_normal = normal
                        ii += 1

                        func.update_progress_post(ii, total_length)

                func.final_progress()
                print('Generate PLA3D layer')


        if calc_mode == 'Tsp':
            print('Post processing start...')
            print('len(coordinates_path):', len(coordinates_path))
            print('len(coordinates):', len(coordinates))

            num_layers = 1
            folder_path = "outputs"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            output_path = os.path.join(folder_path, f"{env_name}_output.txt")
            with open(output_path, 'w') as f:

                for layer in range(num_layers):
                    adjacency_matrix = init_adjacency_matrix.copy()
                    f.write(f'layer\n')
                    ii = 0

                    for coord in coordinates_path:
                        if ii == 0:
                            f.write(f'{coord[0]:0.2f},{coord[1]:0.2f} 0.00 0 0 1\n')
                        elif ii == len(coordinates_path) - 1:
                            if adjacency_matrix[best_path[ii - 1]][best_path[ii]] == 0:
                                pass
                            else:
                                f.write(f'{coord[0]:0.2f},{coord[1]:0.2f} 0.00 0 0 1\n')
                        else:
                            if adjacency_matrix[best_path[ii - 1]][best_path[ii]] == 0 and adjacency_matrix[best_path[ii]][best_path[ii + 1]] == 0:
                                pass
                            elif adjacency_matrix[best_path[ii - 1]][best_path[ii]] == 0 and adjacency_matrix[best_path[ii]][best_path[ii + 1]] != 0:
                                f.write(f'{coord[0]:0.2f},{coord[1]:0.2f} 0.00 0 0 1\n')
                            elif adjacency_matrix[best_path[ii - 1]][best_path[ii]] != 0 and adjacency_matrix[best_path[ii]][best_path[ii + 1]] == 0:
                                adjacency_matrix[best_path[ii - 1]][best_path[ii]] = 0.0
                                adjacency_matrix[best_path[ii]][best_path[ii - 1]] = 0.0
                                f.write(f'{coord[0]:0.2f},{coord[1]:0.2f} 0.00 0 0 1\n')
                            else:
                                adjacency_matrix[best_path[ii - 1]][best_path[ii]] = 0.0
                                adjacency_matrix[best_path[ii]][best_path[ii - 1]] = 0.0
                                if func.calculate_angle(coordinates_path[ii-1], coordinates_path[ii], coordinates_path[ii+1]) > 1:
                                    f.write(f'{coord[0]:0.2f},{coord[1]:0.2f} 0.00 0 0 1\n')

                        ii += 1

                        func.update_progress_post(ii, total_length)

            func.final_progress()
            print('Generate PBF layer')
