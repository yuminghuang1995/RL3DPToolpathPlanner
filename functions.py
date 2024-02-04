import time
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
from collections import defaultdict
from typing import Tuple
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.path as mpath
from shapely.geometry import LineString, Point
from matplotlib.collections import LineCollection
from scipy.spatial import KDTree
from matplotlib.colors import Normalize
from matplotlib import cm
from scipy.sparse import csr_matrix, coo_matrix
import random
import os
import pickle
from mpl_toolkits.mplot3d import Axes3D
import subprocess
import beam_FEA as fea
import sys


def get_inter_coords(pre_coord, coord, step=2):
    distance = calculate_distance(pre_coord, coord)
    vector = [(c - p) / distance for p, c in zip(pre_coord, coord)]  # 归一化向量
    inter_coords = [tuple(p + s * vector[i] for i, p in enumerate(pre_coord)) for s in range(step, int(distance), step)]
    return inter_coords

def beam_fea(G_new, adjacency_matrix_flow_, node_dict, boundary_nodes_array, draw):
    lines = get_merged_diff_ga(G_new, adjacency_matrix_flow_)

    lines = np.array(lines, dtype=int)

    needed_nodes = set(lines.flatten())

    old_to_new = {old: new for new, old in enumerate(sorted(needed_nodes))}

    total_force = calculate_global_force_fea(node_dict, lines, boundary_nodes_array, A=1.0e-6, rho=1250.0, g=9.8)

    total_force_dict = {row[0]: row[1:] for row in total_force}

    total_force_new = []
    for old, new in old_to_new.items():
        if old in total_force_dict:
            total_force_new.append(np.append(new, total_force_dict[old]))

    total_force_new = np.array(total_force_new)

    g_coord = np.array([node_dict[old] for old in sorted(needed_nodes)])
    num_big_node = g_coord.shape[0]

    g_num = np.vectorize(old_to_new.get)(lines)

    boundary_nodes_array_fea = [old_to_new[node] for node in boundary_nodes_array if node in old_to_new]

    total_deformation = fea.beam_fea_calculate(g_num, g_coord, total_force_new, boundary_nodes_array_fea, draw)

    return total_deformation

def beam_fea_dfs(lines, node_dict, boundary_nodes_array, draw):

    lines = np.array(lines, dtype=int)

    needed_nodes = set(lines.flatten())

    old_to_new = {old: new for new, old in enumerate(sorted(needed_nodes))}

    total_force = calculate_global_force_fea(node_dict, lines, boundary_nodes_array, A=1.0e-6, rho=1250.0, g=9.8)

    total_force_dict = {row[0]: row[1:] for row in total_force}

    total_force_new = []
    for old, new in old_to_new.items():
        if old in total_force_dict:
            total_force_new.append(np.append(new, total_force_dict[old]))

    total_force_new = np.array(total_force_new)

    g_coord = np.array([node_dict[old] for old in sorted(needed_nodes)])
    num_big_node = g_coord.shape[0]
    g_num = np.vectorize(old_to_new.get)(lines)
    boundary_nodes_array_fea = [old_to_new[node] for node in boundary_nodes_array if node in old_to_new]
    total_deformation = fea.beam_fea_calculate(g_num, g_coord, total_force_new, boundary_nodes_array_fea, draw)

    return total_deformation


def beam_fea_graph(lines, node_dict, boundary_nodes_array, draw):

    needed_nodes = set(lines.flatten())
    old_to_new = {old: new for new, old in enumerate(sorted(needed_nodes))}
    total_force = calculate_global_force_fea(node_dict, lines, boundary_nodes_array, A=3.14e-6, rho=1250.0, g=9.8)
    total_force_dict = {row[0]: row[1:] for row in total_force}

    total_force_new = []
    for old, new in old_to_new.items():
        if old in total_force_dict:
            total_force_new.append(np.append(new, total_force_dict[old]))

    total_force_new = np.array(total_force_new)

    g_coord = np.array([node_dict[old] for old in sorted(needed_nodes)])
    num_big_node = g_coord.shape[0]
    g_num = np.vectorize(old_to_new.get)(lines)
    boundary_nodes_array_fea = [old_to_new[node] for node in boundary_nodes_array if node in old_to_new]
    total_deformation = fea.beam_fea_calculate(g_num, g_coord, total_force_new, boundary_nodes_array_fea, draw)

    return total_deformation

def beam_fea_graph_simu(lines, node_dict, boundary_nodes_array, index, draw):
    needed_nodes = set(lines.flatten())
    old_to_new = {old: new for new, old in enumerate(sorted(needed_nodes))}
    total_force = calculate_global_force_fea(node_dict, lines, boundary_nodes_array, A=3.14e-6, rho=1250.0, g=9.8)
    total_force_dict = {row[0]: row[1:] for row in total_force}
    total_force_new = []
    for old, new in old_to_new.items():
        if old in total_force_dict:
            total_force_new.append(np.append(new, total_force_dict[old]))

    total_force_new = np.array(total_force_new)

    g_coord = np.array([node_dict[old] for old in sorted(needed_nodes)])
    num_big_node = g_coord.shape[0]
    g_num = np.vectorize(old_to_new.get)(lines)
    boundary_nodes_array_fea = [old_to_new[node] for node in boundary_nodes_array if node in old_to_new]
    total_deformation = fea.beam_fea_calculate(g_num, g_coord, total_force_new, boundary_nodes_array_fea, draw, index)

    return total_deformation


def graph_preprocessing(G_new, adjacency_matrix_flow_, node_dict, boundary_nodes_array):
    lines = get_merged_diff_ga(G_new, adjacency_matrix_flow_)
    lines = np.array(lines, dtype=int)
    needed_nodes = set(lines.flatten())

    old_to_new = {old: new for new, old in enumerate(sorted(needed_nodes))}
    g_coord = np.array([node_dict[old] for old in sorted(needed_nodes)])
    num_big_node = g_coord.shape[0]
    g_num = np.vectorize(old_to_new.get)(lines)

    boundary_nodes_array_fea = [old_to_new[node] for node in boundary_nodes_array if node in old_to_new]

    g_num_new = np.zeros((g_num.shape[0], 3), dtype=int)
    g_num_new[:, [0, 2]] = g_num

    for i, edge in enumerate(g_num):
        midpoint = (g_coord[edge[0]] + g_coord[edge[1]]) / 2
        g_coord = np.vstack([g_coord, midpoint])

        g_num_new[i, 1] = g_coord.shape[0] - 1

    g_coord = np.hstack([np.arange(1, len(g_coord) + 1).reshape(-1, 1), g_coord])
    g_num_new = g_num_new + 1

    boundary_nodes_array_fea = [i + 1 for i in boundary_nodes_array_fea]
    boundary_nodes_array_fea_new = boundary_nodes_array_fea.copy()
    for row in g_num_new:
        if row[0] in boundary_nodes_array_fea and row[2] in boundary_nodes_array_fea:
            boundary_nodes_array_fea_new.append(row[1])

    g_num_new = np.hstack([np.arange(1, len(g_num_new) + 1).reshape(-1, 1), g_num_new])

    with open('D:/Abaqus/temp/Job-beam3d.inp', 'r') as f:
        lines = f.readlines()

    start_node = lines.index('*Node\n') + 1
    end_node = lines.index('*Element, type=B32\n')
    new_node = ["{0:.0f}, {1:.2f}, {2:.2f}, {3:.2f}\n".format(*row) for row in g_coord]
    lines[start_node:end_node] = new_node

    start_element = lines.index('*Element, type=B32\n') + 1
    end_element = lines.index('*Nset, nset=Set-all, generate\n')
    new_element = ["{0:.0f}, {1:.0f}, {2:.0f}, {3:.0f}\n".format(*row) for row in g_num_new]
    lines[start_element:end_element] = new_element

    nset_index = lines.index('*Nset, nset=Set-all, generate\n') + 1
    elset_index = lines.index('*Elset, elset=Set-all, generate\n') + 1

    lines[nset_index] = " 1,  {0},  1\n".format(len(g_coord))
    lines[elset_index] = " 1,  {0},  1\n".format(len(g_num_new))

    start_fix = lines.index('*Nset, nset=Set-fix\n') + 1
    end_fix = lines.index('** Section: Section-2  Profile: Profile-1\n')

    new_fix = [str(node) + ', ' for node in boundary_nodes_array_fea_new]
    lines[start_fix:end_fix] = [' '.join(new_fix).rstrip(', ') + '\n']

    with open('D:/Abaqus/temp/Job-beam3d.inp', 'w') as f:
        f.writelines(lines)

    original_wd = os.getcwd()
    script_path = 'D:/Abaqus/temp/beam-3d.py'
    os.chdir('D:/Abaqus/temp')
    os.system('python {} {}'.format(script_path, num_big_node))
    os.chdir(original_wd)

    while not os.path.exists('D:/Abaqus/temp/Job-beam3d.txt'):
        time.sleep(0.3)
    with open('D:/Abaqus/temp/Job-beam3d.txt', 'r') as f:
        total_u = f.read()
    total_u = float(total_u)
    if os.path.exists('D:/Abaqus/temp/Job-beam3d.txt'):
        os.remove('D:/Abaqus/temp/Job-beam3d.txt')

    return total_u


def calculate_global_stiffness_matrix(coordinates, lines, E=10**6, A=0.000004):
    K_global = np.zeros((3*len(coordinates), 3*len(coordinates)))

    for line in lines:
        point1 = np.array(coordinates[line[0]])
        point2 = np.array(coordinates[line[1]])

        L = np.linalg.norm(point2 - point1) * 0.001
        direction = (point2 - point1) * 0.001 / L

        K_local = E * A / L * np.outer(direction, direction)

        T = np.zeros((3, len(coordinates)*3))
        T[:, line[0]*3:line[0]*3+3] = np.eye(3)
        T[:, line[1]*3:line[1]*3+3] = -np.eye(3)

        K_global += T.T @ K_local @ T

    return K_global

def calculate_global_force_vector(coordinates, lines, boundary_nodes_array, A=0.000004, rho=1000.0, g=9.8):
    F_global = np.zeros(3*len(coordinates))

    for line in lines:
        point1 = np.array(coordinates[line[0]])
        point2 = np.array(coordinates[line[1]])

        L = np.linalg.norm(point2 - point1) * 0.001

        F_self = rho * A * L * g

        F_global[line[0]*3+2] -= F_self / 2
        F_global[line[1]*3+2] -= F_self / 2

    for node in boundary_nodes_array:
        F_global[node*3:node*3+3] = 0

    return F_global


def calculate_displacement(coordinates, lines, boundary_nodes_array, E=10**4, A=0.0016, rho=1250.0, g=9.81):

    K = calculate_global_stiffness_matrix(coordinates, lines, E, A)
    F = calculate_global_force_vector(coordinates, lines, boundary_nodes_array, A, rho, g)

    constrained_dof = []
    for node in boundary_nodes_array:
        constrained_dof.extend([node * 3, node * 3 + 1, node * 3 + 2])

    all_dof = list(range(3 * len(coordinates)))
    free_dof = list(set(all_dof) - set(constrained_dof))

    K_free = K[np.ix_(free_dof, free_dof)]
    F_free = F[free_dof]

    U_free = np.dot(np.linalg.pinv(K_free), F_free)

    U_global = np.zeros(3 * len(coordinates))

    for i in range(len(free_dof)):
        U_global[free_dof[i]] = U_free[i] * 1000

    for dof in constrained_dof:
        U_global[dof] = 0

    U_total = 0
    U_max = 0
    for i in range(len(coordinates)):
        displacement = np.linalg.norm(U_global[i * 3:i * 3 + 3])
        U_total += displacement
        if displacement > U_max:
            U_max = displacement

    return U_max, U_global


def graph_conversion(G_new, adjacency_matrix_flow_, node_dict, boundary_nodes_array, elasticity, cross_area, rho):
    lines = get_merged_diff_ga(G_new, adjacency_matrix_flow_)

    lines = np.array(lines, dtype=int)
    total_force = calculate_global_force_fea(node_dict, lines, boundary_nodes_array, A=cross_area, rho=rho, g=9.8)

    needed_nodes = set(lines.flatten())

    old_to_new = {old: new for new, old in enumerate(sorted(needed_nodes))}
    g_coord = np.array([node_dict[old] for old in sorted(needed_nodes)])

    g_num = np.vectorize(old_to_new.get)(lines)

    is_last_column_all_zero = np.all(total_force[:, -1] == 0)

    if is_last_column_all_zero:
        needed_force = np.array([])
    else:
        needed_force = total_force[np.isin(total_force[:, 0], list(needed_nodes))]
        needed_force[:, 0] = np.vectorize(old_to_new.get)(needed_force[:, 0].astype(int))

    boundary_nodes_array_fea = [old_to_new[node] for node in boundary_nodes_array if node in old_to_new]

    return g_coord, g_num, needed_force, boundary_nodes_array_fea

def get_merged_diff_ga(G_new, adjacency_matrix_flow_):

    edges_from_matrix = set()
    size = adjacency_matrix_flow_.shape[0]
    for i in range(size):
        for j in range(i+1, size):
            if adjacency_matrix_flow_[i][j] != 0:
                edges_from_matrix.add((i, j))

    edges_from_G_new = set()
    for edge in G_new.edges():
        edges_from_G_new.add(tuple(sorted(edge)))

    diff_edges = edges_from_G_new - edges_from_matrix

    return list(diff_edges)

def calculate_global_force_fea(coordinates, lines, boundary_nodes_array, A=0.0001, rho=1250.0, g=9.8):

    F_global = np.zeros((len(coordinates), 4))

    F_global[:, 0] = np.arange(0, len(coordinates))

    for line in lines:
        point1 = np.array(coordinates[line[0]])
        point2 = np.array(coordinates[line[1]])

        L = np.linalg.norm(point2 - point1) * 0.001

        F_self = rho * A * L * g

        F_global[line[0], 3] -= F_self / 2
        F_global[line[1], 3] -= F_self / 2

    mask = np.ones(len(coordinates), dtype=bool)
    if len(boundary_nodes_array) != 0:
        mask[np.array(boundary_nodes_array)] = False

    F_global = F_global[mask]

    return F_global


def extract_nodes_and_edges(node_dict, lines, total_force):

    needed_nodes = set(lines.flatten())

    old_to_new = {old: new for new, old in enumerate(sorted(needed_nodes))}
    g_coord = np.array([node_dict[old] for old in sorted(needed_nodes)])

    g_num = np.vectorize(old_to_new.get)(lines)

    needed_force = total_force[np.isin(total_force[:, 0], list(needed_nodes))]
    needed_force[:, 0] = np.vectorize(old_to_new.get)(needed_force[:, 0].astype(int))

    return g_coord, g_num, needed_force

def calculate_total_displacement(lines, coordinates):
    G = 9.8
    mass_per_unit_length = 1
    elastic_modulus = 1
    forces = {key: [0, 0, 0] for key in coordinates.keys()}
    displacements = {key: [0, 0, 0] for key in coordinates.keys()}
    gravity_vector = [0, 0, -1]

    for line in lines:
        node1, node2 = line
        x1, y1, z1 = coordinates[node1]
        x2, y2, z2 = coordinates[node2]
        length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
        mass = mass_per_unit_length * length
        force_gravity = mass * G

        unit_vector = [(x2 - x1) / length, (y2 - y1) / length, (z2 - z1) / length]
        force_along_edge = force_gravity * abs(unit_vector[2])

        for i in range(3):
            forces[node1][i] += (-force_along_edge / 2) * unit_vector[i] + (force_gravity / 2) * gravity_vector[i]
            forces[node2][i] += (force_along_edge / 2) * unit_vector[i] + (force_gravity / 2) * gravity_vector[i]

    for node, force in forces.items():
        for i in range(3):
            displacements[node][i] = force[i] / elastic_modulus
            if coordinates[node][2] <= 0.1 and i == 2:
                displacements[node][i] = max(0, displacements[node][i])

    total_displacement = 0
    for displacement in displacements.values():
        total_displacement += math.sqrt(sum([i**2 for i in displacement]))

    return total_displacement

def compute_centroid(node_list, coordinates):
    sum_x = sum_y = sum_z = 0
    for node in node_list:
        pos = coordinates[node]
        sum_x += pos[0]
        sum_y += pos[1]
        sum_z += pos[2]
    n = len(node_list)
    return np.array([sum_x / n, sum_y / n, sum_z / n])

def load_data(filename, dataset_path):
    with open(os.path.join(dataset_path, filename + ".pkl"), 'rb') as f:
        new_adjacency_matrix, new_seq = pickle.load(f)
    return new_adjacency_matrix, new_seq

def get_max_index(dataset_path):
    max_index = -1
    for filename in os.listdir(dataset_path):
        if filename.endswith(".pkl"):
            index = int(filename.split('.')[0])
            max_index = max(max_index, index)
    return max_index

def save_data(new_adjacency_matrix, new_seq, dataset_path, env_name, index, i):
    similar_seq = judge_similarity(new_adjacency_matrix, dataset_path)
    if similar_seq:
        return

    new_index = get_max_index(dataset_path) + 1
    with open(os.path.join(dataset_path, str(new_index) + ".pkl"), 'wb') as f:
        pickle.dump((new_adjacency_matrix, new_seq), f)


def judge_similarity(input_adjacency_matrix, dataset_path):
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    for filename in os.listdir(dataset_path):
        if filename.endswith(".pkl"):
            with open(os.path.join(dataset_path, filename), 'rb') as f:
                new_adjacency_matrix, new_seq = pickle.load(f)

                if new_adjacency_matrix.shape == input_adjacency_matrix.shape \
                        and np.allclose(input_adjacency_matrix, new_adjacency_matrix, rtol=0.02):

                    return new_seq
    return []

def get_new_boundary_nodes(boundary_nodes_array, node_mapping):

    boundary_nodes_array_new = []

    for node_new, node in node_mapping.items():

        if node in boundary_nodes_array:
            boundary_nodes_array_new.append(node_new)

    return boundary_nodes_array_new

def distance_coord(coord1, coord2):
    return math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)

def update_matrix(matrix, idx1, idx2):
    if matrix[idx1][idx2] < 0 or matrix[idx2][idx1] < 0:
        matrix[idx1][idx2] = -matrix[idx1][idx2]
        matrix[idx2][idx1] = -matrix[idx2][idx1]
    else:
        matrix[idx1][idx2] = 0
        matrix[idx2][idx1] = 0
    return matrix

def update_new_adjacency_matrix(new_adj_matrix, init_adj_matrix, new_state, state, idx1, idx2):
    if init_adj_matrix[state[idx1]][state[idx2]] < 0 and new_adj_matrix[new_state[idx1]][new_state[idx2]] == 0:
        new_adj_matrix[new_state[idx1]][new_state[idx2]] = -init_adj_matrix[state[idx1]][state[idx2]]
        new_adj_matrix[new_state[idx2]][new_state[idx1]] = -init_adj_matrix[state[idx2]][state[idx1]]
    else:
        new_adj_matrix[new_state[idx1]][new_state[idx2]] = init_adj_matrix[state[idx1]][state[idx2]]
        new_adj_matrix[new_state[idx2]][new_state[idx1]] = init_adj_matrix[state[idx2]][state[idx1]]
    return new_adj_matrix


def calculate_angle(node1, node2, node3):
    node1 = np.array(node1)
    node2 = np.array(node2)
    node3 = np.array(node3)
    vec1 = node2 - node1
    vec2 = node3 - node2
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        cos_angle = 1
    else:
        cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    if cos_angle > 1:
        angle_ = 0.0
    elif cos_angle < -1:
        angle_ = 3.14159
    else:
        angle_ = np.arccos(cos_angle)

    if np.isnan(angle_):
        angle_ = 3.14159
    return np.degrees(angle_)

def calculate_angle_vector(vector1, vector2):
    unit_vector1 = vector1 / np.linalg.norm(vector1)
    unit_vector2 = vector2 / np.linalg.norm(vector2)
    dot_product = np.dot(unit_vector1, unit_vector2)

    dot_product = np.clip(dot_product, -1.0, 1.0)
    angle = np.arccos(dot_product)

    if np.isnan(angle):
        angle = np.pi
    return angle


def calculate_angle_along_stress(node2, node3, vec1):
    node2 = np.array(node2)
    node3 = np.array(node3)

    vec2 = node3 - node2
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        cos_angle = 1
    else:
        cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    if cos_angle > 1:
        angle_ = 0.0
    elif cos_angle < -1:
        angle_ = 3.14159
    else:
        angle_ = np.arccos(cos_angle)

    if np.isnan(angle_):
        angle_ = 3.14159
    angle_ = np.degrees(angle_)
    if angle_ > 90:
        angle_ = 180 - angle_
    return angle_

def draw_graph(env_name, G, node_dict, next_state_, adjacency_matrix_flow_, i=100000, index=100000, output=True, draw=True, subgraph=False, mode='Euler', show='PLA3D', block=0):
    node_colors = []
    if mode == 'Euler':

        sorted_keys = sorted(node_dict.keys())
        node_colors = ["red" if n == next_state_[-1] else "lightblue" for n in sorted_keys]

    if mode == 'Tsp':

        heat_values = [G.nodes[node]['heat'] for node in G.nodes()]
        if np.max(heat_values) == np.min(heat_values):

            norm_heat_values = np.zeros(len(heat_values))
        else:

            norm_heat_values = (heat_values - np.min(heat_values)) / (np.max(heat_values) - np.min(heat_values))

        color_mapping = {0: 'lightblue', 1: 'cyan', 2: 'lightgreen', 3: 'yellow', 4: 'orange', 5: 'red'}

        node_colors = [color_mapping[int(value * 5)] for value in norm_heat_values]

    edge_colors = {}

    if show == 'PLA3D':

        for u, v in G.edges():

            if adjacency_matrix_flow_[u, v] == 0:
                edge_colors[(u, v)] = "black"
            elif adjacency_matrix_flow_[u, v] > 0:
                edge_colors[(u, v)] = "none"
            else:
                edge_colors[(u, v)] = "none"

        fig = plt.figure(figsize=(10, 10), dpi=150)
        ax = fig.add_subplot(111, projection='3d')

        x_coords = [coordinates[0] for coordinates in node_dict.values()]
        y_coords = [coordinates[1] for coordinates in node_dict.values()]
        z_coords = [coordinates[2] for coordinates in node_dict.values()]

        colors = [node_colors[node] for node in node_dict.keys()]

        if subgraph:
            for node, coordinates in node_dict.items():
                ax.text(coordinates[0], coordinates[1], coordinates[2], str(node), fontsize=8)

        for edge in G.edges():
            xs, ys, zs = zip(*[(node_dict[node][0], node_dict[node][1], node_dict[node][2]) for node in edge])
            ax.plot(xs, ys, zs, color=edge_colors.get(edge, 'gray'), lw=1)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')

        ax.set_box_aspect([1, 1, 1])

        ax.axis('off')

    else:

        for u, v in G.edges():

            if adjacency_matrix_flow_[u, v] == 0:
                edge_colors[(u, v)] = "black"
            elif adjacency_matrix_flow_[u, v] > 0:
                edge_colors[(u, v)] = "darkgrey"  # "darkblue"  'darkgrey'
                if mode == 'Tsp':
                    edge_colors[(u, v)] = "black"
            else:
                edge_colors[(u, v)] = "none"  # "lightblue" "gray"  white "none"

        if mode == 'Tsp':
            temp_dpi = 500
        else:
            temp_dpi = 200
        fig, ax = plt.subplots(dpi=temp_dpi)

        x_coords = [coordinates[0] for coordinates in node_dict.values()]
        y_coords = [coordinates[1] for coordinates in node_dict.values()]

        colors = [node_colors[node] for node in node_dict.keys()]

        if mode == 'Tsp':

            ax.scatter(x_coords, y_coords, s=0.01, c='none')  # 0.01  12
        else:
            ax.scatter(x_coords, y_coords, s=2, c=colors)    # 32  46

        if subgraph:
            if mode == 'Tsp':

                for node, coordinates in node_dict.items():
                    ax.text(coordinates[0], coordinates[1], str(node), fontsize=4)
            else:

                for node, coordinates in node_dict.items():
                    ax.text(coordinates[0], coordinates[1], str(node), fontsize=4)

        lines = []
        colors = []
        for edge in G.edges():
            xs, ys = zip(*[(node_dict[node][0], node_dict[node][1]) for node in edge])
            lines.append(list(zip(xs, ys)))
            colors.append(edge_colors.get(edge, 'lightgray'))  # gray

        if mode == 'Tsp':
            lc = LineCollection(lines, colors=colors, linewidths=0.5)  # 0.2
        else:
            lc = LineCollection(lines, colors=colors, linewidths=1.4)  # 2.4  3.8

        ax.add_collection(lc)

        ax.axis('off')
        ax.set_aspect('equal')

    if output:

        if mode == 'Tsp':
            output_file = f"./figure/{env_name}_{block}_{index}_{i}.png"
        else:
            output_file = f"./figure/{env_name}_{index}_{i}.png"

        plt.savefig(output_file)

    if draw:
        plt.show(block=False)
    plt.close()


def draw_multi_layers_graph(env_name, G, node_dict, next_state_, adjacency_matrix_flow_, layers, layer_height=1, i=100000, index=100000, output=True, draw=True, mode='Tsp'):
    node_colors = []
    if mode == 'Euler':
        node_colors = ["red" if n == next_state_[-1] else "lightblue" for n in G.nodes()]

    if mode == 'Tsp':

        heat_values = [G.nodes[node]['heat'] for node in G.nodes()]
        if np.max(heat_values) == np.min(heat_values):

            norm_heat_values = np.zeros(len(heat_values))
        else:

            norm_heat_values = (heat_values - np.min(heat_values)) / (np.max(heat_values) - np.min(heat_values))

        color_mapping = {0: 'blue', 1: 'cyan', 2: 'lightgreen', 3: 'yellow', 4: 'orange', 5: 'red'}

        node_colors = [color_mapping[int(value * 5)] for value in norm_heat_values]


    edge_colors = {}

    for u, v in G.edges():

        if adjacency_matrix_flow_[u, v] == 0:
            edge_colors[(u, v)] = "red"
        elif adjacency_matrix_flow_[u, v] > 0:
            edge_colors[(u, v)] = "darkblue"
        else:
            edge_colors[(u, v)] = "lightblue"

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for layer in range(layers):
        z_offset = layer * layer_height

        x_coords = [coordinates[0] for coordinates in node_dict.values()]
        y_coords = [coordinates[1] for coordinates in node_dict.values()]
        z_coords = [coordinates[2] + z_offset for coordinates in node_dict.values()]

        colors = [node_colors[node] for node in node_dict.keys()]

        ax.scatter(x_coords, y_coords, z_coords, s=30, c=colors, depthshade=False)

        for edge in G.edges():
            xs, ys, zs = zip(*[(node_dict[node][0], node_dict[node][1], node_dict[node][2] + z_offset) for node in edge])
            ax.plot(xs, ys, zs, color=edge_colors.get(edge, 'gray'), lw=3)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if output:

        output_file = f"./figure/{env_name}_{index}_{i}.png"

        plt.savefig(output_file)

    if draw:
        plt.show(block=False)

def find_k_hop_neighbors_for_list(G, beam_seq, k):
    neighbors = set()
    for node in beam_seq:

        k_hop_subgraph = nx.ego_graph(G, node, radius=k)

        neighbors.update(k_hop_subgraph.nodes)

    neighbors_array = np.array(list(neighbors))
    return neighbors_array

def find_k_hop_neighbors(G, start_node, k):
    neighbors = {start_node}
    for _ in range(k):
        next_neighbors = set()
        for node in neighbors:
            next_neighbors.update(G.neighbors(node))
        neighbors.update(next_neighbors)
    return neighbors

def create_new_graph(G_orig, coordinates_, edges_, adjacency_matrix_, state_, init_adjacency_matrix_, radius_, state_dim, heat_radius, mode='Tsp'):

    k_hop_neighbors = find_k_hop_neighbors(G_orig, state_[-1], radius_)

    subgraph = G_orig.subgraph(k_hop_neighbors)

    if mode == 'Euler':

        node1 = coordinates_[state_[-2]]
        node2 = coordinates_[state_[-1]]

        if node1 == node2:
            orig_vector = np.array([1, 0, 0])
        else:
            orig_vector = np.array(node2) - np.array(node1)

        angles = []
        for node in subgraph.nodes:
            if node == state_[-1]:
                continue
            node_coord = coordinates_[node]
            new_vector = np.array(node_coord) - np.array(node2)
            angle = calculate_angle_vector(orig_vector, new_vector)
            vector_magnitude = np.linalg.norm(new_vector)
            angles.append((node, angle, vector_magnitude))

        sorted_nodes = sorted(angles, key=lambda x: (x[1], x[2]))

        node_mapping_ = {0: state_[-1]}

        for index, (orig_node, angle, magnitude) in enumerate(sorted_nodes):
            node_mapping_[index + 1] = orig_node

        G_new_ = nx.relabel_nodes(subgraph, {v: k for k, v in node_mapping_.items()}, copy=True)

    else:

        node_mapping_ = {index: orig_index for index, orig_index in enumerate(subgraph.nodes)}
        G_new_ = nx.relabel_nodes(subgraph, {v: k for k, v in node_mapping_.items()}, copy=True)

    for new_index, orig_index in node_mapping_.items():
        G_new_.nodes[new_index]['pos'] = coordinates_[orig_index]
        G_new_.nodes[new_index]['heat'] = G_orig.nodes[orig_index]['heat']
        G_new_.nodes[new_index]['p_heat'] = G_orig.nodes[orig_index]['p_heat']
        G_new_.nodes[new_index]['p_p_heat'] = G_orig.nodes[orig_index]['p_p_heat']

    node_dict_ = nx.get_node_attributes(G_new_, 'pos')

    num_nodes = len(node_dict_)
    coords_array = np.zeros((3, num_nodes))

    for index, coord in node_dict_.items():
        coords_array[:, index] = coord

    new_state_ = np.zeros_like(state_)
    for ii, old_index in enumerate(state_):
        new_index_ = next((k for k, v in node_mapping_.items() if v == old_index), None)
        if new_index_ is not None:
            new_state_[ii] = new_index_

    heat_array = np.zeros((1, num_nodes))
    p_heat_array = np.zeros((1, num_nodes))
    p_p_heat_array = np.zeros((1, num_nodes))

    for idx, (node, attrs) in enumerate(G_new_.nodes(data=True)):

        heat_array[0, idx] = attrs['heat']
        p_heat_array[0, idx] = attrs['p_heat']
        p_p_heat_array[0, idx] = attrs['p_p_heat']

    coords_array_repeated = np.vstack((coords_array, heat_array))
    if mode == 'Tsp':

        pre_coords_array = np.vstack((coords_array, p_heat_array))
        pre_pre_coords_array = np.vstack((coords_array, p_p_heat_array))
        coords_array = np.stack((pre_pre_coords_array, pre_coords_array, coords_array_repeated), axis=0)
    else:
        coords_array = np.repeat(coords_array_repeated[np.newaxis, :, :], state_dim, axis=0)

    new_adjacency_matrix_ = np.zeros((len(node_mapping_), len(node_mapping_)), dtype=float)

    for new_index_a, old_index_a in node_mapping_.items():
        for new_index_b, old_index_b in node_mapping_.items():
            new_adjacency_matrix_[new_index_a, new_index_b] = adjacency_matrix_[old_index_a, old_index_b]

    for i in range(-2, -state_dim - 1, -1):

        new_adjacency_matrix_ = update_new_adjacency_matrix(new_adjacency_matrix_, init_adjacency_matrix_, new_state_, state_, i, i + 1)
    new_state_adjacency_ = [new_adjacency_matrix_]
    for i in range(-state_dim, -1, 1):
        new_adjacency_matrix_ = update_matrix(new_adjacency_matrix_, new_state_[i], new_state_[i + 1])
        new_state_adjacency_.append(new_adjacency_matrix_)


    return G_new_, new_state_, new_adjacency_matrix_, np.array(new_state_adjacency_), node_mapping_, node_dict_, coords_array


def create_current_state(lst, state_dim, init_adjacency_matrix, mode='Euler', material='CCF'):
    arr = np.zeros(state_dim, dtype=int)
    for i in range(min(3, len(lst))):
        arr[-(i + 1)] = lst[-(i + 1)]

    if mode == 'Tsp' or material == 'PLA3D':

        if len(lst) > 1 and init_adjacency_matrix[lst[-1]][lst[-2]] == 0:
            for ii in range(len(arr)):
                arr[ii] = lst[-1]
    return arr


def create_adj_matrix(adjacency_matrix, beam_seq, mode='Euler'):
    for i in range(len(beam_seq) - 1):
        if adjacency_matrix[beam_seq[i]][beam_seq[i + 1]] < 0:
            adjacency_matrix[beam_seq[i]][beam_seq[i + 1]] = -adjacency_matrix[beam_seq[i]][beam_seq[i + 1]]
            adjacency_matrix[beam_seq[i + 1]][beam_seq[i]] = -adjacency_matrix[beam_seq[i + 1]][beam_seq[i]]
        else:
            adjacency_matrix[beam_seq[i]][beam_seq[i + 1]] = 0
            adjacency_matrix[beam_seq[i + 1]][beam_seq[i]] = 0


        if mode == 'Tsp' and len(beam_seq) > 1:
            if adjacency_matrix[beam_seq[i]][beam_seq[i - 1]] == 0:
                adjacency_matrix[beam_seq[i]][beam_seq[i - 1]] = 1000
                adjacency_matrix[beam_seq[i - 1]][beam_seq[i]] = 1000

    return adjacency_matrix


def o2n_seq_trans(old_seq, node_mapping):
    new_seq = old_seq.copy()
    for ii, old_index in enumerate(old_seq):
        new_index_ = next((k for k, v in node_mapping.items() if v == old_index), None)
        if new_index_ is not None:
            new_seq[ii] = new_index_
    return new_seq


def n2o_seq_trans(new_seq, node_mapping):
    old_seq = new_seq.copy()
    for ii, new_index in enumerate(new_seq):
        old_seq[ii] = node_mapping[new_index]
    return old_seq


def remove_duplicates_and_keep_max(input_list):

    input_list.sort(key=lambda x: x[1], reverse=True)

    result_dict = {}
    for item in input_list:
        key = tuple(item[0])
        if key not in result_dict:
            result_dict[key] = item

    result = list(result_dict.values())
    return result


def calc_radius(adjacency_matrix, n, max_edge_length):

    arr = np.abs(adjacency_matrix)

    flattened_array = arr.flatten()

    sorted_array = np.sort(flattened_array)[::-1]

    top_n_sum = np.sum(sorted_array[:n]) + 1

    if np.count_nonzero(adjacency_matrix) < 4:
        top_n_sum = max_edge_length * 3 + 1

    return top_n_sum


def adjacency_matrix_to_graph(matrix):
    graph = nx.from_numpy_array(np.array(matrix), create_using=nx.Graph())
    return graph


def calculate_maximum_common_subgraph_similarity(matrix1, matrix2):
    graph1 = adjacency_matrix_to_graph(matrix1)
    graph2 = adjacency_matrix_to_graph(matrix2)

    gm = nx.algorithms.isomorphism.GraphMatcher(graph1, graph2)
    mcs = max(gm.subgraph_isomorphisms_iter(), key=lambda x: len(x))

    similarity = len(mcs) / min(len(graph1.nodes), len(graph2.nodes))
    return similarity


def find_most_similar_graph(target_matrix, other_matrices):
    max_similarity = -1
    most_similar_index = None

    for i, matrix in enumerate(other_matrices):
        similarity = calculate_maximum_common_subgraph_similarity(target_matrix, matrix)

        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_index = i

    return most_similar_index, max_similarity

def dfs_cycles(G, source, target, visited, depth, max_depth, cycles):
    visited[source] = True

    if depth == max_depth:
        if target in G[source]:
            cycles.append(tuple(visited.keys()))
    else:
        for neighbor in G[source]:
            if neighbor not in visited:
                dfs_cycles(G, neighbor, target, visited.copy(), depth + 1, max_depth, cycles)


def find_cycles_of_length(G, length):
    cycles = []

    for node in G.nodes:
        dfs_cycles(G, node, node, {}, 0, length - 1, cycles)

    cycles = list(set(cycles))

    return cycles

def remove_duplicate_cycles(cycles):
    unique_cycle_sets = set()
    unique_cycles = []

    for cycle in cycles:
        cycle_set = frozenset(cycle)
        if cycle_set not in unique_cycle_sets:
            unique_cycle_sets.add(cycle_set)
            unique_cycles.append(cycle)

    return unique_cycles


def remove_nested_cycles(cycles_length_3, cycles_length_4):
    nested_cycles = set()

    for cycle3_a in cycles_length_3:
        for cycle3_b in cycles_length_3:
            if cycle3_a == cycle3_b:
                continue
            for cycle4 in cycles_length_4:
                if set(cycle3_a).issubset(cycle4) and set(cycle3_b).issubset(cycle4):
                    nested_cycles.add(cycle4)
                    break

    filtered_cycles_length_4 = [cycle4 for cycle4 in cycles_length_4 if cycle4 not in nested_cycles]

    return cycles_length_3 + filtered_cycles_length_4

def find_faces(G):
    cycles_length_3 = find_cycles_of_length(G, 3)
    cycles_length_4 = find_cycles_of_length(G, 4)
    unique_cycles_length_3 = remove_duplicate_cycles(cycles_length_3)
    unique_cycles_length_4 = remove_duplicate_cycles(cycles_length_4)
    cycles = remove_nested_cycles(unique_cycles_length_3, unique_cycles_length_4)
    return cycles

def find_boundary_edges(G):

    cycles = find_faces(G)

    edge_count = {}
    for cycle in cycles:
        for i in range(len(cycle)):
            edge = frozenset((cycle[i], cycle[(i + 1) % len(cycle)]))
            if edge in edge_count:
                edge_count[edge] += 1
            else:
                edge_count[edge] = 1

    boundary_nodes = set()
    boundary_edges_list = []
    for edge, count in edge_count.items():
        if count == 1:
            boundary_nodes.update(edge)
            boundary_edges_list.append(list(edge))
    boundary_edges = np.array(boundary_edges_list)
    boundary_nodes = np.array(list(boundary_nodes))

    return boundary_edges, boundary_nodes

def find_boundary_nodes(G):

    cycles = find_faces(G)

    edge_count = {}
    for cycle in cycles:
        for i in range(len(cycle)):
            edge = frozenset((cycle[i], cycle[(i + 1) % len(cycle)]))
            if edge in edge_count:
                edge_count[edge] += 1
            else:
                edge_count[edge] = 1

    boundary_nodes = set()
    for edge, count in edge_count.items():
        if count == 1:
            boundary_nodes.update(edge)

    single_degree_nodes = {node for node in G.nodes() if G.degree(node) == 1}

    boundary_nodes.update(single_degree_nodes)


    return boundary_nodes

def find_fix_nodes(G, G_new, adjacency_matrix, new_adjacency_matrix, node_mapping):

    boundary_nodes_new = list(G_new.nodes())
    new_counts = count_zeros_in_adjacent_nodes(G_new, new_adjacency_matrix, boundary_nodes_new)
    old_counts = count_zeros_in_adjacent_nodes(G, adjacency_matrix, [node_mapping[node] for node in boundary_nodes_new])
    difference_nodes = [node for node in boundary_nodes_new if new_counts[node] != old_counts[node_mapping[node]]]
    return difference_nodes


def count_zeros_in_adjacent_nodes(G_new, matrix, rows):
    counts = {}
    for row in rows:
        adjacent_nodes = list(G_new.adj[row])
        count = 0
        for node in adjacent_nodes:

            if matrix[row, node] == 0:
                count += 1
        counts[row] = count
    return counts

def find_smallest_z_set(G):

    z_values = {}
    for node in G.nodes():

        z_values[node] = G.nodes[node]['pos'][-1]

    nodes_with_z_smaller_than_one = {node for node, z_value in z_values.items() if z_value < 1}

    return nodes_with_z_smaller_than_one

def load_checkpoint():
    checkpoint = torch.load('checkpoint/model2_0.pt')
    adj_matrix = checkpoint['adj_matrix']
    print(adj_matrix)
    return adj_matrix


def calculate_distance(pos1, pos2):
    pos1_np, pos2_np = np.array(pos1), np.array(pos2)
    return np.linalg.norm(pos1_np - pos2_np)


def create_heat_field(beam_seq, G_orig, heat_radius, mode='Tsp'):

    nodes = G_orig.nodes
    for node in nodes:
        nodes[node]['heat'] = 0
        nodes[node]['p_heat'] = 0
        nodes[node]['p_p_heat'] = 0

    if mode == 'Tsp':
        ini_temp = 0

        node_positions = [nodes[node]['pos'] for node in nodes]
        kdtree = KDTree(node_positions)

        for i, center_node in enumerate(beam_seq[-8:], start=1):

            for node in nodes:

                nodes[node]['heat'] = max(nodes[node]['heat'] * 0.52, 0)


            center_node_pos = nodes[center_node]['pos']

            indices = kdtree.query_ball_point(center_node_pos, heat_radius + 0.01)

            for index in indices:
                node = list(nodes)[index]
                distance_ = calculate_distance(center_node_pos, nodes[node]['pos'])

                heat_value = heat_radius * (1 - (distance_ / heat_radius) ** 0.8) + ini_temp

                nodes[node]['heat'] = min(nodes[node]['heat'] + heat_value, heat_radius)

            if len(beam_seq) < 3:
                for node in nodes:
                    nodes[node]['p_heat'] = 0
                    nodes[node]['p_p_heat'] = 0

            if i == len(beam_seq[-8:]) - 2:
                for node in nodes:
                    nodes[node]['p_p_heat'] = nodes[node]['heat']

            elif i == len(beam_seq[-8:]) - 1:
                for node in nodes:
                    nodes[node]['p_heat'] = nodes[node]['heat']

    return G_orig

def update_heat_field(next_state, coords_array, heat_radius, mode='Tsp'):

    next_coords_array = np.copy(coords_array)
    if mode == 'Tsp':
        new_heat_values = np.maximum(coords_array[3] * 0.52, 0)

        next_coords_array[3] = new_heat_values

        center_node_idx = next_state
        center_node_coords = next_coords_array[:3, center_node_idx]

        tree = KDTree(next_coords_array[:3].T)

        in_radius_indices = tree.query_ball_point(center_node_coords, heat_radius + 0.01)

        distances = np.sqrt(np.sum((next_coords_array[:3, in_radius_indices] - center_node_coords.reshape(3, 1)) ** 2, axis=0))

        new_heat_values = np.maximum(heat_radius * (1 - (distances / heat_radius) ** 0.8), 0)

        next_coords_array[3, in_radius_indices] += new_heat_values

        next_coords_array[3, in_radius_indices] = np.minimum(next_coords_array[3, in_radius_indices], heat_radius)

    return next_coords_array

def backward_heat_field(pre_state, coords_array, heat_radius, mode='Tsp'):

    previous_coords_array = np.copy(coords_array)

    if mode == 'Tsp':

        center_node_idx = pre_state
        center_node_coords = previous_coords_array[:3, center_node_idx]

        tree = KDTree(previous_coords_array[:3].T)

        in_radius_indices = tree.query_ball_point(center_node_coords, heat_radius + 0.01)

        distances = np.sqrt(np.sum((previous_coords_array[:3, in_radius_indices] - center_node_coords.reshape(3, 1)) ** 2, axis=0))

        subtracted_heat_values = np.maximum(heat_radius * (1 - (distances / heat_radius) ** 0.8), 0)

        previous_coords_array[3, in_radius_indices] = np.maximum(previous_coords_array[3, in_radius_indices] - subtracted_heat_values, 0)

        previous_heat_values = np.where(previous_coords_array[3] != 0, previous_coords_array[3] / 0.52, 0)
        previous_coords_array[3] = previous_heat_values

    return previous_coords_array

def extend_graph_layers(G, layers, layer_height):
    G_extended = G.copy()

    for layer in range(1, layers):
        z_offset = layer * layer_height

        for node, attr in G.nodes(data=True):
            new_node = node + layer * len(G.nodes())
            new_attr = attr.copy()
            new_attr['pos'] = (attr['pos'][0], attr['pos'][1], attr['pos'][2] + z_offset)
            G_extended.add_node(new_node, **new_attr)

        for edge in G.edges():
            new_edge = (edge[0] + layer * len(G.nodes()), edge[1] + layer * len(G.nodes()))
            G_extended.add_edge(*new_edge)

    return G_extended

def save_obj_file(G, file_name, cycles, layer=0, layer_height=0.9):

    for node in G.nodes:
        if 'pos' not in G.nodes[node]:
            raise ValueError(f"Node {node} does not have 'pos' attribute")

    with open(file_name, 'w') as f:

        for node in G.nodes:
            pos = G.nodes[node]['pos']
            f.write(f"v {pos[0]} {pos[1]} {pos[2] + layer_height * layer}\n")

        for cycle in cycles:
            f.write("f")
            for vertex in cycle:
                f.write(f" {vertex + 1}")
            f.write("\n")

def expand_obj_file(input_file, output_file, expansion=10.0):
    vertices = []
    faces = []

    with open(input_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            if parts[0] == 'v':
                vertices.append(np.array([float(parts[1]), float(parts[2]), float(parts[3])]))
            elif parts[0] == 'f':
                faces.append(list(map(int, parts[1:])))

    center = np.mean(vertices, axis=0)

    expanded_vertices = []
    for vertex in vertices:
        direction = vertex - center
        normalized_direction = direction / np.linalg.norm(direction)
        expanded_vertex = vertex + normalized_direction * expansion
        expanded_vertices.append(expanded_vertex)

    with open(output_file, 'w') as f:
        for vertex in expanded_vertices:
            f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        for face in faces:
            f.write("f" + "".join([f" {v}" for v in face]) + "\n")

def create_new_graph_with_grid_nodes_block(G, divisions=3, interval=0.1, max_nodes=4000):

    pos = nx.get_node_attributes(G, 'pos')

    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')

    for _, position in pos.items():
        x, y, _ = position
        min_x = min(min_x, x)
        max_x = max(max_x, x)
        min_y = min(min_y, y)
        max_y = max(max_y, y)

    width = max_x - min_x
    height = max_y - min_y

    grid_width = width / divisions
    grid_height = height / divisions

    G_new_list = [nx.Graph() for _ in range(divisions * divisions)]

    new_node_id = 0
    for x in np.arange(min_x, max_x + interval, interval):
        for y in np.arange(min_y, max_y + interval, interval):
            if x > max_x or y > max_y:
                continue
            grid_x = int((x - min_x) // grid_width)
            grid_y = int((y - min_y) // grid_height)

            grid_x = min(divisions - 1, grid_x)
            grid_y = min(divisions - 1, grid_y)

            grid_index = grid_y * divisions + grid_x

            G_new_list[grid_index].add_node(new_node_id, pos=(x, y, 0))
            new_node_id += 1

    G_new_list_final = []
    for G_new in G_new_list:
        num_nodes = G_new.number_of_nodes()
        if num_nodes > max_nodes:
            G_sub_list = create_new_graph_with_grid_nodes_block(G_new, divisions=2, interval=interval, max_nodes=4000)
            G_new_list_final.extend(G_sub_list)
        else:
            G_new_list_final.append(G_new)

    return G_new_list_final

def ray_intersects_segment(ray_origin, segment):
    p1, p2 = segment
    if p1[1] == p2[1]:
        return False
    if ray_origin[1] < min(p1[1], p2[1]) or ray_origin[1] >= max(p1[1], p2[1]):
        return False
    x_intersect = p1[0] + (ray_origin[1] - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1])
    return x_intersect > ray_origin[0]

def extract_nodes_inside_boundary(G, G_new, boundary_edges):
    coordinates = nx.get_node_attributes(G, 'pos')
    segments = [(coordinates[edge[0]][:2], coordinates[edge[1]][:2]) for edge in boundary_edges]

    G_new_new = nx.Graph()

    for node in G_new.nodes:
        node_pos = G_new.nodes[node]['pos'][:2]
        intersections = sum(ray_intersects_segment(node_pos, segment) for segment in segments)
        if intersections % 2 == 1:
            G_new_new.add_node(node, pos=G_new.nodes[node]['pos'])

    return G_new_new

def extract_nodes_inside_boundary_list(G, G_new_list, boundary_edges, interval=0.1, max_nodes=2000):
    G_new_new_list = []

    for G_new in G_new_list:
        G_new_new = extract_nodes_inside_boundary(G, G_new, boundary_edges)

        if len(G_new_new.nodes) > 0:

            if G_new_new.number_of_nodes() > max_nodes:
                G_sub_list = create_new_graph_with_grid_nodes_block(G_new_new, divisions=2, interval=interval)
                G_sub_list_new_new = [extract_nodes_inside_boundary(G, G_sub, boundary_edges) for G_sub in G_sub_list]
                G_new_new_list.extend(G_sub_list_new_new)
            else:
                G_new_new_list.append(G_new_new)

    return G_new_new_list


def remove_isolated_nodes(G, interval):

    nodes_to_remove = []

    for node in G.nodes():
        isolated = False
        adj_node_num = 0
        for neighbor in G.nodes():
            if neighbor == node:
                continue
            if np.linalg.norm(np.array(G.nodes[node]['pos']) - np.array(G.nodes[neighbor]['pos'])) < interval * 1.1:
                adj_node_num += 1
        if adj_node_num < 2:
            isolated = True
        if isolated:
            nodes_to_remove.append(node)

    G.remove_nodes_from(nodes_to_remove)

    node_dict = {}
    for i, node in enumerate(G.nodes()):
        node_dict[node] = i

    G = nx.relabel_nodes(G, node_dict)

    return G

def calculate_turn(node_dict, node1, node2, node3, radius):
    p1, p2, p3 = np.array(node_dict[node1]), np.array(node_dict[node2]), np.array(node_dict[node3])
    v1, v2 = p1 - p2, p3 - p2
    l1, l2 = np.linalg.norm(v1), np.linalg.norm(v2)
    r = min(radius, l1 / 2, l2 / 2)
    p4 = p2 + r * v1 / l1
    p5 = p2 + r * v2 / l2
    return p2.tolist(), p4.tolist(), p5.tolist()


def draw_path_turn_ori(env_name, node_dict, best_path, radius=2, index=10000):

    line_width = 1
    edge_count = defaultdict(int)
    for i in range(1, len(best_path)):
        edge = str(sorted([best_path[i - 1], best_path[i]]))
        edge_count[edge] += 1

    fig, ax = plt.subplots(dpi=300)
    if len(best_path) == 1:
        color = 'b'
    else:
        edge = str(sorted([best_path[0], best_path[1]]))
        color = 'b' if edge_count[edge] == 1 else 'r'

    prev_point = node_dict[best_path[0]]
    for i in range(1, len(best_path) - 1):


        edge = str(sorted([best_path[i - 1], best_path[i]]))
        color = 'b' if edge_count[edge] == 1 else 'r'

        turn_points = calculate_turn(node_dict, best_path[i - 1], best_path[i], best_path[i + 1], radius)
        x = [prev_point[0], turn_points[1][0]]
        y = [prev_point[1], turn_points[1][1]]
        ax.plot(x, y, color, linewidth=line_width)
        x = [turn_points[1][0], turn_points[2][0]]
        y = [turn_points[1][1], turn_points[2][1]]
        ax.plot(x, y, color, linewidth=line_width)
        prev_point = turn_points[2]

    x = [prev_point[0], node_dict[best_path[-1]][0]]
    y = [prev_point[1], node_dict[best_path[-1]][1]]
    edge = str(sorted([best_path[-2], best_path[-1]]))
    color = 'b' if edge_count[edge] == 1 else 'r'
    ax.plot(x, y, color, linewidth=line_width)


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    plt.title(f"{env_name}_turning_{index}, Edges Processed: {len(best_path) - 1}")

    output_file = f"./figure/{env_name}_turning_{index}.png"

    plt.savefig(output_file)

    plt.show()
    print('output turning picture')
    plt.close()

def calculate_offset(p1, p2, distance, direction):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    norm = (dy, -dx) if direction else (-dy, dx)
    norm_length = (norm[0] * norm[0] + norm[1] * norm[1]) ** 0.5
    norm = (norm[0] / norm_length * distance, norm[1] / norm_length * distance)
    return norm

def draw_arrow(ax, start_point, end_point, color='darkblue', line_width=3):

    mid_point = ((start_point[0] + end_point[0]) / 2, (start_point[1] + end_point[1]) / 2)

    direction = (end_point[0] - start_point[0], end_point[1] - start_point[1])

    length = (direction[0]*direction[0] + direction[1]*direction[1])**0.5
    direction = (direction[0] / length, direction[1] / length)

    arrow_head_length = 0.5
    arrow_head = (direction[0] * arrow_head_length, direction[1] * arrow_head_length)

    ax.annotate("",
                xy=(mid_point[0] + arrow_head[0], mid_point[1] + arrow_head[1]),
                xytext=mid_point,
                arrowprops=dict(arrowstyle="->", color=color, lw=line_width))

def draw_path_turn(env_name, node_dict, best_path, radius=2, index=10000, offset_distance=0.8):
    line_width = 2
    edge_count = defaultdict(int)

    norm = Normalize(vmin=0, vmax=len(best_path))

    for i in range(1, len(best_path)):
        edge = tuple(sorted((best_path[i - 1], best_path[i])))
        edge_count[edge] += 1

    edge_processed = {}
    fig, ax = plt.subplots(dpi=400)

    for i in range(1, len(best_path)):
        start_node = best_path[i - 1]
        end_node = best_path[i]
        start_point = node_dict[start_node][:2]
        end_point = node_dict[end_node][:2]
        edge = tuple(sorted((start_node, end_node)))

        color = cm.viridis(norm(i))

        if edge_count[edge] == 1:

            ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], color=color, linewidth=line_width)
            draw_arrow(ax, start_point, end_point, color=color, line_width=line_width)
        else:

            if edge not in edge_processed:
                offset = calculate_offset(start_point, end_point, offset_distance, True)
                edge_processed[edge] = offset
            else:
                offset = edge_processed[edge]
                offset = (-offset[0], -offset[1])

            start_point_offset = [start_point[0] + offset[0], start_point[1] + offset[1]]
            end_point_offset = [end_point[0] + offset[0], end_point[1] + offset[1]]
            ax.plot([start_point_offset[0], end_point_offset[0]], [start_point_offset[1], end_point_offset[1]],
                    color=color, linewidth=line_width)
            draw_arrow(ax, start_point_offset, end_point_offset, color=color, line_width=line_width)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    plt.title(f"{env_name}_turning_{index}, Edges Processed: {len(best_path) - 1}")

    output_file = f"./figure/{env_name}_turning_{index}.png"
    plt.savefig(output_file)
    plt.show()
    print('output turning picture')
    plt.close()

def relabel_nodes_randomly(G):

    original_nodes = list(G.nodes())

    new_labels = original_nodes.copy()

    random.shuffle(new_labels)

    mapping = {old: new for old, new in zip(original_nodes, new_labels)}

    G_new = nx.relabel_nodes(G, mapping)
    return G_new

def write_node_positions_to_file(G, filename):
    with open(filename, 'w') as f:
        for node in sorted(G.nodes(data=True), key=lambda x: x[0]):
            pos = node[1]['pos']
            f.write(f'{pos[0]}\t{pos[1]}\t{pos[2]}\n')

def write_edge_info_to_file(G, filename):
    with open(filename, 'w') as f:
        for edge in G.edges():

            f.write(f'{edge[0]}\t{edge[1]}\n')
        for edge in G.edges():

            f.write(f'{edge[0]}\t{edge[1]}\n')


def traverse_matrix_euler(matrix, row_index):
    if not any(matrix[row_index, :]):
        return
    columns = np.where(matrix[row_index, :] != 0)[0]
    matrix[row_index, :] = 0
    matrix[:, row_index] = 0

    for col_index in columns:
        traverse_matrix_euler(matrix, col_index)

def traverse_matrix_euler_nonrecursive(matrix, row_index):
    stack = [row_index]
    while stack:
        row_index = stack.pop()
        columns = np.where(matrix[row_index, :] != 0)[0]
        matrix[row_index, :] = 0
        matrix[:, row_index] = 0
        for col_index in columns:
            if np.any(matrix[:, col_index]) or np.any(matrix[col_index, :]):
                stack.append(col_index)

    non_zero_count_next = np.count_nonzero(matrix)
    if non_zero_count_next > 0:
        connectivity = False
    else:
        connectivity = True
    return connectivity

def recursive_search(G, node, node_dict):
    neighbors = [n for n in G.neighbors(node) if node_dict[n] == 0]
    for neighbor in neighbors:
        node_dict[neighbor] = 1
        recursive_search(G, neighbor, node_dict)

def connectivity_tsp(adjacency_matrix_, row_index, state_, G, G_new, len_beam_seq):
    if len_beam_seq >= G.number_of_nodes()-2:
        connectivity = True
        return connectivity

    adjacency_matrix_next = adjacency_matrix_.copy()
    if adjacency_matrix_next[state_[-1]][row_index] < 0 or adjacency_matrix_next[row_index][state_[-1]] < 0:
        adjacency_matrix_next[state_[-1]][row_index] = -adjacency_matrix_next[state_[-1]][row_index]
        adjacency_matrix_next[row_index][state_[-1]] = -adjacency_matrix_next[row_index][state_[-1]]
    else:
        adjacency_matrix_next[state_[-1]][row_index] = 0.0
        adjacency_matrix_next[row_index][state_[-1]] = 0.0

    optional_action_next = np.where(np.array(adjacency_matrix_[row_index]) != 0)[0]
    row_col_array = np.concatenate(np.argwhere(adjacency_matrix_next > 0).T)

    optional_action_next = optional_action_next[~np.isin(optional_action_next, row_col_array)]

    node_dict = {node: 0 for node in G_new.nodes}

    for node in row_col_array:
        node_dict[node] = 1
    node_dict[row_index] = 1

    connectivity = False

    for start_node in optional_action_next:
        new_node_dict = node_dict.copy()
        new_node_dict[start_node] = 1
        recursive_search(G_new, start_node, new_node_dict)

        all_labels_are_one = all(value == 1 for value in new_node_dict.values())
        if all_labels_are_one:
            connectivity = True
    return connectivity


def angle_between(v1, v2):

    dot_product = np.dot(v1, v2)
    norms = np.linalg.norm(v1) * np.linalg.norm(v2)
    cos_theta = dot_product / norms

    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    angle = np.arccos(cos_theta)
    return np.degrees(angle)


def is_within_sector(v_base, v1, v2):

    angle_v1 = angle_between(v_base, v1)
    angle_v2 = angle_between(v_base, v2)

    if angle_v2 >= angle_v1 - 0.01:
        return False

    cross_v1 = np.cross(v_base, v1)
    cross_v2 = np.cross(v_base, v2)

    if angle_between(cross_v1, cross_v2) < 89:
        return True
    else:
        return False


def anti_self_locking_subgraph(G, G_new, adjacency_matrix_, node_mapping_, state_, beam_seq, new_adjacency_matrix_,
                               new_state_, coordinates, heat_radius, init_adjacency_matrix_, rays, train_mode, mode='Tsp', material='CCF'):

    optional_action = np.array([])
    lifting = False
    lines = get_merged_diff(init_adjacency_matrix_, adjacency_matrix_)

    if mode == 'Tsp':
        optional_action = np.where(np.array(adjacency_matrix_[state_[-1]]) != 0)[0]

        row_col_array = np.concatenate(np.argwhere(adjacency_matrix_ > 0).T)

        optional_action = np.setdiff1d(optional_action, row_col_array)

        if optional_action.size == 0:
            lifting = True

            candidate_node = np.array(list(G.nodes()))

            unique_nodes = np.setdiff1d(candidate_node, row_col_array)

            distances = [nx.shortest_path_length(G, state_[-1], node) for node in unique_nodes]

            sorted_nodes = [node for _, node in sorted(zip(distances, unique_nodes), reverse=True)]

            for index in sorted_nodes:
                connectivity = True
                optional_action_jump = np.where(np.array(adjacency_matrix_[index]) != 0)[0]

                optional_action_jump = np.setdiff1d(optional_action_jump, row_col_array)

                if len(optional_action_jump) == 2:
                    angle = calculate_angle(coordinates[optional_action_jump[0]], coordinates[index], coordinates[optional_action_jump[1]])
                    if angle < 1:
                        connectivity = False

                if connectivity:
                    optional_action = np.append(optional_action, index)
                    break

        new_optional_action_ = np.zeros_like(optional_action)
        if lifting:
            new_optional_action_ = optional_action
        else:

            for ii, old_index in enumerate(optional_action):
                new_index_ = next((k for k, v in node_mapping_.items() if v == old_index), None)
                if new_index_ is not None:
                    new_optional_action_[ii] = new_index_

        optional_action = new_optional_action_

    if mode == 'Euler':
        optional_action = np.where(np.array(adjacency_matrix_[state_[-1]]) != 0)[0]  # 找到所有可选的动作

        if material == 'CCF':

            for row_index in optional_action:
                adjacency_matrix_next = adjacency_matrix_.copy()

                if adjacency_matrix_next[state_[-1]][row_index] < 0 or adjacency_matrix_next[row_index][state_[-1]] < 0:
                    adjacency_matrix_next[state_[-1]][row_index] = -adjacency_matrix_next[state_[-1]][row_index]
                    adjacency_matrix_next[row_index][state_[-1]] = -adjacency_matrix_next[row_index][state_[-1]]
                else:
                    adjacency_matrix_next[state_[-1]][row_index] = 0.0
                    adjacency_matrix_next[row_index][state_[-1]] = 0.0


                connectivity = True
                # connectivity = traverse_matrix_euler_nonrecursive(adjacency_matrix_next, row_index)

                if not connectivity:
                    optional_action = optional_action[optional_action != row_index]

        if material == 'PLA3D':

            collision_check = False

            optional_action_temp = optional_action.copy()
            for action in optional_action:
                temp_lines = lines.copy()

                if coordinates[state_[-1]][2] > coordinates[action][2]:
                    temp_lines.append((state_[-1], action))

                if collision_check:
                    collision, _, _ = collision_check_simulation(coordinates, state_[-1], action, rays, temp_lines, norm_output=False)
                else:
                    collision = False

                if collision:
                    optional_action_temp = optional_action_temp[optional_action_temp != action]

            optional_action = optional_action_temp

            if optional_action.size == 0:
                lifting = True

                row_col_array = np.concatenate(np.argwhere(adjacency_matrix_ > 0).T)

                unique_nodes = np.unique(row_col_array)

                distances = [nx.shortest_path_length(G, state_[-1], node) for node in unique_nodes]

                sorted_nodes = [node for _, node in sorted(zip(distances, unique_nodes))]
                # sorted_nodes = [node for _, node in sorted(zip(distances, unique_nodes), reverse=True)]

                for row_index in sorted_nodes:
                    connectivity_euler = False
                    append_node = False
                    optional_action_jump = np.where(np.array(adjacency_matrix_[row_index]) != 0)[0]
                    for index in optional_action_jump:

                        temp_lines = lines.copy()
                        if coordinates[row_index][2] > coordinates[index][2]:
                            temp_lines.append((row_index, index))

                        if collision_check:
                            collision, _, _ = collision_check_simulation(coordinates, row_index, index, rays, temp_lines, norm_output=False)
                        else:
                            collision = False

                        if row_index in beam_seq and not collision:
                            append_node = True
                            break

                    if append_node:
                        optional_action = np.append(optional_action, row_index)
                        break


        new_optional_action_ = np.zeros_like(optional_action)
        if lifting:
            new_optional_action_ = optional_action
        else:

            for ii, old_index in enumerate(optional_action):
                new_index_ = next((k for k, v in node_mapping_.items() if v == old_index), None)
                if new_index_ is not None:
                    new_optional_action_[ii] = new_index_

        optional_action = new_optional_action_

    return optional_action, lifting, lines


def aabb_tree(nodes, lines, coordinates):

    x_min = min(nodes[0][0], nodes[-1][0]) - 180
    x_max = max(nodes[0][0], nodes[-1][0]) + 180

    y_min = min(nodes[0][1], nodes[-1][1]) - 180
    y_max = max(nodes[0][1], nodes[-1][1]) + 180

    z_min = min(nodes[0][2], nodes[-1][2]) - 30
    z_max = max(nodes[0][2], nodes[-1][2]) + 180

    filtered_lines = []

    for line in lines:
        x1, y1, z1 = coordinates[line[0]]
        x2, y2, z2 = coordinates[line[1]]

        if (x_min <= x1 <= x_max and y_min <= y1 <= y_max and z_min <= z1 <= z_max and
                x_min <= x2 <= x_max and y_min <= y2 <= y_max and z_min <= z2 <= z_max):
            filtered_lines.append(line)

    return filtered_lines

def collision_check_simulation(coordinates, current, action, rays, lines, interval=5, norm_output=True, specified_vector=np.array([0., 0., 1.])):

    current_coords = np.array(coordinates[current])
    action_coords = np.array(coordinates[action])

    distance = np.linalg.norm(current_coords - action_coords)
    nodes = []
    if distance <= interval:

        nodes = [current_coords, action_coords]
    else:

        num_of_new_nodes = int(distance / interval)

        new_nodes = [current_coords + i * interval * (action_coords - current_coords) / distance for i in range(1, num_of_new_nodes)]

        nodes = [current_coords] + new_nodes + [action_coords]

    lines = aabb_tree(nodes, lines, coordinates)

    radius = 32
    height_cylinder = 150
    height_cone = 41

    node_rays = [[] for _ in nodes]
    node_index = 0

    for node in nodes:

        for ray_dir in rays:

            cone_center = node - height_cone * ray_dir
            cylinder_center = node - (height_cylinder + height_cone) * ray_dir
            collision = False

            for line in lines:
                line_nodes = sample_line_nodes(line, coordinates, step=5)

                for P in line_nodes:
                    if if_in_cone(P, cone_center, ray_dir, radius, height_cone):
                        collision = True
                        break
                    if if_in_cylinder(P, cylinder_center, ray_dir, radius, height_cylinder):
                        collision = True
                        break

                if collision:

                    break

            if not collision:

                node_rays[node_index].append(ray_dir)

        node_index = node_index + 1

    all_rays = [ray for rays in node_rays for ray in rays]
    duplicate_rays = []
    threshold = 1e-6

    for i, ray_i in enumerate(all_rays):

        if any(np.linalg.norm(dup - ray_i) < threshold for dup in duplicate_rays):
            continue

        count = sum(1 for ray_j in all_rays if np.linalg.norm(ray_i - ray_j) < threshold)

        if count == len(node_rays):
            duplicate_rays.append(ray_i)

    final_collision = True
    normal = None
    min_angle = 0
    if len(duplicate_rays) != 0:

        final_collision = False
        if norm_output:

            neg_duplicate_rays = [-1 * ray for ray in duplicate_rays]

            angles = [np.arccos(np.clip(np.dot(ray, specified_vector) / (np.linalg.norm(ray) * np.linalg.norm(specified_vector)), -1, 1)) for ray in neg_duplicate_rays]

            min_angle_index = np.argmin(angles)

            normal = neg_duplicate_rays[min_angle_index]

            normal = normal.tolist()

            min_angle = angles[min_angle_index]

    return final_collision, normal, min_angle

def generate_rays(degrees):
    rays = []

    init_vector = np.array([1, 0, 0])

    for elev in range(-35, -65, -13):

        rot_mat_y = np.array([
            [math.cos(math.radians(elev)), 0, -math.sin(math.radians(elev))],
            [0, 1, 0],
            [math.sin(math.radians(elev)), 0, math.cos(math.radians(elev))]
        ])

        curr_vector = np.matmul(rot_mat_y, init_vector)

        for azim in range(0, 360, 30):

            rot_mat_z = np.array([
                [math.cos(math.radians(azim)), -math.sin(math.radians(azim)), 0],
                [math.sin(math.radians(azim)), math.cos(math.radians(azim)), 0],
                [0, 0, 1]
            ])

            ray_dir = np.matmul(rot_mat_z, curr_vector)

            rays.append(ray_dir)

    rays.append(np.array([0, 0, -1]))

    return rays


def normalize(v):

    return v / np.linalg.norm(v)


def slerp(v1, v2, t):

    v1 = np.array(v1)
    v2 = np.array(v2)
    dot = np.dot(v1, v2)
    # 避免出现除以0的情况
    dot = np.clip(dot, -1.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)

    s0 = np.sin((1.0 - t) * theta) / sin_theta
    s1 = np.sin(t * theta) / sin_theta

    return (s0 * v1) + (s1 * v2)

def sample_line_nodes(line, coordinates, step=2):
    start, end = line
    start_coords = np.array(coordinates[start])
    end_coords = np.array(coordinates[end])
    length = np.linalg.norm(end_coords - start_coords)
    nodes_count = int(length / step)
    new_nodes_coords = [start_coords + (end_coords - start_coords) * i / nodes_count for i in range(1, nodes_count)]
    nodes = [start_coords] + new_nodes_coords + [end_coords]
    return nodes


def get_merged_diff(init_adjacency_matrix_, adjacency_matrix_):

    diff = np.where(init_adjacency_matrix_ != adjacency_matrix_)

    lines = list(zip(diff[0], diff[1]))

    merged_lines = set()
    for line in lines:
        line = tuple(sorted(line))
        merged_lines.add(line)

    return list(merged_lines)


def collision_checking(coordinates, adjacency_matrix_, vector_z, current, action, optional_action, beam_seq, rays):
    collision = False

    optional_action_lo = optional_action[optional_action != action]
    for other_action in optional_action_lo:

        vector1 = np.array(coordinates[action]) - np.array(coordinates[current])
        vector2 = np.array(coordinates[other_action]) - np.array(coordinates[current])
        if is_within_sector(vector_z, vector1, vector2):

            collision = True


    if len(beam_seq) > 1:

        optional_action_right = np.where(np.array(adjacency_matrix_[action]) != 0)[0]
        optional_action_ro = optional_action_right[optional_action_right != current]
        for other_action in optional_action_ro:

            vector1 = np.array(coordinates[current]) - np.array(coordinates[action])
            vector2 = np.array(coordinates[other_action]) - np.array(coordinates[action])
            if is_within_sector(vector_z, vector1, vector2):

                collision = True

    return collision


def if_in_cylinder(P, cylinder_center, ray_dir, radius, height):

    ray_dir = ray_dir / np.linalg.norm(ray_dir)

    P_prime = P - cylinder_center
    distance_along_ray = np.dot(P_prime, ray_dir)

    if distance_along_ray < 0 or distance_along_ray > height - 0.01:
        return False

    P_proj = distance_along_ray * ray_dir
    distance_to_ray = np.linalg.norm(P_proj - P_prime)

    return distance_to_ray < radius - 0.01


def if_in_cone(P, node, ray_dir, radius, height):

    ray_dir = ray_dir / np.linalg.norm(ray_dir)

    P_prime = P - node
    distance_along_ray = np.dot(P_prime, ray_dir)

    if distance_along_ray < 0 or distance_along_ray > height - 0.01:
        return False

    P_proj = distance_along_ray * ray_dir
    distance_to_ray = np.linalg.norm(P_proj - P_prime)

    cone_radius_at_P = (height - distance_along_ray) * radius / height
    return distance_to_ray < cone_radius_at_P - 0.01


def load_point(save_dir='save'):
    checkpoints = [f for f in os.listdir(save_dir) if f.endswith(".pkl")]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda f: os.path.getctime(os.path.join(save_dir, f)))
        filename = os.path.join(save_dir, latest_checkpoint)
        with open(filename, 'rb') as f:
            checkpoint_data = pickle.load(f)
            return checkpoint_data
    return None


def save_point(beam_seq, index, edge_pass, save_dir='save'):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for filename in os.listdir(save_dir):
        if filename.endswith(".pkl"):
            os.remove(os.path.join(save_dir, filename))

    filename = os.path.join(save_dir, f"point_{index}.pkl")
    with open(filename, 'wb') as f:
        checkpoint_data = {'beam_seq': beam_seq, 'index': index, 'edge_pass': edge_pass}
        pickle.dump(checkpoint_data, f)


def choose_start_nodes(G, boundary_nodes_array, material):
    if material == 'PLA3D':

        selected_node = np.random.choice(boundary_nodes_array)

        neighbors = list(G.neighbors(selected_node))

        boundary_neighbors = [node for node in neighbors if node in boundary_nodes_array]

        selected_boundary_neighbor = np.random.choice(boundary_neighbors)
        start_nodes = [selected_node, selected_boundary_neighbor]
    elif material == 'CCF':

        random_node = random.choice(list(G.nodes))
        start_nodes = [random_node]

    else:
        start_nodes = [0]

    return start_nodes


def accumulate_and_save(directory, output_file):
    accumulated_model_state_dict = None
    accumulated_optimizer_state_dict = None

    filenames = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".pt")]
    num_checkpoints = len(filenames)

    for filename in filenames:
        checkpoint = torch.load(filename)
        model_state_dict = checkpoint['model_state_dict']


        if accumulated_model_state_dict is None:
            accumulated_model_state_dict = {k: torch.zeros_like(v) for k, v in model_state_dict.items()}

        for key in accumulated_model_state_dict:
            accumulated_model_state_dict[key] += model_state_dict[key] / num_checkpoints

    torch.save({
        'model_state_dict': accumulated_model_state_dict,

    }, output_file)


def plot_normalized_heatmaps(tensor, channel_indices=None):

    # Check if channel indices are provided, else use all channels
    if channel_indices is None:
        channel_indices = list(range(tensor.shape[1]))

    # Ensure tensor is on CPU and detached from the graph
    tensor = tensor.cpu().detach().numpy()

    # Plot heatmaps for specified channels
    for i in channel_indices:
        # Extract the channel data
        channel_data = tensor[0, i, :, :]

        # Normalize the channel data
        channel_min = channel_data.min()
        channel_max = channel_data.max()
        channel_data_normalized = (channel_data - channel_min) / (channel_max - channel_min)

        # Plot the heatmap
        plt.imshow(channel_data_normalized, cmap='Blues_r')
        plt.colorbar()  # Show color scale
        plt.title(f"Channel {i + 1}")
        plt.xlim(-0.5, 15.5)
        plt.ylim(19.5, -0.5)  # Inverted y-axis for image coordinate system

        # Set integer ticks for the x and y axes
        plt.xticks(np.arange(0, 16, 1))
        plt.yticks(np.arange(0, 20, 1))
        # plt.show()
        plt.savefig(f'./mnt/heatmap_{i}.png')
        plt.close()

def plot_path_2d(coordinates_path, new_coordinates_path):
    # 提取原始路径的 x 和 y 坐标
    original_x = [p[0] for p in coordinates_path]
    original_y = [p[1] for p in coordinates_path]

    new_x = [p[0] for p in new_coordinates_path]
    new_y = [p[1] for p in new_coordinates_path]

    plt.plot(original_x, original_y, 'b-', label='Original Path')

    plt.plot(new_x, new_y, 'r-', label='New Path with Corners')

    plt.scatter(new_x, new_y, s=10, c='green', label='Nodes')

    plt.legend()

    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')

    plt.grid(True)

    plt.show()


def normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def calculate_mid_vector(v1, v2):

    v1_norm = normalize_vector(v1)
    v2_norm = normalize_vector(v2)

    return normalize_vector(v1_norm + v2_norm)


def calculate_offset_point(point, mid_vector, offset_distance):

    return point + mid_vector * offset_distance


def generate_new_path_with_corners(coordinates_path, offset_distance):
    new_path = [coordinates_path[0]]

    for index in range(1, len(coordinates_path) - 1):
        p0 = np.array(coordinates_path[index - 1])
        p1 = np.array(coordinates_path[index])
        p2 = np.array(coordinates_path[index + 1])

        v1 = p1 - p0
        v2 = p2 - p1

        vm1 = p1 - p0
        vm2 = p1 - p2

        mid_vector = calculate_mid_vector(vm1, vm2)

        dot_product = np.dot(normalize_vector(v1), normalize_vector(v2))
        angle_acute = dot_product > 0

        if angle_acute:

            offset = -offset_distance
        else:

            offset = offset_distance

        extra_node_3 = calculate_offset_point(p1, mid_vector, offset_distance)

        extra_node_1 = calculate_offset_point(p1, -normalize_vector(v1), offset_distance + 3)
        extra_node_2 = calculate_offset_point(p1, normalize_vector(v2), offset_distance + 3)

        new_path.extend([extra_node_1, extra_node_3, extra_node_2])

    new_path.append(coordinates_path[-1])
    return np.array(new_path)

def generate_beam_sequence(env_name, existing_coordinates):
    parts = env_name.split('-')
    common_name = '-'.join(parts[:-1]) + '_wireframe'
    common_name_2 = parts[-1]

    best_path_file = f'data/{parts[0]}/{common_name}.txt'
    node_coordinates_file = f'data/{parts[0]}/Node{common_name}.txt'
    second_node_coordinates_file = f'data/{parts[0]}/Node{parts[0]}-{common_name_2}.txt'

    def extract_best_path(file_path):
        with open(file_path, 'r') as file:
            for line in file:
                if line.startswith("Best path:"):
                    best_path_str = line.split(":")[1].strip()
                    best_path = eval(best_path_str)
                    return best_path
        return None

    def extract_coordinates(file_path):
        coordinates = {}
        with open(file_path, 'r') as file:
            for i, line in enumerate(file):
                coords = [float(x) for x in line.split()]
                coordinates[i] = coords
        return coordinates

    def compare_coordinates(best_path, node_coordinates, existing_coordinates):
        beam_seq = []
        close_nodes = []

        for index in best_path:
            if index in node_coordinates:
                x, y, z = node_coordinates[index]
                for key, value in existing_coordinates.items():
                    if all(abs(a - b) <= 0.01 for a, b in zip(value, [x, y, z])):
                        close_nodes.append(key)

        if close_nodes:
            beam_seq.append([close_nodes, 0.0])

        return beam_seq

    def add_min_z_coordinate_node(beam_seq, second_node_coordinates, existing_coordinates):
        if beam_seq:
            min_z_node = min(second_node_coordinates, key=lambda k: second_node_coordinates[k][2])
            min_z_coords = second_node_coordinates[min_z_node]

            corresponding_node = next((key for key, value in existing_coordinates.items() if all(abs(a - b) <= 0.01 for a, b in zip(value, min_z_coords))), None)
            if corresponding_node is not None:
                beam_seq[0][0].append(corresponding_node)

    best_path = extract_best_path(best_path_file)
    node_coordinates = extract_coordinates(node_coordinates_file)
    second_node_coordinates = extract_coordinates(second_node_coordinates_file)
    beam_seq = compare_coordinates(best_path, node_coordinates, existing_coordinates)
    add_min_z_coordinate_node(beam_seq, second_node_coordinates, existing_coordinates)

    return beam_seq


def calculate_total_max_deformation(node_file, path_file):


    def read_coordinates(filename):
        coordinates = {}
        with open(filename, 'r') as file:
            for index, line in enumerate(file):
                x, y, z = map(float, line.split())
                coordinates[index] = [x, y, z]
        return coordinates

    def read_path(filename):
        path = []
        with open(filename, 'r') as file:
            for line in file:
                if line.startswith("Best path:"):
                    path = line.split(':')[1].strip().strip('[]').split(', ')
                    path = list(map(int, path))
                    break
        return [(path[i], path[i + 1]) for i in range(len(path) - 1)]

    coordinates = read_coordinates(node_file)
    path = read_path(path_file)

    boundary_nodes_array = np.array([index for index, coord in coordinates.items() if coord[2] < 1])
    lines = []
    max_deformation = 0
    lines_max_deformation = []
    index = 0
    max_index = 0
    for edge in path:
        if edge not in lines:
            lines.append(edge)
            current_lines = np.array(lines)
            current_deformation = beam_fea_graph(current_lines, coordinates, boundary_nodes_array, draw=False)

            if current_deformation > max_deformation:
                max_deformation = current_deformation
                lines_max_deformation = list(lines)
                max_index = index

        index += 1

    lines_max_deformation_array = np.array(lines_max_deformation)
    beam_fea_graph(lines_max_deformation_array, coordinates, boundary_nodes_array, draw=False)


    return max_deformation, max_index


def save_average_to_file(return_list, env_name, savept):

    average = round(sum(return_list[:100]) / 100, 3)

    file_suffix = "P" if savept else "I"
    file_name = f"checkpoint/100_average_{env_name}_{file_suffix}.txt"

    with open(file_name, "a") as file:
        file.write(f"{average}\n")


def update_progress(beam_seq, total_nodes, bar_length=50):

    unique_nodes = len(set(node for path in beam_seq for node in path[0]))

    progress = unique_nodes / total_nodes

    arrow = int(round(progress * bar_length - 1)) if unique_nodes != total_nodes else bar_length
    spaces = bar_length - arrow

    progress_bar = 'Progress: [{0}{1}] {2}%'.format('>' * arrow, ' ' * spaces, round(progress * 100, 2))

    sys.stdout.write("\r" + progress_bar)
    sys.stdout.flush()


def update_progress_post(ii, total, bar_length=50):

    progress = ii / total

    arrow = int(round(progress * bar_length - 1)) if ii < total else bar_length
    spaces = bar_length - arrow

    progress_bar = 'Progress: [{0}{1}] {2}%'.format('>' * arrow, ' ' * spaces, round(progress * 100, 2))

    sys.stdout.write("\r" + progress_bar)
    sys.stdout.flush()


def final_progress(bar_length=50):

    progress_bar = 'Progress: [{0}] {1}%'.format('>' * bar_length, 100.0)

    print("\r" + progress_bar)

def preprocess_max_edge_pass(max_edge_pass, material):
    max_edge_pass_after = max_edge_pass
    if max_edge_pass > 3:
        max_edge_pass_after = max_edge_pass - 1

    return max_edge_pass_after


def euclidean_distance(a, b):

    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def find_zigzag_path(G, env_name):
    def find_start_node():

        min_y = min(G.nodes, key=lambda n: G.nodes[n]['pos'][1])
        return max(filter(lambda n: G.nodes[n]['pos'][1] == G.nodes[min_y]['pos'][1], G.nodes),
                   key=lambda n: G.nodes[n]['pos'][0])

    def zigzag_path(start_node):

        current_node = start_node
        direction = 0
        interval = 0.05
        coordinates = []

        while True:
            x, y, _ = G.nodes[current_node]['pos']
            coordinates.append((x, y))

            if direction in [0, 2]:
                next_nodes = [n for n in G.nodes if y - 0.01 < G.nodes[n]['pos'][1] < y + 0.01]
            else:
                next_nodes = [n for n in G.nodes if y + interval - 0.01 < G.nodes[n]['pos'][1] < y + interval + 0.01]

            boundary_nodes = []
            for n in next_nodes:
                close_nodes = [m for m in next_nodes if
                               m != n and euclidean_distance(G.nodes[n]['pos'][:2], G.nodes[m]['pos'][:2]) < 0.06]
                if len(close_nodes) < 2:
                    boundary_nodes.append(n)

            boundary_nodes = [n for n in boundary_nodes if n != current_node]

            if direction == 0:
                filtered_nodes = [n for n in boundary_nodes if G.nodes[n]['pos'][0] < x]
            elif direction == 2:
                filtered_nodes = [n for n in boundary_nodes if G.nodes[n]['pos'][0] > x]
            else:
                filtered_nodes = boundary_nodes

            if not filtered_nodes:
                break

            next_node = min(filtered_nodes, key=lambda n: euclidean_distance(G.nodes[current_node]['pos'][:2], G.nodes[n]['pos'][:2]))

            if next_node == current_node:
                break

            current_node = next_node
            if direction == 3:
                direction = 0
            else:
                direction += 1

        return coordinates

    start_node = find_start_node()

    path_coordinates = zigzag_path(start_node)

    file_name = f"outputs/{env_name}_zigzag.txt"
    with open(file_name, 'w') as file:
        file.write("layer\n")
        file.write(','.join(f'{x:.2f},{y:.2f}' for x, y in path_coordinates) + ';')

    new_start_x = 2.63
    new_start_y = 7.60
    new_start_node = None

    for node in G.nodes:
        x, y, _ = G.nodes[node]['pos']
        if abs(x - new_start_x) < 0.01 and abs(y - new_start_y) < 0.01:
            new_start_node = node
            break

    if new_start_node:

        new_path_coordinates = zigzag_path(new_start_node)

        with open(file_name, 'a') as file:
            file.write(','.join(f'{x:.2f},{y:.2f}' for x, y in new_path_coordinates) + ';')

    return file_name

