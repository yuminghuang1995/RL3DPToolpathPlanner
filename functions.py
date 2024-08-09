from __future__ import division
import numpy as np
import torch
import matplotlib.pyplot as plt
import math
import networkx as nx
from matplotlib.collections import LineCollection
from scipy.spatial import KDTree
import random
import os
import pickle
import sys
import igl

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
    g_num = np.vectorize(old_to_new.get)(lines)

    boundary_nodes_array_fea = [old_to_new[node] for node in boundary_nodes_array if node in old_to_new]

    total_deformation = beam_fea_calculate(g_num, g_coord, total_force_new, boundary_nodes_array_fea, draw)

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
    g_num = np.vectorize(old_to_new.get)(lines)

    boundary_nodes_array_fea = [old_to_new[node] for node in boundary_nodes_array if node in old_to_new]
    total_deformation = beam_fea_calculate(g_num, g_coord, total_force_new, boundary_nodes_array_fea, draw)

    return total_deformation

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
                edge_colors[(u, v)] = "lightgray"
            else:
                edge_colors[(u, v)] = "none"

        fig = plt.figure(figsize=(10, 10), dpi=150)
        ax = fig.add_subplot(111, projection='3d')

        if subgraph:
            for node, coordinates in node_dict.items():
                ax.text(coordinates[0], coordinates[1], coordinates[2], str(node), fontsize=4)

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
                edge_colors[(u, v)] = "blue"
            else:
                edge_colors[(u, v)] = "lightgray"

        if mode == 'Tsp':
            temp_dpi = 1200
        else:
            temp_dpi = 200
        fig, ax = plt.subplots(dpi=temp_dpi)
        x_coords = [coordinates[0] for coordinates in node_dict.values()]
        y_coords = [coordinates[1] for coordinates in node_dict.values()]
        colors = [node_colors[node] for node in node_dict.keys()]
        if mode == 'Tsp':
            ax.scatter(x_coords, y_coords, s=0.01, c='none')
        else:
            ax.scatter(x_coords, y_coords, s=2, c=colors)

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
            colors.append(edge_colors.get(edge, 'lightgray'))
        if mode == 'Tsp':
            lc = LineCollection(lines, colors=colors, linewidths=0.35)
        else:
            lc = LineCollection(lines, colors=colors, linewidths=1.4)
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

def transform_faces(faces, node_mapping):
    reversed_mapping = {v: k for k, v in node_mapping.items()}
    faces_new = []

    for face in faces:
        if all(node in reversed_mapping for node in face):
            new_face = [reversed_mapping[node] for node in face]
            faces_new.append(new_face)
        else:
            missing_nodes = [node for node in face if node not in reversed_mapping]

    faces_new_np = np.array(faces_new, dtype=int)
    return faces_new_np

def lscm_parameterization_libigl(G, faces, new_state_):
    max_node_id = max(G.nodes)
    vertices = np.zeros((max_node_id + 1, 3))
    for n in G.nodes:
        vertices[n] = G.nodes[n]['pos']

    faces = np.array(faces, dtype=np.int32)

    if new_state_[1] != new_state_[2]:
        b = np.array([new_state_[1], new_state_[2]], dtype=np.int32)
    else:
        target_node = new_state_[2]
        min_distance = float('inf')
        nearest_node = None

        for neighbor in G.neighbors(target_node):
            pos_target = np.array(G.nodes[target_node]['pos'])
            pos_neighbor = np.array(G.nodes[neighbor]['pos'])
            distance = np.linalg.norm(pos_target - pos_neighbor)

            if distance < min_distance:
                min_distance = distance
                nearest_node = neighbor

        if nearest_node is None:
            raise ValueError("No neighbors found or no valid nearest node found")

        b = np.array([nearest_node, target_node], dtype=np.int32)

    bc = np.array([[-1.0, 0.0], [0.0, 0.0]])

    success, u = igl.lscm(vertices, faces, b, bc)

    if not success:
        raise ValueError("LSCM parameterization failed")

    return u

def reorder_nodes(G, uv_coordinates):
    uv_coordinates = np.array(uv_coordinates)
    indices_sorted = np.lexsort((uv_coordinates[:, 0], uv_coordinates[:, 1]))  # 先x后y
    old_to_new_mapping = {old_index: new_index for new_index, old_index in enumerate(indices_sorted)}
    G_new = nx.relabel_nodes(G, old_to_new_mapping)

    return G_new, old_to_new_mapping

def combine_mappings(new_temp_to_new_mapping, old_to_new_temp_mapping_):
    final_mapping = {}
    for temp_index, new_index in new_temp_to_new_mapping.items():
        if temp_index in old_to_new_temp_mapping_:
            orig_index = old_to_new_temp_mapping_[temp_index]
            final_mapping[new_index] = orig_index
        else:
            print(f"Warning: No original mapping found for temp index {temp_index}")

    return final_mapping

def create_standard_graph(coordinates_, state_, subgraph):
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

    return node_mapping_

def align_graph(G, state):
    pos = nx.get_node_attributes(G, 'pos')

    if state[1] == state[2] or state[0] == state[1]:
        min_distance = float('inf')
        closest_node = None
        origin_pos = np.array(pos[state[2]])
        for node, node_pos in pos.items():
            if node != state[2]:
                distance = np.linalg.norm(np.array(node_pos) - origin_pos)
                if distance < min_distance:
                    min_distance = distance
                    closest_node = node
        index1 = closest_node
    else:
        index1 = state[1]

    origin = np.array(pos[state[2]])
    for node in pos:
        pos[node] = np.array(pos[node]) - origin

    target = np.array(pos[index1])
    angle = np.arctan2(target[1], target[0])
    rotation_matrix = np.array([
        [np.cos(angle), np.sin(angle), 0],
        [-np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])

    for node in pos:
        pos[node] = rotation_matrix @ np.array(pos[node]).reshape(-1, 1)
        pos[node] = pos[node].flatten()

    if pos[index1][0] > 0:
        for node in pos:
            pos[node][0] *= -1

    result = np.array([pos[node] for node in sorted(G.nodes())])
    return result

def find_k_hop_neighbors(G, start_node, k):
    neighbors = {start_node}
    for _ in range(k):
        next_neighbors = set()
        for node in neighbors:
            next_neighbors.update(G.neighbors(node))
        neighbors.update(next_neighbors)
    return neighbors

def create_new_graph(G_orig, coordinates_, edges_, adjacency_matrix_, state_, init_adjacency_matrix_, radius_, state_dim, heat_radius, faces, env_name, index, mode='Tsp', material='PLA3D'):
    k_hop_neighbors = find_k_hop_neighbors(G_orig, state_[-1], radius_)
    subgraph = G_orig.subgraph(k_hop_neighbors)

    if material == 'PLA3D':
        old_to_new_temp_mapping_ = {index: orig_index for index, orig_index in enumerate(subgraph.nodes)}
        G_new_temp = nx.relabel_nodes(subgraph, {v: k for k, v in old_to_new_temp_mapping_.items()}, copy=True)

        faces_new = transform_faces(faces, old_to_new_temp_mapping_)
        new_state_temp = np.zeros_like(state_)
        for ii, old_index in enumerate(state_):
            new_index_ = next((k for k, v in old_to_new_temp_mapping_.items() if v == old_index), None)
            if new_index_ is not None:
                new_state_temp[ii] = new_index_

        uv_coordinates = lscm_parameterization_libigl(G_new_temp, faces_new, new_state_temp)
        G_new_, new_temp_to_new_mapping = reorder_nodes(G_new_temp, uv_coordinates)
        node_mapping_ = combine_mappings(new_temp_to_new_mapping, old_to_new_temp_mapping_)

    elif material == 'CCF':
        old_to_new_temp_mapping_ = {index: orig_index for index, orig_index in enumerate(subgraph.nodes)}
        G_new_temp = nx.relabel_nodes(subgraph, {v: k for k, v in old_to_new_temp_mapping_.items()}, copy=True)
        new_state_temp = np.zeros_like(state_)
        for ii, old_index in enumerate(state_):
            new_index_ = next((k for k, v in old_to_new_temp_mapping_.items() if v == old_index), None)
            if new_index_ is not None:
                new_state_temp[ii] = new_index_

        aligned_positions = align_graph(G_new_temp, new_state_temp)
        G_new_, new_temp_to_new_mapping = reorder_nodes(G_new_temp, aligned_positions)
        node_mapping_ = combine_mappings(new_temp_to_new_mapping, old_to_new_temp_mapping_)

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
    new_state_adjacency_ = [new_adjacency_matrix_.copy()]
    for i in range(-state_dim, -1, 1):
        new_adjacency_matrix_ = update_matrix(new_adjacency_matrix_, new_state_[i], new_state_[i + 1])
        new_state_adjacency_.append(new_adjacency_matrix_.copy())

    new_state_adjacency_ = np.array(new_state_adjacency_)

    return G_new_, new_state_, new_adjacency_matrix_, new_state_adjacency_, node_mapping_, node_dict_, coords_array


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

    for i in range(len(cycles)):
        cycle = cycles[i]
        if len(cycle) == 3:
            A, B, C = cycle
            pos_A = np.array(G.nodes[A]['pos'])
            pos_B = np.array(G.nodes[B]['pos'])
            pos_C = np.array(G.nodes[C]['pos'])
            AB = pos_B - pos_A
            AC = pos_C - pos_A
            cross_product = np.cross(AB, AC)
            if cross_product[2] < 0:
                cycles[i] = (A, C, B)


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

def find_faces(G):
    cycles_length_3 = find_cycles_of_length(G, 3)
    cycles_length_4 = find_cycles_of_length(G, 4)
    unique_cycles_length_3 = remove_duplicate_cycles(cycles_length_3)
    unique_cycles_length_4 = remove_duplicate_cycles(cycles_length_4)
    cycles = remove_nested_cycles(unique_cycles_length_3, unique_cycles_length_4)
    return cycles

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

def traverse_matrix_euler(matrix, row_index):
    if not any(matrix[row_index, :]):
        return
    columns = np.where(matrix[row_index, :] != 0)[0]
    matrix[row_index, :] = 0
    matrix[:, row_index] = 0

    for col_index in columns:
        traverse_matrix_euler(matrix, col_index)

def recursive_search(G, node, node_dict):
    neighbors = [n for n in G.neighbors(node) if node_dict[n] == 0]
    for neighbor in neighbors:
        node_dict[neighbor] = 1
        recursive_search(G, neighbor, node_dict)

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
        optional_action = np.where(np.array(adjacency_matrix_[state_[-1]]) != 0)[0]

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
                for row_index in sorted_nodes:
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

def choose_start_nodes(G, boundary_nodes_array, material):
    if material == 'PLA3D':
        selected_node = np.random.choice(boundary_nodes_array)
        neighbors = list(G.neighbors(selected_node))
        boundary_neighbors = [node for node in neighbors if node in boundary_nodes_array]
        selected_boundary_neighbor = np.random.choice(boundary_neighbors)
        start_nodes = [selected_node, selected_boundary_neighbor]
    elif material == 'CCF':
        available_nodes = list(set(G.nodes) - set(boundary_nodes_array))
        random_node = random.choice(available_nodes)
        start_nodes = [random_node]
    else:
        start_nodes = [0]

    return start_nodes

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
                    best_path_ = eval(best_path_str)
                    return best_path_
        return None

    def extract_coordinates(file_path):
        coordinates = {}
        with open(file_path, 'r') as file:
            for i, line in enumerate(file):
                coords = [float(x) for x in line.split()]
                coordinates[i] = coords
        return coordinates

    def compare_coordinates(best_path_, node_coordinates_, existing_coordinates_):
        beam_seq_ = []
        close_nodes = []

        for index in best_path_:
            if index in node_coordinates_:
                x, y, z = node_coordinates_[index]
                for key, value in existing_coordinates_.items():
                    if all(abs(a - b) <= 0.01 for a, b in zip(value, [x, y, z])):
                        close_nodes.append(key)

        if close_nodes:
            beam_seq_.append([close_nodes, 0.0])

        return beam_seq_

    def add_min_z_coordinate_node(beam_seq_, second_node_coordinates_, existing_coordinates_):
        if beam_seq_:
            min_z_node = min(second_node_coordinates_, key=lambda k: second_node_coordinates_[k][2])
            min_z_coords = second_node_coordinates_[min_z_node]

            corresponding_node = next((key for key, value in existing_coordinates_.items() if all(abs(a - b) <= 0.01 for a, b in zip(value, min_z_coords))), None)
            if corresponding_node is not None:
                beam_seq_[0][0].append(corresponding_node)

    best_path = extract_best_path(best_path_file)
    node_coordinates = extract_coordinates(node_coordinates_file)
    second_node_coordinates = extract_coordinates(second_node_coordinates_file)
    beam_seq = compare_coordinates(best_path, node_coordinates, existing_coordinates)
    add_min_z_coordinate_node(beam_seq, second_node_coordinates, existing_coordinates)

    return beam_seq


def calculate_total_max_deformation(node_file, path_file):
    def read_coordinates(filename):
        coordinates_ = {}
        with open(filename, 'r') as file:
            for index, line in enumerate(file):
                x, y, z = map(float, line.split())
                coordinates_[index] = [x, y, z]
        return coordinates_

    def read_path(filename):
        path_ = []
        with open(filename, 'r') as file:
            for line in file:
                if line.startswith("Best path:"):
                    path_ = line.split(':')[1].strip().strip('[]').split(', ')
                    path_ = list(map(int, path_))
                    break
        return [(path_[i], path_[i + 1]) for i in range(len(path_) - 1)]

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
    """Update the progress bar based on the iteration index and total length."""
    progress = ii / total

    arrow = int(round(progress * bar_length - 1)) if ii < total else bar_length
    spaces = bar_length - arrow

    progress_bar = 'Progress: [{0}{1}] {2}%'.format('>' * arrow, ' ' * spaces, round(progress * 100, 2))

    sys.stdout.write("\r" + progress_bar)
    sys.stdout.flush()


def final_progress(bar_length=50):
    """Function to display the progress bar at 100% completion and then move to a new line."""
    progress_bar = 'Progress: [{0}] {1}%'.format('>' * bar_length, 100.0)
    print("\r" + progress_bar)

def find_most_similar_checkpoint(current_adj_matrix, checkpoint_dir="checkpoint"):
    max_similarity = -1
    most_similar_checkpoint_path = None
    normalized_matrix = normalize_matrix(current_adj_matrix)

    for filename in os.listdir(checkpoint_dir):
        if filename.endswith(".pt"):
            filepath = os.path.join(checkpoint_dir, filename)
            checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
            adj_matrix = checkpoint['adj_matrix']

            if adj_matrix.shape != normalized_matrix.shape:
                larger_shape = tuple(max(s1, s2) for s1, s2 in zip(adj_matrix.shape, normalized_matrix.shape))
                adj_matrix = adjust_and_pad_matrix(adj_matrix, larger_shape)
                normalized_matrix = adjust_and_pad_matrix(normalized_matrix, larger_shape)

            similarity = calculate_similarity(normalized_matrix, adj_matrix)

            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_checkpoint_path = filepath

    return most_similar_checkpoint_path, max_similarity

def normalize_matrix(matrix):
    max_val = matrix.max()
    if max_val > 0:
        normalized_matrix = matrix / max_val
    else:
        normalized_matrix = matrix
    return normalized_matrix

def calculate_similarity(matrix_a, matrix_b):
    if not isinstance(matrix_a, torch.Tensor):
        matrix_a = torch.tensor(matrix_a, dtype=torch.float32)
    if not isinstance(matrix_b, torch.Tensor):
        matrix_b = torch.tensor(matrix_b, dtype=torch.float32)

    diff = torch.norm(matrix_a - matrix_b, p='fro') * 0.2
    similarity = 1 / (1 + diff)
    return similarity.item()

def adjust_and_pad_matrix(matrix, target_shape):
    padding = [(0, max(0, t - c)) for t, c in zip(target_shape, matrix.shape)]

    if isinstance(matrix, np.ndarray):
        return np.pad(matrix, padding, mode='constant', constant_values=0)
    elif isinstance(matrix, torch.Tensor):
        pad = (0, padding[2][1], 0, padding[1][1], 0, padding[0][1])
        return torch.nn.functional.pad(matrix, pad, "constant", 0)
    else:
        raise TypeError("Unsupported type. The input must be either a numpy array or a torch tensor.")

def extract_faces_from_obj(env_name):
    file_name = f"data/{env_name}.obj"
    faces = []
    try:
        with open(file_name, 'r') as file:
            for line in file:
                if line.startswith('f '):
                    face_indices = line.strip().split()[1:]
                    face_indices = [int(index) - 1 for index in face_indices]
                    faces.append(face_indices)

        faces_np = np.array(faces, dtype=int)
        return faces_np
    except FileNotFoundError:
        print(f"file {file_name} not found。")
        return None
    except Exception as e:
        print(f"read eroor：{e}")
        return None

def pad_matrix(matrix, target_dim):
    current_depth, current_height, current_width = matrix.shape
    padding_height = max(0, target_dim - current_height)
    padding_width = max(0, target_dim - current_width)
    new_height = current_height + padding_height
    new_width = current_width + padding_width
    new_shape = (current_depth, new_height, new_width)
    padded_matrix = np.zeros(new_shape, dtype=matrix.dtype)
    padded_matrix[:, :current_height, :current_width] = matrix
    return padded_matrix

def compute_edges_length(V, Edge):
    diff = V[Edge[:, 1], :] - V[Edge[:, 0], :]
    return np.sqrt(np.sum(diff ** 2, axis=1))

def beam_fea_calculate(g_num, g_coord, total_force_new, boundary_nodes_array_fea, draw, index=0):
    E = 2636
    rho = 1250
    A = 3.14e-6
    G = 1419
    r = 0.001

    G = G * 2.22
    E = E + 700

    V = g_coord * 0.001
    Edge = g_num

    le = compute_edges_length(V, Edge)

    F = np.zeros(6 * V.shape[0])
    for force in total_force_new:
        F[int(force[0]) * 6 + 2] = force[-1]
    F = F * 0.001

    Iz = np.pi * r ** 4 / 4
    Iy = np.pi * r ** 4 / 4
    J = np.pi * r ** 4 / 2

    K = np.zeros((6 * V.shape[0], 6 * V.shape[0]))
    for i in range(Edge.shape[0]):
        v1 = Edge[i, 0]
        v2 = Edge[i, 1]
        DOF = np.hstack((np.arange(6 * v1, 6 * (v1 + 1)), np.arange(6 * v2, 6 * (v2 + 1))))
        kk = np.zeros((12, 12))

        kk[0, 0] = E * A / le[i]
        kk[6, 0] = -kk[0, 0]
        kk[0, 6] = -E * A / le[i]
        kk[6, 6] = -kk[0, 6]
        kk[1, 1] = 12 * E * Iz / le[i] ** 3
        kk[7, 1] = -kk[1, 1]
        kk[1, 5] = 6 * E * Iz / le[i] ** 2
        kk[7, 5] = -kk[1, 5]
        kk[1, 7] = -12 * E * Iz / le[i] ** 3
        kk[7, 7] = -kk[1, 7]
        kk[1, 11] = 6 * E * Iz / le[i] ** 2
        kk[7, 11] = -kk[1, 11]
        kk[2, 2] = 12 * E * Iy / le[i] ** 3
        kk[8, 2] = -kk[2, 2]
        kk[2, 4] = -6 * E * Iy / le[i] ** 2
        kk[8, 4] = -kk[2, 4]
        kk[2, 8] = -12 * E * Iy / le[i] ** 3
        kk[8, 8] = -kk[2, 8]
        kk[2, 10] = -6 * E * Iy / le[i] ** 2
        kk[8, 10] = -kk[2, 10]
        kk[3, 3] = G * J / le[i]
        kk[9, 3] = -kk[3, 3]
        kk[3, 9] = -G * J / le[i]
        kk[9, 9] = -kk[3, 9]
        kk[4, 2] = -6 * E * Iy / le[i] ** 2
        kk[10, 2] = kk[4, 2]
        kk[4, 4] = 4 * E * Iy / le[i]
        kk[10, 4] = 2 * E * Iy / le[i]
        kk[4, 8] = 6 * E * Iy / le[i] ** 2
        kk[10, 8] = kk[4, 8]
        kk[4, 10] = 2 * E * Iy / le[i]
        kk[10, 10] = 4 * E * Iy / le[i]
        kk[5, 1] = 6 * E * Iz / le[i] ** 2
        kk[11, 1] = kk[5, 1]
        kk[5, 5] = 4 * E * Iz / le[i]
        kk[11, 5] = 2 * E * Iz / le[i]
        kk[5, 7] = -6 * E * Iz / le[i] ** 2
        kk[11, 7] = kk[5, 7]
        kk[5, 11] = 2 * E * Iz / le[i]
        kk[11, 11] = 4 * E * Iz / le[i]

        l = (V[v2, 0] - V[v1, 0]) / le[i]
        m = (V[v2, 1] - V[v1, 1]) / le[i]
        n = (V[v2, 2] - V[v1, 2]) / le[i]
        D = np.sqrt(l ** 2 + m ** 2)

        if D == 0:
            if n > 0:
                R = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
            else:
                R = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
        else:
            R = np.array([[l, m, n], [-m / D, l / D, 0], [-l * n / D, -m * n / D, D]])

        RR = np.zeros((12, 12))
        RR[:3, :3] = R
        RR[3:6, 3:6] = R
        RR[6:9, 6:9] = R
        RR[9:, 9:] = R

        K[np.ix_(DOF, DOF)] += RR.T @ kk @ RR

    U = np.zeros(6 * V.shape[0])
    fixed_node = boundary_nodes_array_fea
    fixed_free_dof = np.unique(np.hstack([np.arange(6 * node, 6 * (node + 1)) for node in fixed_node]))
    all_free_dof = np.arange(6 * V.shape[0])
    free_dof = np.setdiff1d(all_free_dof, fixed_free_dof)
    U[free_dof] = np.linalg.solve(K[np.ix_(free_dof, free_dof)], F[free_dof])
    U = U.reshape(-1, 6)
    s = U[:, :3]
    VV = V + s

    if draw:
        distances = np.linalg.norm(V - VV, axis=1)

        color_factor = 100 * distances / 8
        color_factor = np.clip(color_factor, 0, 1)

        fig = plt.figure(dpi=500)
        ax = fig.add_subplot(111, projection='3d')

        for i in range(Edge.shape[0]):
            ax.plot(*V[Edge[i, :], :].T, color='grey', linewidth=1.4, alpha=0.9)

        for i in range(V.shape[0]):
            ax.scatter(*V[i], color='grey', s=2, alpha=0.9)

        ax.axis('off')

        output_dir = 'FEA_simu/ori'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        file_name = f"{output_dir}/plot_{index}.png"

        elev = 30
        azim = 0
        ax.view_init(elev=elev, azim=azim)

        ax.set_xlim([0, 0.1])
        ax.set_ylim([0, 0.1])
        ax.set_zlim([0, 0.1])
        plt.savefig(file_name)

        plt.clf()

    max_deformation_length = 0
    for i in range(VV.shape[0]):
        deformation_length = np.linalg.norm(V[i] - VV[i]) * 100
        if deformation_length > max_deformation_length:
            max_deformation_length = deformation_length

    return max_deformation_length

