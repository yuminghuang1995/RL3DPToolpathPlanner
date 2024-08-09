import random
import numpy as np
import torch
import functions as func
import dqn as dqn
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def calculate(k, state_dim, init_adjacency_matrix, G, coordinates, edges, max_edge_pass, env_name, angle_limit,
              beam_num, boundary_nodes_array, heat_radius, mode, material, block, rl, random_beam, rays, train_mode, max_edge_length, savept, faces):

    start_nodes = func.choose_start_nodes(G, boundary_nodes_array, material)
    beam_seq = [[start_nodes, 0.0]]
    done = False
    edge_pass = 0
    index = 0
    total_nodes = len(G.nodes)

    if material == 'PLA3D' and block == 1:
        beam_seq = func.generate_beam_sequence(env_name, coordinates)

    while not done:

        func.update_progress(beam_seq, total_nodes)

        temp_beam_seq = []
        if len(beam_seq) == 0:
            break

        for i in range(len(beam_seq)):
            state = func.create_current_state(beam_seq[i][0], state_dim, init_adjacency_matrix, mode=mode, material=material)
            adjacency_matrix = func.create_adj_matrix(init_adjacency_matrix.copy(), beam_seq[i][0], mode=mode)
            G_orig = func.create_heat_field(beam_seq[i][0], G.copy(), heat_radius, mode=mode)

            if mode == 'Euler':
                if material == 'CCF':
                    if np.sum(adjacency_matrix < 0) == 0:
                        done = True
                        beam_seq = [beam_seq[i]]
                        func.final_progress()
                        print('Find final path!')
                        break

                if material == 'PLA3D':
                    if np.sum(adjacency_matrix != 0) == 0:
                        done = True
                        beam_seq = [beam_seq[i]]
                        func.final_progress()
                        print('Find final path!')
                        break


            if mode == 'Tsp':
                all_integers = set(range(len(coordinates)))
                contains_all_integers = all_integers.issubset(set(beam_seq[i][0]))
                if contains_all_integers:
                    done = True
                    beam_seq = [beam_seq[i]]
                    func.final_progress()
                    break

            G_new, new_state, new_adjacency_matrix, new_state_adjacency, node_mapping, new_node_dict, new_coords_array = func.create_new_graph(
                G_orig, coordinates, edges, adjacency_matrix, state, init_adjacency_matrix, max_edge_pass, state_dim, heat_radius, faces, env_name, index, mode=mode, material=material)

            if index % 100 == 0:
                func.draw_graph(env_name, G_orig, coordinates, state, adjacency_matrix, 1, index, output=True, draw=False, subgraph=False, mode=mode, show=material)

            optional_action, lifting, lines = func.anti_self_locking_subgraph(G_orig, G_new, adjacency_matrix, node_mapping, state, beam_seq[i][0], new_adjacency_matrix, new_state,
                                                                              coordinates, heat_radius, init_adjacency_matrix, rays, train_mode, mode=mode, material=material)

            if optional_action.size == 0:
                func.final_progress()
                if mode == 'Euler':
                    # func.draw_graph(env_name, G_orig, coordinates, state, adjacency_matrix, i, index, output=True, draw=True, subgraph=False, mode=mode)
                    if material == 'CCF':
                        if i == len(beam_seq) - 1 and not temp_beam_seq:
                            beam_seq = [beam_seq[0]]
                            done = True
                            break

                    if material == 'PLA3D':
                        if i == len(beam_seq) - 1 and not temp_beam_seq:
                            beam_seq = [beam_seq[0]]
                            done = True
                            break
                continue

            new_seq = list()
            if lifting:
                state = np.repeat(optional_action, state_dim)
                G_new, new_state, new_adjacency_matrix, new_state_adjacency, node_mapping, new_node_dict, new_coords_array = func.create_new_graph(
                    G_orig, coordinates, edges, adjacency_matrix, state, init_adjacency_matrix, max_edge_pass, state_dim, heat_radius, faces, env_name, index, mode=mode, material=material)
                optional_action = new_state[-1:]

                new_seq = [[[optional_action[0]], 0]]

            if not new_seq and not lifting:
                new_beam_seq_i = beam_seq[i].copy()
                if material == 'CCF':
                    boundary_nodes_array_new = func.get_new_boundary_nodes(boundary_nodes_array, node_mapping)
                elif material == 'PLA3D':
                    boundary_nodes_ori = func.get_new_boundary_nodes(boundary_nodes_array, node_mapping)
                    boundary_nodes_new = func.find_fix_nodes(G, G_new, adjacency_matrix, new_adjacency_matrix, node_mapping)
                    boundary_nodes_array_new = list(set(boundary_nodes_ori + boundary_nodes_new))
                else:
                    boundary_nodes_array_new = []

                new_seq = dqn.train(env_name, coordinates, node_mapping, lines, rays, G, G_new, new_adjacency_matrix, new_state, new_state_adjacency, state_dim, new_node_dict, optional_action,
                                    angle_limit, new_beam_seq_i, max_edge_pass, i, index, new_coords_array, boundary_nodes_array_new, heat_radius, mode, material, max_edge_length, savept)

            new_seq = [item for item in new_seq if item[0][0] in optional_action]

            if material == 'CCF' and not new_seq:
                beam_seq = [beam_seq[0]]
                done = True
                break

            new_seq = [[item[0], round(item[1] + beam_seq[i][1], 4)] for item in new_seq]

            orig_seq = new_seq.copy()
            for jj in range(len(new_seq)):
                orig_seq[jj][0] = func.n2o_seq_trans(new_seq[jj][0], node_mapping)
                orig_seq[jj][0] = beam_seq[i][0] + orig_seq[jj][0]
            temp_beam_seq += orig_seq

        if not done:
            beam_seq = func.remove_duplicates_and_keep_max(temp_beam_seq)
            new_beam_seq = beam_seq[:beam_num]
            remain_beam_seq = beam_seq[beam_num:]
            if remain_beam_seq:
                if len(remain_beam_seq) < random_beam:
                    new_beam_seq.extend(remain_beam_seq)
                else:
                    random_elements = random.sample(remain_beam_seq, random_beam)
                    new_beam_seq.extend(random_elements)

            beam_seq = new_beam_seq
            edge_pass += 1

        index += 1

    return beam_seq
