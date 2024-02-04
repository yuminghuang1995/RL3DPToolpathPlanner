import random
import time
# import gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib as mpl
import rl_utils
import math
import networkx as nx
import functions as func
import subprocess
import os
from torch_geometric.data import Data
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def train(env_name, coordinates, node_mapping, lines, rays, G, G_new, new_adjacency_matrix, new_state, new_state_adjacency, new_state_dim, new_node_dict, anti_lock_optional_action,
          new_angle_limit, beam_seq_i, max_edge_pass_, ori_i, ori_index, new_coords_array, boundary_nodes_array, heat_radius, calc_mode, material, max_edge_length, savept=False):

    class ReplayBuffer:

        def __init__(self, capacity):
            self.buffer = collections.deque(maxlen=capacity)

        def add(self, state_, state_adjacency_, action_, action_indexes_, reward_, next_state_, next_state_adjacency_, coords_array_, next_coords_array_, done_):  # 将数据加入buffer
            self.buffer.append((state_, state_adjacency_, action_, action_indexes_, reward_, next_state_, next_state_adjacency_, coords_array_, next_coords_array_, done_))

        def sample(self, batch_size_):
            transitions = random.sample(self.buffer, batch_size_)
            state_, state_adjacency_, action_, action_indexes_, reward_, next_state_, next_state_adjacency_, coords_array_, next_coords_array_, done_ = zip(*transitions)
            return np.array(state_), np.array(state_adjacency_), action_, action_indexes_, reward_, \
                np.array(next_state_), np.array(next_state_adjacency_), np.array(coords_array_), np.array(next_coords_array_), done_

        def size(self):
            return len(self.buffer)

    class Qnet(torch.nn.Module):

        def __init__(self, state_dim_, hidden_dim_, action_dim_):
            super(Qnet, self).__init__()

            self.conv1 = torch.nn.Conv2d(state_dim_, state_dim_, 4)
            self.conv2 = torch.nn.Conv2d(state_dim_, state_dim_, 4)
            self.conv3 = torch.nn.Conv2d(state_dim_, state_dim_, 4)
            self.ReLu = torch.nn.ReLU()
            self.Sigmoid = torch.nn.Sigmoid()

            x_dummy = torch.zeros(1, state_dim_, max_sa_size + 4, max_sa_size)
            x_dummy = self.conv1(x_dummy)
            x_dummy = self.ReLu(x_dummy)
            x_dummy = F.max_pool2d(x_dummy, 2)

            x_dummy = self.conv2(x_dummy)
            x_dummy = self.ReLu(x_dummy)
            x_dummy = F.max_pool2d(x_dummy, 2)

            fc1_input_size = x_dummy.numel()

            if max_edge_pass_ <= 4:
                self.fc1 = nn.Linear(fc1_input_size, 256)
                self.fc2 = torch.nn.Linear(256, 128)
                self.fc_A = torch.nn.Linear(128, action_dim_)
                self.fc_V = torch.nn.Linear(128, 1)
            else:
                self.fc1 = nn.Linear(fc1_input_size, 512)
                self.fc2 = torch.nn.Linear(512, 256)
                self.fc_A = torch.nn.Linear(256, action_dim_)
                self.fc_V = torch.nn.Linear(256, 1)

        def forward(self, x):
            batch_size_ = x.size(0)
            x_padded = torch.zeros(batch_size_, state_dim, max_sa_size+4, max_sa_size, device=x.device)
            x_padded[:, :, :(node_num+4), :node_num] = x
            x = x_padded

            x = self.conv1(x)
            x = self.ReLu(x)

            if ori_index == 99999 and i_episode == num_episodes - 1:
                func.plot_normalized_heatmaps(x)

            x = F.max_pool2d(x, 2)

            x = self.conv2(x)
            x = self.ReLu(x)
            x = F.max_pool2d(x, 2)

            x = x.view(batch_size_, -1)
            x = self.ReLu(x)
            x = self.fc1(x)
            x = self.ReLu(x)
            x = self.fc2(x)
            x = self.ReLu(x)
            A = self.fc_A(x)
            V = self.fc_V(x)
            x = (V + A - A.mean(1, keepdim=True))
            return x

    class DQN:

        def __init__(self, state_dim_, hidden_dim_, action_dim_, learning_rate, gamma_, epsilon_, target_update_, device_):
            self.action_dim = action_dim_
            self.q_net = Qnet(state_dim_, hidden_dim_, self.action_dim).to('cuda')
            self.target_q_net = Qnet(state_dim_, hidden_dim_, self.action_dim).to('cuda')
            self.optimizer = torch.optim.AdamW(self.q_net.parameters(), lr=learning_rate)
            self.gamma = gamma_
            self.epsilon = epsilon_
            self.target_update = target_update_
            self.count = 0
            self.device = device_
            self.loss = 0

        def take_action(self, state_, state_adjacency_, pre_pre_action_, adjacency_matrix_flow_, edge_pass_, coords_array_, greedy=False):  # epsilon-贪婪策略采取动作
            optional_action = np.where(np.array(adjacency_matrix_flow_[state_[-1]]) != 0)[0]  # 找到所有可选的动作(公用)

            if calc_mode == 'Tsp':
                row_col_set = np.concatenate(np.argwhere(adjacency_matrix_flow_ > 0).T)
                optional_action = optional_action[~np.isin(optional_action, row_col_set)]

            if edge_pass_ == 0:
                optional_action = anti_lock_optional_action

            action_index_ = 0
            if np.random.random() < self.epsilon and not greedy:
                action_ = np.random.choice(optional_action)

                action_index_ = action_
            else:

                state_adjacency_add_ = np.concatenate((state_adjacency_, coords_array_), axis=1)

                state_adjacency_add_ = torch.as_tensor(np.array(state_adjacency_add_), dtype=torch.float).to(self.device)
                state_adjacency_add_ = state_adjacency_add_.unsqueeze(0)
                q_values = self.q_net(state_adjacency_add_)[0]

                selected_q_values = q_values[optional_action]
                max_index = torch.argmax(selected_q_values).item()
                action_ = optional_action[max_index]
                action_index_ = action_

            return action_, action_index_

        def update(self, transition_dict_):

            states_adjacency = np.concatenate((transition_dict_['states_adjacency'], transition_dict_['coor_array']), axis=2)
            states_adjacency = torch.tensor(states_adjacency).to(self.device)

            action_indexes = torch.tensor(transition_dict_['action_indexes']).view(-1, 1).to(self.device)
            rewards = torch.tensor(transition_dict_['rewards'], dtype=torch.float).view(-1, 1).to(self.device)

            next_states_adjacency = np.concatenate((transition_dict_['next_states_adjacency'], transition_dict_['next_coor_array']), axis=2)
            next_states_adjacency = torch.tensor(next_states_adjacency).to(self.device)

            dones = torch.tensor(transition_dict_['dones'], dtype=torch.float).view(-1, 1).to(self.device)

            q_values = self.q_net(states_adjacency.float()).gather(1, action_indexes)

            max_next_q_values, _ = self.target_q_net(next_states_adjacency.float()).max(dim=1, keepdim=True)

            q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
            dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))

            self.optimizer.zero_grad()
            dqn_loss.backward()
            self.optimizer.step()
            self.loss = dqn_loss
            if self.count % self.target_update == 0:
                self.target_q_net.load_state_dict(self.q_net.state_dict())
            self.count += 1

        def save_checkpoint(self, name, ori_index_):
            torch.save({
                'model_state_dict': self.q_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': self.loss,
            }, "checkpoint/model" + ".pt")

        def load_checkpoint(self, name, ori_index_):
            filepath = "checkpoint/model" + ".pt"
            if not os.path.isfile(filepath):
                print('Prior mode start: No checkpoint at first')
                return

            checkpoint = torch.load(filepath)

            self.q_net.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    def damping(edge_pass_):
        if Gaussian_damp:
            sigma = 0.865
            mu = 1.0
            damp = 0.9 * math.exp(-(edge_pass_ ** 2) / (2 * sigma ** 2)) + 0.1
        else:
            if edge_pass_ < 4:
                damp = (1 / (edge_pass_ + 1) ** 2)
            else:
                damp = 0.0625

        return damp

    def cal_reward(angle_, edge_pass_, double_pass_, lifting_):
        x1 = (angle_ - 60) / 30
        x2 = (angle_ - 120) / 30

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        reward_ = sigmoid(-10 * x1) + 0.9 * (sigmoid(10 * x1) - sigmoid(10 * x2))

        reward_ = reward_ - double_pass_
        reward_ = reward_ - lifting_

        return reward_


    class Environment:
        """ 环境搭建 """

        def __init__(self, new_adjacency_matrix_, new_node_dict_, new_state_, new_state_adjacency_, new_coords_array_):
            self.adjacency_matrix = new_adjacency_matrix_
            self.node_dict = new_node_dict_
            self.state = new_state_
            self.state_adjacency = new_state_adjacency_
            self.coords_array = new_coords_array_

        def obser_space(self):
            return int(np.count_nonzero(self.adjacency_matrix) / 2)

        def action_space(self):
            return len(self.node_dict)

        def reset(self):
            state_ = self.state.copy()
            adjacency_matrix_flow_ = self.adjacency_matrix.copy()
            state_adjacency_ = self.state_adjacency.copy()
            coords_array_ = self.coords_array.copy()
            return state_, adjacency_matrix_flow_, state_adjacency_, coords_array_

        def step(self, action_, state_, state_adjacency_, edge_pass_, adjacency_matrix_flow_, coords_array_, temp_path_, temp_lines_):

            adjacency_matrix_flow_last = adjacency_matrix_flow_.copy()
            if adjacency_matrix_flow_[state[-1]][action_] < 0 or adjacency_matrix_flow_[action_][state[-1]] < 0:
                adjacency_matrix_flow_[state[-1]][action_] = -adjacency_matrix_flow_[state[-1]][action_]
                adjacency_matrix_flow_[action_][state[-1]] = -adjacency_matrix_flow_[action_][state[-1]]
            else:
                adjacency_matrix_flow_[state[-1]][action_] = 0.0
                adjacency_matrix_flow_[action_][state[-1]] = 0.0

            next_state_ = state_.copy()
            next_state_ = np.append(next_state_[1:], action_)
            next_state_adjacency_ = state_adjacency_.copy()

            adjacency_matrix_flow_ = np.expand_dims(adjacency_matrix_flow_, axis=0)
            next_state_adjacency_ = np.concatenate([next_state_adjacency_[1:], adjacency_matrix_flow_], axis=0)

            adjacency_matrix_flow_ = np.squeeze(adjacency_matrix_flow_, axis=0)

            next_lines_ = temp_lines_ + [(node_mapping[next_state_[-2]], node_mapping[next_state_[-1]])]

            done_ = False
            reward_ = 0
            angle_ = 0
            angle_pre_ = 0
            node0_ = self.node_dict[state_[len(next_state_) - 3]]
            node1_ = self.node_dict[next_state_[len(next_state_) - 3]]
            node2_ = self.node_dict[next_state_[len(next_state_) - 2]]
            node3_ = self.node_dict[next_state_[len(next_state_) - 1]]

            if calc_mode == 'Euler':
                if tuple(temp_path_) in path_reward_dict:

                    reward_ = path_reward_dict[tuple(temp_path_)]
                else:

                    if material == 'CCF':

                        angle_ = func.calculate_angle(node1_, node2_, node3_)
                        angle_pre_ = func.calculate_angle(node0_, node1_, node2_)

                        double_pass = 0
                        if adjacency_matrix_flow_[next_state_[-2]][next_state_[-1]] == 0:
                            if distance_reward:
                                double_pass = func.calculate_distance(node1_, node2_) / max_edge_length
                            else:
                                double_pass = 1.0

                        lifting = 0
                        if (next_state_[-2] in boundary_nodes_array) and next_state_[-1] == next_state_[-3]:
                            lifting = 0.5
                        if (next_state_[-2] not in boundary_nodes_array) and next_state_[-1] == next_state_[-3]:
                            lifting = 1

                        reward_ = cal_reward(angle_, edge_pass_, double_pass, lifting)

                        damping_ = damping(edge_pass_)
                        reward_ = reward_ * damping_

                    if material == 'PLA3D':
                        damping_ = damping(edge_pass_)

                        reward_ = func.beam_fea(G_new, adjacency_matrix_flow_, self.node_dict, boundary_nodes_array, draw=False)
                        reward_ = -reward_ * 10

                        gravity_center = 0.1 * ((node3_[2] + node2_[2]) / 2 - (node2_[2] + node1_[2]) / 2)
                        reward_ = reward_ - gravity_center

                        collision_punish = 0
                        if collision_reward:
                            collision, _, min_angle = func.collision_check_simulation(coordinates, node_mapping[next_state_[-2]], node_mapping[next_state_[-1]], rays, next_lines_, norm_output=False)
                            collision_punish = 0
                            if collision:
                                collision_punish = 1000
                            else:
                                collision_punish = min_angle
                        reward_ = reward_ - collision_punish

                        reward_ = reward_ * damping_

                    path_reward_dict[tuple(temp_path_)] = reward_

                if np.sum(adjacency_matrix_flow_ != 0) == 0:
                    done_ = True

                if np.sum(adjacency_matrix_flow_ != 0) != 0 and np.sum(adjacency_matrix_flow_[action_] != 0) == 0:
                    done_ = True

                if edge_pass_ >= (max_edge_pass - 1):
                    done_ = True

            if calc_mode == 'Tsp':
                if tuple(temp_path_) in path_reward_dict:

                    reward_ = path_reward_dict[tuple(temp_path_)]
                else:
                    if material == 'Metal':

                        damping_ = damping(edge_pass_)

                        if np.count_nonzero(next_state_ == 0) >= 2:
                            angle_ = 0
                        else:
                            angle_ = func.calculate_angle(node1_, node2_, node3_)

                        reward_ = heat_radius - coords_array_[-1][3][next_state_[-1]]

                        reward_ = damping_ * reward_

                        unique_elements = np.unique(state_)
                        if len(unique_elements) == 1:
                            reward_ = reward_ - 1.0 * heat_radius * damping_

                    if material == 'Clay':
                        node2_ = self.node_dict[next_state_[len(next_state_) - 2]]
                        node3_ = self.node_dict[next_state_[len(next_state_) - 1]]
                        angle_ = func.calculate_angle_along_stress(node2_, node3_, np.array([10, 0, 0]))
                        if edge_pass_ == 0:
                            damping_ = 1
                        elif edge_pass_ == 1:
                            damping_ = 0.01
                        elif edge_pass_ == 2:
                            damping_ = 0.0
                        else:
                            damping_ = 0.1

                        if angle_ > angle_limit:
                            reward_ = 0 * damping_
                        elif 58 < angle_ <= angle_limit:
                            reward_ = 0.9 * damping_
                        else:
                            reward_ = 1 * damping_

                    if material == 'Tsp':
                        reward_ = -func.calculate_distance(node2_, node3_)

                    path_reward_dict[tuple(temp_path_)] = reward_

                if edge_pass_ >= (max_edge_pass - 1):
                    done_ = True

                row_col_array = np.concatenate(np.argwhere(adjacency_matrix_flow_ > 0).T)

                row_col_set = set(row_col_array)

                if all_integers.issubset(row_col_set):
                    done_ = True

                optional_action = np.where(np.array(adjacency_matrix_flow_[next_state_[-1]]) != 0)[0]

                optional_action = optional_action[~np.isin(optional_action, row_col_array)]
                if optional_action.shape == (0,):
                    done_ = True

            single_next_coords_array_ = func.update_heat_field(next_state_[-1], coords_array_[-1], heat_radius, calc_mode)

            next_coords_array_ = np.concatenate((coords_array_[1:], single_next_coords_array_[None]), axis=0)

            return next_state_, next_state_adjacency_, reward_, done_, angle_, next_coords_array_, next_lines_

    Iter = 1

    if material == 'CCF':
        action_dim = max_edge_pass_ * 35
    else:
        action_dim = max_edge_pass_ * 40

    lr = 5e-4
    num_episodes = 500
    hidden_dim = 256
    gamma = 0.98
    epsilon = 0.5
    target_update = 3
    buffer_size = 500
    minimal_size = 2
    batch_size = 3
    collision_reward = False
    distance_reward = False
    Gaussian_damp = False
    if material == 'CCF':
        max_sa_size = max_edge_pass_ * 35
    else:
        max_sa_size = max_edge_pass_ * 40

    state_dim = new_state_dim
    angle_limit = new_angle_limit
    max_edge_pass = max_edge_pass_
    death_reward = -4
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # random.seed(0)
    # np.random.seed(0)
    # torch.manual_seed(0)

    replay_buffer = ReplayBuffer(buffer_size)
    plot = False

    env = Environment(new_adjacency_matrix, new_node_dict, new_state, new_state_adjacency, new_coords_array)
    path_length = env.obser_space()
    node_num = env.action_space()

    agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device)
    return_list = []
    edge_pass_list = []
    ave_angle_list = []
    limit_angle_list = []

    counter = 0
    best_return = -100

    all_candidates = list()
    all_integers = set(range(len(new_adjacency_matrix[0])))

    path_reward_dict = {}
    new_sequence = [[[0], 0.0]]

    if savept:
        agent.load_checkpoint(env_name, ori_index)

    for i in range(Iter):
        for i_episode in range(int(num_episodes)):
            edge_pass = 0
            episode_return = 0
            total_angle = 0
            limit_angle_num = 0
            temp_path = []
            state, adjacency_matrix_flow, state_adjacency, coords_array = env.reset()
            pre_action = 0
            pre_pre_action = 0
            done = False
            first_reward = 0

            temp_lines = lines.copy()

            if material == 'CCF':
                m = 40
            else:
                m = 30

            if counter >= 5 * max_edge_pass_ and i_episode >= m * max_edge_pass_:

                sequences = func.remove_duplicates_and_keep_max(all_candidates)
                sequences = sorted(sequences, key=lambda tup: tup[1], reverse=True)
                action, action_index = agent.take_action(state, state_adjacency, pre_pre_action, adjacency_matrix_flow, edge_pass, coords_array, greedy=True)
                next_state, next_state_adjacency, reward, done, angle, next_coords_array, next_lines = env.step(action, state, state_adjacency, edge_pass, adjacency_matrix_flow,
                                                                                                                coords_array, temp_path, temp_lines)

                if reward != sequences[0][1]:
                    new_sequence = [[sequences[0][0], sequences[0][1]]]
                else:
                    new_sequence = [[[action], reward]]

                break

            while not done:

                action, action_index = agent.take_action(state, state_adjacency, pre_pre_action, adjacency_matrix_flow, edge_pass, coords_array, greedy=False)
                temp_path.append(action)

                next_state, next_state_adjacency, reward, done, angle, next_coords_array, next_lines = env.step(action, state, state_adjacency, edge_pass, adjacency_matrix_flow, coords_array, temp_path, temp_lines)

                replay_buffer.add(state, state_adjacency, action, action_index, reward, next_state, next_state_adjacency, coords_array, next_coords_array, done)

                state = next_state
                state_adjacency = next_state_adjacency
                pre_pre_action = pre_action
                pre_action = action
                coords_array = next_coords_array
                temp_lines = next_lines
                episode_return += reward

                total_angle += angle
                if angle > angle_limit:
                    limit_angle_num += 1

                if edge_pass == 0:
                    first_reward = episode_return

                edge_pass += 1

                if done:
                    if episode_return > best_return:
                        best_return = episode_return

                        counter = 0

                    else:
                        counter += 1

                    candidate = [[int(temp_path[0])], episode_return, first_reward]

                    all_candidates.append(candidate)

                if replay_buffer.size() > minimal_size:
                    b_s, b_sa, b_a, b_i, b_r, b_ns, b_nsa, b_ca, b_nca, b_d = replay_buffer.sample(batch_size)
                    transition_dict = {
                        'states': b_s,
                        'states_adjacency': b_sa,
                        'actions': b_a,
                        'action_indexes': b_i,
                        'next_states': b_ns,
                        'next_states_adjacency': b_nsa,
                        'rewards': b_r,
                        'coor_array': b_ca,
                        'next_coor_array': b_nca,
                        'dones': b_d
                    }
                    agent.update(transition_dict)

            return_list.append(episode_return)
            edge_pass_list.append(edge_pass)
            ave_angle_list.append(total_angle / edge_pass)
            limit_angle_list.append(limit_angle_num / edge_pass)

        if savept:
            agent.save_checkpoint(env_name, ori_index)


    if plot:
        episodes_list = list(range(len(return_list)))

        mpl.rcParams['font.family'] = 'Times New Roman'
        mpl.rcParams['font.size'] = 24
        mv_return_reward = rl_utils.moving_average(return_list, 9)
        fig = plt.figure(num=1, figsize=(14, 11))
        ax1 = fig.add_subplot(211)
        ax1.spines['top'].set_linewidth(4)
        ax1.spines['bottom'].set_linewidth(4)
        ax1.spines['left'].set_linewidth(4)
        ax1.spines['right'].set_linewidth(4)
        ax1.set_ylim(-0.42, 0.02)
        ax1.plot(episodes_list, mv_return_reward, color='blue', linewidth=3)
        plt.show()
        print('output reward')

    return new_sequence
















