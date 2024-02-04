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
import rl_utils
import math
import networkx as nx
import functions as func
from scipy.spatial import ConvexHull
from scipy.spatial import KDTree



class Environ:
    """ 环境搭建 """

    def __init__(self, env_name_, env_node_, env_edge_, plot_finished_path_, mode='Tsp'):
        self.env_name = env_name_
        self.env_node = np.loadtxt(env_node_)
        self.mode = mode
        if mode == 'Euler':
            self.env_edge = np.loadtxt(env_edge_).astype(int)
        if mode == 'Tsp':
            self.env_edge = np.array([])
        self.adjacency_matrix = np.zeros((len(self.env_node), len(self.env_node)))
        self.node_dict = {}
        self.plot_finished_path = plot_finished_path_
        self.G = nx.Graph()

    def build_graph(self, interval, material):
        for kk in range(len(self.env_node)):
            self.node_dict[kk] = [self.env_node[kk][0], self.env_node[kk][1], self.env_node[kk][2]]

        if self.mode == 'Euler':

            for node, coord in self.node_dict.items():
                self.G.add_node(node, pos=coord)
            self.G.add_edges_from(self.env_edge)

        if self.mode == 'Tsp':

            for i in range(len(self.env_node)):
                self.G.add_node(i, pos=self.env_node[i])

            kdtree = KDTree(self.env_node)
            max_distance = interval * 1.1

            for i in range(len(self.env_node)):

                nearby_indices = kdtree.query_ball_point(self.env_node[i], max_distance)

                for j in nearby_indices:
                    if i != j:
                        self.G.add_edge(i, j)

            self.env_edge = np.array(list(self.G.edges()))

            self.env_edge = np.concatenate((self.env_edge, self.env_edge), axis=0)

        for edge in self.env_edge:
            node1, node2 = edge
            node1 = int(node1)
            node2 = int(node2)
            length = np.linalg.norm(self.env_node[int(node1)] - self.env_node[int(node2)])

            if self.adjacency_matrix[node1][node2] != 0 or self.adjacency_matrix[node2][node1] != 0:
                self.adjacency_matrix[node1][node2] = -self.adjacency_matrix[node1][node2]
                self.adjacency_matrix[node2][node1] = -self.adjacency_matrix[node2][node1]
            else:
                self.adjacency_matrix[node1][node2] = length
                self.adjacency_matrix[node2][node1] = length

        max_edge = 0.05

        if material == 'CCF':
            positive_adj_matrix = np.abs(self.adjacency_matrix)
            max_edge = np.max(positive_adj_matrix)

        if material == 'CCF':

            boundary_nodes = []
        elif material == 'PLA3D':
            boundary_nodes = func.find_smallest_z_set(self.G)
        else:
            boundary_nodes = []

        boundary_nodes_array = np.array(list(boundary_nodes))

        return self.node_dict, np.array(self.adjacency_matrix), self.env_edge, self.G, max_edge, boundary_nodes_array
