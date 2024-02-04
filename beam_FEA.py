import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import networkx as nx
import random
import math
from networkx.algorithms import isomorphism
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os


def compute_edges_length(V, Edge):
    diff = V[Edge[:, 1], :] - V[Edge[:, 0], :]
    return np.sqrt(np.sum(diff ** 2, axis=1))

def beam_fea_calculate(g_num, g_coord, total_force_new, boundary_nodes_array_fea, draw, index=0):

    E = 3836
    rho = 1250
    A = 3.14e-6
    G = 1419
    r = 0.001

    V = g_coord * 0.001
    Edge = g_num

    G = G * 2.22
    E = E + 700

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

