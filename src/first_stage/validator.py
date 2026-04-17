import torch

import numpy as np
import re
from scipy.io import loadmat

from .generator import testing_list, max_length, max_G, max_k, stress_min
from .reader import stress_range
from . import config
from .parser import args, device

# Validation grid
if(len(testing_list)>5):
    testing_list = testing_list[-5:]
# testing_list.append(1)
valid_truth_list = []
valid_X_list = []
valid_T_list = []
valid_L_list = []
valid_W_list = []
valid_G_list = []
valid_k1_list = []
valid_k2_list = []
for file in testing_list:
    valid_truth_single_case_list = []
    valid_X_single_case_list = []
    valid_T_single_case_list = []
    valid_L_single_case_list = []
    valid_W_single_case_list = []
    valid_G_single_case_list = []
    valid_k1_single_case_list = []
    valid_k2_single_case_list = []
    segment_length_list = []
    G_list = []
    idx_MSE_single_case = np.empty([0,7])
    truth_MSE_single_case = np.empty([0,1])
    file_path = args.data_path + str(file)

    # Read .geo file for wire geometries
    with open(file_path+".geo") as f:
        lines = f.readlines()
    for line in lines:
        if line[0:9] == 'Rectangle':
            # rect_vertices = re.findall('-*[0-9]+', line)
            # current_segment_length = float(rect_vertices[4]+"e-6")
            rect_vertices = re.findall('-*[0-9]+\.*[0-9]*', line)
            current_segment_length = max(abs(float(rect_vertices[4]+"e-6")),abs(float(rect_vertices[5]+"e-6")))
            segment_length_list.append(current_segment_length)
    n_segments = len(segment_length_list)
    L_list = [L/max_length for L in segment_length_list]

    # Read .mat file for stress results and current
    data_mat = loadmat(file_path+".mat")
    current = data_mat['J']
    stress = data_mat['sVC']

    # Current density
    for J in current[0]:
        G = config.e * config.Z * config.rou * J / config.Omega
        G_list.append(G)

    # Stress
    idx_MSE_list = []
    truth_MSE_list = []
    for i in range(n_segments):
        truth = stress[i,0] # shape = [L_n_points, T_n_points]

        wire_interval = segment_length_list[i]/(truth.shape[0]-1)

        # calculate k1 and k2
        k1 = truth[:2, :]
        k2 = truth[-2:, :]
        k1 = (k1[1,:]-k1[0,:])/wire_interval
        k2 = (k2[1,:]-k2[0,:])/wire_interval
        k1 += G_list[i]
        k2 += G_list[i] 
        k1 = k1 / max_k
        k2 = k2 / max_k
        # k1_list.append(k1)
        # k2_list.append(k2)

        # Genrate indexes for colocation points [x,t,L,W,G,k1,k2]
        truth = 2*((truth-stress_min) / stress_range)-1 # convert to [-1, 1]
        x = np.linspace(0,1, num=truth.shape[0], endpoint=True)
        t = np.linspace(0,1, num=truth.shape[1], endpoint=True)
        X, T = np.meshgrid(x, t) # shape=[T_n_points, L_n_points]
        L = np.tile(L_list[i], [X.shape[0],X.shape[1]])
        W = np.tile(1.0, [X.shape[0],X.shape[1]])
        G = np.tile(G_list[i]/max_G, [X.shape[0],X.shape[1]])
        if(i==0):
            k1[:] = 0
        if(i == n_segments-1):
            k2[:] = 0
        # plt.plot(k1, label = f"k1_{i}")
        # plt.plot(k2, label = f"k2_{i}")
        k1 = np.tile(k1,[truth.shape[0],1]).T # shape=[T_n_points, L_n_points]
        k2 = np.tile(k2,[truth.shape[0],1]).T # shape=[T_n_points, L_n_points]

        X = np.expand_dims(X.flatten(), axis=-1)
        T = np.expand_dims(T.flatten(), axis=-1)
        L = np.expand_dims(L.flatten(), axis=-1)
        W = np.expand_dims(W.flatten(), axis=-1)
        G = np.expand_dims(G.flatten(), axis=-1)
        k1 = np.expand_dims(k1.flatten(), axis=-1)
        k2 = np.expand_dims(k2.flatten(), axis=-1)

        X = torch.FloatTensor(X).to(device)
        T = torch.FloatTensor(T).to(device)
        L = torch.FloatTensor(L).to(device)
        W = torch.FloatTensor(W).to(device)
        G = torch.FloatTensor(G).to(device)
        k1 = torch.FloatTensor(k1).to(device)
        k2 = torch.FloatTensor(k2).to(device)

        valid_truth_single_case_list.append(truth)
        valid_X_single_case_list.append(X)
        valid_T_single_case_list.append(T)
        valid_L_single_case_list.append(L)
        valid_W_single_case_list.append(W)
        valid_G_single_case_list.append(G)
        valid_k1_single_case_list.append(k1)
        valid_k2_single_case_list.append(k2)

    valid_truth_list.append(valid_truth_single_case_list)
    valid_X_list.append(valid_X_single_case_list)
    valid_T_list.append(valid_T_single_case_list)
    valid_L_list.append(valid_L_single_case_list)
    valid_W_list.append(valid_W_single_case_list)
    valid_G_list.append(valid_G_single_case_list)
    valid_k1_list.append(valid_k1_single_case_list)
    valid_k2_list.append(valid_k2_single_case_list)