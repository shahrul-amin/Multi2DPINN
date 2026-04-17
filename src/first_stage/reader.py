import numpy as np
from scipy.io import loadmat
import re

from .config import e, Z, rou, Omega
from .generator import training_list, max_length, max_G, max_k, stress_max, stress_min
from .parser import args

# params for domain
stress_range = stress_max - stress_min

# Read training data and convert to input arrays
idx_MSE_list = []
truth_MSE_list = []
idx_MSE = np.empty([0,7])
truth_MSE = np.empty([0,1])
# file_num = 0
for file in training_list:
    # file_num += 1
    # print(file_num)
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
        G = e * Z * rou * J / Omega
        G_list.append(G)

    # Stress
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

        idx = [X,T,L,W,G,k1,k2]

        idx_MSE_single_segment = [np.expand_dims(col[:,:].flatten(), axis=-1) for col in idx] # ALL POINTS
        idx_MSE_single_segment = np.hstack(idx_MSE_single_segment)
        truth_MSE_single_segment = np.expand_dims(truth.T.flatten(), axis=-1)

        idx_MSE_single_case = np.vstack([idx_MSE_single_case, idx_MSE_single_segment])
        truth_MSE_single_case = np.vstack([truth_MSE_single_case, truth_MSE_single_segment])

    idx_MSE_list.append(idx_MSE_single_case)
    truth_MSE_list.append(truth_MSE_single_case)

    # plt.show()
    # mpl.use('Agg')

    # # Plot k values for all segments
    # for i in range(n_segments):
    #     plt.plot(k1_list[i], label = f"k1_{i}")
    #     plt.plot(k2_list[i], label = f"k2_{i}")
    # plt.legend()
    # plt.show()
idx_MSE = np.vstack(idx_MSE_list)
truth_MSE = np.vstack(truth_MSE_list)