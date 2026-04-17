import torch
import numpy as np
from scipy.io import loadmat
import re

from .reader import testing_list, max_length, max_G, max_k, stress_min, stress_range
from .config import T_n_points, e, Z, rou, Omega
from .parser import args, device

# Validation grid
node_segments_list = []
segment_endopoints_and_direction_list = []

flux_predictor_inputs_Ni_list = []
flux_predictor_inputs_Gl_list = []
flux_predictor_inputs_Gu_list = []
flux_predictor_inputs_Gr_list = []
flux_predictor_inputs_Gd_list = []
flux_predictor_inputs_T_list = []

boundary_X_list = []
boundary_T_list = []
boundary_L_list = []
boundary_W_list = []
boundary_G_list = []

valid_truth_list = []
valid_X_list = []
valid_T_list = []
valid_L_list = []
valid_W_list = []
valid_G_list = []
valid_k1_list = []
valid_k2_list = []
for file in testing_list:
    node_dict_single_case = {}
    node_count = 0
    node_segments_list_single_case = [] # connected segments in four derictions for all nodes [left, up, right, down]
    segment_endopoints_and_direction_single_case = [] # [(x1,y1),(x2,y2),(x3,y3),(x4,y4),horizontal:0 or vertical:1]

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

            node_0 = (float(rect_vertices[1]), float(rect_vertices[2]))
            node_1 = (node_0[0] + float(rect_vertices[4]), node_0[1] + float(rect_vertices[5]))

            if abs(float(rect_vertices[4])) > abs(float(rect_vertices[5])): # horizontal segment
                node_0_2nd = (node_0[0], node_0[1] + float(rect_vertices[5]))
                node_1_2nd = (node_0[0] + float(rect_vertices[4]), node_0[1])
                node_0 = ((node_0[0]+node_0_2nd[0])/2, (node_0[1]+node_0_2nd[1])/2)
                node_1 = ((node_1[0]+node_1_2nd[0])/2, (node_1[1]+node_1_2nd[1])/2)
                if(float(rect_vertices[4]) > 0):
                    segment_endopoints_and_direction_single_case.append([node_0,node_1,0])
                else:
                    segment_endopoints_and_direction_single_case.append([node_1,node_0,0])
            else: # vertical segment
                node_0_2nd = (node_0[0] + float(rect_vertices[4]), node_0[1])
                node_1_2nd = (node_0[0], node_0[1] + float(rect_vertices[5]))
                node_0 = ((node_0[0]+node_0_2nd[0])/2, (node_0[1]+node_0_2nd[1])/2)
                node_1 = ((node_1[0]+node_1_2nd[0])/2, (node_1[1]+node_1_2nd[1])/2)
                if(float(rect_vertices[5]) > 0):
                    segment_endopoints_and_direction_single_case.append([node_0,node_1,1])
                else:
                    segment_endopoints_and_direction_single_case.append([node_1,node_0,1])
            
            if (node_0 not in node_dict_single_case):
                node_dict_single_case[node_0] = node_count
                node_count += 1
                node_segments_list_single_case.append([None, None, None, None]) # segment numbers arranged in the order of: [left, up, right, down]

            if (node_1 not in node_dict_single_case):
                node_dict_single_case[node_1] = node_count
                node_count += 1
                node_segments_list_single_case.append([None, None, None, None]) # segment numbers arranged in the order of: [left, up, right, down]

            current_segment_n = len(segment_length_list)-1
            node_0_n = node_dict_single_case[node_0]
            node_1_n = node_dict_single_case[node_1]
            if(abs(float(rect_vertices[4])) > abs(float(rect_vertices[5]))): # horizontal rectangle
                if(float(rect_vertices[4])>0): # pointing right
                    node_segments_list_single_case[node_0_n][2] = current_segment_n # current segment is attached to the right of node_0
                    node_segments_list_single_case[node_1_n][0] = current_segment_n # current segment is attached to the left of node_1
                else:
                    node_segments_list_single_case[node_0_n][0] = current_segment_n
                    node_segments_list_single_case[node_1_n][2] = current_segment_n
            else: # vertical rectangle
                if(float(rect_vertices[5])>0): # pointing up
                    node_segments_list_single_case[node_0_n][1] = current_segment_n # current segment is attached to the upside of node_0
                    node_segments_list_single_case[node_1_n][3] = current_segment_n # current segment is attached to the downside of node_1
                else:
                    node_segments_list_single_case[node_0_n][3] = current_segment_n
                    node_segments_list_single_case[node_1_n][1] = current_segment_n

    # Read .mat file for stress results and current
    data_mat = loadmat(file_path+".mat")
    current = data_mat['J']
    stress = data_mat['sVC']       

    n_segments = len(segment_length_list)
    n_nodes = len(node_segments_list_single_case)
    L_list = [L/max_length for L in segment_length_list]

    # Current density
    for J in current[0]:
        G = e * Z * rou * J / Omega
        G_list.append(G)


    # Generate input for flux predictor net
    flux_predictor_inputs_Ni_single_case_list = []
    flux_predictor_inputs_Gl_single_case_list = []
    flux_predictor_inputs_Gu_single_case_list = []
    flux_predictor_inputs_Gr_single_case_list = []
    flux_predictor_inputs_Gd_single_case_list = []
    flux_predictor_inputs_T_single_case_list = []
    for i in range(n_nodes):
        connected_segments = node_segments_list_single_case[i]
        if(sum(segment is not None for segment in connected_segments) == 1): # this is a boundary node, not an internal node, no need to predict k
            continue
        Ni = (i+1)/n_nodes # Number of internal nodes: 0.1, 0.2, ... , 0.9

        G_l = 0 if connected_segments[0]==None else G_list[connected_segments[0]]/max_G
        G_u = 0 if connected_segments[1]==None else G_list[connected_segments[1]]/max_G
        G_r = 0 if connected_segments[2]==None else G_list[connected_segments[2]]/max_G
        G_d = 0 if connected_segments[3]==None else G_list[connected_segments[3]]/max_G

        T = np.linspace(0,1, num=T_n_points, endpoint=True)

        Ni = Ni*np.ones([T.shape[0], 1])
        G_l = G_l*np.ones([T.shape[0], 1])
        G_u = G_u*np.ones([T.shape[0], 1])
        G_r = G_r*np.ones([T.shape[0], 1])
        G_d = G_d*np.ones([T.shape[0], 1])
        T = np.expand_dims(T, axis=-1)

        Ni = torch.FloatTensor(Ni).to(device)
        G_l = torch.FloatTensor(G_l).to(device)
        G_u = torch.FloatTensor(G_u).to(device)
        G_r = torch.FloatTensor(G_r).to(device)
        G_d = torch.FloatTensor(G_d).to(device)
        T = torch.FloatTensor(T).to(device)

        flux_predictor_inputs_Ni_single_case_list.append(Ni)
        flux_predictor_inputs_Gl_single_case_list.append(G_l)
        flux_predictor_inputs_Gu_single_case_list.append(G_u)
        flux_predictor_inputs_Gr_single_case_list.append(G_r)
        flux_predictor_inputs_Gd_single_case_list.append(G_d)
        flux_predictor_inputs_T_single_case_list.append(T)

    node_segments_list.append(node_segments_list_single_case)
    segment_endopoints_and_direction_list.append(segment_endopoints_and_direction_single_case)

    flux_predictor_inputs_Ni_list.append(flux_predictor_inputs_Ni_single_case_list)
    flux_predictor_inputs_Gl_list.append(flux_predictor_inputs_Gl_single_case_list)
    flux_predictor_inputs_Gu_list.append(flux_predictor_inputs_Gu_single_case_list)
    flux_predictor_inputs_Gr_list.append(flux_predictor_inputs_Gr_single_case_list)
    flux_predictor_inputs_Gd_list.append(flux_predictor_inputs_Gd_single_case_list)
    flux_predictor_inputs_T_list.append(flux_predictor_inputs_T_single_case_list)


    # Generate input for stress net
    boundary_X_single_case_list = []
    boundary_T_single_case_list = []
    boundary_L_single_case_list = []
    boundary_W_single_case_list = []
    boundary_G_single_case_list = []
    for i in range(n_segments):
        truth = stress[i,0] # shape = [L_n_points, T_n_points]

        wire_interval = segment_length_list[i]/(truth.shape[0]-1)

        # Genrate indexes for colocation points [x,t,L,W,G,k1,k2]
        truth = 2*((truth-stress_min) / stress_range)-1 # convert to [-1, 1]
        x = np.linspace(0,1, num=2, endpoint=True) # only generate points on boudaries, i.e. x = 0 and 1
        t = np.linspace(0,1, num=truth.shape[1], endpoint=True)
        X, T = np.meshgrid(x, t) # shape=[T_n_points, L_n_points]
        L = np.tile(L_list[i], [X.shape[0],X.shape[1]])
        W = np.tile(1.0, [X.shape[0],X.shape[1]])
        G = np.tile(G_list[i]/max_G, [X.shape[0],X.shape[1]])

        X = np.expand_dims(X.flatten(), axis=-1)
        T = np.expand_dims(T.flatten(), axis=-1)
        L = np.expand_dims(L.flatten(), axis=-1)
        W = np.expand_dims(W.flatten(), axis=-1)
        G = np.expand_dims(G.flatten(), axis=-1)

        X = torch.FloatTensor(X).to(device)
        T = torch.FloatTensor(T).to(device)
        L = torch.FloatTensor(L).to(device)
        W = torch.FloatTensor(W).to(device)
        G = torch.FloatTensor(G).to(device)

        boundary_X_single_case_list.append(X)
        boundary_T_single_case_list.append(T)
        boundary_L_single_case_list.append(L)
        boundary_W_single_case_list.append(W)
        boundary_G_single_case_list.append(G)

    boundary_X_list.append(boundary_X_single_case_list)
    boundary_T_list.append(boundary_T_single_case_list)
    boundary_L_list.append(boundary_L_single_case_list)
    boundary_W_list.append(boundary_W_single_case_list)
    boundary_G_list.append(boundary_G_single_case_list)


    # Generate validation grid
    valid_truth_single_case_list = []
    valid_X_single_case_list = []
    valid_T_single_case_list = []
    valid_L_single_case_list = []
    valid_W_single_case_list = []
    valid_G_single_case_list = []
    valid_k1_single_case_list = []
    valid_k2_single_case_list = []
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