import random
import csv
import re

import numpy as np
from scipy.io import loadmat

from . import config
from .parser import args

# Generate trainig and testing datasets
if args.read_csv:
    # read dataset from csv start
    reader = csv.reader(open(args.sample_dir+'/training.csv', "r"), delimiter=",")
    training_list = list(reader)
    training_list = [int(file[0]) for file in training_list]
    random.shuffle(training_list)

    reader = csv.reader(open(args.sample_dir+'/testing.csv', "r"), delimiter=",")
    testing_list = list(reader)
    testing_list = [int(file[0]) for file in testing_list]
    random.shuffle(testing_list)

    reader = csv.reader(open(args.sample_dir+'/statistics.csv', "r"), delimiter=",")
    statistics_list = list(reader)
    statistics_list = [float(value) for value in statistics_list[1]]
    max_time, max_length, max_G, max_k, stress_max, stress_min = statistics_list
    # read dataset from csv end
else:
    files_list = [i for i in range(config.total_cases)]
    # random.shuffle(files_list)
    training_list = files_list[:config.training_cases]
    testing_list = files_list[config.training_cases:]
    np.savetxt(args.run_dir+"/training.csv", training_list, fmt='%i', delimiter=',')
    np.savetxt(args.run_dir+"/testing.csv", testing_list, fmt='%i', delimiter=',')

    max_G = 0.0
    max_k = 0.0
    max_length = 0.0
    max_time = config.time_length
    stress_max = 0.0
    stress_min = 0.0
    All_k = np.empty(1)
    for file in files_list:
        segment_length_list = []
        G_list = []
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
                max_length = max(max_length, current_segment_length)

        # Read .mat file for stress results and current
        data_mat = loadmat(file_path+".mat")
        current = data_mat['J']
        stress = data_mat['sVC']

        # Current density
        for J in current[0]:
            G = config.e * config.Z * config.rou * J / config.Omega
            G_list.append(G)
            max_G = max(max_G, abs(G))

        # Stress
        for i in range(len(segment_length_list)):
            truth = stress[i,0] # shape = [L_n_points, T_n_points]
            stress_max = max(stress_max, truth.max())
            stress_min = min(stress_min, truth.min())

            wire_interval = segment_length_list[i]/(truth.shape[0]-1)

            # calculate k1 and k2
            k1 = truth[:2, :]
            k2 = truth[-2:, :]
            k1 = (k1[1,:]-k1[0,:])/wire_interval
            k2 = (k2[1,:]-k2[0,:])/wire_interval
            k1 += G_list[i]
            k2 += G_list[i]
            All_k = np.hstack([All_k, k1, k2])

            max_k =  max(max_k, abs(k1).max(), abs(k2).max())

    # Plot all k values from lowest to largest
    # All_k.sort()
    # plt.plot(All_k, label = f"All k values")
    # plt.show()

    with open(args.run_dir+"/statistics.csv", "w", newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(["max_time", "max_length", "max_G", "max_k", "stress_max", "stress_min"])
        writer.writerow([max_time, max_length, max_G, max_k, stress_max, stress_min])

# training_list = training_list[:10]