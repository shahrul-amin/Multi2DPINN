import os
import csv
import random
from shutil import copyfile

from .parser import args

# read dataset from csv start
if not os.path.exists(args.sample_dir+'/training.csv'):
    copyfile(args.model_path[:args.model_path.rfind('/')]+'/training.csv', args.sample_dir+'/training.csv')
    copyfile(args.model_path[:args.model_path.rfind('/')]+'/testing.csv', args.sample_dir+'/testing.csv')
    copyfile(args.model_path[:args.model_path.rfind('/')]+'/statistics.csv', args.sample_dir+'/statistics.csv')

# reader = csv.reader(open(args.sample_dir+'/training.csv', "r"), delimiter=",")
# training_list = list(reader)
# training_list = [int(file[0]) for file in training_list]
# random.shuffle(training_list)

reader = csv.reader(open(args.sample_dir+'/testing.csv', "r"), delimiter=",")
testing_list = list(reader)
testing_list = [int(file[0]) for file in testing_list]
random.shuffle(testing_list)

reader = csv.reader(open(args.sample_dir+'/statistics.csv', "r"), delimiter=",")
statistics_list = list(reader)
statistics_list = [float(value) for value in statistics_list[1]]
max_time, max_length, max_G, max_k, stress_max, stress_min = statistics_list
# read dataset from csv end

# testing_list=[training_list[0]]
if(args.testcase > -1):
    testing_list=[args.testcase]
    print(args.testcase)

# params for domain
stress_range = stress_max - stress_min

k_x = 1/max_length
k_t = 1/max_time # 3600*24*365*10=315360000