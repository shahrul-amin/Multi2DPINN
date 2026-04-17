import torch
import os
from pathlib import Path

import argparse
from datetime import datetime
from shutil import copyfile

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def mkdirs(*paths):
    if isinstance(paths, list) or isinstance(paths, tuple):
        for path in paths:
            mkdir(path)
    else:
        raise ValueError

class Parser(argparse.ArgumentParser):
  def __init__(self): 
    super(Parser, self).__init__(description='EM 1D Simnet Solver')
    self.add_argument('--sample-dir', type=str, default="./run/second_stage/EMdataset_10seg_1n2/", help='directory to save output') # tested on 8000-10000 cases in the dataset
    self.add_argument('--data-path', type=str, default="/data/test_trees/", help='path to data')  # server
    self.add_argument('--model-path', type=str, default="./ckpt/trial_function_mlp_20.pkl", help='path to trained mynet model') # fisrt stage trained on 8000 ten-segment straight lines with layer configuration [7, 256, 512, 1024, 512, 256, 1]    
    
    self.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    self.add_argument('--lr-div', type=float, default=10., help='lr div factor to get the initial lr')
    self.add_argument('--lr-pct', type=float, default=0.3, help='percentage to reach the maximun lr, which is args.lr')
    self.add_argument('--weight-decay', type=float, default=0., help="weight decay")

    self.add_argument('--n-epochs', type=int, default=1000, help='Toatal training epochs')
    self.add_argument('--cuda', type=int, default=0, choices=[0, 1, 2, 3], help='cuda index')
    self.add_argument('--save-freq', type=int, default=100, help='Epochs between plot')
    self.add_argument('--fig-format', type=str, default='1d2d3d', help='Figure plot format, 1d, 2d, 3d or mixed, e.g. 2d3d')

    self.add_argument('--testcase', type=int, default=0, help='Testcase number')
    self.add_argument('--timer', action='store_true', default=False, help='Print the training time')

  def parse(self):
    args = self.parse_args()

    # Create output dir
    dt = datetime.now()
    args.date = dt.strftime("%Y%m%d%H%M%S")
    args.hparams = f'EM1d_Epochs{args.n_epochs}_TreeModel_{args.date}'
    args.run_dir = args.sample_dir + '/' + args.hparams
    args.code_bkup_dir = args.run_dir + '/code_bkup'
    mkdirs(args.run_dir, args.code_bkup_dir)

    # backup the code of current model
    code_path_src = Path(__file__).resolve()
    code_path_dst = Path(args.code_bkup_dir) / code_path_src.name
    copyfile(code_path_src, code_path_dst)

    # print('Arguments:')
    # pprint(vars(args))
    # with open(args.run_dir + "/args.txt", 'w') as args_file:
    #     json.dump(vars(args), args_file, indent=4)

    return args

args = Parser().parse()

device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}")