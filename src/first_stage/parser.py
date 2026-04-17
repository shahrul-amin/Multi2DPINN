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
    self.add_argument('--sample-dir', type=str, default="./run/first_stage/EMdataset_10seg_1n2", help='directory to save output')
    self.add_argument('--data-path', type=str, default="/data/EMdataset_10seg_1n2/", help='path to data') 

    self.add_argument('--read-csv', type=bool, default=False, help='Read training and testing data from .csv file')    
    self.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    self.add_argument('--weight-decay', type=float, default=0., help="weight decay")

    self.add_argument('--w-mse', type=float, default=1, help='Weight of MSE loss. (Ground Truth)')

    self.add_argument('--n-epochs', type=int, default=10000, help='Total training epochs')
    self.add_argument('--n-batches', type=int, default=10000, help='Number of batches in every epoch')
    self.add_argument('--cuda', type=int, default=0, choices=[0, 1, 2, 3], help='cuda index')
    self.add_argument('--save-freq', type=int, default=5, help='Model save frequency, i.e. epochs between model save and plot')

  def parse(self):
    args = self.parse_args()

    # Create output dir
    dt = datetime.now()
    args.date = dt.strftime("%Y%m%d%H%M%S")
    args.hparams = f'EM1d_Epochs{args.n_epochs}_{args.date}'
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