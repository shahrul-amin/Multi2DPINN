import random

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim

from .config import mynet_mlp
from .reader import idx_MSE
from .parser import device, args

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

# mynet_mlp = [7, 256, 512, 512, 256, 1]
class Net(nn.Module):
    # initializers
    def __init__(self, mlp_layers):
        super(Net, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(mlp_layers)-1):
            self.layers.append(nn.Linear(mlp_layers[i], mlp_layers[i+1]))

        self.weight_init(mean=0.0, std=0.02)

    # weight_init
    def weight_init(self, mean, std):
        # for m in self._modules:
        for m in self.layers:
            normal_init(m, mean, std)

    # forward method
    def forward(self, x,t,L,W,G,k1,k2):
        he = torch.cat((x,t,L,W,G,k1,k2), 1)
        for l, m in enumerate(self.layers):
            he = m(he)
            if l!=len(self.layers)-1:
                he = F.relu(he)
                # he = F.tanh(he)

        return he

def MSE_loss(input_batch, truth_batch):
    x=input_batch[:,0:1]
    t=input_batch[:,1:2]
    L=input_batch[:,2:3]
    W=input_batch[:,3:4]
    G=input_batch[:,4:5]
    k1=input_batch[:,5:6]
    k2=input_batch[:,6:7]
    u = mynet(x,t,L,W,G,k1,k2)
    loss = u - truth_batch
    loss = (loss**2).mean()
    return loss

mynet = Net(mynet_mlp)
print(mynet)
mynet = mynet.to(device)

optimizer = optim.Adam(mynet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
MSE_batches = list(range(idx_MSE.shape[0]))
MSE_bz = int(idx_MSE.shape[0]/args.n_batches)
random.shuffle(MSE_batches)    