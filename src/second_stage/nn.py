import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad
import numpy as np

from .config import mynet_mlp, flux_mlp
from .parser import args, device

def nth_derivative(f, wrt, n):
    for i in range(n):
        grads = grad(f, wrt, create_graph=True, allow_unused=True)[0]
        f = grads
        if grads is None:
            print('bad grad')
            return torch.tensor(0.)
    return grads

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

class FluxPredictNet(nn.Module):
    # initializers
    def __init__(self, mlp_layers):
        super(FluxPredictNet, self).__init__()
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
    def forward(self, ni,Gl,Gu,Gr,Gd,t):
        he = torch.cat((ni,Gl,Gu,Gr,Gd,t), 1)
        for l, m in enumerate(self.layers):
            he = m(he)
            if l!=len(self.layers)-1:
                # he = F.sigmoid(he)
                # he = F.relu(he)
                he = F.tanh(he)
        
        # output layer
        # tanh
        he = F.tanh(he)

        #sigmoid
        # he = F.sigmoid(he)
        # he = (2*he)-1

        return he

def annealing_linear(start, end, pct):
    return start + pct * (end-start)


def annealing_cos(start, end, pct):
    "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
    cos_out = np.cos(np.pi * pct) + 1
    return end + (start-end)/2 * cos_out

class OneCycleScheduler(object):
    """
    (0, pct_start) -- linearly increase lr
    (pct_start, 1) -- cos annealing
    """
    def __init__(self, lr_max, div_factor=25., pct_start=0.3):
        super(OneCycleScheduler, self).__init__()
        self.lr_max = lr_max
        self.div_factor = div_factor
        self.pct_start = pct_start
        self.lr_low = self.lr_max / self.div_factor
    
    def step(self, pct):
        # pct: [0, 1]
        if pct <= self.pct_start:
            return annealing_linear(self.lr_low, self.lr_max, pct / self.pct_start)

        else:
            return annealing_cos(self.lr_max, self.lr_low / 1e4, (
                pct - self.pct_start) / (1 - self.pct_start))

def flat(x):
    m = x.shape[0]
    return [x[i] for i in range(m)]

def interior_loss(input_batch,kappa=4.0675706e-18, k_x=1, k_t=1, stress_range = 1):
    x=input_batch[:,0:1]
    t=input_batch[:,1:2]
    L=input_batch[:,2:3]
    W=input_batch[:,3:4]
    G=input_batch[:,4:5]
    k1=input_batch[:,5:6]
    k2=input_batch[:,6:7]
    k_x =k_x/L.detach() # input k_x = 1/maxLength, input L = wireLength/maxLength, real k_x = 1/((wireLength/maxLength)*maxLength) = 1/maxLength / (wireLength/maxlength)
    u = mynet(x,t,L,W,G,k1,k2)
    #u = [v[0], v[1]]
    u_t = nth_derivative(flat(u), wrt=t, n=1)
    u_x = nth_derivative(flat(u), wrt=x, n=1)
    u_xx = nth_derivative(flat(u_x), wrt=x, n=1)
    loss = stress_range*u_t*k_t - stress_range*kappa*u_xx*(k_x**2)
    loss = (loss**2).mean()
    # w = torch.tensor(0.01/np.pi)
    # f = u_t + u*u_x - w*u_xx
    return loss

def boundary_loss(input_batch, k_x=1, stress_range = 1, max_G=1, max_k=1, is_left=True):
    x=input_batch[:,0:1]
    t=input_batch[:,1:2]
    L=input_batch[:,2:3]
    W=input_batch[:,3:4]
    G=input_batch[:,4:5]
    k1=input_batch[:,5:6]
    k2=input_batch[:,6:7]
    k_x =k_x/L.detach() # input k_x = 1/maxLength, input L = wireLength/maxLength, real k_x = 1/((wireLength/maxLength)*maxLength) = 1/maxLength / (wireLength/maxlength)
    u = mynet(x,t,L,W,G,k1,k2)
    u_x = nth_derivative(flat(u), wrt=x, n=1)
    G = G.detach()*max_G # G:[-1,1]
    if is_left:
        k = k1.detach()*max_k
    else:
        k = k2.detach()*max_k
    # loss = u_x*(k_x*stress_range)+G - k
    loss = u_x+G/(k_x*stress_range) - k/(k_x*stress_range)
    loss = (loss**2).mean()
    return loss

def initial_loss(input_batch):
    x=input_batch[:,0:1]
    t=input_batch[:,1:2]
    L=input_batch[:,2:3]
    W=input_batch[:,3:4]
    G=input_batch[:,4:5]
    k1=input_batch[:,5:6]
    k2=input_batch[:,6:7]
    u = mynet(x,t,L,W,G,k1,k2)
    loss = u
    loss = (loss**2).mean()
    return loss

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

mynet.load_state_dict(torch.load(args.model_path, map_location=device))

mynet = mynet.to(device)
mynet.eval()

flux_predictor = FluxPredictNet(flux_mlp)
flux_predictor = flux_predictor.to(device)

optimizer = optim.Adam(flux_predictor.parameters(), lr=args.lr, weight_decay=args.weight_decay)
if(args.lr_pct >= 0):
    scheduler = OneCycleScheduler(lr_max=args.lr, div_factor=args.lr_div, pct_start=args.lr_pct)
# criterion = nn.MSELoss()