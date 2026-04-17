import math

time_interval = 1.e4
T_n_points = 101
time_length = time_interval * (T_n_points-1)
e = 1.69e-19
Ea=0.84*e
kB=1.3806e-23
T=353.0
D0 = 7.5e-8
Da=D0*math.exp(-Ea/(kB*T))
B = 1e11
Omega=1.182e-29
Z=10
rou = 3.e-08
kappa = Da*B*Omega/(kB*T)

# [X,Time,L,W,G,k1,k2]
mynet_mlp = [7, 256, 512, 1024, 512, 256, 1] # 7-layer
# mynet_mlp = [7, 256, 512, 1024, 2048, 2048, 2048, 2048, 1024, 512, 256, 1] # 12-layer

#[Number_of_node, Gl, Gu, Gr, Gd, Time]
flux_mlp = [6, 256, 512, 1024, 512, 256, 3]