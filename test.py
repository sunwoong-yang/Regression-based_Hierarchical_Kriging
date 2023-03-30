git_Test2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from surrogate_model.MLP import MLP
from surrogate_model.DE import DeepEnsemble
from surrogate_model.GPR import GPR
from surrogate_model.GPRs import GPRs
from data_mining.DM import DM


from data_mining.optimize import optimize
import matplotlib.pyplot as plt
from PrePost.PrePost import *


class ToyProblem(torch.utils.data.Dataset):
    def __init__(self, size=1000):
        self.size = size
        self.x = torch.rand(size, 1)
        self.y = torch.sin(2 * torch.pi * self.x)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# Instantiate the toy problem dataset
# dataset = ToyProblem(5)
# for x, y in dataset:
#     print(x.item(),y.item())
N_inp, N_out = 1, 2
# header, dataloader = csv2Dat(N_inp=N_inp, dir="ToyProblem3.csv")
header, train_x, train_y = csv2Num(N_inp=N_inp, dir="ToyProblem3.csv")
# header2, dataloader2 = csv2Dat(N_inp=1, dir="ToyProblem3.csv")


# def normalize(dataloader):
#     STD = Scaler
#     STD.fit(X=dataloader)
#
# normalize(dataloader)
#
# a

# Create a data loader for the dataset
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=dataset.size, shuffle=True)

# dataset2 = ToyProblem(5)
# dataloader2 = torch.utils.data.DataLoader(dataset2, batch_size=dataset.size, shuffle=True)




mlp = MLP(N_inp, [30, 30, 30], N_out, nn.ReLU())
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
mlp = mlp.to(device)

criterion_ = nn.MSELoss()
optimizer_ = optim.Adam(mlp.parameters(), lr=1e-3)

# mlp.fit(dataloader, 1000, criterion_, optimizer_)
mlp.fit(train_x, train_y, 1000, criterion_, optimizer_)


DE = DeepEnsemble(N_inp, [30, 30, 30], N_out, nn.ReLU(), num_models=5)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
DE = DE.to(device)

criterion_ = nn.MSELoss()
optimizer_ = optim.Adam(DE.parameters(), lr=1e-3)

DE.fit(train_x, train_y, 1000, optimizer_)


# gpr = GPR()
# gpr.fit(train_x, train_y)

gprs = GPRs()
gprs.fit(train_x, train_y)

# DE_SA = DM(DE)
# MLP_SA = DM(mlp)
# GPR_SA = DM(gpr)
# print(DE_SA.Sobol([[0, 1]], 1024))
# print(DE_SA.ANOVA([[0, 1]], 1024))
# print(MLP_SA.Sobol([[0, 1]], 1024))
# print(MLP_SA.ANOVA([[0, 1]], 1024))
# print(GPR_SA.Sobol([[0, 1]], 1024))
# print(GPR_SA.ANOVA([[0, 1]], 1024))

X_test = np.linspace(0, 1, 100).reshape(-1, 1)

Y_de, std, alea_std, epis_std = DE.predict(X_test, return_var=True)
Y_mlp = mlp.predict(X_test)
Y_s, STD_s = gprs.predict(X_test, True)

# Generating upper and lower bound of 68% confidence interval


plt.scatter(gprs.models[0].train_x, gprs.models[0].train_y, c='r', label='G1')
plt.plot(X_test, Y_s[:,0], c='r', label=f"{header[1][0]}")
plt.fill_between(X_test.flatten(), Y_s[:,0]-30*STD_s[:,0], Y_s[:,0]+30*STD_s[:,0], color='r', alpha=.5)
plt.fill_between(X_test.flatten(), Y_de[:,0]-30*epis_std[:,0], Y_de[:,0]+30*epis_std[:,0], color='b', alpha=.5)
plt.plot(X_test, Y_mlp[:,0], c='g', ls='-', label=f"MLP {header[1][0]}")
plt.legend()
plt.show()

plt.scatter(gprs.models[1].train_x, gprs.models[1].train_y, c='r', label='G2')
plt.plot(X_test, Y_s[:,1], c='r', label=f"GPR {header[1][1]}")
plt.fill_between(X_test.flatten(), Y_s[:,1]-30*STD_s[:,1], Y_s[:,1]+30*STD_s[:,1], color='r', alpha=.5)
plt.plot(X_test, Y_de[:,1], c='g', ls='--', label=f"DE {header[1][1]}")
plt.fill_between(X_test.flatten(), Y_de[:,1]-30*epis_std[:,1], Y_de[:,1]+30*epis_std[:,1], color='g', alpha=.5)
plt.plot(X_test, Y_mlp[:,1], c='b', ls='-', label=f"MLP {header[1][0]}")
plt.legend()
plt.show()