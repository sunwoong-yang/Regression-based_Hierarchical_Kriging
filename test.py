import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from surrogate_model.MLP import MLP
from surrogate_model.DE import DeepEnsemble
from surrogate_model.GPR import GPR
from surrogate_model.GPRs import GPRs
from surrogate_model.MFDNN import MFDNN
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



# N_inp, N_out = 1, 2
# inp_header, out_header, train_x, train_y = csv2Num(N_inp=N_inp, dir="ToyProblem3.csv")
N_inp, N_out = 1, 1
inp_header, out_header, train_x, train_y = csv2Num(N_inp=N_inp, dir="HF2.csv")
LF_inp_header, LF_out_header, LF_train_x, LF_train_y = csv2Num(N_inp=N_inp, dir="LF2.csv")
HF_inp_header, HF_out_header, HF_train_x, HF_train_y = csv2Num(N_inp=N_inp, dir="HF2.csv")
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




mlp = MLP(N_inp, [5, 5, 5], "GELU", N_out)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
mlp = mlp.to(device)

criterion_ = nn.MSELoss()
optimizer_ = optim.Adam(mlp.parameters(), lr=1e-3)

mlp.fit(train_x, train_y, 1000, criterion_, optimizer_)


DE = DeepEnsemble(N_inp, [5, 5, 5], "GELU", N_out, num_models=5)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
DE = DE.to(device)

criterion_ = nn.MSELoss()
optimizer_ = optim.Adam(DE.parameters(), lr=1e-3)

DE.fit(train_x, train_y, 3000, optimizer_)

gpr = GPR()
gpr.fit(train_x, train_y)

gprs = GPRs()
gprs.fit(train_x, train_y)

mfdnn = MFDNN(input_dim=1, output_dim=1)
criterion_ = nn.MSELoss()
mfdnn.add_fidelity(hidden_layers=[20, 20], activation="Tanh", criterion=criterion_, lr=1e-3, epochs=1000)
mfdnn.add_fidelity(hidden_layers=[10, 10], activation="Tanh", criterion=criterion_, lr=1e-3, epochs=1000)

mfdnn.fit(train_x=[LF_train_x, HF_train_x], train_y=[LF_train_y, HF_train_y])

mlp_hf = MLP(1, [10, 10], "GELU", 1)
mlp_hf = mlp_hf.to(device)
criterion_ = nn.MSELoss()
optimizer_ = optim.Adam(mlp_hf.parameters(), lr=1e-3)

mlp_hf.fit(HF_train_x, HF_train_y, 1000, criterion_, optimizer_)

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
Y_gpr, STD_gpr = gprs.predict(X_test, True)
Y_LF = mfdnn.predict(X_test, pred_fidelity=0)
Y_HF = mfdnn.predict(X_test, pred_fidelity=1)
Y_mlp_HF = mlp_hf.predict(X_test)

plt.scatter(gprs.models[0].train_x, gprs.models[0].train_y, c='k', label=f'Train data {out_header[0]}')
plt.plot(X_test, Y_gpr[:,0], c='r', label=f"GPR {out_header[0]}")
plt.plot(X_test, Y_mlp[:,0], c='g', ls='-', label=f"MLP {out_header[0]}")
plt.plot(X_test, Y_de[:,0], c='b', ls='--', label=f"DE {out_header[0]}")
plt.fill_between(X_test.flatten(), Y_gpr[:,0]-30*STD_gpr[:,0], Y_gpr[:,0]+30*STD_gpr[:,0], color='r', alpha=.5)
plt.fill_between(X_test.flatten(), Y_de[:,0]-30*epis_std[:,0], Y_de[:,0]+30*epis_std[:,0], color='b', alpha=.5)
plt.legend()
plt.title("Y1")
plt.show()

# plt.scatter(gprs.models[1].train_x, gprs.models[1].train_y, c='k', label=f'Train data {out_header[1]}')
# plt.plot(X_test, Y_gpr[:,1], c='r', label=f"GPR {out_header[1]}")
# plt.plot(X_test, Y_mlp[:,1], c='g', label=f"MLP {out_header[1]}")
# plt.plot(X_test, Y_de[:,1], c='b', label=f"DE {out_header[1]}")
# plt.fill_between(X_test.flatten(), Y_gpr[:,1]-30*STD_gpr[:,1], Y_gpr[:,1]+30*STD_gpr[:,1], color='r', alpha=.5)
# plt.fill_between(X_test.flatten(), Y_de[:,1]-30*epis_std[:,1], Y_de[:,1]+30*epis_std[:,1], color='b', alpha=.5)
# plt.legend()
# plt.title("Y2")
# plt.show()
# function to sum to numbers

def HF(x):
    return (6 * x - 2) ** 2 * np.sin(12 * x - 4)
plt.scatter(mfdnn.train_x[0],mfdnn.train_y[0], c='b', label='LF')
plt.scatter(mfdnn.train_x[1],mfdnn.train_y[1], c='r', label='HF')
plt.plot(X_test, Y_LF, c='b', label="LF pred")
plt.plot(X_test, Y_HF, c='r', label="HF pred")
plt.plot(X_test, Y_mlp_HF, c='g', ls='--', label="Only HF ped")
plt.plot(X_test, HF(X_test), c='k', label="Ground truth")
plt.legend()
plt.show()


