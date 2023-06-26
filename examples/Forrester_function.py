import numpy as np

import torch.nn as nn
import torch.optim as optim
from surrogate_model.MFDNN import MFDNN
from surrogate_model.HK import HK
import matplotlib.pyplot as plt

"""
3 level Forrester functions can be found here
Xiao, M., Zhang, G., Breitkopf, P., Villon, P., & Zhang, W. (2018). Extended Co-Kriging interpolation method based on multi-fidelity data. Applied Mathematics and Computation, 323, 120-131.
[Link] https://www.sciencedirect.com/science/article/pii/S0096300317307646
"""

def LF_function(x):
    return 0.5 * HF_function(x) + 10 * (x - 0.5) + 5

def MF_function(x):
    return 0.4 * HF_function(x) - x - 1

def HF_function(x):
    return ((6 * x - 2)**2) * np.sin(12 * x - 4)


LF_x = np.linspace(0, 1, 21).reshape(-1,1)
MF_x = np.linspace(0, 1, 11).reshape(-1,1)
HF_x = np.array([0, 0.4, 0.6, 1]).reshape(-1,1)

LF_y = LF_function(LF_x)
MF_y = MF_function(MF_x)
HF_y = HF_function(HF_x)

hk = HK(x=[LF_x, MF_x, HF_x], y=[LF_y, MF_y, HF_y], n_pop=[100,100,100], n_gen=[100,100,100], HKtype="r")
hk.fit(history=False)

criterion_ = nn.MSELoss()
mfdnn = MFDNN(input_dim=1, output_dim=1)
mfdnn.add_fidelity(hidden_layers=[10, 10], activation="Tanh", criterion=criterion_, lr=1e-3, epochs=3000)
mfdnn.add_fidelity(hidden_layers=[10, 10], activation="Tanh", criterion=criterion_, lr=1e-3, epochs=3000)
mfdnn.add_fidelity(hidden_layers=[10, 10], activation="Tanh", criterion=criterion_, lr=1e-3, epochs=3000)

hfdnn = MFDNN(input_dim=1, output_dim=1)
hfdnn.add_fidelity(hidden_layers=[10, 10], activation="Tanh", criterion=criterion_, lr=1e-3, epochs=3000)

mfdnn.fit(train_x=[LF_x, MF_x, HF_x], train_y=[LF_y, MF_y, HF_y])
hfdnn.fit(train_x=[HF_x], train_y=[HF_y])

test_x = np.linspace(0, 1, 100).reshape(-1, 1)
pred_LF_y = mfdnn.predict(test_x, pred_fidelity=0)
pred_MF_y = mfdnn.predict(test_x, pred_fidelity=1)
pred_HF_y = mfdnn.predict(test_x, pred_fidelity=2)

pred_LF_y_HK = hk.predict(test_x, pred_fidelity=0, return_std=False)
pred_MF_y_HK = hk.predict(test_x, pred_fidelity=1, return_std=False)
pred_HF_y_HK = hk.predict(test_x, pred_fidelity=2, return_std=False)

fig, ax = plt.subplots(dpi=300)
ax.plot(test_x, HF_function(test_x), c='k', ls='--', label="Ground truth", zorder=99)
ax.scatter(mfdnn.train_x[0],mfdnn.train_y[0], c='C0', label='LF')
ax.scatter(mfdnn.train_x[1],mfdnn.train_y[1], c='C1', label='MF')
ax.scatter(mfdnn.train_x[2],mfdnn.train_y[2], c='C2', label='HF')
# ax.plot(test_x, pred_LF_y, c='C0', label="LF pred")
# ax.plot(test_x, pred_MF_y, c='C1', label="MF pred")
ax.plot(test_x, pred_HF_y, c='C2', label="HF pred")
ax.plot(test_x, hfdnn.predict(test_x), c='C3', label="Only HF pred")
# ax.plot(test_x, pred_LF_y_HK, c='C4', label="LF RHK")
# ax.plot(test_x, pred_MF_y_HK, c='C5', label="MF RHK")
ax.plot(test_x, pred_HF_y_HK, c='C6', label="HF RHK")
ax.legend()

plt.show()