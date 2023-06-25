import numpy as np

import torch.nn as nn
from surrogate_model.MFDNN import MFDNN
from surrogate_model.HK import HK
import matplotlib.pyplot as plt
from pyDOE import lhs

"""
Test_function_3 can be found here
Xiong, F., Ren, C., Mo, B., Li, C., & Hu, X. (2023). A new adaptive multi-fidelity metamodel method using meta-learning and Bayesian deep learning. Structural and Multidisciplinary Optimization, 66(3), 58.
[Link] https://link.springer.com/article/10.1007/s00158-023-03518-8
"""
in_dim=1
out_dim=1
def LF_function(x): # low-fidelity function
    return np.sin(8 * np.pi * x)

def MF_function(x): # MF function does not exist in the refrence paper, but is arbitrariliy defined
    return LF_function(x)**2

def HF_function(x): # high-fidelity function
    return (x-np.sqrt(2)) * LF_function(x)**2

LF_x = lhs(in_dim, samples=60, criterion='maximin')
MF_x = lhs(in_dim, samples=30, criterion='maximin')
HF_x = lhs(in_dim, samples=14, criterion='maximin')

# LF_x = np.linspace(0, 1, 11).reshape(-1,1)
# HF_x = np.array([0, 0.4, 0.6, 1]).reshape(-1,1)

LF_y = LF_function(LF_x)
MF_y = MF_function(MF_x)
HF_y = HF_function(HF_x)

hk = HK(x=[LF_x, MF_x, HF_x], y=[LF_y, MF_y, HF_y], n_pop=[100,100,100], n_gen=[100,100,100], HKtype="r")
hk.fit()

criterion_ = nn.MSELoss()
mfdnn = MFDNN(input_dim=in_dim, output_dim=out_dim)
mfdnn.add_fidelity(hidden_layers=[25, 25], activation="Tanh", criterion=criterion_, lr=1e-3, epochs=3000)
mfdnn.add_fidelity(hidden_layers=[25, 25], activation="Tanh", criterion=criterion_, lr=1e-3, epochs=3000)
mfdnn.add_fidelity(hidden_layers=[25, 25], activation="Tanh", criterion=criterion_, lr=1e-3, epochs=3000)

mfdnn.fit(train_x=[LF_x, MF_x, HF_x], train_y=[LF_y, MF_y, HF_y])

hfdnn = MFDNN(input_dim=in_dim, output_dim=out_dim)
hfdnn.add_fidelity(hidden_layers=[25, 25], activation="Tanh", criterion=criterion_, lr=1e-3, epochs=3000)

hfdnn.fit(train_x=[HF_x], train_y=[HF_y])

test_x = np.linspace(0, 1, 101).reshape(-1, 1)
pred_LF_y = mfdnn.predict(test_x, pred_fidelity=0)
pred_HF_y = mfdnn.predict(test_x, pred_fidelity=2)

pred_HF_y_HK = hk.predict(test_x, pred_fidelity=2, return_std=False)

fig, ax = plt.subplots(dpi=300)
ax.plot(test_x, HF_function(test_x), c='k', ls='--', label="Ground truth")
ax.scatter(mfdnn.train_x[0],mfdnn.train_y[0], c='C0', label='LF')
ax.scatter(mfdnn.train_x[2],mfdnn.train_y[2], c='C1', label='HF')
ax.plot(test_x, pred_LF_y, c='C0', label="LF pred")
ax.plot(test_x, pred_HF_y, c='C1', label="HF pred")
ax.plot(test_x, hfdnn.predict(test_x), c='C2', label="Only HF pred")
ax.plot(test_x, pred_HF_y_HK, c='C3', label="RHK")
ax.legend()

plt.show()