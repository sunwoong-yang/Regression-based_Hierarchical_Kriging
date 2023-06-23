import numpy as np

import torch.nn as nn
from surrogate_model.MFDNN import MFDNN
from surrogate_model.HK import HK
import matplotlib.pyplot as plt
from pyDOE import lhs

"""
Test_function_2 can be found here
Xiong, F., Ren, C., Mo, B., Li, C., & Hu, X. (2023). A new adaptive multi-fidelity metamodel method using meta-learning and Bayesian deep learning. Structural and Multidisciplinary Optimization, 66(3), 58.
[Link] https://link.springer.com/article/10.1007/s00158-023-03518-8
"""
in_dim=1
out_dim=1
def LF_function(x): # high-fidelity function
    return HF_function(x) + 0.3 - 0.03 * (x-7)**2

def MF_function(x): # high-fidelity function
    return HF_function(x) + 0.3 - 0.03 * (x-3)**2

def HF_function(x): # high-fidelity function
    return - np.sin(x) - np.exp(x / 100) + 10


LF_x = lhs(in_dim, samples=20, criterion='maximin') * 10
MF_x = lhs(in_dim, samples=10, criterion='maximin') * 10
HF_x = lhs(in_dim, samples=5, criterion='maximin') * 10

LF_y = LF_function(LF_x).reshape(-1,1)
MF_y = MF_function(MF_x).reshape(-1,1)
HF_y = HF_function(HF_x).reshape(-1,1)

hk = HK(x=[LF_x, MF_x, HF_x], y=[LF_y, MF_y, HF_y], n_pop=[100,100,100], n_gen=[100,100,100], HKtype="r")

hk.fit()

criterion_ = nn.MSELoss()
mfdnn = MFDNN(input_dim=in_dim, output_dim=out_dim)
mfdnn.add_fidelity(hidden_layers=[10, 10], activation="GELU", criterion=criterion_, lr=1e-3, epochs=3000)
mfdnn.add_fidelity(hidden_layers=[10, 10], activation="GELU", criterion=criterion_, lr=1e-3, epochs=3000)
mfdnn.add_fidelity(hidden_layers=[10, 10], activation="GELU", criterion=criterion_, lr=1e-3, epochs=3000)

mfdnn.fit(train_x=[LF_x, MF_x, HF_x], train_y=[LF_y, MF_y, HF_y])

hfdnn = MFDNN(input_dim=in_dim, output_dim=out_dim)
hfdnn.add_fidelity(hidden_layers=[10, 10], activation="GELU", criterion=criterion_, lr=1e-3, epochs=3000)

hfdnn.fit(train_x=[HF_x], train_y=[HF_y])

test_x = np.linspace(0, 1, 101).reshape(-1, 1) * 10

pred_LF_y = mfdnn.predict(test_x, pred_fidelity=0)
pred_MF_y = mfdnn.predict(test_x, pred_fidelity=1)
pred_HF_y = mfdnn.predict(test_x, pred_fidelity=2)

pred_HF_y_HK = hk.predict(test_x, pred_fidelity=2, return_std=False)

fig, ax = plt.subplots(dpi=300)
ax.plot(test_x, HF_function(test_x), c='k', ls='--', label="Ground truth")
ax.scatter(mfdnn.train_x[0],mfdnn.train_y[0], c='C0', label='LF')
ax.scatter(mfdnn.train_x[1],mfdnn.train_y[1], c='C1', label='MF')
ax.scatter(mfdnn.train_x[2],mfdnn.train_y[2], c='C2', label='HF')
ax.plot(test_x, pred_LF_y, c='C0', label="LF pred")
ax.plot(test_x, pred_MF_y, c='C1', label="MF pred")
ax.plot(test_x, pred_HF_y, c='C2', label="HF pred")
ax.plot(test_x, hfdnn.predict(test_x), c='C3', label="Only HF pred")
ax.plot(test_x, pred_HF_y_HK, c='C4', label="RHK")
ax.legend()
plt.show()
