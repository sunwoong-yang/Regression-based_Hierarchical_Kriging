import numpy as np

import torch.nn as nn
from surrogate_model.MFDNN import MFDNN
from surrogate_model.HK import HK
import matplotlib.pyplot as plt
from pyDOE import lhs

"""
Test_function_8 can be found here
Xiong, F., Ren, C., Mo, B., Li, C., & Hu, X. (2023). A new adaptive multi-fidelity metamodel method using meta-learning and Bayesian deep learning. Structural and Multidisciplinary Optimization, 66(3), 58.
[Link] https://link.springer.com/article/10.1007/s00158-023-03518-8
However, LF and MF functions in the above ref are erroneous. Therefore, in this code, sigma from i=1 to 20 is changed to "i=1 ~ 19"
"""
in_dim = 20
out_dim = 1
def LF_function(x):
    y = 0.5 * (x[:, 0] - 1)**2
    for i in range(1, in_dim):
        y += 0.6 * (x[:, i] - x[:, i-1])**2
    for i in range(0, in_dim - 1):
        y += -0.5 * x[:, i] * x[:, i+1]
    return y

def MF_function(x):
    y = 0.8 * (x[:, 0] - 1)**2
    for i in range(1, in_dim):
        y += 0.8 * (x[:, i] - x[:, i-1])**2
    for i in range(0, in_dim - 1):
        y += -0.2 * x[:, i] * x[:, i+1]
    return y

def HF_function(x):
    y = 1.0 * (x[:, 0] - 1)**2
    for i in range(1, in_dim):
        y += 1.0 * (x[:, i] - x[:, i-1])**2
    return y


LF_x = lhs(in_dim, samples=300, criterion='maximin') * 1 + 0.5
MF_x = lhs(in_dim, samples=200, criterion='maximin') * 1 + 0.5
HF_x = lhs(in_dim, samples=100, criterion='maximin') * 1 + 0.5

LF_y = LF_function(LF_x).reshape(-1,1)
MF_y = MF_function(MF_x).reshape(-1,1)
HF_y = HF_function(HF_x).reshape(-1,1)

hk = HK(x=[LF_x, MF_x, HF_x], y=[LF_y, MF_y, HF_y], n_pop=[100,100,100], n_gen=[100,100,100], HKtype="r")
hk.fit(history=True)

criterion_ = nn.MSELoss()
mfdnn = MFDNN(input_dim=in_dim, output_dim=out_dim)
mfdnn.add_fidelity(hidden_layers=[20, 20], activation="GELU", criterion=criterion_, lr=1e-3, epochs=3000)
mfdnn.add_fidelity(hidden_layers=[15, 15], activation="GELU", criterion=criterion_, lr=1e-3, epochs=3000)
mfdnn.add_fidelity(hidden_layers=[10, 10], activation="GELU", criterion=criterion_, lr=1e-3, epochs=3000)

mfdnn.fit(train_x=[LF_x, MF_x, HF_x], train_y=[LF_y, MF_y, HF_y])

hfdnn = MFDNN(input_dim=in_dim, output_dim=out_dim)
hfdnn.add_fidelity(hidden_layers=[10, 10], activation="GELU", criterion=criterion_, lr=1e-3, epochs=3000)

hfdnn.fit(train_x=[HF_x], train_y=[HF_y])

test_x = lhs(in_dim, samples=100, criterion='maximin') * 1 + 0.5

pred_LF_y = mfdnn.predict(test_x, pred_fidelity=0)
pred_MF_y = mfdnn.predict(test_x, pred_fidelity=1)
pred_HF_y = mfdnn.predict(test_x, pred_fidelity=2)

pred_HF_y_HK = hk.predict(test_x, pred_fidelity=2, return_std=False)

fig, ax = plt.subplots(dpi=300)
# ax.scatter(HF_function(test_x), pred_LF_y, edgecolors='C0', label="LF", facecolors='none')
# ax.scatter(HF_function(test_x), pred_MF_y, edgecolors='C1', label="MF", facecolors='none')
ax.scatter(HF_function(test_x), pred_HF_y, edgecolors='C2', label="HF", facecolors='none')
# ax.scatter(HF_function(test_x), hfdnn.predict(test_x), edgecolors='C3', label="Only HF pred", facecolors='none')
ax.scatter(HF_function(test_x),pred_HF_y_HK, edgecolors='C4', label="RHK", facecolors='none')
lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
ax.plot(lims, lims, '--k')
ax.set_xlim(lims)
ax.set_ylim(lims)

ax.legend(fontsize=15, frameon=False)
plt.show()
