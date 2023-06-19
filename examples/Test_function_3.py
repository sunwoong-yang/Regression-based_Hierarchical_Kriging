import numpy as np

import torch.nn as nn
from surrogate_model.MFDNN import MFDNN
import matplotlib.pyplot as plt
from pyDOE import lhs

"""
Test_function_3 can be found here
Xiong, F., Ren, C., Mo, B., Li, C., & Hu, X. (2023). A new adaptive multi-fidelity metamodel method using meta-learning and Bayesian deep learning. Structural and Multidisciplinary Optimization, 66(3), 58.
[Link] https://link.springer.com/article/10.1007/s00158-023-03518-8
"""

def LF_function(x): # low-fidelity function
    return np.sin(8 * np.pi * x)

def HF_function(x): # high-fidelity function
    return (x-np.sqrt(2)) * LF_function(x)**2

LF_x = lhs(1, samples=60, criterion='maximin')
HF_x = lhs(1, samples=14, criterion='maximin')

# LF_x = np.linspace(0, 1, 11).reshape(-1,1)
# HF_x = np.array([0, 0.4, 0.6, 1]).reshape(-1,1)

LF_y = LF_function(LF_x)
HF_y = HF_function(HF_x)

criterion_ = nn.MSELoss()
mfdnn = MFDNN(input_dim=1, output_dim=1)
mfdnn.add_fidelity(hidden_layers=[25, 25], activation="Tanh", criterion=criterion_, lr=1e-3, epochs=3000)
mfdnn.add_fidelity(hidden_layers=[25, 25], activation="Tanh", criterion=criterion_, lr=1e-3, epochs=3000)

mfdnn.fit(train_x=[LF_x, HF_x], train_y=[LF_y, HF_y])

hfdnn = MFDNN(input_dim=1, output_dim=1)
hfdnn.add_fidelity(hidden_layers=[25, 25], activation="Tanh", criterion=criterion_, lr=1e-3, epochs=3000)

hfdnn.fit(train_x=[HF_x], train_y=[HF_y])

test_x = np.linspace(0, 1, 100).reshape(-1, 1)
pred_LF_y = mfdnn.predict(test_x, pred_fidelity=0)
pred_HF_y = mfdnn.predict(test_x, pred_fidelity=1)

fig, ax = plt.subplots(dpi=300)
ax.plot(test_x, HF_function(test_x), c='k', ls='--', label="Ground truth")
ax.scatter(mfdnn.train_x[0],mfdnn.train_y[0], c='C0', label='LF')
ax.scatter(mfdnn.train_x[1],mfdnn.train_y[1], c='C1', label='HF')
ax.plot(test_x, pred_LF_y, c='C0', label="LF pred")
ax.plot(test_x, pred_HF_y, c='C1', label="HF pred")
ax.plot(test_x, hfdnn.predict(test_x), c='C2', label="Only HF pred")
ax.legend()

plt.show()