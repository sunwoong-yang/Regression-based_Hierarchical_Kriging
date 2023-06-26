import numpy as np
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

