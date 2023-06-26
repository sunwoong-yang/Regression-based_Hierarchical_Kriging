import numpy as np
from surrogate_model.HK import HK
import matplotlib.pyplot as plt

"""
Test_function_6 can be found here
Xiong, F., Ren, C., Mo, B., Li, C., & Hu, X. (2023). A new adaptive multi-fidelity metamodel method using meta-learning and Bayesian deep learning. Structural and Multidisciplinary Optimization, 66(3), 58.
[Link] https://link.springer.com/article/10.1007/s00158-023-03518-8
"""

def LF_function(x):
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5], x[:, 6], x[:, 7], x[:, 8], x[:, 9]
    return 0.5 * x1**2 + 0.6 * x2**2 + 0.3 * x1*x2 - 3 * x1 - 5 * x2 + (x3 - 2)**2 + 4.5 * (x4 - 5)**2 + 1.2 * (x5 - 3)**2 + \
        2 * (x6-1)**2 + 3 * x7**2 + 7 * (x8-3)**2 + 2 * (x9-2)**2 + (x10 - 1)**2 + 10

def MF_function(x):
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5], x[:, 6], x[:, 7], x[:, 8], x[:, 9]
    return 0.8 * x1**2 + 0.7 * x2**2 + 0.5 * x1*x2 - 4 * x1 - 6 * x2 + (x3 - 2)**2 + 4 * (x4 - 5)**2 + 1.1 * (x5 - 3)**2 + \
        2 * (x6-1)**2 + 4.5 * x7**2 + 7 * (x8-3)**2 + 2 * (x9-2)**2 + (x10 - 1)**2 + 10

def HF_function(x):
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5], x[:, 6], x[:, 7], x[:, 8], x[:, 9]
    return 1 * x1**2 + 1 * x2**2 + 1 * x1*x2 - 4 * x1 - 6 * x2 + (x3 - 2)**2 + 4 * (x4 - 5)**2 + 1 * (x5 - 3)**2 + \
        2 * (x6-1)**2 + 5 * x7**2 + 7 * (x8-3)**2 + 2 * (x9-2)**2 + (x10 - 1)**2 + 11
