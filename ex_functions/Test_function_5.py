import numpy as np

"""
Test_function_5 can be found here
Xiong, F., Ren, C., Mo, B., Li, C., & Hu, X. (2023). A new adaptive multi-fidelity metamodel method using meta-learning and Bayesian deep learning. Structural and Multidisciplinary Optimization, 66(3), 58.
[Link] https://link.springer.com/article/10.1007/s00158-023-03518-8
"""

def LF_function(x):
    x1, x2, x3, x4, x5, x6 = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5]
    return 15 * (x1-2)**2 + 0.85 * (x2-2)**2 + 0.6 * (x3-1)**2 + 1.35 * (x4-4)**2 + 0.6 * (x5-1)**2 + 0.6 * (x6-4)**2

def MF_function(x):
    x1, x2, x3, x4, x5, x6 = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5]
    return 20 * (x1-2)**2 + 0.95 * (x2-2)**2 + 0.8 * (x3-1)**2 + 1.05 * (x4-4)**2 + 0.8 * (x5-1)**2 + 0.7 * (x6-4)**2

def HF_function(x):
    x1, x2, x3, x4, x5, x6 = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5]
    return 25 * (x1-2)**2 + 1.00 * (x2-2)**2 + 1.00 * (x3-1)**2 + 1.00 * (x4-4)**2 + 1.00 * (x5-1)**2 + 1.00 * (x6-4)**2
