import numpy as np

"""
Test_function_4 can be found here
Xiong, F., Ren, C., Mo, B., Li, C., & Hu, X. (2023). A new adaptive multi-fidelity metamodel method using meta-learning and Bayesian deep learning. Structural and Multidisciplinary Optimization, 66(3), 58.
[Link] https://link.springer.com/article/10.1007/s00158-023-03518-8
"""
tf, th, tl = 0.2, 0.3, 0.1
def LF_function(x):
    x1, x2 = x[:, 0], x[:, 1]
    return (1 - np.exp(-1 / 2 / x2)) * ((1000 * tf * x1 ** 3 + 1900 * x1 ** 2 + 2092 * x1 + 60) / (
                1000 * tl * x1 ** 3 + 500 * x1 ** 2 + 4 * x1 + 20)) + 5 * np.exp(-tf) * x1 ** (th / 2) / (x2 ** (2 + th) + 1) + (10*x1**2 + 4*x2**2) / (50*x1*x2 + 10)

def MF_function(x):
    x1, x2 = x[:, 0], x[:, 1]
    return (1 - np.exp(-1 / 2 / x2)) * ((1000 * tf * x1 ** 3 + 1900 * x1 ** 2 + 2092 * x1 + 60) / (
                1000 * tl * x1 ** 3 + 500 * x1 ** 2 + 4 * x1 + 20)) + 5 * np.exp(-tf) * x1 ** (th / 2) / (x2 ** (2 + th) + 1)

def HF_function(x):
    x1, x2 = x[:, 0], x[:, 1]
    return (1 - np.exp(-1/2/x2)) * ((1000*tf*x1**3 + 1900*x1**2 + 2092*x1 + 60) / (1000*tl*x1**3 + 500*x1**2 + 4*x1 + 20))

