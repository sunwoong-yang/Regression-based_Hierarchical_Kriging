import numpy as np

"""
Test_function_1 can be found here
Xiong, F., Ren, C., Mo, B., Li, C., & Hu, X. (2023). A new adaptive multi-fidelity metamodel method using meta-learning and Bayesian deep learning. Structural and Multidisciplinary Optimization, 66(3), 58.
[Link] https://link.springer.com/article/10.1007/s00158-023-03518-8
"""

def LF_function(x): # high-fidelity function
    return np.sin(x) + 0.2 * x + 0.5

def MF_function(x): # high-fidelity function
    return np.sin(x) + 0.8 * x + (x - 0.5)**2 / 45 + 0.5

def HF_function(x): # high-fidelity function
    return np.sin(x) + 0.2 * x + (x - 0.5)**2 / 16 + 0.5