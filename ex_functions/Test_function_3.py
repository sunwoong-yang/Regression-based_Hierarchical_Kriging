import numpy as np
from surrogate_model.HK import HK
import matplotlib.pyplot as plt
from pyDOE import lhs

"""
Test_function_3 can be found here
Xiong, F., Ren, C., Mo, B., Li, C., & Hu, X. (2023). A new adaptive multi-fidelity metamodel method using meta-learning and Bayesian deep learning. Structural and Multidisciplinary Optimization, 66(3), 58.
[Link] https://link.springer.com/article/10.1007/s00158-023-03518-8
"""

def LF_function(x): # low-fidelity function
    return np.sin(8 * np.pi * x)

def MF_function(x): # MF function does not exist in the refrence paper, but is arbitrariliy defined
    return LF_function(x)**2

def HF_function(x): # high-fidelity function
    return (x-np.sqrt(2)) * LF_function(x)**2
