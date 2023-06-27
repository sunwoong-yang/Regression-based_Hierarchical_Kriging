import numpy as np

"""
3 level Branin function can be found here
[Ref] Perdikaris, P., Raissi, M., Damianou, A., Lawrence, N. D., & Karniadakis, G. E. (2017). Nonlinear information fusion algorithms for data-efficient multi-fidelity modelling. Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences, 473(2198), 20160751.
"""


def LF_function(x): # low-fidelity function
    x1, x2 = x[:, 0], x[:, 1]
    return MF_function(1.2 * (x+2)) - 3 * x2 + 1

def MF_function(x): # high-fidelity function
    x1, x2 = x[:, 0], x[:, 1]
    return 10 * (HF_function(x-2))**0.5 + 2 * (x1 - 0.5) - 3 * (3*x2 - 1) - 1

def HF_function(x): # high-fidelity function
    x1, x2 = x[:,0], x[:,1]
    return (-1.275 * x1**2 / np.pi**2 + 5 * x1 / np.pi + x2 - 6)**2 + (10 - 5/4/np.pi) * np.cos(x1) + 10

def scaling_x(x):
    """
    Since design space of x1 & x2 = [-5, 10] & [0, 15]
    """
    x[:, 0] = 15 * x[:, 0] - 5
    x[:, 1] = 15 * x[:, 1]

    return x


