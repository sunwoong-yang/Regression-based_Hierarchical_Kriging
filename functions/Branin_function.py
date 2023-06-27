import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

def plot_Branin(test_x, ground_truth, i_model, r_model):
    fig, ax = plt.subplots(dpi=300)
    current_palette = sns.color_palette("Set2")
    # current_palette = sns.color_palette()
    ax.scatter(i_model.x[0], i_model.y[0], color=current_palette[0], label='Low-fidelity data')
    ax.scatter(i_model.x[1], i_model.y[1], color=current_palette[1], label='Mid-fidelity data')
    ax.scatter(i_model.x[2], i_model.y[2], color=current_palette[2], label='High-fidelity data')

    ax.plot(test_x, i_model.predict(test_x, pred_fidelity=0, return_std=False),
            color=current_palette[0], label='Low-fidelity (IHK)', linestyle='--')
    ax.plot(test_x, i_model.predict(test_x, pred_fidelity=1, return_std=False),
            color=current_palette[1], label='Mid-fidelity (IHK)', linestyle='--')
    ax.plot(test_x, i_model.predict(test_x, pred_fidelity=2, return_std=False),
            color=current_palette[2], label='High-fidelity (IHK)', linestyle='--')

    ax.plot(test_x, r_model.predict(test_x, pred_fidelity=0, return_std=False),
            color=current_palette[0], label='Low-fidelity (RHK)', linestyle='-')
    ax.plot(test_x, r_model.predict(test_x, pred_fidelity=1, return_std=False),
            color=current_palette[1], label='Mid-fidelity (RHK)', linestyle='-')
    ax.plot(test_x, r_model.predict(test_x, pred_fidelity=2, return_std=False),
            color=current_palette[2], label='High-fidelity (RHK)', linestyle='-')

    ax.legend(fontsize=12, frameon=False, ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1.0), columnspacing=0.4)
    plt.tight_layout()
    plt.show()