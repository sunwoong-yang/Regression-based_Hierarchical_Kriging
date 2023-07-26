import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""
3 level Forrester functions can be found here
Ha, H., Oh, S., & Yee, K. (2014). Feasibility study of hierarchical kriging model in the design optimization process. Journal of the Korean Society for Aeronautical & Space Sciences, 42(2), 108-118.
[Link] http://koreascience.or.kr/article/JAKO201409150678130.pdf
"""

def LF_function(x):
    return 0.2 * MF_function(x) + 10 * x
    # return 0.2 * MF_function(x) + 5 * np.sin(x)

def MF_function(x):
    return 0.5 * HF_function(x) + 10 * (x - 0.5) - 5

def HF_function(x):
    return ((6 * x - 2)**2) * np.sin(12 * x - 4)

def plot_Forrester(test_x, ground_truth, i_model, r_model):
    fig, ax = plt.subplots(dpi=300)
    current_palette = sns.color_palette("Set2")
    # current_palette = sns.color_palette()

    ax.plot(test_x, ground_truth, color='k', linestyle='-', alpha=0., label=' ')


    ax.scatter(i_model.x_original[0], i_model.y[0], color=current_palette[0], label='Low-fidelity data', edgecolors='k', zorder=101)
    ax.scatter(i_model.x_original[1], i_model.y[1], color=current_palette[1], label='Mid-fidelity data', edgecolors='k', zorder=101)
    ax.scatter(i_model.x_original[2], i_model.y[2], color=current_palette[2], label='High-fidelity data', edgecolors='k', zorder=101)

    ax.plot(test_x, ground_truth, color='k', linestyle='-', zorder=100, alpha=0.7, label='Ground truth')

    ax.plot(test_x, i_model.predict(test_x, pred_fidelity=0, return_std=False),
            color=current_palette[0], label='Low-fidelity (IHK)', linestyle='--')
    ax.plot(test_x, i_model.predict(test_x, pred_fidelity=1, return_std=False),
            color=current_palette[1], label='Mid-fidelity (IHK)', linestyle='--')
    ax.plot(test_x, i_model.predict(test_x, pred_fidelity=2, return_std=False),
            color=current_palette[2], label='High-fidelity (IHK)', linestyle='--')

    ax.plot(test_x, ground_truth, color='k', linestyle='-', alpha=0., label=' ')

    ax.plot(test_x, r_model.predict(test_x, pred_fidelity=0, return_std=False),
            color=current_palette[0], label='Low-fidelity (RHK)', linestyle='-')
    ax.plot(test_x, r_model.predict(test_x, pred_fidelity=1, return_std=False),
            color=current_palette[1], label='Mid-fidelity (RHK)', linestyle='-')
    ax.plot(test_x, r_model.predict(test_x, pred_fidelity=2, return_std=False),
            color=current_palette[2], label='High-fidelity (RHK)', linestyle='-')



    ax.legend(fontsize=12, frameon=False, ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1.0), columnspacing=0.4)
    plt.tight_layout()
    plt.show()

    return ax