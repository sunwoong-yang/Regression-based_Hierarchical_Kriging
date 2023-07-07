# https://mf2.readthedocs.io/en/v2022.06.0/functions/six_hump_camelback.html

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""
2 level six hump function can be found here
[Ref] Hao, P., Feng, S., Li, Y., Wang, B., & Chen, H. (2020). Adaptive infill sampling criterion for multi-fidelity gradient-enhanced kriging model. Structural and Multidisciplinary Optimization, 62, 353-373.
"""


def LF_function(x): # low-fidelity function
    x1, x2 = x[:, 0], x[:, 1]
    return HF_function(0.5 * x) + 0.5 * x1 * x2 -15

def MF_function(x): # high-fidelity function
    x1, x2 = x[:, 0], x[:, 1]
    return HF_function(0.7 * x) + x1 * x2 -15

def HF_function(x): # high-fidelity function
    x1, x2 = x[:,0], x[:,1]
    return 4 * x1**2 - 2.1 * x1**4 + x1**6 / 3 + x1 * x2 - 4 * x2**2 + 4 * x2**4

def scaling_x(x):
    """
    Since design space of x1 & x2 = [-1, 1]
    """
    x[:, 0] = 2 * x[:, 0] - 1
    x[:, 1] = 2 * x[:, 1] - 1

    return x

def plot_camel(i_model, r_model, HF_function):
    lin_pts=51
    x1, x2 = np.linspace(-1, 1, lin_pts), np.linspace(-1, 1, lin_pts)
    xx, yy = np.meshgrid(x1, x2)
    test_x = np.hstack((xx.reshape(-1,1), yy.reshape(-1,1)))
    i_pred = i_model.predict(test_x, pred_fidelity=2, return_std=False)
    r_pred = r_model.predict(test_x, pred_fidelity=2, return_std=False)
    ground_truth = HF_function(test_x)
    # xx, yy, i_pred, r_pred, ground_truth = xx.reshape(-1,lin_pts), yy.reshape(-1,lin_pts), i_pred.reshape(-1,lin_pts), r_pred.reshape(-1,lin_pts), ground_truth.reshape(-1,lin_pts)
    i_pred, r_pred, ground_truth = i_pred.reshape(-1,lin_pts), r_pred.reshape(-1,lin_pts), ground_truth.reshape(-1,lin_pts)

    current_palette = sns.color_palette("viridis", as_cmap=True)

    fig, ax = plt.subplots(dpi=300)
    contour1 = ax.contour(xx, yy, ground_truth, levels=np.linspace(-1, 3, 11), colors='k', linewidths=1, linestyles='--', extend='both')  ## 등고선
    contour2 = ax.contourf(xx, yy, ground_truth, levels=np.linspace(-1, 3, 251), cmap=current_palette, extend='both')
    ax.clabel(contour1, contour1.levels, inline=True)  ## contour 라벨 #0-250
    fig.colorbar(contour2)
    plt.show()

    fig, ax = plt.subplots(dpi=300)
    contour1 = ax.contour(xx, yy, i_pred, levels=np.linspace(-1, 3, 11), colors='k', linewidths=1, linestyles='--', extend='both')  ## 등고선
    contour2 = ax.contourf(xx, yy, i_pred, levels=np.linspace(-1, 3, 251), cmap=current_palette, extend='both')
    ax.scatter(i_model.x_original[0][:,0], i_model.x_original[0][:,1], color='r')
    ax.clabel(contour1, contour1.levels, inline=True)  ## contour 라벨 #0-250
    fig.colorbar(contour2)
    plt.show()

    fig, ax = plt.subplots(dpi=300)
    contour1 = ax.contour(xx, yy, r_pred, levels=np.linspace(-1, 3, 11), colors='k', linewidths=1, linestyles='--', extend='both')  ## 등고선
    contour2 = ax.contourf(xx, yy, r_pred, levels=np.linspace(-1, 3, 251), cmap=current_palette, extend='both')
    ax.scatter(i_model.x_original[0][:, 0], i_model.x_original[0][:, 1], color='r')
    ax.clabel(contour1, contour1.levels, inline=True)  ## contour 라벨
    fig.colorbar(contour2)
    plt.show()

    fig, ax = plt.subplots(dpi=300)
    contour1 = ax.contour(xx, yy, np.abs(i_pred-ground_truth), levels=np.linspace(0, .1, 6), colors='k', linewidths=1, linestyles='--',
                          extend='both')  ## 등고선
    contour2 = ax.contourf(xx, yy, np.abs(i_pred-ground_truth), levels=np.linspace(0, .1, 251), cmap=current_palette, extend='both')
    ax.clabel(contour1, contour1.levels, inline=True)  ## contour 라벨 #0-250
    fig.colorbar(contour2)
    plt.show()

    fig, ax = plt.subplots(dpi=300)
    contour1 = ax.contour(xx, yy, np.abs(r_pred-ground_truth), levels=np.linspace(0, .1, 6), colors='k', linewidths=1, linestyles='--',
                          extend='both')  ## 등고선
    contour2 = ax.contourf(xx, yy, np.abs(r_pred-ground_truth), levels=np.linspace(0, .1, 251), cmap=current_palette, extend='both')
    ax.clabel(contour1, contour1.levels, inline=True)  ## contour 라벨
    fig.colorbar(contour2)
    plt.show()
