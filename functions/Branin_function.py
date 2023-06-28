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

def plot_Branin(i_model, r_model, HF_function):
    lin_pts=51
    x1, x2 = np.linspace(-5, 10, lin_pts), np.linspace(0, 15, lin_pts)
    xx, yy = np.meshgrid(x1, x2)
    test_x = np.hstack((xx.reshape(-1,1), yy.reshape(-1,1)))
    i_pred = i_model.predict(test_x, pred_fidelity=2, return_std=False)
    r_pred = r_model.predict(test_x, pred_fidelity=2, return_std=False)
    ground_truth = HF_function(test_x)
    # xx, yy, i_pred, r_pred, ground_truth = xx.reshape(-1,lin_pts), yy.reshape(-1,lin_pts), i_pred.reshape(-1,lin_pts), r_pred.reshape(-1,lin_pts), ground_truth.reshape(-1,lin_pts)
    i_pred, r_pred, ground_truth = i_pred.reshape(-1,lin_pts), r_pred.reshape(-1,lin_pts), ground_truth.reshape(-1,lin_pts)

    current_palette = sns.color_palette("viridis", as_cmap=True)

    fig, ax = plt.subplots(dpi=300)
    contour1 = ax.contour(xx, yy, ground_truth, levels=np.linspace(0, 250, 11), colors='k', linewidths=1, linestyles='--', extend='both')  ## 등고선
    contour2 = ax.contourf(xx, yy, ground_truth, levels=np.linspace(0, 250, 251), cmap=current_palette, extend='both')
    ax.clabel(contour1, contour1.levels, inline=True)  ## contour 라벨 #0-250
    fig.colorbar(contour2)
    plt.show()

    fig, ax = plt.subplots(dpi=300)
    contour1 = ax.contour(xx, yy, i_pred, levels=np.linspace(0, 250, 11), colors='k', linewidths=1, linestyles='--', extend='both')  ## 등고선
    contour2 = ax.contourf(xx, yy, i_pred, levels=np.linspace(0, 250, 251), cmap=current_palette, extend='both')
    ax.scatter(i_model.x_original[0][:,0], i_model.x_original[0][:,1], color='r')
    ax.clabel(contour1, contour1.levels, inline=True)  ## contour 라벨 #0-250
    fig.colorbar(contour2)
    plt.show()

    fig, ax = plt.subplots(dpi=300)
    contour1 = ax.contour(xx, yy, r_pred, levels=np.linspace(0, 250, 11), colors='k', linewidths=1, linestyles='--', extend='both')  ## 등고선
    contour2 = ax.contourf(xx, yy, r_pred, levels=np.linspace(0, 250, 251), cmap=current_palette, extend='both')
    ax.scatter(i_model.x_original[0][:, 0], i_model.x_original[0][:, 1], color='r')
    ax.clabel(contour1, contour1.levels, inline=True)  ## contour 라벨
    fig.colorbar(contour2)
    plt.show()

    fig, ax = plt.subplots(dpi=300)
    contour1 = ax.contour(xx, yy, np.abs(i_pred-ground_truth), levels=np.linspace(0, 30, 6), colors='k', linewidths=1, linestyles='--',
                          extend='both')  ## 등고선
    contour2 = ax.contourf(xx, yy, np.abs(i_pred-ground_truth), levels=np.linspace(0, 30, 251), cmap=current_palette, extend='both')
    ax.clabel(contour1, contour1.levels, inline=True)  ## contour 라벨 #0-250
    fig.colorbar(contour2)
    plt.show()

    fig, ax = plt.subplots(dpi=300)
    contour1 = ax.contour(xx, yy, np.abs(r_pred-ground_truth), levels=np.linspace(0, 30, 6), colors='k', linewidths=1, linestyles='--',
                          extend='both')  ## 등고선
    contour2 = ax.contourf(xx, yy, np.abs(r_pred-ground_truth), levels=np.linspace(0, 30, 251), cmap=current_palette, extend='both')
    ax.clabel(contour1, contour1.levels, inline=True)  ## contour 라벨
    fig.colorbar(contour2)
    plt.show()
