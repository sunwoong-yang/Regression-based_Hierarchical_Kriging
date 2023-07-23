import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
def plot_2d_RAE(i_model, r_model, entire_x, aoa_data, qoi_data, index_qoi):
	# test_x = x
	# xx = test_x[:,0]
    # lin_pts=51
    xx, yy = entire_x[:,0].reshape(9,-1), entire_x[:,1].reshape(9,-1)
    # xx, yy = np.meshgrid(x1, x2)
    test_x = np.hstack((xx.reshape(-1,1), yy.reshape(-1,1)))
    i_pred = i_model.predict(test_x, pred_fidelity=2, return_std=False)
    r_pred = r_model.predict(test_x, pred_fidelity=2, return_std=False)
    i_pred, r_pred = i_pred.reshape(-1, aoa_data.shape[0]), r_pred.reshape(-1, aoa_data.shape[0])
    ground_truth = qoi_data[2, index_qoi, :, :]

    current_palette = sns.color_palette("viridis", as_cmap=True)

    fig, ax = plt.subplots(dpi=300)
    contour1 = ax.contour(xx, yy, ground_truth, levels=np.linspace(-0.3, 0., 11), colors='k', linewidths=1, linestyles='--', extend='both')  ## 등고선
    contour2 = ax.contourf(xx, yy, ground_truth, levels=np.linspace(-0.3, 0., 251), cmap=current_palette, extend='both')
    ax.clabel(contour1, contour1.levels, inline=True)  ## contour 라벨 #0-250
    fig.colorbar(contour2)
    plt.show()

    fig, ax = plt.subplots(dpi=300)
    contour1 = ax.contour(xx, yy, i_pred, levels=np.linspace(-0.3, 0., 11), colors='k', linewidths=1, linestyles='--', extend='both')  ## 등고선
    contour2 = ax.contourf(xx, yy, i_pred, levels=np.linspace(-0.3, 0., 251), cmap=current_palette, extend='both')
    # ax.scatter(i_model.x_original[0][:,0], i_model.x_original[0][:,1], color='r')
    ax.clabel(contour1, contour1.levels, inline=True)  ## contour 라벨 #0-250
    fig.colorbar(contour2)
    plt.show()

    fig, ax = plt.subplots(dpi=300)
    contour1 = ax.contour(xx, yy, r_pred, levels=np.linspace(-0.3, 0., 11), colors='k', linewidths=1, linestyles='--', extend='both')  ## 등고선
    contour2 = ax.contourf(xx, yy, r_pred, levels=np.linspace(-0.3, 0., 251), cmap=current_palette, extend='both')
    # ax.scatter(i_model.x_original[0][:, 0], i_model.x_original[0][:, 1], color='r')
    ax.clabel(contour1, contour1.levels, inline=True)  ## contour 라벨
    fig.colorbar(contour2)
    plt.show()

    fig, ax = plt.subplots(dpi=300)
    contour1 = ax.contour(xx, yy, np.abs(i_pred-ground_truth), levels=np.linspace(0, 0.05, 11), colors='k', linewidths=1, linestyles='--',
                          extend='both')  ## 등고선
    contour2 = ax.contourf(xx, yy, np.abs(i_pred-ground_truth), levels=np.linspace(0, 0.05, 251), cmap=current_palette, extend='both')
    ax.clabel(contour1, contour1.levels, inline=True)  ## contour 라벨 #0-250
    fig.colorbar(contour2)
    plt.show()

    fig, ax = plt.subplots(dpi=300)
    contour1 = ax.contour(xx, yy, np.abs(r_pred-ground_truth), levels=np.linspace(0, 0.05, 11), colors='k', linewidths=1, linestyles='--',
                          extend='both')  ## 등고선
    contour2 = ax.contourf(xx, yy, np.abs(r_pred-ground_truth), levels=np.linspace(0, 0.05, 251), cmap=current_palette, extend='both')
    ax.clabel(contour1, contour1.levels, inline=True)  ## contour 라벨
    fig.colorbar(contour2)
    plt.show()