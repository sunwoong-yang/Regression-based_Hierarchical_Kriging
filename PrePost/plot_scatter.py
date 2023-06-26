import matplotlib.pyplot as plt
import numpy as np


def plot_scatter(test_x, ground_truth, i_model, r_model, title="Function 1"):

	i_pred_HF = i_model.predict(test_x, return_std=False)
	r_pred_HF = r_model.predict(test_x, return_std=False)

	fig, ax = plt.subplots(dpi=300)
	ax.scatter(ground_truth, i_pred_HF, edgecolors='C0', label="IHK", facecolors='none')
	ax.scatter(ground_truth, r_pred_HF, edgecolors='C1', label="RHK", facecolors='none')
	lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
	ax.plot(lims, lims, '--k')
	ax.set_xlim(lims)
	ax.set_ylim(lims)
	ax.legend(fontsize=15)
	ax.set_title(title, fontsize=20)
	plt.show()