import matplotlib.pyplot as plt
import numpy as np


def plot_scatter(ground_truth, i_pred, r_pred, title="Function 1"):

	fig, ax = plt.subplots(dpi=300)
	ax.scatter(ground_truth, i_pred, edgecolors='C0', label="IHK", facecolors='none')
	ax.scatter(ground_truth, r_pred, edgecolors='C1', label="RHK", facecolors='none')
	lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
	ax.plot(lims, lims, '--k')
	ax.set_xlim(lims)
	ax.set_ylim(lims)
	ax.legend(fontsize=20)
	ax.set_title(title, fontsize=20)
	plt.show()