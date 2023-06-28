import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_scatter(ground_truth, i_pred, r_pred, title="Function 1"):

	fig, ax = plt.subplots(dpi=300)
	current_palette = sns.color_palette("Set2")

	ax.scatter(ground_truth, i_pred, edgecolors='k', facecolors=current_palette[0], label="IHK", alpha=0.7)
	ax.scatter(ground_truth, r_pred, edgecolors='k', facecolors=current_palette[1], label="RHK", alpha=0.7)

	lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
	ax.plot(lims, lims, '--k')
	ax.set_xlim(lims)
	ax.set_ylim(lims)
	ax.legend(fontsize=20, frameon=False)
	ax.set_title(title, fontsize=20)
	plt.show()