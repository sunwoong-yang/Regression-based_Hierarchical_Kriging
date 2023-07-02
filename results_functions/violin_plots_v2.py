import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

IHK_Forrester = np.load("../error_functions/IHK_Forrester.npy", )
RHK_Forrester = np.load("../error_functions/RHK_Forrester.npy", )
IHK_Branin = np.load("../error_functions/IHK_Branin.npy", )
RHK_Branin = np.load("../error_functions/RHK_Branin.npy", )
IHK_Func1 = np.load("../error_functions/IHK_Func1.npy", )
RHK_Func1 = np.load("../error_functions/RHK_Func1.npy", )
IHK_Func2 = np.load("../error_functions/IHK_Func2.npy", )
RHK_Func2 = np.load("../error_functions/RHK_Func2.npy", )
IHK_Func3 = np.load("../error_functions/IHK_Func3.npy", )
RHK_Func3 = np.load("../error_functions/RHK_Func3.npy", )
IHK_Func4 = np.load("../error_functions/IHK_Func3.npy", )
RHK_Func4 = np.load("../error_functions/RHK_Func3.npy", )
IHK_Func5 = np.load("../error_functions/IHK_Func3.npy", )
RHK_Func5 = np.load("../error_functions/RHK_Func3.npy", )
IHK_Func6 = np.load("../error_functions/IHK_Func3.npy", )
RHK_Func6 = np.load("../error_functions/RHK_Func3.npy", )

error_set = ["RMSE", "MAE"]
error_name = np.array([i for i in error_set for j in range(15)]).reshape(-1, 1)
error_name = np.vstack((error_name, error_name))
IorR_name = np.array([i for i in ["IHK", "RHK"] for j in range(15 * len(error_set))]).reshape(-1, 1)
func_name = ["Forrester", "Branin", "Func1", "Func2", "Func3", "Func4", "Func5", "Func6"]


def make_df_per_func(IHK_results, RHK_results):
	IHK_err = np.vstack((IHK_results[:, [0]], IHK_results[:, [1]]))
	RHK_err = np.vstack((RHK_results[:, [0]], RHK_results[:, [1]]))
	error = np.vstack((IHK_err, RHK_err))
	error = np.hstack((error, error_name, IorR_name))

	df_error = pd.DataFrame(error, columns=["Error", "Error_name", "Type"])
	df_error['Error'] = df_error['Error'].astype('float')

	return df_error


df_Forrester = make_df_per_func(IHK_Forrester, RHK_Forrester)
df_Branin = make_df_per_func(IHK_Branin, RHK_Branin)
df_Func1 = make_df_per_func(IHK_Func1, RHK_Func1)
df_Func2 = make_df_per_func(IHK_Func2, RHK_Func2)
df_Func3 = make_df_per_func(IHK_Func3, RHK_Func3)
df_Func4 = make_df_per_func(IHK_Func4, RHK_Func4)
df_Func5 = make_df_per_func(IHK_Func5, RHK_Func5)
df_Func6 = make_df_per_func(IHK_Func6, RHK_Func6)

df_set = [df_Forrester, df_Branin, df_Func1, df_Func2, df_Func3, df_Func4, df_Func5, df_Func6]

# https://coding-kindergarten.tistory.com/134
fig = plt.figure(figsize=(10, 6), dpi=350)
ax2_ylim = [
	[1000, 1600],  # Forrester
	[3, 7],
	[765, 770],
	[220, 230],
	[110, 130],
	[80, 160],
	[80, 160],
	[80, 160]
]
for idx, df_ in enumerate(df_set):
	ax1 = fig.add_subplot(2, 4, idx + 1)
	# ax1.set_aspect('equal')
	sns.violinplot(data=df_[df_["Error_name"] == "RMSE"], x="Error_name", y="Error", palette="Set2", split=True,
	               hue="Type", ax=ax1, linewidth=1, )
	ax2 = ax1.twinx()
	sns.violinplot(data=df_, x="Error_name", y="Error", palette="Set2", split=True, hue="Type", ax=ax2, linewidth=1, )
	plt.setp(ax1.collections, alpha=.7)
	plt.setp(ax2.collections, alpha=.7)
	ax2.set_ylim(ax2_ylim[idx])
	ax2.axvline(0.5, color='k', linestyle='--')
	ax1.legend('', frameon=False)
	ax2.legend('', frameon=False)
	ax1.set_ylabel("")
	ax2.set_ylabel("")
	ax1.set_xlabel("")
	plt.title(func_name[idx], fontweight='bold', fontsize=14)
# plt.gca().set_aspect('equal')
# ax1.set_aspect(1)
# ax2.set_aspect(1)


lines, labels = fig.axes[-1].get_legend_handles_labels()
fig.legend(lines, labels, fontsize=18, loc="upper center", bbox_to_anchor=(0.5, 1.02), frameon=False, ncol=2)
plt.tight_layout()
fig.subplots_adjust(top=0.9)
plt.show()

# fig, ax = plt.subplots(dpi=300)
# sns.violinplot(data=df_mae, x="Function", y="MAE", ax=ax, palette="Set2", split=True, hue="Type")
# plt.setp(ax.collections, alpha=.7)
# plt.show()
#
# fig, ax = plt.subplots(dpi=300)
# sns.violinplot(data=df_rsq, x="Function", y="Rsq", ax=ax, palette="Set2", split=True, hue="Type")
# plt.setp(ax.collections, alpha=.7)
# plt.show()
