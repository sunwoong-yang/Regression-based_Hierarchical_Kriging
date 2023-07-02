import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

IHK_Forrester = np.load("../results_functions/error/IHK_Forrester.npy", )
RHK_Forrester = np.load("../results_functions/error/RHK_Forrester.npy", )
IHK_Branin = np.load("../results_functions/error/IHK_Forrester.npy", )
RHK_Branin = np.load("../results_functions/error/RHK_Forrester.npy", )
# IHK_Func1 = np.load("../results_functions/error/IHK_Func1.npy", )
# RHK_Func1 = np.load("../results_functions/error/RHK_Func1.npy", )
# IHK_Func2 = np.load("../results_functions/error/IHK_Func2.npy", )
# RHK_Func2 = np.load("../results_functions/error/RHK_Func2.npy", )
IHK_Func3 = np.load("../results_functions/error/IHK_Func3.npy", )
RHK_Func3 = np.load("../results_functions/error/RHK_Func3.npy", )
IHK_Func4 = np.load("../results_functions/error/IHK_Func4.npy", )
RHK_Func4 = np.load("../results_functions/error/RHK_Func4.npy", )
IHK_Func5 = np.load("../results_functions/error/IHK_Func5.npy", )
RHK_Func5 = np.load("../results_functions/error/RHK_Func5.npy", )
IHK_Func6 = np.load("../results_functions/error/IHK_Func3.npy", )
RHK_Func6 = np.load("../results_functions/error/RHK_Func3.npy", )

error_set = ["RMSE", "MAE"]
error_name = np.array([i for i in error_set for j in range(15)]).reshape(-1, 1)
error_name = np.vstack((error_name, error_name))
IorR_name = np.array([i for i in ["IHK", "RHK"] for j in range(15 * len(error_set))]).reshape(-1, 1)
func_name = ["Forrester", "Branin", "Func3", "Func4", "Func5", "Func6"] # 나중에 func 1,2,3,4로 당기기 (기존 func1,2을 안쓰게됨)


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
# df_Func1 = make_df_per_func(IHK_Func1, RHK_Func1)
# df_Func2 = make_df_per_func(IHK_Func2, RHK_Func2)
df_Func3 = make_df_per_func(IHK_Func3, RHK_Func3)
df_Func4 = make_df_per_func(IHK_Func4, RHK_Func4)
df_Func5 = make_df_per_func(IHK_Func5, RHK_Func5)
df_Func6 = make_df_per_func(IHK_Func6, RHK_Func6)

df_set = [df_Forrester, df_Branin, df_Func3, df_Func4, df_Func5, df_Func6]

# https://coding-kindergarten.tistory.com/134
# https://gmnam.tistory.com/252
fig = plt.figure(figsize=(8.5, 6), dpi=350)
# ax2_ylim = [
# 	[1000, 1600],  # Forrester
# 	# [3, 7],
# 	# [765, 770],
# 	[220, 230],
# 	[110, 130],
# 	[80, 160],
# 	[80, 160],
# 	[80, 160]
# ]
current_palette = sns.color_palette("Set2")
box_setting_IHK = dict(boxprops=dict(facecolor=current_palette[0]), patch_artist=True, medianprops=dict(color='k'), showfliers=True)
box_setting_RHK = dict(boxprops=dict(facecolor=current_palette[1]), patch_artist=True, medianprops=dict(color='k'), showfliers=True)
for idx, df_ in enumerate(df_set):
	ax1 = fig.add_subplot(2, 3, idx + 1)
	# ax1.set_aspect('equal')
	temp = df_[df_["Error_name"] == "RMSE"]
	p1 = ax1.boxplot(temp[temp["Type"] == "IHK"]["Error"].values, positions=[-0.2],
	            **box_setting_IHK)
	p2 = ax1.boxplot(temp[temp["Type"] == "RHK"]["Error"].values, positions=[+0.2],
	            **box_setting_RHK)
	ax2 = ax1.twinx()
	temp = df_[df_["Error_name"] == "MAE"]
	ax2.boxplot(temp[temp["Type"] == "IHK"]["Error"].values, positions=[1-0.2],
	            **box_setting_IHK)
	ax2.boxplot(temp[temp["Type"] == "RHK"]["Error"].values, positions=[1+0.2],
	            **box_setting_RHK)

	ax1.set_xlim([-0.5, 1.5])
	ax2.axvline(0.5, color='k', linestyle='--')
	ax1.legend('', frameon=False)
	ax2.legend('', frameon=False)
	ax1.set_ylabel("")
	ax2.set_ylabel("")
	ax1.set_xlabel("")
	colors = [current_palette[0], current_palette[1]]
	ax1.set_xticks([0,1], ["RMSE","MAE"])
	plt.tick_params(bottom=False)
	plt.title(func_name[idx], fontweight='bold', fontsize=14)
# plt.gca().set_aspect('equal')
# ax1.set_aspect(1)
# ax2.set_aspect(1)


fig.legend([p1["boxes"][0], p2["boxes"][0]], ["IHK","RHK"], fontsize=18, loc="upper center", bbox_to_anchor=(0.5, 1.02), frameon=False, ncol=2,)
plt.tight_layout()
fig.subplots_adjust(top=0.88)
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

