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
IHK_Func4 = np.load("../error_functions/IHK_Func4.npy", )
RHK_Func4 = np.load("../error_functions/RHK_Func4.npy", )
IHK_Func5 = np.load("../error_functions/IHK_Func5.npy", )
RHK_Func5 = np.load("../error_functions/RHK_Func5.npy", )
IHK_Func6 = np.load("../error_functions/IHK_Func6.npy", )
RHK_Func6 = np.load("../error_functions/RHK_Func6.npy", )

func_set = ["Forrester", "Branin", "Func1", "Func2", "Func3", "Func4", "Func5", "Func6"]
function_name = np.array([i for i in func_set for j in range(30) ]).reshape(-1,1)
function_name = np.vstack((function_name, function_name))
IorR_name = np.array([i for i in ["IHK", "RHK"] for j in range(30*len(func_set))]).reshape(-1,1)

i = 0
IHK_rmse = np.vstack((IHK_Forrester[:,[i]], IHK_Branin[:,[i]], IHK_Func1[:,[i]], IHK_Func2[:,[i]], IHK_Func3[:,[i]], IHK_Func4[:,[i]], IHK_Func5[:,[i]], IHK_Func6[:,[i]]))
RHK_rmse = np.vstack((RHK_Forrester[:,[i]], RHK_Branin[:,[i]], RHK_Func1[:,[i]], RHK_Func2[:,[i]], RHK_Func3[:,[i]], RHK_Func4[:,[i]], RHK_Func5[:,[i]], RHK_Func6[:,[i]]))
rmse = np.vstack((IHK_rmse, RHK_rmse))
rmse = np.hstack((rmse, function_name, IorR_name))
i = 1
IHK_mae = np.vstack((IHK_Forrester[:,[i]], IHK_Branin[:,[i]], IHK_Func1[:,[i]], IHK_Func2[:,[i]], IHK_Func3[:,[i]], IHK_Func4[:,[i]], IHK_Func5[:,[i]], IHK_Func6[:,[i]]))
RHK_mae = np.vstack((RHK_Forrester[:,[i]], RHK_Branin[:,[i]], IHK_Func1[:,[i]], IHK_Func2[:,[i]], IHK_Func3[:,[i]], IHK_Func4[:,[i]], IHK_Func5[:,[i]], IHK_Func6[:,[i]]))
mae = np.vstack((IHK_mae, RHK_mae))
mae = np.hstack((mae, function_name, IorR_name))
i = 2
IHK_rsq = np.vstack((IHK_Forrester[:,[i]], IHK_Branin[:,[i]], IHK_Func1[:,[i]], IHK_Func2[:,[i]], IHK_Func3[:,[i]], IHK_Func4[:,[i]], IHK_Func5[:,[i]], IHK_Func6[:,[i]]))
RHK_rsq = np.vstack((RHK_Forrester[:,[i]], RHK_Branin[:,[i]], IHK_Func1[:,[i]], IHK_Func2[:,[i]], IHK_Func3[:,[i]], IHK_Func4[:,[i]], IHK_Func5[:,[i]], IHK_Func6[:,[i]]))
rsq = np.vstack((IHK_rsq, RHK_rsq))
rsq = np.hstack((rsq, function_name, IorR_name))

df_rmse = pd.DataFrame(rmse, columns=["RMSE", "Function", "Type"])
df_rmse['RMSE'] = df_rmse['RMSE'].astype('float')
df_mae = pd.DataFrame(mae, columns=["MAE", "Function", "Type"])
df_mae['MAE'] = df_mae['MAE'].astype('float')
df_rsq = pd.DataFrame(rsq, columns=["Rsq", "Function", "Type"])
df_rsq['Rsq'] = df_rsq['Rsq'].astype('float')
# df_IHK_mae = pd.DataFrame(IHK_mae, columns=["RMSE", "Function"])
# df_IHK_mae['RMSE'] = df_IHK_mae['RMSE'].astype('float')
# df_IHK_rsq = pd.DataFrame(IHK_rsq, columns=["RMSE", "Function"])
# df_IHK_rsq['RMSE'] = df_IHK_rsq['RMSE'].astype('float')

# df_RHK_rmse = pd.DataFrame(RHK_rmse, columns=["RMSE", "Function"])
# df_RHK_rmse['RMSE'] = df_RHK_rmse['RMSE'].astype('float')
# df_RHK_mae = pd.DataFrame(RHK_mae, columns=["RMSE", "Function"])
# df_RHK_mae['RMSE'] = df_RHK_mae['RMSE'].astype('float')
# df_RHK_rsq = pd.DataFrame(RHK_rsq, columns=["RMSE", "Function"])
# df_RHK_rsq['RMSE'] = df_RHK_rsq['RMSE'].astype('float')

# https://coding-kindergarten.tistory.com/134
fig, ax = plt.subplots(dpi=300)
sns.violinplot(data=df_rmse, x="Function", y="RMSE", ax=ax, palette="Set2", split=True, hue="Type")
plt.setp(ax.collections, alpha=.7)
plt.show()

fig, ax = plt.subplots(dpi=300)
sns.violinplot(data=df_mae, x="Function", y="MAE", ax=ax, palette="Set2", split=True, hue="Type")
plt.setp(ax.collections, alpha=.7)
plt.show()

fig, ax = plt.subplots(dpi=300)
sns.violinplot(data=df_rsq, x="Function", y="Rsq", ax=ax, palette="Set2", split=True, hue="Type")
plt.setp(ax.collections, alpha=.7)
plt.show()