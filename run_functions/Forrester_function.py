from functions.Forrester_function import *
from run_functions.train_models import train_models
from PrePost.plot_scatter import plot_scatter

import numpy as np

in_dim = 1
function_name = "Forrester"
# LF_x = lhs(in_dim, samples=80, criterion='maximin')
# MF_x = lhs(in_dim, samples=40, criterion='maximin')
# HF_x = lhs(in_dim, samples=20, criterion='maximin')
LF_x = np.linspace(0, 1, 21).reshape(-1,1)
MF_x = np.linspace(0, 1, 8).reshape(-1,1)
HF_x = np.array([0, 0.4, 0.6, 1]).reshape(-1,1)

LF_y = LF_function(LF_x).reshape(-1, 1)
MF_y = MF_function(MF_x).reshape(-1, 1)
HF_y = HF_function(HF_x).reshape(-1, 1)

test_x = np.linspace(0, 1, 300).reshape(-1, 1)
ground_truth = HF_function(test_x)


IHKs, RHKs, i_errors, r_errors, IHK_likeli, RHK_likeli, IHK_time, RHK_time = train_models([LF_x, MF_x, HF_x], [LF_y, MF_y, HF_y],
                                              test_x=test_x, test_y=ground_truth,
                                              history=False, repetition=15, add_noise=[[0, 0.2, 0.22], [1, 0.1, 0.11]], rand_seed=42)

print("IHK likelihood: ", np.mean(IHK_likeli, axis=0))
print("IHK error: ", np.mean(i_errors, axis=0))
print("IHK time: ", np.sum(IHK_time))
print("********************")
print("RHK likelihood: ", np.mean(RHK_likeli, axis=0))
print("RHK error: ", np.mean(r_errors, axis=0))
print("RHK time: ", np.sum(RHK_time))

np.save(f"../results_functions/likeli/IHK_{function_name}.npy", IHK_likeli)
np.save(f"../results_functions/likeli/RHK_{function_name}.npy", RHK_likeli)
np.save(f"../results_functions/error/IHK_{function_name}.npy", i_errors)
np.save(f"../results_functions/error/RHK_{function_name}.npy", r_errors)
np.save(f"../results_functions/time/IHK_{function_name}.npy", IHK_time)
np.save(f"../results_functions/time/RHK_{function_name}.npy", RHK_time)

i_pred = IHKs[0].predict(test_x, return_std=False)
r_pred = RHKs[0].predict(test_x, return_std=False)

plot_scatter(ground_truth, i_pred, r_pred, title="Forrester function")
ax = plot_Forrester(test_x, ground_truth, IHKs[4], RHKs[4])
ax.figure.savefig("../results_functions/Forrester_pedagogical.png")


#########################################################################################################

# IHK, RHK = train_models([LF_x, MF_x, HF_x], [LF_y, MF_y, HF_y], history=True)
# # IHK_2level, RHK_2level = train_models([LF_x, HF_x], [LF_y, HF_y])
# i_pred = IHK.predict(test_x, return_std=False)
# r_pred = RHK.predict(test_x, return_std=False)
#
# plot_scatter(ground_truth, i_pred, r_pred, title="Forrester function")
#
# i_error, r_error = cal_error(ground_truth, i_pred, r_pred)
# print("IHK error: ", i_error)
# print("RHK error: ", r_error)
#
# plot_Forrester(test_x, ground_truth, IHK, RHK)
