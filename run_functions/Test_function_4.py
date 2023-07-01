from functions.Test_function_4 import *
from run_functions.train_models import train_models
from PrePost.plot_scatter import plot_scatter
from pyDOE import lhs

in_dim = 2

LF_x = lhs(in_dim, samples=100, criterion='maximin') * 1
MF_x = lhs(in_dim, samples=60, criterion='maximin') * 1
HF_x = lhs(in_dim, samples=20, criterion='maximin') * 1

LF_y = LF_function(LF_x).reshape(-1, 1)
MF_y = MF_function(MF_x).reshape(-1, 1)
HF_y = HF_function(HF_x).reshape(-1, 1)

test_x = lhs(in_dim, samples=300, criterion='maximin') * 1
ground_truth = HF_function(test_x)

IHKs, RHKs, i_errors, r_errors, IHK_time, RHK_time = train_models([LF_x, MF_x, HF_x], [LF_y, MF_y, HF_y],
                                              test_x=test_x, test_y=ground_truth,
                                              history=False, repetition=15, add_noise=[[0, 0.2, 0.11], [1, 0.1, 0.055]], rand_seed=42)

print("IHK error: ", np.mean(i_errors, axis=0))
print("IHK time: ", np.sum(IHK_time))
print("********************")
print("RHK error: ", np.mean(r_errors, axis=0))
print("RHK time: ", np.sum(RHK_time))

np.save("../error_functions/IHK_Func4.npy", i_errors)
np.save("../error_functions/RHK_Func4.npy", r_errors)
np.save("../time_functions/IHK_Func4.npy", IHK_time)
np.save("../time_functions/RHK_Func4.npy", RHK_time)

i_pred = IHKs[0].predict(test_x, return_std=False)
r_pred = RHKs[0].predict(test_x, return_std=False)

plot_scatter(ground_truth, i_pred, r_pred, title="Function 4")