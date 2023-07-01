from functions.Branin_function import *
from run_functions.train_models import train_models
from PrePost.plot_scatter import plot_scatter
from PrePost.cal_error import cal_error
from PrePost.uniform_sampling import uniform
from pyDOE import lhs

in_dim = 2

LF_x = scaling_x(lhs(in_dim, samples=100, criterion='maximin'))
MF_x = scaling_x(lhs(in_dim, samples=50, criterion='maximin'))
HF_x = scaling_x(lhs(in_dim, samples=20, criterion='maximin'))

# LF_x = scaling_x(uniform(in_dim, n_pts=150))
# MF_x = scaling_x(uniform(in_dim, n_pts=100))
# HF_x = scaling_x(uniform(in_dim, n_pts=50))


LF_y = LF_function(LF_x).reshape(-1, 1)
LF_y *= np.random.normal(loc=1, scale=0.3, size=(len(LF_x),1))
MF_y = MF_function(MF_x).reshape(-1, 1)
MF_y *= np.random.normal(loc=1, scale=0.2, size=(len(MF_x),1))
HF_y = HF_function(HF_x).reshape(-1, 1)

test_x = scaling_x(lhs(in_dim, samples=300, criterion='maximin'))
ground_truth = HF_function(test_x)

IHKs, RHKs, i_errors, r_errors, IHK_time, RHK_time = train_models([LF_x, MF_x, HF_x], [LF_y, MF_y, HF_y],
                                              test_x=test_x, test_y=ground_truth,
                                              history=False, repetition=15, add_noise=[[0, 0.2, 2.25], [1, 0.1, 1.125]], rand_seed=42)

print("IHK error: ", np.mean(i_errors, axis=0))
print("IHK time: ", np.sum(IHK_time))
print("********************")
print("RHK error: ", np.mean(r_errors, axis=0))
print("RHK time: ", np.sum(RHK_time))

np.save("../error_functions/IHK_Branin.npy", i_errors)
np.save("../error_functions/RHK_Branin.npy", r_errors)
np.save("../time_functions/IHK_Branin.npy", IHK_time)
np.save("../time_functions/RHK_Branin.npy", RHK_time)

i_pred = IHKs[0].predict(test_x, return_std=False)
r_pred = RHKs[0].predict(test_x, return_std=False)

plot_scatter(ground_truth, i_pred, r_pred, title="Forrester function")
plot_Branin(IHKs[0], RHKs[0], HF_function)


# IHK_2level, RHK_2level = train_models([LF_x, HF_x], [LF_y, HF_y])
# i_pred = IHK.predict(test_x, return_std=False)
# r_pred = RHK.predict(test_x, return_std=False)
#
# plot_scatter(ground_truth, i_pred, r_pred, title="Branin function")
#
# i_error, r_error = cal_error(ground_truth, i_pred, r_pred)
# print("IHK error: ", i_error)
# print("RHK error: ", r_error)
#
# plot_Branin(IHK, RHK, HF_function)