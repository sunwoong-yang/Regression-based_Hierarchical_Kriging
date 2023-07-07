from functions.Camel_function import *
from run_functions.train_models import train_models
from PrePost.plot_scatter import plot_scatter
from PrePost.cal_error import cal_error
from PrePost.uniform_sampling import uniform
from pyDOE import lhs

in_dim = 2
function_name = "Camel"

np.random.seed(42)
LF_x = scaling_x(lhs(in_dim, samples=80, criterion='maximin'))
MF_x = scaling_x(lhs(in_dim, samples=50, criterion='maximin'))
HF_x = scaling_x(lhs(in_dim, samples=25, criterion='maximin'))

# LF_x = scaling_x(uniform(in_dim, n_pts=150))
# MF_x = scaling_x(uniform(in_dim, n_pts=100))
# HF_x = scaling_x(uniform(in_dim, n_pts=50))


LF_y = LF_function(LF_x).reshape(-1, 1)
MF_y = MF_function(MF_x).reshape(-1, 1)
HF_y = HF_function(HF_x).reshape(-1, 1)

test_x = scaling_x(lhs(in_dim, samples=300, criterion='maximin'))
ground_truth = HF_function(test_x)
# print(np.min(ground_truth), np.max(ground_truth), np.max(ground_truth) - np.min(ground_truth))
IHKs, RHKs, i_errors, r_errors, IHK_likeli, RHK_likeli, IHK_time, RHK_time = train_models([LF_x, MF_x, HF_x], [LF_y, MF_y, HF_y],
                                              test_x=test_x, test_y=ground_truth,
                                              history=True, repetition=15, add_noise=[[0, 0.2, 0.03], [1, 0.1, 0.015]], rand_seed=42)

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

plot_scatter(ground_truth, i_pred, r_pred, title="Camel function")
plot_camel(IHKs[0], RHKs[0], HF_function)
