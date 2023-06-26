from ex_functions.Test_function_1 import *
from run_functions.train_models import train_models
from PrePost.plot_scatter import plot_scatter
from PrePost.cal_error import cal_error
from pyDOE import lhs

in_dim = 1

LF_x = lhs(in_dim, samples=20, criterion='maximin') * 10
MF_x = lhs(in_dim, samples=10, criterion='maximin') * 10
HF_x = lhs(in_dim, samples=5, criterion='maximin') * 10

LF_y = LF_function(LF_x).reshape(-1, 1)
MF_y = MF_function(MF_x).reshape(-1, 1)
HF_y = HF_function(HF_x).reshape(-1, 1)

test_x = np.linspace(0, 1, 100).reshape(-1, 1) * 10
ground_truth = HF_function(test_x)

IHK, RHK = train_models([LF_x, MF_x, HF_x], [LF_y, MF_y, HF_y])
# IHK_2level, RHK_2level = train_models([LF_x, HF_x], [LF_y, HF_y])
i_pred = IHK.predict(test_x, return_std=False)
r_pred = RHK.predict(test_x, return_std=False)

plot_scatter(ground_truth, i_pred, r_pred, title="Function 1")

i_error, r_error = cal_error(ground_truth, i_pred, r_pred)
print("IHK error: ", i_error)
print("RHK error: ", r_error)