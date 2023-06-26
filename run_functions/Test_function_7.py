from functions.Test_function_7 import *
from run_functions.train_models import train_models
from PrePost.plot_scatter import plot_scatter
from PrePost.cal_error import cal_error
from pyDOE import lhs

in_dim = 15

LF_x = lhs(in_dim, samples=250, criterion='maximin') * 1 + 0.5
MF_x = lhs(in_dim, samples=200, criterion='maximin') * 1 + 0.5
HF_x = lhs(in_dim, samples=150, criterion='maximin') * 1 + 0.5

LF_y = LF_function(LF_x).reshape(-1, 1)
MF_y = MF_function(MF_x).reshape(-1, 1)
HF_y = HF_function(HF_x).reshape(-1, 1)

test_x = lhs(in_dim, samples=100, criterion='maximin') * 1 + 0.5
ground_truth = HF_function(test_x)

IHK, RHK = train_models([LF_x, MF_x, HF_x], [LF_y, MF_y, HF_y])
i_pred = IHK.predict(test_x, return_std=False)
r_pred = RHK.predict(test_x, return_std=False)

plot_scatter(ground_truth, i_pred, r_pred, title="Function 7")

i_error, r_error = cal_error(ground_truth, i_pred, r_pred)
print("IHK error: ", i_error)
print("RHK error: ", r_error)