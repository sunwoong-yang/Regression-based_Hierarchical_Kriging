from functions.Branin_function import *
from run_functions.train_models import train_models
from PrePost.plot_scatter import plot_scatter
from PrePost.cal_error import cal_error
from pyDOE import lhs
import numpy as np

in_dim = 2

LF_x = scaling_x(lhs(in_dim, samples=80, criterion='maximin'))
MF_x = scaling_x(lhs(in_dim, samples=40, criterion='maximin'))
HF_x = scaling_x(lhs(in_dim, samples=20, criterion='maximin'))

LF_y = LF_function(LF_x).reshape(-1, 1) #* np.random.normal(loc=1, scale=0.3, size=(len(LF_x),1))
MF_y = MF_function(MF_x).reshape(-1, 1)
HF_y = HF_function(HF_x).reshape(-1, 1)

test_x = scaling_x(lhs(in_dim, samples=100, criterion='maximin'))
ground_truth = HF_function(test_x)

IHK, RHK = train_models([LF_x, MF_x, HF_x], [LF_y, MF_y, HF_y])
# IHK_2level, RHK_2level = train_models([LF_x, HF_x], [LF_y, HF_y])
i_pred = IHK.predict(test_x, return_std=False)
r_pred = RHK.predict(test_x, return_std=False)

plot_scatter(ground_truth, i_pred, r_pred, title="Branin function")

i_error, r_error = cal_error(ground_truth, i_pred, r_pred)
print("IHK error: ", i_error)
print("RHK error: ", r_error)

plot_Branin(test_x, ground_truth, IHK, RHK)