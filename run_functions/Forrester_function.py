from functions.Forrester_function import *
from run_functions.train_models import train_models
from PrePost.plot_scatter import plot_scatter
from PrePost.cal_error import cal_error
import matplotlib.pyplot as plt

in_dim = 1

# LF_x = lhs(in_dim, samples=80, criterion='maximin')
# MF_x = lhs(in_dim, samples=40, criterion='maximin')
# HF_x = lhs(in_dim, samples=20, criterion='maximin')
LF_x = np.linspace(0, 1, 21).reshape(-1,1)
MF_x = np.linspace(0, 1, 8).reshape(-1,1)
HF_x = np.array([0, 0.4, 0.6, 1]).reshape(-1,1)

LF_y = LF_function(LF_x).reshape(-1, 1)
MF_y = MF_function(MF_x).reshape(-1, 1)
HF_y = HF_function(HF_x).reshape(-1, 1)

test_x = np.linspace(0, 1, 100).reshape(-1, 1)
ground_truth = HF_function(test_x)

IHK, RHK = train_models([LF_x, MF_x, HF_x], [LF_y, MF_y, HF_y], history=True)
# IHK_2level, RHK_2level = train_models([LF_x, HF_x], [LF_y, HF_y])
i_pred = IHK.predict(test_x, return_std=False)
r_pred = RHK.predict(test_x, return_std=False)

plot_scatter(ground_truth, i_pred, r_pred, title="Forrester function")

i_error, r_error = cal_error(ground_truth, i_pred, r_pred)
print("IHK error: ", i_error)
print("RHK error: ", r_error)

plot_Forrester(test_x, ground_truth, IHK, RHK)