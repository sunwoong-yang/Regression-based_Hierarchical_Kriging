from ex_functions.Test_function_7 import *
from run_functions.train_models import train_models
from PrePost.plot_scatter import plot_scatter
from pyDOE import lhs

in_dim = 15

LF_x = lhs(in_dim, samples=300, criterion='maximin') * 1 + 0.5
MF_x = lhs(in_dim, samples=200, criterion='maximin') * 1 + 0.5
HF_x = lhs(in_dim, samples=100, criterion='maximin') * 1 + 0.5

LF_y = LF_function(LF_x).reshape(-1, 1)
MF_y = MF_function(MF_x).reshape(-1, 1)
HF_y = HF_function(HF_x).reshape(-1, 1)

test_x = lhs(in_dim, samples=100, criterion='maximin') * 1 + 0.5
ground_truth = HF_function(test_x)

IHK, RHK = train_models([LF_x, MF_x, HF_x], [LF_y, MF_y, HF_y])
plot_scatter(test_x, ground_truth, IHK, RHK, title="Function 7")