from ex_functions.Test_function_4 import *
from run_functions.train_models import train_models
from PrePost.plot_scatter import plot_scatter
from pyDOE import lhs

in_dim = 2

LF_x = lhs(in_dim, samples=100, criterion='maximin') * 1
MF_x = lhs(in_dim, samples=50, criterion='maximin') * 1
HF_x = lhs(in_dim, samples=10, criterion='maximin') * 1

LF_y = LF_function(LF_x).reshape(-1, 1)
MF_y = MF_function(MF_x).reshape(-1, 1)
HF_y = HF_function(HF_x).reshape(-1, 1)

test_x = lhs(in_dim, samples=100, criterion='maximin') * 1
ground_truth = HF_function(test_x)

IHK, RHK = train_models([LF_x, MF_x, HF_x], [LF_y, MF_y, HF_y])
plot_scatter(test_x, ground_truth, IHK, RHK)