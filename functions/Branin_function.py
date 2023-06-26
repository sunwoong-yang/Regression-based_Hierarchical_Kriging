import numpy as np
from surrogate_model.HK import HK
import matplotlib.pyplot as plt
from pyDOE import lhs


"""
3 level Branin function can be found here
[Ref] Perdikaris, P., Raissi, M., Damianou, A., Lawrence, N. D., & Karniadakis, G. E. (2017). Nonlinear information fusion algorithms for data-efficient multi-fidelity modelling. Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences, 473(2198), 20160751.
"""


def LF_function(x): # low-fidelity function
    x1, x2 = x[:, 0], x[:, 1]
    return MF_function(1.2 * (x+2)) - 3 * x2 + 1

def MF_function(x): # high-fidelity function
    x1, x2 = x[:, 0], x[:, 1]
    return 10 * (HF_function(x-2))**0.5 + 2 * (x1 - 0.5) - 3 * (3*x2 - 1) - 1

def HF_function(x): # high-fidelity function
    x1, x2 = x[:,0], x[:,1]
    return (-1.275 * x1**2 / np.pi**2 + 5 * x1 / np.pi + x2 - 6)**2 + (10 - 5/4/np.pi) * np.cos(x1) + 10


LF_x = lhs(2, samples=80, criterion='maximin')
MF_x = lhs(2, samples=40, criterion='maximin')
HF_x = lhs(2, samples=20, criterion='maximin')

for x in [LF_x, MF_x, HF_x]:
    """
    Since design space of x1 & x2 = [-5, 10] & [0, 15]
    """
    x[:, 0] = 15 * x[:, 0] - 5
    x[:, 1] = 15 * x[:, 1]

LF_y = LF_function(LF_x).reshape(-1,1)
MF_y = MF_function(MF_x).reshape(-1,1)
HF_y = HF_function(HF_x).reshape(-1,1)
# print(HF_y)

hk = HK(x=[LF_x, MF_x, HF_x], y=[LF_y, MF_y, HF_y], n_pop=[100,100,100], n_gen=[100,100,100], HKtype="r")
hk.fit(history=False)

test_x = lhs(2, samples=100, criterion='maximin')
test_x[:, 0] = 15 * test_x[:, 0] - 5
test_x[:, 1] = 15 * test_x[:, 1]


pred_LF_y_HK = hk.predict(test_x, pred_fidelity=0, return_std=False)
pred_MF_y_HK = hk.predict(test_x, pred_fidelity=1, return_std=False)
pred_HF_y_HK = hk.predict(test_x, pred_fidelity=2, return_std=False)

fig, ax = plt.subplots(dpi=300)
ax.scatter(HF_function(test_x),pred_HF_y_HK, edgecolors='C4', label="RHK", facecolors='none')
lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
ax.plot(lims, lims, '--k')
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.legend(fontsize=15)
plt.show()