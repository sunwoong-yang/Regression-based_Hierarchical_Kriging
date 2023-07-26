import numpy as np
import scipy
import math

class HKVariable:
    def __init__(self):
        self.__total_opt_theta = []
        self.__total_opt_nugget =[]
        self.__total_F = []
        self.__total_R = []
        self.__total_invR = []
        self.__total_beta = []
        self.__total_sigmaSQ = []
        self.__total_MLE = []

        return

    def getParameter(self):
        args = [self.__total_opt_theta, self.__total_opt_nugget, self.__total_F, self.__total_R,
                self.__total_invR, self.__total_beta, self.__total_sigmaSQ, self.__total_MLE]

        return args

    def setParameter(self, args):
        self.__total_opt_theta, self.__total_opt_nugget, self.__total_F, self.__total_R, \
        self.__total_invR, self.__total_beta, self.__total_sigmaSQ, self.__total_MLE = args

        return

# def cal_r(x1, x2, X, HKtype): # cubic spline
#     if HKtype == "i" or HKtype == "I":
#         theta = X
#
#     elif HKtype == "r" or HKtype == "R":
#         theta = X[:-1]
#
#     eps = theta * np.abs(x1 - x2)
#     for enu, temp in enumerate(eps):
#         eps[eps <= 0.2] = 1 - 15 * eps[eps <= 0.2] ** 2 + 30 * eps[eps <= 0.2] ** 3
#         eps[0.2 < eps < 1] = 1.25 * (1 - eps[eps <= 0.2]) ** 3
#         eps[epse >= 1] = 0
#         if temp <= 0.2:
#             eps[enu] = 1 - 15 * temp ** 2 + 30 * temp ** 3
#         elif 0.2 < temp < 1:
#             eps[enu] = 1.25 * (1 - temp) ** 3
#         else:
#             eps[enu] = 0
#
#     return np.prod(eps)

def cal_r(x1, x2, X, HKtype): # Matern (nu=1.5)

    if HKtype == "i" or HKtype == "I":
        theta = X

    elif HKtype == "r" or HKtype == "R":
        theta = X[:-1]

    # https: // scikit - learn.org / stable / modules / gaussian_process.html  # gp-kernels
    # d(x_i, x_j) / l : this term is expressed as eps in this code
    eps = theta * np.abs(x1 - x2)
    eps = (1 + np.sqrt(3) * eps) * np.exp(-np.sqrt(3) * eps)

    return np.prod(eps)

def cal_r_vector (x1, x2, X, HKtype) : # x_test shape : (test 데이터 개수, N_dimension)
    r_vector = np.zeros((x1.shape[0], x2.shape[0])) # r_vector shape : ( 해당 레벨에서 실제 샘플 데이터 개수,test 데이터 개수) --> 이렇게해야 추후 식들에서 r의 차원이 맞음

    for enu1, temp1 in enumerate(x2) :
        r_vector[:,enu1] = np.array([cal_r(temp1,temp2,X,HKtype) for temp2 in x1])

    return r_vector


def cal_R(x, y, X, HKtype):  # x : (N_pts,N_dv)의 shape을 가지는 dv값 array, shape : 1차원으로 N_dv개수만큼 원소있음
    N_pts = x.shape[0]
    R = np.zeros((N_pts, N_pts))

    for i in range(N_pts):
        for j in range(i + 1, N_pts):
            R[i, j] = cal_r(x[i], x[j], X, HKtype)

    if HKtype == "i" or HKtype == "I":
        nugget = 10 ** -12
    elif HKtype == "r" or HKtype == "R":
        nugget = X[-1]
        # nugget = 10 ** -12

    return R + R.transpose() + (1 + nugget) * np.identity(N_pts)

def cal_beta(Y, F, invR, transF) :
    temp_beta = 1 / ((F @ invR @ transF))

    return temp_beta * transF @ invR @ Y

def cal_sigmaSQ(N_pts, Y, F, beta, invR):
    return 1 / N_pts * (Y - F * beta).transpose() @ invR @ (Y - F * beta)

def cal_regression_sigmaSQ(N_pts, Y, F, beta, R, invR, nugget):
    return 1 / N_pts * (Y - F * beta).transpose() @ invR @ ( R - nugget * np.identity(N_pts)) @ invR @ (Y - F * beta)

def cal_MLE(N_pts, sigmaSQ,R):
    det_R = np.linalg.det(R)

    if det_R == 0 :
        return np.float64(-9999.) # det_R = 0 나오면 MLE가 inf가 나와서 해당 theta가 최적해로 뽑혀버림. 이를 방지위해
    else :
        MLE = - (N_pts/2) * np.log(sigmaSQ) - 1 / 2 * np.log(np.linalg.det(R)) # MEDOC version Log-likelihood

    return MLE