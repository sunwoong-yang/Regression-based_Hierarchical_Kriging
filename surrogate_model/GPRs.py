import numpy as np
from surrogate_model.GPR import GPR
from PrePost.PrePost import *


class GPRs():  # 얘를 그냥 기존 GPR에 넣어서 output dim 알아서 감지하고, predict할때는 y_idx넣어서 한 gpr 모델의 output만 내뱉도록 하자
    def __init__(self, n_restarts=None, alpha=None, kernel=None, **kwargs):
        self.models = []
        self.kwargs = kwargs

    def fit(self, train_x, train_y):
        # for X, Y in dataloader:  # 각 QoI dimension마다 적용되는 for loop가 아님. 수정 필요
        self.train_x, self.train_y = train_x, train_y
        for y_idx in range(self.train_y.shape[1]):
            # individual_data = Ten2Dat(X, Y[:,y_idx])
            individual_gpr = GPR(**self.kwargs)
            individual_gpr.fit(self.train_x, self.train_y[:,[y_idx]])
            self.models.append(individual_gpr)

    def predict(self, X, return_std=False):
        y_set, std_set = [], []

        for model in self.models:
            pred = model.predict(X, return_std=return_std)
            if return_std:
                y_set.append(pred[0].reshape(-1,1))
                std_set.append(pred[1].reshape(-1,1))
            else:
                y_set.append(pred.reshape(-1,1))
        if return_std:
            return np.hstack(y_set), np.hstack(std_set)
        else:
            return np.hstack(y_set)