import numpy as np

from surrogate_model.HK import HK
from PrePost.PrePost import *


class HKs():  # Multi-output HK
    def __init__(self, **kwargs):
        # self.train_x, self.train_y = train_x, train_y
        self.models = []
        self.kwargs = kwargs
        self.x, self.y = self.kwargs.pop("x"), self.kwargs.pop("y")
    def fit(self):
        # for X, Y in dataloader:  # 각 QoI dimension마다 적용되는 for loop가 아님. 수정 필요
        for y_idx in range(self.y[0].shape[1]):
            # Define y_train which has only output of (y_idx)th dimension
            y_train = []
            for y_fidelity in self.y:
                y_train.append(y_fidelity[:, y_idx])
            individual_HK = HK(x=self.x, y=y_train, **self.kwargs)
            individual_HK.fit()
            self.models.append(individual_HK)

    # def predict(self, X, return_std=False): , **kwargs

    def predict(self, **kwargs):
        y_set, std_set = [], []

        for model in self.models:
            pred = model.predict(**kwargs)
            if kwargs['return_std']:
                y_set.append(pred[0].reshape(-1,1))
                std_set.append(pred[1].reshape(-1,1))
            else:
                y_set.append(pred.reshape(-1,1))
        if kwargs['return_std']:
            return np.hstack(y_set), np.hstack(std_set)
        else:
            return np.hstack(y_set)