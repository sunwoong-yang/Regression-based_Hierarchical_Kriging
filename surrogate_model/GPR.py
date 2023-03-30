from sklearn.gaussian_process import GaussianProcessRegressor as sklearn_GPR
import sklearn.gaussian_process.kernels as sklearn_kernels
from PrePost.PrePost import *


class GPR(sklearn_GPR):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = sklearn_GPR(**self.kwargs)

    def fit(self, train_x, train_y):
        self.train_x, self.train_y = train_x, train_y
        train_x_normalized, self.x_scaler = normalize(train_x)
        train_y_normalized, self.y_scaler = normalize(train_y)
        self.input_dim = train_x.shape[1]

        if ("kernel" not in self.kwargs) or (self.kwargs["kernel"] is None):
            self.kwargs["kernel"] = sklearn_kernels.ConstantKernel(1, constant_value_bounds=[(1e-1,1e3)] * 1) * \
                                    sklearn_kernels.RBF(np.ones(self.input_dim), length_scale_bounds=[(1e-1, 1e3)] * self.input_dim)

        self.model.fit(train_x_normalized, train_y_normalized) # 모델학습은 scaling된 후의 데이터로 수행되어야함

    def predict(self, X, return_std=False):
        scaled_X = self.x_scaler.transform(X)
        if return_std:
            scaled_Y, scaled_std = self.model.predict(scaled_X, return_std=return_std)
            return self.y_scaler.inverse_transform(scaled_Y.reshape(-1,1)), self.y_scaler.scale_ * scaled_std # scaled된 std에 scaling std가 곱해져서 나와야
        else:
            scaled_Y = self.model.predict(scaled_X, return_std=return_std)
            return self.y_scaler.inverse_transform(scaled_Y.reshape(-1,1))
        # return self.model.predict(scaled_X, return_std=return_std)