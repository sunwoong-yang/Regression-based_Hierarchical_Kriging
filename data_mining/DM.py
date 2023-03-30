from SALib.analyze import sobol
from SALib.sample.sobol import sample as sobol_sampling
import numpy as np

class DM():
    """
    input, output으로 들어가는건 Python 사용자의 다양성을 고려하여 tensor가 아닌 np.array
    """

    def __init__(self, model):
        self.model = model

    def sweep_1d(self, x_sweep, idx_sweep=0):
        x = np.zeros((len(x_sweep), self.model.input_dim))
        x[:, idx_sweep] = x_sweep
        y = self.model.predict(x)

        return y

    def Sobol(self, explored_space, N_sample):
        problem, sample, predictions = self.sample_saltelli(explored_space, N_sample)

        return sobol.analyze(problem, predictions, calc_second_order=True)

    def ANOVA(self, explored_space, N_sample):
        problem, sample, predictions = self.sample_saltelli(explored_space, N_sample)
        model_mean = predictions.mean()
        dof = len(sample) - 1
        predictions_squared = predictions ** 2
        mean_squared_predictions = predictions_squared.mean()
        ss_residual = ((predictions - model_mean) ** 2).sum()
        ss_total = ((predictions - predictions.mean()) ** 2).sum()
        ms_regression = ss_total / dof
        ms_residual = ss_residual / dof
        f_value = ms_regression / ms_residual

        return f_value, ms_regression, ms_residual

    def sample_saltelli(self, explored_space, N_sample):
        num_vars = len(explored_space)
        problem = {
            'num_vars': num_vars,
            'names': [f'x{i + 1}' for i in range(num_vars)],
            'bounds': explored_space
        }
        sample = sobol_sampling(problem, N_sample)
        predictions = self.model.predict(sample).reshape(-1)

        return problem, sample, predictions