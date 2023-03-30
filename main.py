import torch!
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
# !pip install SALib
from SALib.analyze import sobol
from SALib.sample.sobol import sample as sobol_sampling
from statsmodels.stats.anova import anova_lm
import scipy.optimize as optimize
from sklearn.gaussian_process import GaussianProcessRegressor as sklearn_GPR
import sklearn.gaussian_process.kernels as sklearn_kernels

# !pip install -U pymoo
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

#@title Pre-requisites
def Dat2Ten(dataloader):
    x_total, y_total = [], []
    for x,y in dataloader:
        x_total.append(x)
        y_total.append(y)
    return torch.cat(x_total, dim=0), torch.cat(y_total, dim=0)

def Dat2Num(dataloader):
    x_total, y_total = [], []
    for x,y in dataloader:
        x_total.append(x)
        y_total.append(y)
    return np.concatenate(x_total, axis=0), torch.concatenate(y_total, axis=0)

def Num2Ten(x):
    return torch.tensor(x, dtype=torch.float32)

def NLLloss(y_real, y_pred, var):
    return (torch.log(var) + ((y_real - y_pred).pow(2))/var).mean()/2 + 0.5*np.log10(2*np.pi)

class Data_Generation(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

#@title Surrogates
"""
모든 Surrogate는 fit과 predict 메서드를 가지며, 이 두 메서드는 input과 output을 np.array 형태로 받는다
"""
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim, activation):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = nn.ModuleList()

        self.hidden_layers.append(nn.Linear(input_dim, hidden_layers[0]))
        
        for i in range(len(hidden_layers) - 1):
            self.hidden_layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
        
        self.final_layer = nn.Linear(hidden_layers[-1], output_dim)
        
        self.activation = activation
        
    def forward(self, x):
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            x = self.activation(x)
        x = self.final_layer(x)
        return x

    def fit(self, dataloader, num_epochs, criterion, optimizer):
        self.train_x, self.train_y = Dat2Ten(dataloader)
        for epoch in range(num_epochs):
            for inputs, y_real in dataloader:
                inputs, y_real = inputs.to(device), y_real.to(device)
                optimizer.zero_grad()
                y_pred = self.forward(inputs)
                loss = criterion(y_real, y_pred)
                # loss = self.NLLloss(y_real, y_pred, var_pred)
                loss.backward()
                optimizer.step()

    def predict(self, X):
        X = Num2Ten(X)
        return self.forward(X).detach().numpy()

class AUX_MLP(MLP):
    def __init__(self, input_dim, hidden_layers, output_dim, activation):
        super(AUX_MLP, self).__init__(input_dim, hidden_layers, output_dim, activation) # Same as super().__init__(input_dim, hidden_layers, output_dim, activation)
        
        self.final_layer = nn.Linear(hidden_layers[-1], 2*output_dim)

    def forward(self, x):
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            x = self.activation(x)
        
        final_output = self.final_layer(x)
        mean, var = final_output.chunk(2, dim=-1)
        var = nn.functional.softplus(var) + 1e-6
        return mean, var

class DeepEnsemble(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim, activation, num_models):
        super(DeepEnsemble, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.models = nn.ModuleList()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = self.models.to(device)

        for _ in range(num_models):
            self.models.append(AUX_MLP(input_dim, hidden_layers, output_dim, activation))
        
    def forward(self, x):
        # print(model(x))
        means = torch.stack([NN(x)[0] for NN in self.models], dim=-1)
        vars = torch.stack([NN(x)[1] for NN in self.models], dim=-1)
        mean = torch.mean(means, dim=-1)
        alea_var = torch.mean(vars.pow(2), dim=-1)
        epis_var = torch.mean(means.pow(2), dim=-1) - mean.pow(2)
        var = alea_var + epis_var
          
        return mean, var, alea_var, epis_var
    
    def fit(self, dataloader, num_epochs, criterion, optimizer):
        self.train_x, self.train_y = Dat2Ten(dataloader)
        for model in self.models:
            for epoch in range(num_epochs):
                for inputs, y_real in dataloader:
                    inputs, y_real = inputs.to(device), y_real.to(device)
                    optimizer.zero_grad()
                    y_pred, var_pred = model(inputs)
                    # loss = criterion(predict, y_real)
                    loss = NLLloss(y_real, y_pred, var_pred)
                    loss.backward()
                    optimizer.step()
                # print(f'Epoch {epoch+1} Loss: {loss.item()}')

    def predict(self, X, return_var = False):
      X = Num2Ten(X)
      if return_var:
        return tuple(i.detach().numpy() for i in self.forward(X)) # return mean, alea_var, epis_var, var
      else:
        return self.forward(X)[0].detach().numpy() # only prediction values

class GPR(sklearn_GPR):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        if self.kwargs["kernel"] is None:
            self.kwargs["kernel"] = sklearn_kernels.ConstantKernel(1, constant_value_bounds = [(1e-1, 1e3)]*1) * sklearn_kernels.RBF(np.ones(self.input_dim), length_scale_bounds = [(1e-1, 1e3)]*self.input_dim)
        self.model = sklearn_GPR(**self.kwargs)
        
    def fit(self, dataloader):
        self.train_x, self.train_y = Dat2Num(dataloader)
        self.input_dim = self.train_x.shape[1]
        self.model.fit(self.train_x, self.train_y)
        
    def predict(self, X, return_std = False):
        return self.model.predict(X, return_std=return_std)

class GPRs(): # 얘를 그냥 기존 GPR에 넣어서 output dim 알아서 감지하고, predict할때는 y_idx넣어서 한 gpr 모델의 output만 내뱉도록 하자
    def __init__(self, n_restarts=None, alpha=None, kernel=None):
      self.n_restarts = n_restarts
      self.alpha = alpha
      self.kernel = kernel
      self.GPRs_model = [ ]
    
    def fit(self, dataloader):
      
      for data in dataloader: # 각 QoI dimension마다 적용되는 for loop가 아님. 수정 필요
        GPR_temp = GPR()
        GPR_temp.fit(data)
        self.GPRs_model.append(GPR_temp)
      
    def predict(self, X, return_std = False):
      return [model.predict(X, return_std=return_std) for model in self.GPRs_model]

###########################################################################################################################

#@title Data MIning
class DM():
    """
    input, output으로 들어가는건 Python 사용자의 다양성을 고려하여 tensor가 아닌 np.array
    """
    def __init__(self, model):
        self.model = model
    
    def sweep_1d(self, x_sweep, idx_sweep = 0):
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
        'names': [f'x{i+1}' for i in range(num_vars)],
        'bounds': explored_space
        }
        sample = sobol_sampling(problem, N_sample)
        predictions = self.model.predict(sample).reshape(-1)
        
        return problem, sample, predictions

def Optimize_Scipy(models=[], weights=None, **kwargs): # pymoo: GA, MOGA, scipy.optimize: gradient-based
    """
    Scipy.minimize 참고: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    """
    if weights is None:
        weights = np.ones_like(models, dtype=float)
    def make_scipy_func(x):
        final_func = 0
        for idx, model in enumerate(models):
            final_func += weights[idx] * model.predict(x.reshape(1,-1))
        return final_func

    res = sci_min(make_scipy_func, **kwargs)
    return res

def optimize(self, dv_idx, obj_idx = None, Morm="m", weights = None): # pymoo: GA, MOGA, scipy.optimize: gradient-based
  
  class SO(Problem):
    """
    single-objective
    """
    def __init__(self, dv_idx, Morm = "m"):
        super().__init__(n_var=len(dv_idx),
                        n_obj=1,
                        n_constr=0,
                        xl=lower_bound,
                        xu=upper_bound)

        self.Morm = Morm
        

    def _evaluate(self, x, out, *args, **kwargs):
        obj = self.model.predict(x)
        
        if weights is None:
          if self.Morm == "M":
            out["F"] = -np.sum(obj)
          elif self.Morm == "m":
            out["F"] = np.sum(obj)
        else:
          if self.Morm == "M":
            out["F"] = -np.sum(np.multiply(obj, weights))
          elif self.Morm == "m":
            out["F"] = np.sum(np.multiply(obj, weights))

  class MO(Problem): # Bi-objective optimization

    def __init__(self, dv_idx, obj_idx, Morm = ["M","m"]):
        super().__init__(n_var=len(dv_idx),
                        n_obj=len(obj_idx),
                        n_constr=0,
                        xl=lower_bound,
                        xu=upper_bound)
        
        self.obj_idx = obj_idx
        self.Morm = Morm
        

    def _evaluate(self, x, out, *args, **kwargs):
        obj = self.model.predict(x)
        F = []
        for idx in range(len(self.obj_idx)):
          #여기에 m M 읽고 상황맞게 최적화하도록 수정필요
          if Morm[idx] == "M":
            F.append(-obj[:,idx])
          elif Morm[idx] == "m":
            F.append(obj[:,idx])
          
        out["F"] = np.column_stack(F)
  
  if obj_idx is None:
    SO(dv_idx, Morm = "m")
  else:
    MO(dv_idx, obj_idx, Morm = ["M","m"])

