import torch
import torch.nn as nn

from surrogate_model.AUX_MLP import AUX_MLP
from PrePost.PrePost import *

class DeepEnsemble(nn.Module):
    def __init__(self, input_dim, hidden_layers, activation, output_dim, num_models):
        super(DeepEnsemble, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.models = nn.ModuleList()
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.models = self.models.to(self.device)


        for _ in range(num_models):
            self.models.append(AUX_MLP(input_dim, hidden_layers, activation, output_dim))

    def forward(self, x):
        # print(model(x))
        means = torch.stack([NN(x)[0] for NN in self.models], dim=-1)
        vars = torch.stack([NN(x)[1] for NN in self.models], dim=-1)
        mean = torch.mean(means, dim=-1)
        alea_var = torch.mean(vars.pow(2), dim=-1)
        epis_var = torch.mean(means.pow(2), dim=-1) - mean.pow(2)
        var = alea_var + epis_var

        return mean, var, alea_var, epis_var

    def fit(self, train_x, train_y, num_epochs, optimizer):
        self.train_x, self.train_y = Num2Ten(train_x), Num2Ten(train_y)
        train_x_normalized, self.x_scaler = normalize(train_x)
        train_y_normalized, self.y_scaler = normalize(train_y)
        train_x_normalized, train_y_normalized = Num2Ten(train_x_normalized), Num2Ten(train_y_normalized)
        for model in self.models:
            for epoch in range(num_epochs):
                # for inputs, y_real in dataloader:
                inputs, y_real = train_x_normalized.to(self.device), train_y_normalized.to(self.device)
                optimizer.zero_grad()
                y_pred, var_pred = model(inputs)
                # loss = criterion(predict, y_real)
                loss = NLLloss(y_real, y_pred, var_pred)
                loss.backward()
                optimizer.step()
                # print(f'Epoch {epoch+1} Loss: {loss.item()}')

    def predict(self, X, return_var=False):
        with torch.no_grad():
            scaled_X = Num2Ten(self.x_scaler.transform(X))
            scaled_mean, scaled_var, scaled_alea_var, scaled_epis_var = self.forward(scaled_X)
            scaled_mean, scaled_var, scaled_alea_var, scaled_epis_var = Ten2Num(scaled_mean), Ten2Num(scaled_var), Ten2Num(scaled_alea_var), Ten2Num(scaled_epis_var)
            # X = Num2Ten(X)
            # mean, var, alea_var, epis_var = self.forward(X)
            if return_var:
                return self.y_scaler.inverse_transform(scaled_mean), self.y_scaler.scale_ * scaled_var**0.5, self.y_scaler.scale_ * scaled_alea_var**0.5, self.y_scaler.scale_ * scaled_epis_var**0.5
                # return tuple(y.detach().numpy() for y in self.forward(X))  # return mean, alea_var, epis_var, var
            else:
                return self.y_scaler.inverse_transform(scaled_mean)  # only prediction values