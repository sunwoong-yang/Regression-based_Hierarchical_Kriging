import torch.nn as nn
from PrePost.PrePost import *


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, activation, output_dim):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = nn.ModuleList()

        self.hidden_layers.append(nn.Linear(input_dim, hidden_layers[0]))
        
        for i in range(len(hidden_layers) - 1):
            self.hidden_layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
        
        self.final_layer = nn.Linear(hidden_layers[-1], output_dim)

        if activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU()
        elif activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'Sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'GELU':
            self.activation = nn.GELU()
        elif activation == 'SiLU':
            self.activation = nn.SiLU()
        else:
            assert ('Invalid activation!')
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")

    def forward(self, x):
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            x = self.activation(x)
        x = self.final_layer(x)
        return x

    def fit(self, train_x, train_y, num_epochs, criterion, optimizer):
        self.train_x, self.train_y = Num2Ten(train_x), Num2Ten(train_y)
        train_x_normalized, self.x_scaler = normalize(train_x)
        train_y_normalized, self.y_scaler = normalize(train_y)
        train_x_normalized, train_y_normalized = Num2Ten(train_x_normalized), Num2Ten(train_y_normalized)
        inputs, y_real = train_x_normalized.to(self.device), train_y_normalized.to(self.device)

        for epoch in range(num_epochs):
            # for inputs, y_real in dataloader:
            # inputs, y_real = train_x_normalized.to(self.device), train_y_normalized.to(self.device)
            optimizer.zero_grad()
            y_pred = self.forward(inputs)
            loss = criterion(y_real, y_pred)
            # loss = self.NLLloss(y_real, y_pred, var_pred)
            loss.backward()
            optimizer.step()


    def predict(self, X):
        with torch.no_grad():
            scaled_X = Num2Ten(self.x_scaler.transform(X))
            scaled_Y = Ten2Num(self.forward(scaled_X))
        return self.y_scaler.inverse_transform(scaled_Y)

