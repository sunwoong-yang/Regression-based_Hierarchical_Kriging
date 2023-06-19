from surrogate_model.MLP import *
from PrePost.PrePost import *
from sklearn.preprocessing import StandardScaler as Scaler
import torch.optim as optim
class MFDNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MFDNN, self).__init__()
        self.device = torch.device("cpu")
        self.N_fidelity = 0
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.MLP_list = nn.ModuleList()
        self.criterion = []
        self.lr = []
        self.epochs = []

    def fit(self, train_x=[], train_y=[]):
        if (self.input_dim != len(train_x)) or (self.output_dim != len(train_y)):
            assert ('Mismatch between dim(inp/out) and len(X/Y)!')
        self.train_x = [Num2Ten(x) for x in train_x]
        self.train_y = [Num2Ten(y) for y in train_y]
        # self.train_x, self.train_y = Num2Ten(train_x), Num2Ten(train_y)
        self.x_scaler, self.y_scaler = self.get_scaler(train_x, train_y)
        optimizers = []
        for fidelity in range(self.N_fidelity):
            optimizers.append(optim.Adam(self.MLP_list[fidelity].parameters(), lr=self.lr[fidelity]))
        # train_x_normalized, train_y_normalized = Num2Ten(train_x_normalized), Num2Ten(train_y_normalized)

        for fidelity in range(self.N_fidelity):
            epochs, criterion, optimizer = self.epochs[fidelity], self.criterion[fidelity], optimizers[fidelity]
            # inputs, y_real = train_x_normalized[fidelity].to(self.device), train_y_normalized[fidelity].to(self.device)

            if fidelity == 0:
                # Train the lowest MLP with MLP.py
                self.MLP_list[fidelity].fit(self.use_scaler(self.train_x[fidelity], self.x_scaler[fidelity]),
                                            self.use_scaler(self.train_y[fidelity], self.y_scaler[fidelity]),
                                            epochs, criterion, optimizer)

            else:
                for epoch in range(epochs):
                    #이 아래를 통째로 prediction으로 치환못하려나
                    # mlp_out = self.MLP_list[0](self.use_scaler(self.train_x[fidelity], self.x_scaler[0]))
                    # for sub_fidelity in range(1, fidelity+1):
                    #     mlp_inp = torch.cat([
                    #         mlp_out, self.use_scaler(self.train_x[fidelity], self.x_scaler[sub_fidelity])
                    #     ], dim=1)
                    #     mlp_out = self.MLP_list[sub_fidelity](mlp_inp)
                    mlp_out = self.predict(self.train_x[fidelity], pred_fidelity=fidelity, fit=True)

                    loss = criterion(self.use_scaler(self.train_y[fidelity], self.y_scaler[fidelity]), mlp_out)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

    def predict(self, X, pred_fidelity=None, fit=False):
        if pred_fidelity is None:
            pred_fidelity = self.N_fidelity-1

        with torch.set_grad_enabled(fit):
            mlp_out = self.MLP_list[0](self.use_scaler(X, self.x_scaler[0]))
            for sub_fidelity in range(1, pred_fidelity+1):
                mlp_inp = torch.cat([
                    mlp_out, self.use_scaler(X, self.x_scaler[sub_fidelity])], dim=1)
                mlp_out = self.MLP_list[sub_fidelity](mlp_inp)

            # return Ten2Num(self.use_scaler(mlp_out, self.y_scaler[pred_fidelity]))
        if fit:
            return mlp_out
        else:
            return self.use_scaler(mlp_out, self.y_scaler[pred_fidelity], inv_transform=True)
    def add_fidelity(self, hidden_layers, activation, criterion, lr, epochs):
        self.N_fidelity += 1
        self.criterion.append(criterion)
        self.lr.append(lr)
        self.epochs.append(epochs)

        if self.N_fidelity == 1:
            # NN of the lowest fidelity does not need to change inp/out dimensions
            self.MLP_list.append(MLP(self.input_dim, hidden_layers, activation, self.output_dim).to(self.device))
        else:
            # NN of the other fidelity needs to change inp dimension (inp --> inp+out)
            self.MLP_list.append(MLP(self.input_dim + self.output_dim, hidden_layers, activation, self.output_dim).to(self.device))

    def get_scaler(self, X, Y):
        x_scaler, y_scaler = [], []
        # scaled_x, scaled_y = [], []
        for x, y in zip(X, Y):
            STD_x, STD_y = Scaler(), Scaler()
            STD_x.fit(x)
            STD_y.fit(y)
            x_scaler.append(STD_x)
            y_scaler.append(STD_y)

        return x_scaler, y_scaler

    def use_scaler(self, x, scaler, inv_transform=False):
        if not inv_transform:
            return Num2Ten(scaler.transform(x))
        else:
            return Num2Ten(scaler.inverse_transform(x))