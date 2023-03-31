import torch.nn as nn
from surrogate_model.MLP import MLP

class AUX_MLP(MLP):
    def __init__(self, input_dim, hidden_layers, activation, output_dim):
        super(AUX_MLP, self).__init__(input_dim, hidden_layers, activation, output_dim)  # Same as super().__init__(input_dim, hidden_layers, output_dim, activation)

        self.final_layer = nn.Linear(hidden_layers[-1], 2 * output_dim)

    def forward(self, x):
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            x = self.activation(x)

        final_output = self.final_layer(x)
        mean, var = final_output.chunk(2, dim=-1)
        var = nn.functional.softplus(var) + 1e-6
        return mean, var