import torch
import torch.nn as nn

class SimpleRegressionNet(nn.Module):
    def __init__(self, n_input, n_output, n_hidden, n_layer):
        """
        creates a variable net with linear layers in sequence with dropout and ReLU
        :param n_input: input layer size
        :param n_output: output layer size
        :param n_hidden: hidden layer size
        :param n_layer: number of hidden layers
        """
        super().__init__()

        self.activation = nn.ReLU()  # Activation function
        self.dropout = nn.Dropout(p=0.2)

        self.layers = nn.ModuleList([nn.Linear(n_input, n_hidden), self.activation])
        for i in range(n_layer):
            self.layers.extend([nn.Linear(n_hidden, n_hidden), self.activation])
        self.layers.append(nn.Linear(n_hidden, n_output))

        # init layer weights for faster convergence
        for layer in self.layers:
            if hasattr(layer, "weight"):
                nn.init.xavier_uniform(layer.weight)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.dropout(x)
        return x
