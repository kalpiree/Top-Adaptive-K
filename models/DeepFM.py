import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepFM(nn.Module):
    def __init__(self, num_factors, num_features, deep_layers=None):
        """

        Args:
          num_factors:
          num_features:
          deep_layers:
        """
        super(DeepFM, self).__init__()
        if deep_layers is None:
            deep_layers = [50, 25, 10]  # Default architecture of deep layers

        # FM Part
        self.linear = nn.Linear(num_features, 1)
        self.V = nn.Parameter(torch.randn(num_features, num_factors) * 0.01)

        # Deep Part
        self.deep_layers = nn.ModuleList()
        input_size = num_features
        for layer_size in deep_layers:
            self.deep_layers.append(nn.Linear(input_size, layer_size))
            self.deep_layers.append(nn.ReLU())
            input_size = layer_size
        self.deep_layers.append(nn.Linear(input_size, 1))  # Output layer of the deep part

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.linear.weight, std=0.01)
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0)
        for layer in self.deep_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x):
    # FM Part
        linear_part = self.linear(x)
        interaction_part = 0.5 * (torch.pow(torch.mm(x, self.V), 2) - torch.mm(torch.pow(x, 2), torch.pow(self.V, 2))).sum(1, keepdim=True)
        fm_part = linear_part + interaction_part

        # Deep Part
        deep_output = x
        for layer in self.deep_layers:
            deep_output = layer(deep_output)

        # Combine FM and deep part outputs
        output = fm_part + deep_output.view(-1, 1)  # Ensure this is reshaped properly if necessary
        return torch.sigmoid(output)
