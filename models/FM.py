import torch
from torch import nn


class FactorizationMachine(nn.Module):
    def __init__(self, num_factors, num_features):
        """

        Args:
          num_factors:
          num_features:
        """
        super(FactorizationMachine, self).__init__()
        self.linear = torch.nn.Linear(num_features, 1)
        self.V = torch.nn.Parameter(torch.randn(num_features, num_factors) * 0.01)

    def forward(self, x):
        linear_part = self.linear(x)
        interaction_part = 0.5 * (torch.pow(torch.mm(x, self.V), 2) - torch.mm(torch.pow(x, 2), torch.pow(self.V, 2))).sum(1, keepdim=True)
        output = linear_part + interaction_part
        return torch.sigmoid(output)