import torch
import torch.nn as nn
import torch.nn.functional as F

class WMF(nn.Module):
    def __init__(self, num_users, num_items, num_factors):
        super(WMF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, num_factors)
        self.item_embedding = nn.Embedding(num_items, num_factors)

    def forward(self, user, item, weight):
        """

        Args:
          user:
          item:
          weight:

        Returns:

        """
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)
        interaction = torch.sum(user_emb * item_emb, dim=1)
        output = interaction * weight
        return F.sigmoid(output)  # Apply sigmoid activation here