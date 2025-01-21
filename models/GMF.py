from torch import nn


class GMF(nn.Module):
    def __init__(self, num_users: int, num_items: int, num_factor: int = 8,
                 use_pretrain: bool = False, use_NeuMF: bool = False, pretrained_GMF=None):
        """

        Args:
          num_users:
          num_items:
          num_factor:
          use_pretrain:
          use_NeuMF:
          pretrained_GMF:
        """
        super(GMF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_factor = num_factor
        self.use_pretrain = use_pretrain
        self.use_NeuMF = use_NeuMF
        self.pretrained_GMF = pretrained_GMF

        self.user_embedding = nn.Embedding(num_users, num_factor)
        self.item_embedding = nn.Embedding(num_items, num_factor)
        self.predict_layer = nn.Linear(num_factor, 1)
        self.sigmoid = nn.Sigmoid()

        if use_pretrain and self.pretrained_GMF:
            self._load_pretrained_model()
        else:
            self._init_weight()

    def _init_weight(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        if not self.use_NeuMF:
            nn.init.normal_(self.predict_layer.weight, std=0.01)

    def _load_pretrained_model(self):
        if self.pretrained_GMF:
            self.user_embedding.weight.data.copy_(self.pretrained_GMF.user_embedding.weight)
            self.item_embedding.weight.data.copy_(self.pretrained_GMF.item_embedding.weight)

    def forward(self, users, items):
        user_embedded = self.user_embedding(users)
        item_embedded = self.item_embedding(items)
        interaction = user_embedded * item_embedded
        if not self.use_NeuMF:
            output = self.predict_layer(interaction)
            output = self.sigmoid(output)
            output = output.view(-1)
        else:
            output = interaction
        return output