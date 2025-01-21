import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class SASRec(nn.Module):
    def __init__(self, num_items, embedding_size, num_heads, num_layers, dropout=0.1):
        super(SASRec, self).__init__()
        self.item_embedding = nn.Embedding(num_items, embedding_size)
        encoder_layers = TransformerEncoderLayer(embedding_size, num_heads, dropout=dropout)
        self.transformer = TransformerEncoder(encoder_layers, num_layers)

        # Additional layer to transform the last sequence embedding into a binary output
        self.output_layer = nn.Linear(embedding_size, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        # Initialize output layer weights
        nn.init.xavier_uniform_(self.output_layer.weight)
        if self.output_layer.bias is not None:
            nn.init.constant_(self.output_layer.bias, 0)

    def forward(self, item_sequences):
        item_embeddings = self.item_embedding(item_sequences)
        output = self.transformer(item_embeddings)
        # Assuming we use the last item's embedding for prediction
        last_item_embedding = output[:, -1, :]  # Get the last timestep's embeddings
        output = self.output_layer(last_item_embedding)  # Transform to a binary output
        return torch.sigmoid(output).squeeze()  # Ensure output is between 0 and 1
