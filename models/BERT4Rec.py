import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class BERT4Rec(nn.Module):
    def __init__(self, num_items, embedding_size, num_heads, num_layers, dropout=0.1):
        super(BERT4Rec, self).__init__()
        self.item_embedding = nn.Embedding(num_items, embedding_size)
        encoder_layers = TransformerEncoderLayer(embedding_size, num_heads, dropout=dropout)
        self.transformer = TransformerEncoder(encoder_layers, num_layers)
        self.output_layer = nn.Linear(embedding_size, num_items)  # Assuming output to num_items
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        # Initialize transformer and output layer weights

    def forward(self, item_sequences):
        embeddings = self.item_embedding(item_sequences)
        transformer_output = self.transformer(embeddings)
        output = self.output_layer(transformer_output)
        return torch.sigmoid(output)  # Assuming classification; adjust as needed
