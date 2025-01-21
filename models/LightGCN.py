import torch
import torch.nn as nn


def normalize_adjacency_matrix(adj):
    """
    Normalize the adjacency matrix using D^-1 * A (row normalization).
    """
    rowsum = torch.sparse.sum(adj, dim=1).to_dense()
    inv_rowsum = torch.pow(rowsum, -1).flatten()
    inv_rowsum[torch.isinf(inv_rowsum)] = 0.0

    diag_inv = torch.sparse_coo_tensor(
        indices=torch.stack([torch.arange(len(inv_rowsum)), torch.arange(len(inv_rowsum))]),
        values=inv_rowsum,
        size=(len(inv_rowsum), len(inv_rowsum)),
        device=adj.device
    )
    normalized_adj = torch.sparse.mm(diag_inv, adj)
    return normalized_adj.coalesce()  # Coalesce the tensor


class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_size, n_layers, adjacency_matrix, gpu):
        """
        Initialize LightGCN with embeddings and adjacency matrix propagation.

        Args:
            num_users (int): Number of users.
            num_items (int): Number of items.
            embedding_size (int): Size of embeddings.
            n_layers (int): Number of layers for propagation.
            adjacency_matrix (torch.sparse_coo_tensor): Sparse adjacency matrix.
            gpu (torch.device): Device for computation (CPU or GPU).
        """
        super(LightGCN, self).__init__()
        self.embedding_size = embedding_size
        self.n_layers = n_layers

        # Embedding layers
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.item_embedding = nn.Embedding(num_items, embedding_size)

        # Normalized adjacency matrix
        self.A = normalize_adjacency_matrix(adjacency_matrix.to(gpu))
        self.A_T = self.A.transpose(0, 1).coalesce()  # Transpose for item-user aggregation

        self.user_list = torch.arange(num_users, dtype=torch.long, device=gpu)
        self.item_list = torch.arange(num_items, dtype=torch.long, device=gpu)

        self.u_eval = None
        self.i_eval = None

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Initialize user and item embeddings with a normal distribution.
        """
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

    def forward(self, user_indices, item_indices):
        """
        Forward pass for LightGCN to compute interaction scores.

        Args:
            user_indices (torch.Tensor): Indices of users in the batch.
            item_indices (torch.Tensor): Indices of items in the batch.

        Returns:
            torch.Tensor: Interaction scores for user-item pairs.
        """
        user_embeddings = self.user_embedding(self.user_list)
        item_embeddings = self.item_embedding(self.item_list)

        # Propagate embeddings using the adjacency matrix
        for _ in range(self.n_layers):
            user_embeddings = torch.sparse.mm(self.A, item_embeddings)  # Users aggregate from items
            item_embeddings = torch.sparse.mm(self.A_T, user_embeddings)  # Items aggregate from users

        user_final_embeddings = user_embeddings[user_indices]
        item_final_embeddings = item_embeddings[item_indices]

        # Compute interaction scores
        interaction_scores = (user_final_embeddings * item_final_embeddings).sum(dim=1)
        return torch.sigmoid(interaction_scores)

    def get_embedding(self):
        """
        Compute the final embeddings for users and items after propagation.

        Returns:
            torch.Tensor, torch.Tensor: User embeddings and item embeddings.
        """
        user_embeddings = self.user_embedding(self.user_list)
        item_embeddings = self.item_embedding(self.item_list)

        for _ in range(self.n_layers):
            user_embeddings = torch.sparse.mm(self.A, item_embeddings)
            item_embeddings = torch.sparse.mm(self.A_T, user_embeddings)

        self.u_eval = user_embeddings
        self.i_eval = item_embeddings

        return user_embeddings, item_embeddings

    def forward_eval(self, batch_user_indices):
        """
        Forward evaluation to compute scores for all items for a batch of users.

        Args:
            batch_user_indices (torch.Tensor): Batch of user indices.

        Returns:
            torch.Tensor: Interaction scores for batch users and all items.
        """
        if self.u_eval is None or self.i_eval is None:
            self.get_embedding()

        interaction_scores = torch.matmul(self.u_eval[batch_user_indices], self.i_eval.T)
        return torch.sigmoid(interaction_scores)

    def get_loss(self, output):
        """
        Compute BPR loss for positive and negative interaction pairs.

        Args:
            output (tuple): Positive and negative scores.

        Returns:
            torch.Tensor: BPR loss.
        """
        pos_score, neg_score = output
        loss = -(pos_score - neg_score).sigmoid().log().sum()
        return loss

