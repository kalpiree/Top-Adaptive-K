
import torch


class Train():
    def __init__(self, model, optimizer, epochs, dataloader, criterion, test_obj, device='cpu',
                 print_cost=True, use_features=False, use_weights=False, model_type='pointwise'):
        """
        Args:
            model: The recommendation model (e.g., GMF, NeuMF, LightGCN).
            optimizer: Optimizer for training the model.
            epochs: Number of training epochs.
            dataloader: Training DataLoader.
            criterion: Loss function.
            test_obj: Validation or test DataLoader.
            device: Device to run the model (e.g., 'cpu' or 'cuda').
            print_cost: Whether to print training logs.
            use_features: Use additional features (e.g., for FM/DeepFM).
            use_weights: Use weights for weighted loss.
            model_type: Specifies the type of model ('pointwise', 'pairwise', or 'LightGCN').
        """
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.dataloader = dataloader
        self.criterion = criterion
        self.device = device
        self.print_cost = print_cost
        self.test = test_obj
        self.use_features = use_features
        self.use_weights = use_weights
        self.model_type = model_type  # Added to handle LightGCN or other model-specific behaviors

    def train(self):
        self.model.to(self.device)
        for epoch in range(self.epochs):
            self.model.train()  # Set model to training mode
            total_loss = 0
            for batch in self.dataloader:
                batch = [x.to(self.device) for x in batch]

                if self.model_type == 'LightGCN':  # Special handling for LightGCN
                    users, pos_items, neg_items = batch  # LightGCN uses triplets
                    outputs = self.model(users, pos_items, neg_items)
                    loss = self.model.get_loss(outputs)
                elif self.use_features:
                    users, items, labels, weights, features, _, _ = batch
                    features = features.float()
                    outputs = self.model(features)
                    labels = labels.float()
                    loss = self.criterion(outputs.view(-1), labels)
                elif self.use_weights:
                    users, items, labels, weights, _, _ = batch
                    outputs = self.model(users, items, weights)
                    labels = labels.float()
                    loss = self.criterion(outputs.view(-1), labels)
                else:
                    users, items, labels, _, _, _ = batch
                    outputs = self.model(users, items)
                    labels = labels.float()
                    loss = self.criterion(outputs.view(-1), labels)

                # Backpropagation and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(self.dataloader)

            # Testing phase
            self.model.eval()  # Set model to evaluation mode
            total_test_loss = 0
            hr, ndcg = 0, 0  # Initialize metrics for validation/testing
            with torch.no_grad():
                for batch in self.test:
                    batch = [x.to(self.device) for x in batch]

                    if self.model_type == 'LightGCN':
                        users, pos_items, neg_items = batch
                        outputs = self.model(users, pos_items, neg_items)
                        test_loss = self.model.get_loss(outputs).item()
                    elif self.use_features:
                        users, items, labels, weights, features, _, _ = batch
                        features = features.float()
                        outputs = self.model(features)
                        labels = labels.float()
                        test_loss = self.criterion(outputs.view(-1), labels).item()
                    elif self.use_weights:
                        users, items, labels, weights, _, _ = batch
                        outputs = self.model(users, items, weights)
                        labels = labels.float()
                        test_loss = self.criterion(outputs.view(-1), labels).item()
                    else:
                        users, items, labels, _, _, _ = batch
                        outputs = self.model(users, items)
                        labels = labels.float()
                        test_loss = self.criterion(outputs.view(-1), labels).item()

                    total_test_loss += test_loss

                    # Optional: Add HR/NDCG evaluation for ranking-based models
                    # hr_batch, ndcg_batch = calculate_metrics(outputs, labels)
                    # hr += hr_batch
                    # ndcg += ndcg_batch

            avg_test_loss = total_test_loss / len(self.test)

            if self.print_cost:
                print(f'Epoch {epoch + 1}: Train Loss: {avg_loss:.4f}, Test Loss: {avg_test_loss:.4f}')
                # Optional: Uncomment if HR/NDCG evaluation is implemented
                # print(f'HR@10: {hr:.4f}, NDCG@10: {ndcg:.4f}')

        if self.print_cost:
            print('Learning finished')
