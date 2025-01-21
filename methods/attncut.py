import torch
import torch.nn as nn
import pandas as pd

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Define the AttnCut Model
class AttnCut(nn.Module):
    def __init__(self, input_size: int = 1, d_model: int = 64, n_head: int = 2, num_layers: int = 1,
                 dropout: float = 0.2):
        super(AttnCut, self).__init__()
        self.pre_encoding = nn.LSTM(input_size=input_size, hidden_size=32, num_layers=1, batch_first=True,
                                    bidirectional=True)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dropout=dropout)
        self.encoding_layer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decison_layer = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=1),
            nn.Sigmoid()  # Outputs probabilities for each position
        )

    def forward(self, x):
        x = self.pre_encoding(x)[0]  # Pre-encoding using LSTM
        x = self.encoding_layer(x)  # Transformer Encoder
        output = self.decison_layer(x)  # Decision Layer (per position probabilities)
        return output


# Load Data
calibration_data = pd.read_csv("/content/validations_with_scores.csv")  # Calibration dataset
test_data = pd.read_csv("/content/tests_with_scores.csv")  # Test dataset

# Ensure consistent UserId sets
common_user_ids = set(calibration_data["userId"]).intersection(set(test_data["userId"]))
calibration_data = calibration_data[calibration_data["userId"].isin(common_user_ids)]
test_data = test_data[test_data["userId"].isin(common_user_ids)]

# Sort data by scores in descending order (from the recommender system)
calibration_data = calibration_data.sort_values(by=["userId", "score"], ascending=[True, False])
test_data = test_data.sort_values(by=["userId", "score"], ascending=[True, False])


# Prepare Data
def prepare_user_data(groups, seq_length=300):
    user_tensors = []
    labels_tensors = []
    user_ids = []
    for user, group in groups:
        scores = torch.tensor(group["score"].values, dtype=torch.float32, device=device)
        labels = torch.tensor(group["label"].values, dtype=torch.float32, device=device)
        # Pad or truncate
        scores = torch.nn.functional.pad(scores, (0, max(0, seq_length - len(scores))))[:seq_length]
        labels = torch.nn.functional.pad(labels, (0, max(0, seq_length - len(labels))))[:seq_length]
        user_tensors.append(scores)
        labels_tensors.append(labels)
        user_ids.append(user)
    return torch.stack(user_tensors).unsqueeze(-1), torch.stack(labels_tensors), user_ids


# Group Data
calibration_groups = calibration_data.groupby("userId")
test_groups = test_data.groupby("userId")

# Prepare Calibration and Test Data
calibration_scores, calibration_labels, calibration_user_ids = prepare_user_data(calibration_groups)
test_scores, test_labels, test_user_ids = prepare_user_data(test_groups)

# Initialize Model
model = AttnCut(input_size=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training Loop
epochs = 20
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    # Forward Pass
    output = model(calibration_scores).squeeze(-1)

    # Ground Truth for Loss
    ideal_k = calibration_labels.argmax(dim=1)  # Ground truth cutoff positions
    selected_probs = output[torch.arange(output.size(0)), ideal_k]  # Selected probabilities for ideal_k
    loss = -torch.log(selected_probs + 1e-8).mean()  # Negative log-probability for selected positions

    # Backpropagation
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# Calibration Phase
model.eval()
with torch.no_grad():
    calibration_probs = model(calibration_scores).squeeze(-1)
    user_k_values = calibration_probs.argmax(dim=1) + 1  # User-specific k values from calibration
    user_k_map = {user_id: k.item() for user_id, k in zip(calibration_user_ids, user_k_values.cpu().numpy())}

# Print user-specific k values
print("\nUser-specific k values after calibration:")
for user_id, k in user_k_map.items():
    print(f"UserId: {user_id}, Optimal k: {k}")

# Test Phase
with torch.no_grad():
    test_probs = model(test_scores).squeeze(-1)


# Metrics Calculation
def evaluate_metrics(user_probs, labels, k):
    k = min(k, 25)  # Clamp k to a maximum of 50 for testing
    hits = labels[:k].sum().item()
    recall = hits / labels.sum().item() if labels.sum().item() > 0 else 0.0
    dcg = (labels[:k] / torch.log2(torch.arange(2, k + 2, device=device))).sum().item()
    idcg = (torch.sort(labels, descending=True).values[:k] / torch.log2(
        torch.arange(2, k + 2, device=device))).sum().item()
    ndcg = dcg / idcg if idcg > 0 else 0.0
    mrr = (labels[:k] / torch.arange(1, k + 1, device=device)).max().item()

    return recall, ndcg, mrr


# Evaluate on Test Data
metrics = []
for i, user_probs in enumerate(test_probs):
    user_id = test_user_ids[i]
    k = user_k_map[user_id]  # Use calibration k
    metrics.append(evaluate_metrics(user_probs, test_labels[i], k))

# Print Metrics
avg_metrics = torch.tensor(metrics, device=device).mean(dim=0)
print(f"\nMetrics -> Recall: {avg_metrics[0]:.4f}, NDCG: {avg_metrics[1]:.4f}, MRR: {avg_metrics[2]:.4f}")
