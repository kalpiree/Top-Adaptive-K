import torch
import torch.nn as nn
import pandas as pd

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define Experts and Towers for MMOECut
class Expert(nn.Module):
    def __init__(self, d_model, n_head, num_layers, dropout=0.2):
        super(Expert, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dropout=dropout)
        self.attention_layer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        return self.attention_layer(x)

class TowerCut(nn.Module):
    def __init__(self, d_model):
        super(TowerCut, self).__init__()
        self.cut_layer = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.cut_layer(x)

class MMOECut(nn.Module):
    def __init__(self, seq_len=300, num_experts=3, num_tasks=1, input_size=1, encoding_size=64, d_model=128, n_head=2, num_layers=1, dropout=0.2):
        super(MMOECut, self).__init__()
        self.seq_len = seq_len
        self.expert_hidden = d_model
        self.pre_encoding = nn.LSTM(input_size=input_size, hidden_size=encoding_size, num_layers=1, batch_first=True, bidirectional=True)
        self.softmax = nn.Softmax(dim=1)
        self.experts = nn.ModuleList([Expert(d_model=self.expert_hidden, n_head=n_head, num_layers=num_layers, dropout=dropout) for _ in range(num_experts)])
        self.gate = nn.Parameter(torch.randn(encoding_size * seq_len * 2, num_experts), requires_grad=True)
        self.tower = TowerCut(self.expert_hidden)

    def forward(self, x):
        experts_in = self.pre_encoding(x)[0]
        experts_out = [e(experts_in) for e in self.experts]
        experts_out_tensor = torch.stack(experts_out)
        batch_size = experts_in.shape[0]
        gate_output = self.softmax(experts_in.reshape(batch_size, -1) @ self.gate)
        weighted_experts = torch.sum(gate_output.t().unsqueeze(2).expand(-1, -1, self.seq_len).unsqueeze(3).expand(-1, -1, -1, self.expert_hidden) * experts_out_tensor, dim=0)
        return self.tower(weighted_experts)

# Load Data
calibration_data = pd.read_csv("/content/validations_with_scores.csv")  # Calibration dataset
test_data = pd.read_csv("/content/tests_with_scores.csv")  # Test dataset

# Ensure consistent UserId sets
common_user_ids = set(calibration_data["userId"]).intersection(set(test_data["userId"]))
calibration_data = calibration_data[calibration_data["userId"].isin(common_user_ids)]
test_data = test_data[test_data["userId"].isin(common_user_ids)]

# Sort data by scores in descending order
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
        scores = torch.nn.functional.pad(scores, (0, max(0, seq_length - len(scores))))[:seq_length]
        labels = torch.nn.functional.pad(labels, (0, max(0, seq_length - len(labels))))[:seq_length]
        user_tensors.append(scores)
        labels_tensors.append(labels)
        user_ids.append(user)
    return torch.stack(user_tensors).unsqueeze(-1), torch.stack(labels_tensors), user_ids

calibration_groups = calibration_data.groupby("userId")
test_groups = test_data.groupby("userId")
calibration_scores, calibration_labels, calibration_user_ids = prepare_user_data(calibration_groups)
test_scores, test_labels, test_user_ids = prepare_user_data(test_groups)

# Initialize Model
model = MMOECut(seq_len=300).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training Loop
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(calibration_scores).squeeze(-1)
    ideal_k = calibration_labels.argmax(dim=1)
    selected_probs = output[torch.arange(output.size(0)), ideal_k]
    loss = -torch.log(selected_probs + 1e-8).mean()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# Calibration Phase
model.eval()
with torch.no_grad():
    calibration_probs = model(calibration_scores).squeeze(-1)
    user_k_values = calibration_probs.argmax(dim=1) + 1
    user_k_map = {user_id: min(k.item(),25) for user_id, k in zip(calibration_user_ids, user_k_values.cpu().numpy())}

# Debugging: Print user-specific k values
print("\nUser-specific k values after calibration:")
for user_id, k in user_k_map.items():
    print(f"UserId: {user_id}, Optimal k: {k}")

# Test Phase
with torch.no_grad():
    test_probs = model(test_scores).squeeze(-1)

# Evaluate Metrics
def evaluate_metrics(user_probs, labels, k):
    k = min(k, 25)
    hits = labels[:k].sum().item()
    recall = hits / labels.sum().item() if labels.sum().item() > 0 else 0.0
    dcg = (labels[:k] / torch.log2(torch.arange(2, k + 2, device=device))).sum().item()
    idcg = (torch.sort(labels, descending=True).values[:k] / torch.log2(torch.arange(2, k + 2, device=device))).sum().item()
    ndcg = dcg / idcg if idcg > 0 else 0.0
    mrr = (labels[:k] / torch.arange(1, k + 1, device=device)).max().item()
    return recall, ndcg, mrr

metrics = []
for i, user_probs in enumerate(test_probs):
    user_id = test_user_ids[i]
    k = user_k_map[user_id]
    metrics.append(evaluate_metrics(user_probs, test_labels[i], k))

# Print Metrics
avg_metrics = torch.tensor(metrics, device=device).mean(dim=0)
print(f"\nMetrics -> Recall: {avg_metrics[0]:.4f}, NDCG: {avg_metrics[1]:.4f}, MRR: {avg_metrics[2]:.4f}")
