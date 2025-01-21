import pandas as pd
import numpy as np
import torch
from itertools import combinations
from scipy.stats import poisson

# Check CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def poisson_binomial_exact(probabilities, m):
    """Compute exact Poisson-Binomial probability for P(S_u^Y = m-1)."""
    total_prob = 0
    for indices in combinations(range(len(probabilities)), m):
        prod = np.prod([probabilities[i] for i in indices]) * \
               np.prod([1 - probabilities[i] for i in range(len(probabilities)) if i not in indices])
        total_prob += prod
    return total_prob

def poisson_binomial_approximation(probabilities, m):
    """Approximate the Poisson-Binomial distribution for P(S_u^Y = m-1)."""
    mean = probabilities.sum().item()
    variance = (probabilities * (1 - probabilities)).sum().item()
    return poisson.pmf(m, mean) if variance > 0 else 1 if m == round(mean) else 0

def expected_ndcg_at_k(scores, k, num_items, method="approximation"):
    """Compute the expected NDCG at k based on calibrated scores."""
    probabilities = torch.sort(scores, descending=True).values  # Sort scores in descending order
    expected_ndcg = 0

    for m in range(1, num_items + 1):
        # Compute P(S_u^Y = m-1) using the chosen method
        if method == "exact":
            prob_su = poisson_binomial_exact(probabilities.cpu().numpy(), m - 1)
        elif method == "approximation":
            prob_su = poisson_binomial_approximation(probabilities, m - 1)
        else:
            raise ValueError("Invalid method. Choose 'exact' or 'approximation'.")

        # Compute the numerator: \sum_{r=1}^{k} P(Y_{u, r} = 1) P(S_u^Y = m-1)
        r_vals = torch.arange(1, k + 1, device=device)
        discounted_gains = probabilities[:k] * prob_su / torch.log2(1 + r_vals)
        numerator = discounted_gains.sum().item()

        # Compute the denominator: \sum_{r=1}^{\min(m, k)} 1 / \log_2(1 + r)
        denominator = (1 / torch.log2(1 + r_vals[:min(m, k)])).sum().item()

        if denominator > 0:
            expected_ndcg += numerator / denominator

    return expected_ndcg

def true_ndcg(scores, labels, k):
    """Calculate true NDCG@k using true labels."""
    # Sort scores and their associated labels in descending order
    sorted_indices = torch.argsort(scores, descending=True)
    sorted_labels = labels[sorted_indices]

    # Compute DCG@k
    r_vals = torch.arange(2, k + 2, device=device)
    gains = sorted_labels[:k] / torch.log2(r_vals)
    dcg = gains.sum().item()

    # Compute Ideal DCG (IDCG@k)
    ideal_sorted_labels = torch.sort(labels, descending=True).values[:k]
    ideal_gains = ideal_sorted_labels / torch.log2(r_vals)
    idcg = ideal_gains.sum().item()

    return dcg / idcg if idcg > 0 else 0

def calculate_user_metrics(user_data, k_range=25, method="approximation"):
    """Calculate optimal k and NDCG for a single user."""
    # Convert to PyTorch tensors and move to GPU
    scores = torch.tensor(user_data['score'].values, device=device)
    labels = torch.tensor(user_data['label'].values, device=device)
    num_items = len(scores)

    # Determine the optimal k using expected NDCG
    max_expected_ndcg = 0
    optimal_k = 1

    print(f"User: {user_data['userId'].iloc[0]} | Total Items: {num_items}")
    for k in range(1, min(k_range, num_items) + 1):
        expected_ndcg = expected_ndcg_at_k(scores, k, num_items, method=method)
        print(f"k: {k}, Expected NDCG: {expected_ndcg}")
        if expected_ndcg > max_expected_ndcg:
            max_expected_ndcg = expected_ndcg
            optimal_k = k

    # Compute true NDCG for the optimal k
    true_ndcg_score = true_ndcg(scores, labels, optimal_k)

    print(f"Optimal K: {optimal_k}, Max Utility: {max_expected_ndcg}, True NDCG: {true_ndcg_score}")

    return {
        'OptimalK': optimal_k,
        'MaxUtility': max_expected_ndcg,
        'TrueNDCG': true_ndcg_score
    }

def calculate_metrics(data, k_range=25, method="approximation"):
    """Calculate metrics for all users sequentially."""
    grouped = data.groupby('userId')
    print(f"Processing {len(grouped)} users...")

    user_metrics = {}

    for user, group in grouped:
        user_metrics[user] = calculate_user_metrics(group, k_range, method=method)

    # Compute dataset-level averages
    total_ndcg = sum(metrics['TrueNDCG'] for metrics in user_metrics.values())
    average_ndcg = total_ndcg / len(user_metrics)

    print("Completed calculations for all users.")
    return user_metrics, {'AverageNDCG': average_ndcg}

# Example usage
# Load a subset of the dataset for testing
# subset_size = 100
data = pd.read_csv("/content/tests_with_scores_movielens_GMF.csv")  # Replace with your dataset path
# data = raw_data[raw_data['userId'].isin(raw_data['userId'].unique()[:subset_size])]

# Use exact method for smaller datasets and approximation for larger datasets
user_metrics, dataset_metrics = calculate_metrics(data, k_range=25, method="approximation")
print(user_metrics)
print(dataset_metrics)
