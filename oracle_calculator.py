import os
import numpy as np
import pandas as pd

def calculate_hit_rate(user_df):
    """Hit Rate: Max value across all k from 1 to 50."""
    return max(int(user_df["label"].iloc[:k].sum() > 0) for k in range(1, min(25, len(user_df) + 1)))

def calculate_recall(user_df):
    """Recall: Max value across all k from 1 to 50."""
    total_relevant = user_df["label"].sum()
    if total_relevant == 0:
        return 0
    return max(user_df["label"].iloc[:k].sum() / total_relevant for k in range(1, min(25, len(user_df) + 1)))

def calculate_mrr(user_df):
    """MRR: Max value across all k from 1 to 50."""
    return max(
        (1 / rank if user_df["label"].iloc[rank - 1] == 1 else 0)
        for rank in range(1, min(25, len(user_df) + 1))
    )

def calculate_ndcg(user_df):
    """NDCG: Max value across all k from 1 to 50."""
    dcg_values = []
    for k in range(1, min(25, len(user_df) + 1)):
        dcg = sum(
            1 / np.log2(rank + 1)
            for rank, label in enumerate(user_df["label"].iloc[:k], start=1)
            if label == 1
        )
        ideal_relevant_items = user_df["label"].sum()
        idcg = sum(1 / np.log2(rank + 1) for rank in range(1, ideal_relevant_items + 1))
        dcg_values.append(dcg / idcg if idcg > 0 else 0)
    return max(dcg_values)

def calculate_f1_score(user_df):
    """F1 Score: Combines Precision and Recall."""
    total_relevant = user_df["label"].sum()
    if total_relevant == 0:
        return 0
    f1_values = []
    for k in range(1, min(25, len(user_df) + 1)):
        retrieved_relevant = user_df["label"].iloc[:k].sum()
        precision = retrieved_relevant / 25
        recall = retrieved_relevant / total_relevant if total_relevant > 0 else 0
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0
        f1_values.append(f1)
    return max(f1_values)

def calculate_oracle_metrics(file_path):
    df = pd.read_csv(file_path)
    oracle_metrics = {"Hit Rate": [], "Recall": [], "MRR": [], "NDCG": [], "F1 Score": []}

    for user_id, user_df in df.groupby("userId"):
        user_df = user_df.sort_values(by="score", ascending=False)
        oracle_metrics["Hit Rate"].append(calculate_hit_rate(user_df))
        oracle_metrics["Recall"].append(calculate_recall(user_df))
        oracle_metrics["MRR"].append(calculate_mrr(user_df))
        oracle_metrics["NDCG"].append(calculate_ndcg(user_df))
        oracle_metrics["F1 Score"].append(calculate_f1_score(user_df))

    oracle_values = {metric: np.mean(values) for metric, values in oracle_metrics.items()}
    return oracle_values

def process_files(input_folder, output_file):
    datasets = ["movielens", "lastfm", "amazonoffice"]
    models = ["MLP", "NeuMF", "GMF", "DeepFM", "LightGCN"]
    results = []

    for dataset in datasets:
        for model in models:
            validation_file = f"validations_with_scores_{dataset}_{model}.csv"
            test_file = f"tests_with_scores_{dataset}_{model}.csv"

            validation_path = os.path.join(input_folder, validation_file)
            test_path = os.path.join(input_folder, test_file)

            if os.path.exists(validation_path) and os.path.exists(test_path):
                print(f"Processing {validation_file} and {test_file}...")

                # Calculate oracle metrics for the validation file
                validation_oracle_values = calculate_oracle_metrics(validation_path)
                validation_oracle_values.update({"Dataset": dataset, "Model": model, "Type": "Validation"})
                results.append(validation_oracle_values)

                # Calculate oracle metrics for the test file
                test_oracle_values = calculate_oracle_metrics(test_path)
                test_oracle_values.update({"Dataset": dataset, "Model": model, "Type": "Test"})
                results.append(test_oracle_values)
            else:
                print(f"Missing files for dataset={dataset}, model={model}")

    # Save results to a single output file
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    input_folder = "input_data_folder"  # Replace with the folder containing your files
    output_file = "oracle_metrics_results.csv"
    process_files(input_folder, output_file)
