import numpy as np
import pandas as pd
from scipy.stats import binom
import time

class KAdaptAlgorithm:
    def __init__(self, calibration_file, test_file, alpha=0.1, eta=0.05, lambda_init=1.0, k=0.9, max_items=50):
        # Load datasets from CSV files
        self.calibration_data = pd.read_csv(calibration_file)
        self.test_data = pd.read_csv(test_file)

        self.alpha = alpha  # Risk threshold
        self.eta = eta  # Confidence level
        self.lambda_init = lambda_init  # Initial lambda
        self.delta_lambda = 0.01  # Lambda decrement step
        self.k = k  # Scaling factor for risk calculation
        self.max_items = max_items  # Max items recommendable per user

        # Pre-sort the calibration and test data for each user by score in descending order
        self.calibration_data.sort_values(by=['userId', 'score'], ascending=[True, False], inplace=True)
        self.test_data.sort_values(by=['userId', 'score'], ascending=[True, False], inplace=True)

    def calculate_epsilon(self, n_users):
        """Calculate epsilon based on user count and hyperparameters."""
        delta = (1 / n_users) * np.sqrt(2 * np.log(2 * 100 / self.eta))  # |Lambda| = 100
        return 2 * delta

    def generate_prediction_set(self, user_data, lambda_threshold):
        """Generate prediction set \pi_lambda(u) for a user based on pre-sorted scores."""
        # Filter items with scores above the lambda threshold
        prediction_set = user_data[user_data['score'] > lambda_threshold]

        # Limit the prediction set to max_items if necessary
        return prediction_set.head(self.max_items)["itemId"].tolist()


    def calculate_utility(self, true_items, predicted_items, metric="hit_rate"):
        """Calculate utility based on the specified metric."""
        k = len(predicted_items)  # Size of the prediction set
        if k == 0:  # No items predicted
            return 0
        if metric == "hit_rate":
            return 1 if len(set(true_items) & set(predicted_items)) > 0 else 0
        elif metric == "recall":
            return len(set(true_items) & set(predicted_items)) / len(true_items)
        elif metric == "ndcg":
            dcg = sum(
                [1 / np.log2(i + 2) if item in true_items else 0 for i, item in enumerate(predicted_items)]
            )
            idcg = sum([1 / np.log2(i + 2) for i in range(len(true_items))])
            return dcg / idcg if idcg > 0 else 0
        elif metric == "mrr":
            for rank, item in enumerate(predicted_items, start=1):
                if item in true_items:
                    return 1 / rank
            return 0

        elif metric == "f1_":
            true_positive = len(set(true_items) & set(predicted_items))
            precision = true_positive / 25 if k > 0 else 0
            recall = true_positive / len(true_items) if len(true_items) > 0 else 0
            if precision + recall > 0:
                return 2 * (precision * recall) / (precision + recall)
            else:
                return 0
        else:
            raise ValueError("Invalid metric specified.")

    def calculate_empirical_loss(self, user_data, lambda_threshold, metric):
        """Calculate empirical loss for a single user."""
        true_items = user_data[user_data["label"] == 1]["itemId"].tolist()
        predicted_items = self.generate_prediction_set(user_data, lambda_threshold)

        utility = self.calculate_utility(true_items, predicted_items, metric)
        return self.k - utility  # Empirical loss is 1 - utility

    def calculate_risk(self, users_data, lambda_threshold, metric):
        """Calculate risk R_n(lambda) based on user utilities."""
        n_users = users_data["userId"].nunique()
        user_losses = []

        for user_id, user_data in users_data.groupby("userId"):
            loss = self.calculate_empirical_loss(user_data, lambda_threshold, metric)
            user_losses.append(loss)

        return (1 / n_users) * sum(user_losses)  # Scaled mean risk across users

    def find_optimal_lambda(self, metric="hit_rate"):
        """Find the optimal lambda using the calibration dataset."""
        lambda_threshold = self.lambda_init
        n_users = self.calibration_data["userId"].nunique()
        epsilon = self.calculate_epsilon(n_users)

        while lambda_threshold > 0:
            risk = self.calculate_risk(self.calibration_data, lambda_threshold, metric)
            print(f"Lambda: {lambda_threshold}, Risk: {risk}, Epsilon: {epsilon}")

            if risk <= self.alpha - (1 * epsilon):
                print(f"Optimal Lambda Found: {lambda_threshold}")
                return lambda_threshold

            old_lambda = lambda_threshold
            lambda_threshold -= self.delta_lambda

            print(
                f"Lambda updated from {old_lambda} to {lambda_threshold}. New Risk: {risk}, Target Risk: {self.alpha - epsilon}"
            )

        print("No optimal lambda found within range.")
        return None

    def evaluate_test_data(self, lambda_threshold, metric="hit_rate"):
        """Evaluate the test dataset using the optimal lambda."""
        results = []

        for user_id, user_data in self.test_data.groupby("userId"):
            true_items = user_data[user_data["label"] == 1]["itemId"].tolist()
            predicted_items = self.generate_prediction_set(user_data, lambda_threshold)

            utility = self.calculate_utility(true_items, predicted_items, metric)
            results.append({
                "userId": user_id,
                "utility": utility,
                "predicted_items": predicted_items,
            })

        # Calculate overall metrics
        overall_utility = np.mean([result["utility"] for result in results])
        print(f"Overall Utility ({metric}): {overall_utility}")
        return results

    def calculate_avg_set_size(self, lambda_threshold):
        """Calculate the average prediction set size for the test dataset."""
        set_sizes = []

        for user_id, user_data in self.test_data.groupby("userId"):
            # Generate the prediction set for the user
            prediction_set = self.generate_prediction_set(user_data, lambda_threshold)
            set_sizes.append(len(prediction_set))  # Record the size of the prediction set

        avg_set_size = np.mean(set_sizes)  # Calculate the average set size
        print(f"Average Prediction Set Size: {avg_set_size}")
        return avg_set_size

# Example Usage
if __name__ == "__main__":
    calibration_file = "calibration_data.csv"  # Generic name for calibration dataset
    test_file = "test_data.csv"  # Generic name for test dataset
    algorithm = KAdaptAlgorithm(calibration_file, test_file, alpha=0.02, k=0.16, eta=0.03, max_items=25)
    lambda_start_time = time.time()
    optimal_lambda = algorithm.find_optimal_lambda(metric="f1_")
    lambda_end_time = time.time()
    print(f"Time taken to find optimal lambda: {lambda_end_time - lambda_start_time:.2f} seconds")
    test_results = algorithm.evaluate_test_data(optimal_lambda, metric="f1_")
    avg_set_size = algorithm.calculate_avg_set_size(optimal_lambda)
