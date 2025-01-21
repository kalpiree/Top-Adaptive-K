import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
import torch
import torch.nn.functional as F
from scipy.optimize import minimize


def apply_calibration_and_evaluate(validation_df, test_df, top_n, file_name):
    results = []

    # Split validation and test data into groups
    for group in validation_df['group'].unique():
        val_group_df = validation_df[validation_df['group'] == group]
        test_group_df = test_df[test_df['group'] == group]

        # Extract scores and labels
        val_scores = val_group_df['score'].values
        val_labels = val_group_df['label'].values
        test_scores = test_group_df['score'].values

        # Apply calibration methods
        calibrated_methods = {
            'non_calibrated': test_scores,  # Non-calibrated method
            'platt': self.apply_platt_scaling(val_scores, val_labels, test_scores),
            'isotonic': apply_isotonic_regression(val_scores, val_labels, test_scores),
            'temperature': self.apply_temperature_scaling(val_scores, val_labels, test_scores),
            'bbq': self.apply_bbq(val_scores, val_labels, test_scores),
            'beta': self.apply_beta_calibration(val_scores, val_labels, test_scores)
        }

        for method, calibrated_probs in calibrated_methods.items():
            if calibrated_probs is not None:
                hit_rate, ndcg, loss = evaluate_metrics(test_group_df, calibrated_probs, top_n)
                results.append({
                    'File': file_name,
                    'Group': group,
                    'Lambda': None,
                    'Method': method,
                    'Loss': loss,
                    'Average Set Size': top_n,
                    'Hit Rate': hit_rate,
                    'NDCG': ndcg,

                })

    return results


def apply_platt_scaling(train_scores, train_labels, test_scores):
    model = LogisticRegression()
    model.fit(train_scores.reshape(-1, 1), train_labels)
    calibrated_probs = model.predict_proba(test_scores.reshape(-1, 1))[:, 1]
    return calibrated_probs


def apply_isotonic_regression(train_scores, train_labels, test_scores):
    model = IsotonicRegression(out_of_bounds='clip')
    model.fit(train_scores, train_labels)
    calibrated_probs = model.transform(test_scores)
    return calibrated_probs


def apply_beta_calibration(train_scores, train_labels, test_scores):
    try:
        from betacal import BetaCalibration
        beta_calibrator = BetaCalibration(parameters="abm")
        beta_calibrator.fit(train_scores, train_labels)
        calibrated_probs = beta_calibrator.predict(test_scores)
        return calibrated_probs
    except ImportError as e:
        print(f"BetaCalibration method could not be imported: {e}")
        return None


def evaluate_metrics(test_data, calibrated_probs, top_n):
    #test_data['calibrated_score'] = calibrated_probs
    test_data.loc[:, 'calibrated_score'] = calibrated_probs
    test_data_sorted = test_data.sort_values(by='calibrated_score', ascending=False)

    hits = []
    ndcgs = []
    losses = []

    for user_id in test_data['userId'].unique():
        user_data = test_data_sorted[test_data_sorted['userId'] == user_id]
        top_n_items = user_data.head(top_n)['itemId'].tolist()
        ground_truth_item = user_data[user_data['label'] == 1]['itemId'].values
        if len(ground_truth_item) > 0:
            ground_truth_item = ground_truth_item[0]
            hits.append(hit(ground_truth_item, top_n_items))
            ndcgs.append(ndcg(ground_truth_item, top_n_items))
            losses.append(calculate_loss(ground_truth_item, top_n_items))

    hit_rate = np.mean(hits)
    avg_ndcg = np.mean(ndcgs)
    avg_loss = np.mean(losses)

    return hit_rate, avg_ndcg, avg_loss


def hit(gt_item, pred_items):
    """Calculate hit rate."""
    return 1 if gt_item in pred_items else 0


def ndcg(gt_item, pred_items):
    """Calculate normalized discounted cumulative gain."""
    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        return 1 / np.log2(index + 2)
    return 0


def calculate_loss(gt_item, pred_items):
    """Calculate loss."""
    return 0 if gt_item in pred_items else 1
