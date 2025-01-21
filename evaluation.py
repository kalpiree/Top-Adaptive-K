import numpy as np
import torch


def hit(gt_item, pred_items):
    """Calculate hit rate."""
    return 1 if gt_item in pred_items else 0


def ndcg(gt_item, pred_items):
    """Calculate normalized discounted cumulative gain."""
    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        return 1 / np.log2(index + 2)
    return 0


def metrics(model, test_loader, top_k, device):
    HR, NDCG = [], []
    model.eval()
    use_weights = 'WMF' in model.__class__.__name__  # Automatically determine weight usage

    with torch.no_grad():

        for batch in test_loader:
            batch = [x.to(device) for x in batch]

            if test_loader.dataset.include_features:
                # When features are included, assume they are at the end of the batch
                users, items, labels, weights, features, _, _ = batch
                features = features.float()
                predictions = model(features)  # Assuming model takes features only
                predictions = predictions.squeeze()
            elif use_weights:
                # When weights are used
                users, items, labels, weights, _, _ = batch
                weights = weights.float()
                predictions = model(users, items, weights)  # Assuming model takes users, items, and weights
            else:
                # When neither features nor weights are used
                users, items, labels, _, _, _ = batch
                users, items = users.long(), items.long()  ## check this if long needed or not
                predictions = model(users, items)  # Assuming model takes users and items only

            for user_id in users.unique():
                user_mask = users == user_id
                user_items = items[user_mask]
                user_predictions = predictions[user_mask]
                user_labels = labels[user_mask]

                positive_indices = (user_labels == 1).nonzero(as_tuple=True)[0]

                if positive_indices.numel() > 0:
                    for pos_index in positive_indices:
                        gt_item = user_items[pos_index].item()
                        _, indices = torch.topk(user_predictions, min(top_k, len(user_items)))
                        recommends = user_items[indices].cpu().numpy().tolist()

                        HR.append(hit(gt_item, recommends))
                        NDCG.append(ndcg(gt_item, recommends))

    avg_hr = np.mean(HR) if HR else 0
    avg_ndcg = np.mean(NDCG) if NDCG else 0
    return avg_hr, avg_ndcg
