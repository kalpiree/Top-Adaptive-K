import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd


class FlexibleDataLoader:
    """
    Flexible Data Loader for preprocessing explicit and implicit datasets.
    """

    def __init__(self, df, dataset_type='explicit'):
        """
        Args:
          df: Input DataFrame.
          dataset_type: 'explicit' or 'implicit'.
        """
        self.df = df
        self.dataset_type = dataset_type

    def read_data(self):
        """
        Adds a weight column to the dataset.
        Returns: Processed DataFrame.
        """
        if self.dataset_type == 'implicit':
            self.df['weight'] = self.calculate_weights(self.df['rating'])
        else:
            self.df['weight'] = 8.0

        return self.df

    def calculate_weights(self, interactions):
        alpha = 0.5
        return 1 + alpha * interactions

    def split_train_test(self):
        """
        Splits the dataset into training, calibration, and test sets.

        Returns:
            train_df: DataFrame containing the training data.
            cal_df: DataFrame containing the calibration data.
            test_df: DataFrame containing the test data.
        """
        train_list, cal_list, test_list = [], [], []

        # Group by userId
        grouped = self.df.groupby('userId')

        for user_id, group in grouped:
            interactions = group.sample(frac=1, random_state=42)  # Shuffle interactions for randomness
            n_interactions = len(interactions)

            if n_interactions < 2:
                # Less than 2 interactions -> All in train
                train_list.append(interactions)
            else:
                # Split interactions
                train_cutoff = int(0.5 * n_interactions)
                cal_cutoff = train_cutoff + int(0.3 * n_interactions)

                train_list.append(interactions.iloc[:train_cutoff])
                cal_list.append(interactions.iloc[train_cutoff:cal_cutoff])
                test_list.append(interactions.iloc[cal_cutoff:])

        # Concatenate lists into DataFrames
        train_df = pd.concat(train_list, ignore_index=True)
        cal_df = pd.concat(cal_list, ignore_index=True)
        test_df = pd.concat(test_list, ignore_index=True)

        return train_df, cal_df, test_df, self.df


class MovieLens(Dataset):
    def __init__(self, df, total_df, ng_ratio=1, include_features=False):
        """
        Args:
          df: Processed DataFrame (training, validation, or test).
          total_df: Full dataset (used for negative sampling).
          ng_ratio: Ratio of negative samples to positive samples.
          include_features: Whether to include one-hot encoded features.
        """
        super(MovieLens, self).__init__()
        self.df = df
        self.total_df = total_df
        self.ng_ratio = ng_ratio
        self.include_features = include_features

        # Map user and item IDs to a continuous range starting at 0
        self.user_map = {user_id: idx for idx, user_id in enumerate(df['userId'].unique())}
        self.item_map = {item_id: idx for idx, item_id in enumerate(df['itemId'].unique())}
        self.item_map_reverse = {idx: item_id for item_id, idx in self.item_map.items()}  # Reverse mapping

        # Mapping based on total data
        self.user_map_tot = {user_id: idx for idx, user_id in enumerate(sorted(total_df['userId'].unique()))}
        self.item_map_tot = {item_id: idx for idx, item_id in enumerate(sorted(total_df['itemId'].unique()))}
        self.num_users_tot = len(self.user_map_tot)
        self.num_items_tot = len(self.item_map_tot)
        self.num_features_tot = self.num_users_tot + self.num_items_tot

        self.users, self.items, self.labels, self.weights, self.features, self.ratings, self.groups = self._prepare_data()

    def _prepare_data(self):
        """
        Prepares training data including negative samples.
        Returns:
            users, items, labels, weights, features, ratings, groups
        """
        users, items, labels, weights, features, ratings, groups = [], [], [], [], [], [], []

        for row in tqdm(self.df.itertuples(index=False), total=len(self.df), desc="Preparing Data"):
            u = row.userId
            i = row.itemId
            w = row.weight
            rating = row.rating
            group = getattr(row, 'group', 0)  # Use 0 if 'group' column is missing

            mapped_u = self.user_map_tot[u]
            mapped_i = self.item_map_tot[i]
            users.append(mapped_u)
            items.append(mapped_i)
            labels.append(1)
            weights.append(w)
            ratings.append(row.rating)
            groups.append(group)

            if self.include_features:
                user_feature = F.one_hot(torch.tensor(mapped_u), num_classes=self.num_users_tot)
                item_feature = F.one_hot(torch.tensor(mapped_i), num_classes=self.num_items_tot)
                feature = torch.cat((user_feature, item_feature), dim=0)
                features.append(feature)

            user_interacted_items = set(self.total_df[self.total_df['userId'] == u]['itemId'].map(self.item_map_tot))
            potential_negatives = set(self.item_map_tot.values()) - user_interacted_items

            num_negatives = min(len(potential_negatives), self.ng_ratio)
            negative_samples = np.random.choice(list(potential_negatives), num_negatives, replace=False)

            for neg in negative_samples:
                users.append(mapped_u)
                items.append(neg)
                labels.append(0)
                weights.append(1.0)
                ratings.append(0)
                groups.append(group)

                if self.include_features:
                    negative_feature = torch.cat(
                        (user_feature, F.one_hot(torch.tensor(neg), num_classes=self.num_items_tot)), dim=0)
                    features.append(negative_feature)

        users, items, labels, weights = map(torch.tensor, [users, items, labels, weights])
        features = torch.stack(features) if features else None

        return users, items, labels, weights, features, ratings, groups

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        if self.include_features:
            feature_vector = self.features[idx]
            return self.users[idx], self.items[idx], self.labels[idx], self.weights[idx], feature_vector, self.ratings[
                idx], self.groups[idx]
        else:
            return self.users[idx], self.items[idx], self.labels[idx], self.weights[idx], self.ratings[idx], \
                self.groups[idx]

    def get_num_users(self):
        return self.num_users_tot

    def get_num_items(self):
        return self.num_items_tot

    def get_num_features(self):
        return self.num_features_tot
