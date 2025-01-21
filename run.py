import gc
import os
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader
import argparse
from evaluation import metrics
from models.BERT4Rec import BERT4Rec
from models.DeepFM import DeepFM
from models.FM import FactorizationMachine
from models.GMF import GMF
from models.LightGCN import LightGCN
from models.MLP import MLP
from models.NeuMF import NeuMF
from models.SASRec import SASRec
from models.WMF import WMF
from train import Train
from utils_new import FlexibleDataLoader, MovieLens
from scipy.sparse import coo_matrix

def construct_adjacency_matrix(train_df, num_users, num_items):
    user_item_pairs = train_df[['userId', 'itemId']].values
    data = [1] * len(user_item_pairs)
    rows, cols = zip(*user_item_pairs)
    adj_matrix = coo_matrix((data, (rows, cols)), shape=(num_users, num_items))
    return adj_matrix

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train recommender models on different datasets.")
    parser.add_argument('--datasets', nargs='+', default=['movielens'],
                        help="List of datasets to use (default: ['movielens']")
    parser.add_argument('--models', nargs='+', default=['MLP'],
                        help="List of models to train (default: ['MLP']")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size for training (default: 256)")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument('--factor', type=int, default=8, help="Number of factors for the models (default: 8)")
    parser.add_argument('--output_folder', type=str, default='score_files',
                        help="Output folder to save results (default: 'score_files')")
    return parser.parse_args()

def append_scores_and_data(model, data_loader, device):
    model.eval()
    data_collection = []

    with torch.no_grad():
        for batch in data_loader:
            batch = [x.to(device).squeeze() for x in batch]
            if data_loader.dataset.include_features:
                users, items, labels, weights, features, ratings, groups = batch
                features = features.float()
                predictions = model(features).squeeze()
            else:
                users, items, labels, weights, ratings, groups = batch
                users, items = users.long(), items.long()
                predictions = model(users, items)

            batch_data = {
                'userId': users.cpu().numpy(),
                'itemId': items.cpu().numpy(),
                'score': predictions.cpu().numpy(),
                'label': labels.cpu().numpy(),
                'weight': weights.cpu().numpy(),
                'rating': ratings.cpu().numpy(),
                'group': groups.cpu().numpy()
            }
            data_collection.append(pd.DataFrame(batch_data))

    return pd.concat(data_collection, ignore_index=True)

def run_model_for_dataset(model_name, train_df, validation_df, test_df, total_df, dataset, dataset_type, args):
    train_dataset = validation_dataset = test_dataset = None
    train_dataloader = validation_dataloader = test_dataloader = None
    model = optimizer = criterion = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        train_dataset = MovieLens(train_df, total_df, ng_ratio=1, include_features=(model_name in ['FM', 'DeepFM']))
        validation_dataset = MovieLens(validation_df, total_df, ng_ratio=50, include_features=(model_name in ['FM', 'DeepFM']))
        test_dataset = MovieLens(test_df, total_df, ng_ratio=50, include_features=(model_name in ['FM', 'DeepFM']))

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        num_users = train_dataset.get_num_users()
        num_items = train_dataset.get_num_items()
        num_features = train_dataset.get_num_features()

        adj_matrix = construct_adjacency_matrix(train_df, num_users, num_items)
        adj_matrix_torch = torch.tensor(adj_matrix.toarray(), dtype=torch.float32)

        models = {
            'MLP': MLP(num_users=num_users, num_items=num_items, num_factor=args.factor),
            'GMF': GMF(num_users=num_users, num_items=num_items, num_factor=args.factor),
            'NeuMF': NeuMF(num_users=num_users, num_items=num_items, num_factor=args.factor),
            'FM': FactorizationMachine(num_factors=args.factor, num_features=num_features),
            'DeepFM': DeepFM(num_factors=args.factor, num_features=num_features),
        }

        model = models[model_name].to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        criterion = torch.nn.BCELoss()

        trainer = Train(
            model=model,
            optimizer=optimizer,
            epochs=args.epochs,
            dataloader=train_dataloader,
            criterion=criterion,
            test_obj=test_dataloader,
            device=device,
            print_cost=True,
            use_features=model_name in ['FM', 'DeepFM'],
        )
        trainer.train()

        os.makedirs(args.output_folder, exist_ok=True)
        validation_df = append_scores_and_data(model, validation_dataloader, device)
        test_df = append_scores_and_data(model, test_dataloader, device)

        validation_df.to_csv(f"{args.output_folder}/validations_with_scores_{dataset}_{model_name}.csv", index=False)
        test_df.to_csv(f"{args.output_folder}/tests_with_scores_{dataset}_{model_name}.csv", index=False)

        top_k = 50
        avg_hr_test, avg_ndcg_test = metrics(model, test_dataloader, top_k, device)
        avg_hr_val, avg_ndcg_val = metrics(model, validation_dataloader, top_k, device)

        print(f"Dataset: {dataset}, Model: {model_name}")
        print(f"Average Hit Rate Test Set (HR@{top_k}): {avg_hr_test:.3f}")
        print(f"Average NDCG Test Set (NDCG@{top_k}): {avg_ndcg_test:.3f}")
        print(f"Average Hit Rate Validation Set (HR@{top_k}): {avg_hr_val:.3f}")
        print(f"Average NDCG Validation Set (NDCG@{top_k}): {avg_ndcg_val:.3f}")

    finally:
        del train_dataset, validation_dataset, test_dataset
        del train_dataloader, validation_dataloader, test_dataloader
        del model, optimizer, criterion
        gc.collect()
        torch.cuda.empty_cache()

def main():
    args = parse_arguments()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    for dataset in args.datasets:
        dataset_type = 'implicit' if dataset == 'lastfm' else 'explicit'
        print(f"Using dataset: {dataset}, type: {dataset_type}")

        file_path = os.path.join("input_data_folder", f"{dataset}.csv")
        df = pd.read_csv(file_path)

        data_loader = FlexibleDataLoader(df=df, dataset_type=dataset_type)
        processed_data = data_loader.read_data()
        train_df, validation_df, test_df, total_df = data_loader.split_train_test()

        print(f"Dataset: {dataset}, Train: {len(train_df)}, Validation: {len(validation_df)}, Test: {len(test_df)}")

        for model_name in args.models:
            print(f"Processing model: {model_name}")
            try:
                run_model_for_dataset(
                    model_name=model_name,
                    train_df=train_df,
                    validation_df=validation_df,
                    test_df=test_df,
                    total_df=total_df,
                    dataset=dataset,
                    dataset_type=dataset_type,
                    args=args,
                )
            except Exception as e:
                print(f"Error encountered while processing model {model_name} on dataset {dataset}: {e}")

if __name__ == '__main__':
    main()
