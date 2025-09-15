import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from collections import OrderedDict
import matplotlib.pyplot as plt
import json

# Importing custom modules
from dataset import SimpleDataset
from model import cnn_dropout, FC_dropout
from query_strategies import predictive_entropy_sampling, bald_sampling, variation_ratio_sampling
from evaluation import eval_perf


def get_config():
    parser = argparse.ArgumentParser()

    # Paths
    parser.add_argument('--data_dir', type=str, default='/kaggle/input/active-learning-subset')
    parser.add_argument('--backbone_path', type=str, default='/kaggle/input/active-learning-subset/backboneSiteA2019.pth')
    parser.add_argument('--fc_path', type=str, default='/kaggle/input/active-learning-subset/fcSiteA2019.pth')

    # General settings
    parser.add_argument('--rounds', type=int, default=10)
    parser.add_argument('--query_size', type=int, default=50)
    parser.add_argument('--sim', type=int, default=30)
    parser.add_argument('--cold_start', action='store_true')
    
    # Training
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)


    # Strategy
    parser.add_argument('--strategy', type=str, choices=['predictive_entropy', 'bald', 'variation_ratio'], default='predictive_entropy')

    args = parser.parse_args()
    print(args)
    
    return args


def main():
    args = get_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset Path
    X = np.load(f'{args.data_dir}/tar_image_subset.npy')
    y = np.load(f'{args.data_dir}/label_target_subset.npy')

    # Train Test split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)
    
    # For evaluation
    val_dataset = SimpleDataset(X_val, y_val)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    
    # Loading model weights
    backbone_state = torch.load(args.backbone_path)
    fc_state = torch.load(args.fc_path)

    backbone_state = OrderedDict((k.replace('module.', ''), v) for k, v in backbone_state.items())
    fc_state = OrderedDict((k.replace('module.', ''), v) for k, v in fc_state.items())

    # Initialize and load models
    backbone = cnn_dropout()
    backbone.to(device)

    fc = FC_dropout(1024)
    fc.to(device)

    backbone.load_state_dict(backbone_state)
    fc.load_state_dict(fc_state)


    # Create separate pools and histories for each strategy
    labeled_pool = set()
    unlabeled_pool =  set(range(len(X_train)))
    print('Length of Labeled Pool:', len(labeled_pool))
    print('Length of Unlabeled Pool:', len(unlabeled_pool))
        
    history = {
        "rounds": [],
        "num_labeled": [],
        "train_accuracy": [],
        "train_loss": [],
        "val_accuracy": [],
        "val_avg_f1": [],
        "val_f1s": [],
        "confusion_matrices": []
    }

    # Settings
    num_rounds = args.rounds
    query_size = args.query_size
    sim = args.sim

    # Calculate total and effective samples
    total_samples = num_rounds * query_size
    effective_samples = query_size * (num_rounds * (num_rounds + 1)) // 2  # Sum of 1+2+...+num_rounds

    # Nicely formatted print
    print("=" * 50)
    print(" Active Learning Configuration Summary")
    print("=" * 50)
    print(f"Simulations (sim):          {sim}")
    print(f"Number of Rounds:           {num_rounds}")
    print(f"Query Size per Round:       {query_size}")
    print(f"Training Epochs:            {args.epochs}")
    print(f"Total Labeled Samples:      {total_samples}")
    print(f"Effective Samples (Total):  {effective_samples}")
    print("=" * 50)


    # ============ Main Active Learning Loop ============
    for round_num in range(1, num_rounds + 1):
        print(f"\n--- Round {round_num} [{args.strategy}] ---")

        # Cold start: Re-initialize model at every round
        if args.cold_start:
            backbone = cnn_dropout().to(device)
            fc = FC_dropout(1024).to(device)
            backbone.load_state_dict(backbone_state)
            fc.load_state_dict(fc_state)
        
        # 1. Query k samples
        if args.strategy == 'predictive_entropy':
            queried_indices = predictive_entropy_sampling(X_train, backbone, fc, query_size, device, unlabeled_pool, sim=sim)
        elif args.strategy == 'bald':
            queried_indices = bald_sampling(X_train, backbone, fc, query_size, device, unlabeled_pool, sim=sim)
        elif args.strategy == 'variation_ratio':
            queried_indices = variation_ratio_sampling(X_train, backbone, fc, query_size, device, unlabeled_pool, sim=sim)
    
        # 2. Update pools
        new_indices = set(queried_indices) - labeled_pool  # avoid duplication
        labeled_pool.update(new_indices)
        unlabeled_pool.difference_update(new_indices)
        print('Length of Labeled Pool:', len(labeled_pool))
        print('Length of Unlabeled Pool:', len(unlabeled_pool))
        
        # 3. Build train loader
        train_dataset = SimpleDataset(X_train[list(labeled_pool)], y_train[list(labeled_pool)])
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
        # 4. Train + Evaluate            
        train_acc, train_loss, optimizer = train(backbone, fc, train_loader, device, epochs=args.epochs, lr=args.lr)
        val_f1s, val_acc, val_avg_f1, cm = eval_perf(val_dataloader, backbone, fc, device)
        
        print("Per-class F1 scores:", val_f1s)
        print(f"Overall Accuracy: {val_acc * 100:.2f}%")
        print(f"Average F1 Score: {val_avg_f1:.4f}")
    
        # 5. Update history
        history["rounds"].append(round_num)
        history["num_labeled"].append(len(labeled_pool))
        history["train_accuracy"].append(train_acc)
        history["train_loss"].append(train_loss)
        history["val_accuracy"].append(val_acc)
        history["val_avg_f1"].append(val_avg_f1)
        history["val_f1s"].append(val_f1s.tolist())
        history["confusion_matrices"].append(cm.tolist())

        # Logging the Results
        with open(f"history_{args.strategy}.txt", "w") as f:
            json.dump(history, f, indent=4)
        print(f'=== History logged for Round {round_num}')

        # Creating checkpoint
        checkpoint = {
            'round': round_num,
            'backbone_state_dict': backbone.state_dict(),
            'fc_state_dict': fc.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'labeled_pool': list(labeled_pool),
            'unlabeled_pool': list(unlabeled_pool),
            'args': vars(args),
        }
        torch.save(checkpoint, f'checkpoint_{args.strategy}_round_{round_num}.pth')
        print(f'=== Checkpoint created for Round {round_num}')



    print('========= END OF SCRIPT =========')


def train(backbone, fc, train_loader, device, epochs=5, lr=1e-3):
    backbone.train()
    fc.train()

    optimizer = torch.optim.Adam(list(backbone.parameters()) + list(fc.parameters()), lr=lr)
    train_acc_history = []
    train_loss_history = []

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            xt = batch["x"].to(device)
            yt = batch["y"].to(device)

            optimizer.zero_grad()

            features = backbone(xt)
            outputs = fc(features)
            loss = F.cross_entropy(outputs, yt)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Accuracy calculation
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == yt).sum().item()
            total += yt.size(0)

        acc = correct / total
        avg_loss = total_loss / len(train_loader)
        train_acc_history.append(acc)
        train_loss_history.append(avg_loss)
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Train Acc: {acc*100:.2f}%")
              
    return train_acc_history, train_loss_history, optimizer



if __name__ == "__main__":
    main()
