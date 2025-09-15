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
import os
import math
import pandas as pd

# Importing custom modules
from dataset import Conv1dDataset
from model import cnn_dropout, FC_dropout
from utils import enable_dropout
from evaluation import mc_dropout_eval
from acquisition_functions import bald_sampling, bvsb_sampling


def get_config():
    parser = argparse.ArgumentParser()

    # Paths
    parser.add_argument('--input_dir', type=str, default='/kaggle/input/data-usa')
    parser.add_argument('--target_dir', type=str, default='/kaggle/working/')
    parser.add_argument('--backbone_path', type=str, default='/kaggle/input/active-learning-subset/backboneSiteA2019.pth')
    parser.add_argument('--fc_path', type=str, default='/kaggle/input/active-learning-subset/fcSiteA2019.pth')

    # General settings
    parser.add_argument('--site', type=str, default='A')
    parser.add_argument('--year', type=str, default='2019')
    parser.add_argument('--query_size', type=int, default=10)
    parser.add_argument('--initial_labeled', type=int, default=0)
    parser.add_argument('--total_labeled_samples', type=int, default=300)
    
    # Training
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--stochastic_passes', type=int, default=20)



    # Strategy
    parser.add_argument('--strategy', type=str, choices=['bald', 'bvsb'], default='bald')

    args = parser.parse_args()
    print(args)
    
    return args


def main():
    args = get_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ====== Dataset Path ======
    X = np.load(f'{args.input_dir}/Site_{args.site}/x-{args.year}.npy')
    y = np.load(f'{args.input_dir}/Site_{args.site}/y-{args.year}.npy')
    print(f'X: {X.shape}, y: {y.shape}')
    print(f'Successfully Loaded data for Site {args.site} - {args.year}')

    # ====== Preprocessing the data ======
    feature_means = X.mean(axis=(0, 1))  # Shape: (6,)
    feature_stds = X.std(axis=(0, 1))    # Shape: (6,)
    feature_means = feature_means.reshape(1, 1, -1)  # Shape: (1, 1, 6)
    feature_stds = feature_stds.reshape(1, 1, -1)    # Shape: (1, 1, 6)
    X = ((X - feature_means) / feature_stds).astype(np.float32)

    # ====== Train Test split ======
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)
    print(f'Training Data: {X_train.shape}{y_train.shape}')
    print(f'Validation Data: {X_val.shape}{y_val.shape}')
    NUM_CLASSES = len(np.unique(y_train))

    # ====== For validation ======
    val_dataset = Conv1dDataset(X_val, y_val)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # ====== Loading model weights ======
    backbone_state = torch.load(args.backbone_path)
    fc_state = torch.load(args.fc_path)

    backbone_state = OrderedDict((k.replace('module.', ''), v) for k, v in backbone_state.items())
    fc_state = OrderedDict((k.replace('module.', ''), v) for k, v in fc_state.items())

    # ====== Initialize and load models ======
    backbone = cnn_dropout()
    backbone.to(device)

    fc = FC_dropout(1024, NUM_CLASSES)
    fc.to(device)

    backbone.load_state_dict(backbone_state)
    print('Backbone Loaded Successfully')
    fc.load_state_dict(fc_state)
    print('FC Loaded Successfully')

    # ====== Class Weights ======
    all_labels = np.array(y_train)  # shape: (num_samples, H, W) if segmentation

    flattened = all_labels.flatten()
    flattened = flattened[flattened >= 0]
    class_counts = np.bincount(flattened, minlength=NUM_CLASSES)
    total_pixels = class_counts.sum()
    
    class_weights = total_pixels / (NUM_CLASSES * class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    class_weights = torch.log(1 + class_weights)
    
    # ====== Resume or Initialize ======
    optimizer = torch.optim.AdamW(list(backbone.parameters()) + list(fc.parameters()), lr=args.lr)

    # load checkpoint if exists
    start_round, labeled_pool, unlabeled_pool, history = load_checkpoint(args, backbone, fc, optimizer, device)
    start_round = 0 if args.initial_labeled>0 else 1

    if history is None:
        history = {
            "rounds": [],
            "num_labeled": [],
            "train_accuracy": [],
            "train_loss": [],
            "val_accuracy": [],
            "val_loss": [],
            "avg_f1": [],
            "per_class_f1": [],
            "kappa": [],
            "confusion_matrices": []
        }

    # initialize pools if starting fresh
    if len(labeled_pool) == 0 and len(unlabeled_pool) == 0:
        labeled_pool = set()
        unlabeled_pool = set(range(len(X_train)))

    print('Length of Labeled Pool:', len(labeled_pool))
    print('Length of Unlabeled Pool:', len(unlabeled_pool))

    # ====== Settings ======
    QUERY_SIZE = args.query_size
    INITIAL_LABELED = args.initial_labeled
    AL_ROUNDS = math.ceil((args.total_labeled_samples - INITIAL_LABELED) / QUERY_SIZE)
    EPOCHS = args.epochs
    T = args.stochastic_passes

    # ====== Calculate total and effective samples ======
    total_samples = AL_ROUNDS * QUERY_SIZE + INITIAL_LABELED
    effective_samples = INITIAL_LABELED * AL_ROUNDS + QUERY_SIZE * (AL_ROUNDS * (AL_ROUNDS + 1)) // 2  # Sum of 1+2+...+num_rounds
    
    print("=" * 50)
    print(" Active Learning Configuration Summary")
    print("=" * 50)
    print(f"Stochastic Passes:          {T}")
    print(f"Number of Rounds:           {AL_ROUNDS}")
    print(f"Initial Labeled Samples:    {INITIAL_LABELED}")
    print(f"Query Size per Round:       {QUERY_SIZE}")
    print(f"Number of epochs:           {EPOCHS}")
    print(f"Total Labeled Samples:      {total_samples}")
    print(f"Effective Samples (Total):  {effective_samples}")
    print("=" * 50)

    # ============ Main Active Learning Loop ============
    for round_num in range(start_round, AL_ROUNDS + 1):
        print(f"\n--- Round {round_num} [{args.strategy}] ---")

        if round_num == 0:  # Round for initially labeled samples training
            initial_labeled = set(np.random.choice(list(unlabeled_pool), INITIAL_LABELED, replace=False))
            new_indices = set(initial_labeled) - labeled_pool  
        else:
            # 1. Query k samples
            if args.strategy == 'bald':
                queried_indices = bald_sampling(X_train, backbone, fc, args.query_size, args.batch_size, device, unlabeled_pool, T=T)
                new_indices = set(queried_indices) - labeled_pool  
            elif args.strategy == 'bvsb':
                queried_indices = bvsb_sampling(X_train, backbone, fc, args.query_size, args.batch_size, device, unlabeled_pool, T=T)
                new_indices = set(queried_indices) - labeled_pool  

        # 2. Update pools
        labeled_pool.update(new_indices)
        unlabeled_pool.difference_update(new_indices)
        print('Length of Labeled Pool:', len(labeled_pool))
        print('Length of Unlabeled Pool:', len(unlabeled_pool))

        # 3. Build train loader
        train_dataset = Conv1dDataset(X_train[list(labeled_pool)], y_train[list(labeled_pool)])
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        # 4. Train + Evaluate
        train_losses, val_losses, train_accs, val_accs, val_kappas, val_f1s, val_per_class_f1s, cm = train(backbone, fc, train_dataloader, val_dataloader, optimizer, device, class_weights, EPOCHS, T, round_num, AL_ROUNDS)
    
        # 5. Update history
        history["rounds"].append(round_num)
        history["num_labeled"].append(len(labeled_pool))
        history["train_accuracy"].append(train_accs[-1])
        history["train_loss"].append(train_losses[-1])
        history["val_accuracy"].append(val_accs)
        history["val_loss"].append(val_losses)
        history["avg_f1"].append(val_f1s)
        history["per_class_f1"].append(val_per_class_f1s.tolist())
        history["kappa"].append(val_kappas)
        history["confusion_matrices"].append(cm.tolist())

        # Define paths
        results_dir = os.path.join(args.target_dir, "results")
        data_dir = os.path.join(args.target_dir, "data")
        ckpt_dir = os.path.join(args.target_dir, "checkpoints")
        
        # Create directories only if they don't exist
        for d in [args.target_dir, results_dir, data_dir, ckpt_dir]:
            if not os.path.exists(d):
                os.makedirs(d)

        # Logging the Results
        with open(f"{args.target_dir}/results/history_{args.strategy}.json", "w") as f:
            json.dump(history, f, indent=4)
        print(f'=== History logged for Round {round_num}')

        # Also logging in excel for clear understanding
        output_file = f"{args.target_dir}/results/excel_history_{args.strategy}.xlsx"

        df = pd.DataFrame({
            "round": history["rounds"],
            "num_labeled": history["num_labeled"],
            "train_accuracy": history["train_accuracy"],
            "train_loss": history["train_loss"],
            "val_accuracy": history["val_accuracy"],
            "val_loss": history["val_loss"],
            "avg_f1": history["avg_f1"],
            "kappa": history["kappa"],
            # store per-class f1 as string so Excel can hold it
            "per_class_f1": [",".join(map(str, f1)) for f1 in history["per_class_f1"]]
        })
        
        with pd.ExcelWriter(output_file, engine="openpyxl", mode="w") as writer:
            df.to_excel(writer, sheet_name="history", index=False)
        
        print(f"Saved history to {output_file}")

        # Saving the Labeled Pool for each strategy
        data_file = os.path.join(args.target_dir, f"data/approx_BCNN_data_{args.strategy}.npz")
        np.savez(
            data_file,
            X=X_train[list(labeled_pool)],
            y=y_train[list(labeled_pool)]
        )
        print(f"Saved labeled data for {args.strategy} to {data_file}")

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
        torch.save(checkpoint, f'{args.target_dir}/checkpoints/checkpoint_{args.strategy}_round_{round_num}.pth')
        print(f'=== Checkpoint created for Round {round_num}')



    print('========= END OF SCRIPT =========')


        

def train(backbone, fc, train_loader, val_loader, optimizer, device, class_weights, epochs, T, round_num, total_rounds):
    backbone.train()
    fc.train()

    train_losses = []
    train_accs = []

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    for epoch in range(epochs):
        backbone.train()
        fc.train()
        
        running_loss = 0.0
        correct = 0
        total_samples = 0
        
        for batch in train_loader:
            xt = batch["x"].to(device)
            yt = batch["y"].to(device)

            optimizer.zero_grad()

            features = backbone(xt)
            outputs = fc(features)
            loss = F.cross_entropy(outputs, yt, weight=class_weights)
            loss.backward()
            optimizer.step()

            batch_size = yt.size(0)
            running_loss += loss.item() * batch_size
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == yt).sum().item()
            total_samples += batch_size

        train_loss = running_loss / total_samples
        train_acc = correct / total_samples

        train_losses.append(train_loss)
        train_accs.append(train_acc)

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
    # Validation
    val_loss, val_acc, val_avg_f1, val_per_class_f1, val_kappa, cm = mc_dropout_eval(backbone, fc, val_loader, class_weights, device, T, round_num, total_rounds)

    scheduler.step(val_loss)
    
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
              f"F1: {val_avg_f1:.4f}, Kappa: {val_kappa:.4f}")  

    return train_losses, val_loss, train_accs, val_acc, val_kappa, val_avg_f1, val_per_class_f1, cm


def load_checkpoint(args, backbone, fc, optimizer, device):
    """Load the latest checkpoint if it exists, else return defaults."""
    ckpt_dir = os.path.join(args.target_dir, "checkpoints")
    data_dir = os.path.join(args.target_dir, "data")
    results_dir = os.path.join(args.target_dir, "results")

    if not os.path.exists(ckpt_dir) or len(os.listdir(ckpt_dir)) == 0:
        print("No checkpoint found. Starting fresh...")
        return 0, set(), set(), None  # round=0, empty pools, no history

    # Find latest checkpoint
    ckpt_files = [f for f in os.listdir(ckpt_dir) if f.startswith("checkpoint_")]
    ckpt_files.sort(key=lambda x: int(x.split("_round_")[-1].split(".")[0]))
    latest_ckpt = ckpt_files[-1]
    ckpt_path = os.path.join(ckpt_dir, latest_ckpt)

    print(f"Resuming from checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Restore states
    backbone.load_state_dict(checkpoint['backbone_state_dict'])
    fc.load_state_dict(checkpoint['fc_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    labeled_pool = set(checkpoint['labeled_pool'])
    unlabeled_pool = set(checkpoint['unlabeled_pool'])
    start_round = checkpoint['round'] + 1

    # Load history if exists
    history_file = os.path.join(results_dir, f"history_{args.strategy}.json")
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            history = json.load(f)
    else:
        history = None

    return start_round, labeled_pool, unlabeled_pool, history



if __name__ == "__main__":
    main()
