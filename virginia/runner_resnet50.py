"""
runner_resnet50.py
==================
Trains ResNet50 on ISIC 2019 for 30 epochs with W&B tracking.
Logs to W&B project: resnet50-30epochs

Usage:
    python runner_resnet50.py                        # local (no W&B)
    python runner_resnet50.py --wandb                # with W&B
    python runner_resnet50.py --data_dir /path/to/data --wandb
"""

# -------- Imports -------- #
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from TL_resnet50 import get_pretrained_model
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from isic_utils import (
    set_up, 
    make_loaders,
    train_epoch, 
    evaluate, 
    plot_training_history
)
import wandb

# -------- Config -------- #

WANDB        = False        # override with --wandb flag
SEED         = 42
BATCH_SIZE   = 32
NUM_WORKERS  = 4
NUM_CLASSES  = 8
NUM_EPOCHS   = 30
LR           = 0.001
DROPOUT      = 0.3
DATA_DIR     = os.path.expanduser("~/Downloads/ISIC_2019_mini")   # local default
WANDB_PROJECT = "resnet50-30epochs"


# -------- Main # -------- #

def main(args):
    use_wandb = args.wandb or WANDB

    if use_wandb:
        wandb.init(project=WANDB_PROJECT, config={
            "model":       "resnet50",
            "epochs":      args.num_epochs,
            "batch_size":  args.batch_size,
            "lr":          args.lr,
            "dropout":     DROPOUT,
            "seed":        SEED,
            "feature_extract": True
        })

    # ---- Setup / Loaders ---- #
    device, base_ds, train_idx, val_idx = set_up(seed=SEED, data_dir=args.data_dir)
    train_loader, val_loader, class_weights = make_loaders(
        base_ds, 
        train_idx, 
        val_idx,
        batch_size=args.batch_size,
        num_workers=args.num_workers, 
        device=device
    )

    # ---- Model Build ---- #
    model = get_pretrained_model('resnet50',num_classes=NUM_CLASSES,
                                 feature_extract=True, dropout_prob=DROPOUT)
    # sending model                             
    model = model.to(device)
    
    # -- debugging code -- #
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[DEBUG] Trainable params: {trainable:,} / {total:,}")
    # ---------------------#
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        print("-" * 30)

        train_loss, train_acc = train_epoch(
            model, 
            train_loader, 
            criterion, 
            optimizer, 
            device, 
            feature_extract=True
        )
        val_metrics = evaluate(model, val_loader, criterion, device)
        val_loss = val_metrics['loss']
        val_acc  = val_metrics['acc']

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"  Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%")
        print(f"  Val   Loss: {val_loss:.4f}  Val   Acc: {val_acc:.2f}%")

        if use_wandb:
            log_dict = {
                "epoch":epoch + 1,
                "train_loss":train_loss,
                "train_acc":train_acc,
                "val_loss":val_loss,
                "val_acc":val_acc,
                "val_macro_f1":val_metrics['macro_f1'],
                "val_bacc":val_metrics['bacc'],
                "lr": LR,
            }
            for c, auc in enumerate(val_metrics['auc_per_class']):
                log_dict[f"val_auc_class_{c}"] = auc
            wandb.log(log_dict)
            
            # confusion matrix
            fig, ax = plt.subplots(figsize=(10, 8))
            disp = ConfusionMatrixDisplay(
                confusion_matrix=val_metrics['conf_matrix'],
                display_labels=base_ds.classes
            )
            disp.plot(ax=ax, colorbar=False, xticks_rotation=45)
            ax.set_title(f"Confusion Matrix — Epoch {epoch+1}")
            wandb.log({"conf_matrix": wandb.Image(fig)})
            plt.close(fig)

    # save png locally on rivanna
    plot_training_history(history, title="ResNet50_30epochs")
    
    # close w&b run
    if use_wandb:
        wandb.finish()

# instilling flexibility in the SLURM capabilities: 
# (allows passing of arguments from the SLURM file)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',    default=DATA_DIR)
    parser.add_argument('--num_epochs',  type=int,   default=NUM_EPOCHS)
    parser.add_argument('--batch_size',  type=int,   default=BATCH_SIZE)
    parser.add_argument('--num_workers', type=int,   default=NUM_WORKERS)
    parser.add_argument('--lr',          type=float, default=LR)
    parser.add_argument('--wandb',       action='store_true')
    args = parser.parse_args()
    main(args)
