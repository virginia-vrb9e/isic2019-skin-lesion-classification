"""
rn50_base_va.py
Baseline ResNet-50 training script

Usage:
    python rn50_baseline_va.py                                  # local, mini dataset
    python rn50_baseline_va.py --data_dir /path/to/data         # custom data path
    python rn50_baseline_va.py --data_dir /path/to/data --wandb # with W&B logging
"""

import os
import argparse  # for command-line argument parsing
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# ------------------------------------------------------------------
# Expected Directory Structure
# Subdirectories needed for datasets.ImageFolder to work correctly
# ------------------------------------------------------------------
# DATA_DIR/
# ├── MEL/
# ├── NV/
# ├── BCC/
# ├── AK/
# ├── BKL/
# ├── DF/
# ├── VASC/
# └── SCC/


# -------
# Config
# -------
"""
    Config Dictionary:
    - Default values for all training hyperparameters and dataset settings
    
    Overridable from the command line (argparse):
        --data_dir, 
        --num_epochs, 
        --batch_size, 
        --lr, 
        --num_workers, 
        --wandb

    Editable in Config directly:
        freeze_bb, 
        dropout, 
        img_size, 
        val_split, 
        seed, 
        mini-run, 
        num_classes, 
        classes

    Ablation options not yet implemented (commented out in Config):
        patience, 
        loss_fn, 
        scheduler, 
        step_size, 
        gamma
    
"""
# these are defaults that can be overridden by command-line arguments or ablation options in the main function
Config = {
    "architecture": "resnet50",
    "pretrained":   True,
    "freeze_bb": "full",  # "full" = freeze all, "partial" = unfreeze layer4, "none" = train everything
    "num_classes":  8,
    "dropout":      0.3,
    "img_size":     224,
    "batch_size":   32,
    "epochs":       30,
    "lr":           1e-3,
    "num_workers":  4,
    "val_split":    0.20,
    "seed":         42,
    "data_dir":     os.path.expanduser("~/Downloads/ISIC_2019_mini"),
    "classes":      ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"],
    "mini-run":     False,  # True = use 10% of data

    # --- ablation options ---
    # "patience":     10,             # early stopping after N epochs no improvement
    # "loss_fn":      "weighted_ce",  # "ce" | "focal" | "weighted_ce"
    # "scheduler":    "step",         # "step" | "cosine" | None
    # "step_size":    5,              # StepLR step size
    # "gamma":        0.1,            # StepLR decay factor
}


# -----------------
# Model Definition
# -----------------

def get_pretrained_model(num_classes=Config["num_classes"],
                        dropout_prob=Config["dropout"],
                        feature_extract=Config["freeze_bb"]):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    if feature_extract == "full":
        for param in model.parameters():
            param.requires_grad = False
    elif feature_extract == "partial":
        for param in model.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = True
    elif feature_extract == "none":
        for param in model.parameters():
            param.requires_grad = True

    num_in_features = model.fc.in_features  # 2048 for ResNet-50
    model.fc = nn.Sequential(
        nn.Dropout(dropout_prob),
        nn.Linear(num_in_features, num_classes),
    )
    return model


# -------
# Dataset
# -------

class TransformedSubset(Dataset):
    def __init__(self, base: datasets.ImageFolder, indices, transform=None):
        self.base = base
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        base_idx = self.indices[i]
        path, target = self.base.samples[base_idx]
        img = self.base.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.base.target_transform is not None:
            target = self.base.target_transform(target)
        return img, target


# -------------------------
# Transformation & Loaders
# -------------------------

def make_transforms(image_size=Config["img_size"]):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])
    return train_tf, val_tf


def make_loaders(base_ds, train_idx, val_idx, seed, batch_size, num_workers,
                image_size=Config["img_size"]):
    train_tf, val_tf = make_transforms(image_size)
    train_ds = TransformedSubset(base_ds, train_idx, transform=train_tf)
    val_ds = TransformedSubset(base_ds, val_idx, transform=val_tf)
    g = torch.Generator().manual_seed(seed)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, generator=g, pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )
    return train_loader, val_loader


# ------
# Setup
# ------

def set_seeds_to(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def set_up(seed, data_dir):
    set_seeds_to(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Accelerator: {device}")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    base_ds = datasets.ImageFolder(data_dir)
    g = torch.Generator().manual_seed(seed)
    n = len(base_ds)
    train_size = int((1 - Config["val_split"]) * n)
    val_size = n - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        base_ds, [train_size, val_size], generator=g,
    )
    return device, base_ds, train_subset.indices, val_subset.indices


# ----------------------
# Training & Evaluation
# ----------------------

def _set_bn_eval(m):
    if isinstance(m, nn.modules.batchnorm._BatchNorm):
        m.eval()


def train_epoch(model, dataloader, criterion, optimizer, device, feature_extract="full"):
    model.train()
    # keep BN layers frozen during train whenever any part of the backbone is frozen (full or partial feature extraction)
    if feature_extract in ("full", "partial"):
        model.apply(_set_bn_eval)
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / len(dataloader), 100.0 * correct / total


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return running_loss / len(dataloader), 100.0 * correct / total


def train_model(device, model, train_loader, val_loader,
                num_epochs, lr, feature_extract="full"):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
    )
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 30)
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, feature_extract,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%")
        print(f"Val   Loss: {val_loss:.4f}  Val   Acc: {val_acc:.2f}%")

    return history


# --------
# Plotting
# --------

def plot_training_history(history, title="Training History"):
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"], label="Train Loss")
    ax1.plot(history["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title(f"{title} - Loss"); ax1.legend(); ax1.grid(True)
    ax2.plot(history["train_acc"], label="Train Acc")
    ax2.plot(history["val_acc"], label="Val Acc")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.set_title(f"{title} - Accuracy"); ax2.legend(); ax2.grid(True)
    plt.tight_layout()
    plt.savefig(f"training_history_{title.replace(' ', '_')}.png")
    plt.show()


# ----
# Main
# ----

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   type=str, default=Config["data_dir"])
    parser.add_argument("--num_epochs", type=int, default=Config["epochs"])
    parser.add_argument("--batch_size", type=int, default=Config["batch_size"])
    parser.add_argument("--lr",         type=float, default=Config["lr"])
    parser.add_argument("--num_workers", type=int, default=Config["num_workers"])
    parser.add_argument("--wandb",      action="store_true")
    args = parser.parse_args()

    print("\nStarting Main Script...")
    print(f"Config: {Config}")

    device, base_ds, train_idx, val_idx = set_up(
        seed=Config["seed"], data_dir=args.data_dir,
    )
    if Config.get("mini-run"):
        train_idx = train_idx[:len(train_idx)//10]
        val_idx = val_idx[:len(val_idx)//10]
        print(f"Mini-run: {len(train_idx)} training samples; {len(val_idx)} validation samples")
    train_loader, val_loader = make_loaders(
        base_ds, train_idx, val_idx,
        seed=Config["seed"], batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    feature_extract = Config["freeze_bb"]
    tag = feature_extract  # "full", "partial", or "none"
    labels = {"full": "FF Backbone", "partial": "PF Backbone", "none": "Fully UF Backbone"}
    print(f"\nResNet50 — {labels[feature_extract]}")

    model = get_pretrained_model(
        num_classes=Config["num_classes"],
        dropout_prob=Config["dropout"],
        feature_extract=feature_extract,
    )

    if args.wandb:
        import wandb
        wandb.init(project="isic-resnet50", config={**Config, "mode": tag})

    history = train_model(
        device, model, train_loader, val_loader,
        num_epochs=args.num_epochs, lr=args.lr,
        feature_extract=feature_extract,
    )
    plot_training_history(history, f"ResNet50_{tag}")

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
