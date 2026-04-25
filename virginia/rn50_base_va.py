"""
rn50_base_va.py
Baseline ResNet-50 training script
    * with "weighted_ce" loss function as DEFAULT
    * with ablation options for 
        * freezing backbone, 
        * dropout, 
        * augmentation, and 
        * architecture choice
    *with Config dict matching HS & BL for ablation comparison

criterion:  CrossEntropyLoss with class weights to address imbalance (default in Config: "weighted_ce")
    
optimizer:  torch.optim.Adam()

Alternative architectures added in Config: 
    "architecture": "resnet50",    # "mobilenetv3_small" | "efficientnet_b0" | "resnet50
    
Done: 
1. restored suite of metrics and conf matrix on wandb
2. planned ablation studies --> Efrain
# --------------------------------------------------------------------
Next Steps: (4/23): 

3. look at fusion of MLP (metadata) 
4. look at impage preprocessing workflow --> test and propose to team if successful
5. Finish reading and REVISING LITERATURE REVIEW to reflect current state of the art and our approach
# --------------------------------------------------------------------
Possible Steps (if time allows):
1. ensembling combinations of three fully fine tuned and optimized models + MLP metadata model
2. Looking at Fitzpatrick issue
# --------------------------------------------------------------------
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import (balanced_accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix, precision_score, recall_score)




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
        --wandb,
        --freeze_bb, 
        --dropout,
        --loss_fn,

    Editable in Config directly:
        img_size, 
        val_split, 
        seed, 
        mini-run, 
        num_classes, 
        classes, 
        loss_fn

    Ablation options not yet implemented (commented out in Config):
        patience, 
        scheduler, 
        step_size, 
        gamma
    
"""
# defaults that can be overridden by command-line arguments 
# or specifying ablation options in the main function call 
# (IF arparse setup for that arg)
Config = {
    # added from HS abl 1&2
    "architecture": "resnet50",    # "mobilenetv3_small" | "efficientnet_b0" | "resnet50
    
    "pretrained":   True,           # load ImageNet weights | False = start all weights at random and train from scratch
    "freeze_bb":    "full",         # "full" = freeze all, "partial" = unfreeze layer4, "none" = train everything
    "augmentation": "standard",     # "none" | "geometric" | "color" | "standard"
    "num_classes":  8,              # 9: includes 'unk' or images that are none-of-the-known-classes
    "dropout":      0.3,            # 0.0=disabled | E.g.: 0.2=20% RN features dropped b4 classification | to try: 0.05, 0.1, 0,2, 0.3, 0.4, 0.5
    "img_size":     224,
    "batch_size":   32,
    "epochs":       30,
    "lr":           1e-4,           # changed from original 1e-3 (consider 1e-5 for FUF)
    "num_workers":  4,              # may need to change this
    "val_split":    0.20,
    "seed":         42,
    "data_dir":     os.path.expanduser("~/Downloads/ISIC_2019_mini"),
    "classes":      ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"],
    "mini-run":     False,  # True = use 10% of data | False = use 100%

    # --- more ablation options ---
    # "patience":     10,               # early stopping after N epochs no improvement
    "loss_fn":      "weighted_ce",      # "ce" | "focal" | "weighted_ce"
    # "scheduler":    "step",           # "step" | "cosine" | None
    # "step_size":    5,              #  StepLR step size
    # "gamma":        0.1,            # StepLR decay factor
}

# --------------------------------------------------------------------
# Model Definition(s) - Resnet-50, EfficientNet_B0, MobileNetV3_small
# --------------------------------------------------------------------

# -- Establishing Multi-architecture Structure to mirror HS & BL -- #
def get_pretrained_model(arch=Config["architecture"],
                        num_classes=Config["num_classes"],
                        dropout_prob=Config["dropout"],
                        feature_extract=Config["freeze_bb"]):

    if arch == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        final_layer = "fc"
        partial_layers = [model.layer4]  # last layer/block
        
    elif arch == "mobilenetv3_small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        final_layer = "classifier"
        partial_layers = [model.features[-1]]  # last feature block
        
    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        final_layer = "classifier"
        partial_layers = [model.features[-1]]
        
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    # freeze logic
    if feature_extract == "full":
        for param in model.parameters():
            param.requires_grad = False
            
    elif feature_extract == "partial":
        for param in model.parameters():
            param.requires_grad = False
        for layer in partial_layers:
            for param in layer.parameters():
                param.requires_grad = True

    # replace head
    if final_layer == "fc":
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout_prob), 
            nn.Linear(in_features, num_classes)
        )
        
    elif final_layer == "classifier":
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Sequential(
            nn.Dropout(dropout_prob), 
            nn.Linear(in_features, num_classes)
        )
    return model

# def get_pretrained_model(num_classes=Config["num_classes"],
#                       dropout_prob=Config["dropout"],
#                        feature_extract=Config["freeze_bb"]):
#    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
#    if feature_extract == "full":
#        for param in model.parameters():
#            param.requires_grad = False
#    elif feature_extract == "partial":
#        for param in model.parameters():
#            param.requires_grad = False
#        for param in model.layer4.parameters():
#            param.requires_grad = True
#    elif feature_extract == "none":
#        for param in model.parameters():
#            param.requires_grad = True

#    num_in_features = model.fc.in_features  # 2048 for ResNet-50
#    model.fc = nn.Sequential(
#        nn.Dropout(dropout_prob),
#        nn.Linear(num_in_features, num_classes),
#    )
#    return model


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

# adding more complex augmentation to match HS and BL code and choices

def make_transforms(image_size=Config["img_size"], augmentation=Config.get("augmentation", "standard")):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if augmentation == "none":
        aug_tfms = []
    elif augmentation == "geometric":
        aug_tfms = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
        ]
    elif augmentation == "color":
        aug_tfms = [
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ]
    elif augmentation == "standard":
        aug_tfms = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ]

    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        *aug_tfms,
        transforms.ToTensor(),
        normalize,
    ])
    val_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])
    return train_tf, val_tf

# 4-23: altering for argparse control of augmentation strategy
def make_loaders(base_ds, train_idx, val_idx, seed, batch_size, num_workers,
                image_size=Config["img_size"], augmentation=Config["augmentation"]):
    train_tf, val_tf = make_transforms(image_size, augmentation)
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
"""
Contains: 

    * set seed
    * device (cuda) check
    * data splitting with `train_test_split`

"""

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
    base_ds = datasets.ImageFolder(data_dir)  # stratified data directory
    targets = [s[1] for s in base_ds.samples]  # s[1] is class index --> list 'targets'
    # new setup to match team (old commented out, for debug)
    train_idx, val_idx = train_test_split(
        range(len(base_ds)), 
        test_size=Config["val_split"], 
        stratify=targets, 
        random_state=seed,
    )
    
    # g = torch.Generator().manual_seed(seed)
    # n = len(base_ds)
    # train_size = int((1 - Config["val_split"]) * n)
    # val_size = n - train_size
    # train_subset, val_subset = torch.utils.data.random_split(
    #     base_ds, 
    #     [train_size, val_size], 
    #     generator=g,
    # )
    
    return device, base_ds, train_idx, val_idx
    # return device, base_ds, train_subset.indices, val_subset.indices

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


def evaluate(model, dataloader, criterion, device, num_classes=Config["num_classes"]):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # per-class AUC
    auc_per_class = []
    for c in range(num_classes):
        try:
            auc_per_class.append(roc_auc_score((all_labels == c).astype(int), all_probs[:, c]))
        except ValueError:
            auc_per_class.append(0.0)

    # per-class precision, recall, f1
    prec_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
    rec_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)
    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)

    # per-class specificity from confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=range(num_classes))
    spec_per_class = np.zeros(num_classes)
    for c in range(num_classes):
        tn = cm.sum() - cm[c, :].sum() - cm[:, c].sum() + cm[c, c]
        fp = cm[:, c].sum() - cm[c, c]
        spec_per_class[c] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "loss": running_loss / len(dataloader),
        "acc": 100.0 * correct / total,
        "bacc": balanced_accuracy_score(all_labels, all_preds),
        "conf_matrix": cm,
        # per-class arrays (indexed by class)
        "auc_per_class": auc_per_class,
        "precision_per_class": prec_per_class,
        "recall_per_class": rec_per_class,       # recall = sensitivity
        "f1_per_class": f1_per_class,
        "specificity_per_class": spec_per_class,
        # macro / weighted aggregates
        "f1_macro": f1_score(all_labels, all_preds, average="macro"),
        "f1_weighted": f1_score(all_labels, all_preds, average="weighted"),
        "precision_macro": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "precision_weighted": precision_score(all_labels, all_preds, average="weighted", zero_division=0),
        "recall_macro": recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "recall_weighted": recall_score(all_labels, all_preds, average="weighted", zero_division=0),
        "sensitivity_macro": recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "specificity_macro": float(spec_per_class.mean()),
        "auc_macro": float(np.mean(auc_per_class)),
    }

# 4-25: updated logging to match team schema (val/ prefix, per-class metrics, best tracking)
def train_model(device, model, train_loader, val_loader,
                num_epochs, lr, feature_extract="full",
                use_wandb=False, base_ds=None, arch=Config["architecture"]):
    if use_wandb:
        import wandb
    model = model.to(device)
    classes = base_ds.classes if base_ds else Config["classes"]
    # -------------------------------------------------
    # adding ce loss calculations:
    
    targets =   [base_ds.targets[i] for i in train_loader.dataset.indices] # get class indices for training samples
    counts =    np.bincount(targets, minlength=Config["num_classes"]) # count samples per class
    
    weights =   1.0 / (counts + 1e-6) # inverse frequency weighting
    weights =   weights / np.sum(weights) * Config["num_classes"] # normalize to num_classes
    
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # -------------------------------------------------
    
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
    )

    history = {"train_loss": [], "train_acc": [], 
            "val_loss": [], "val_acc": [], 
            "val_bacc": [], "val_f1_macro": [], 
    }

    # --- benchmark: log model size and inference speed once ---
    if use_wandb:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
        wandb.run.summary["benchmark/params_total"] = total_params
        wandb.run.summary["benchmark/params_trainable"] = trainable_params
        wandb.run.summary["benchmark/size_mb"] = round(size_mb, 2)
        # inference timing (single batch)
        model.eval()
        dummy = next(iter(val_loader))[0].to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        inference_ms = (time.time() - t0) * 1000 / dummy.size(0)
        wandb.run.summary["benchmark/inference_ms"] = round(inference_ms, 2)

    # tracking variables for best wandb metrics
    best_bacc = 0.0
    best_epoch = -1
    best_conf_matrix = None
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 30)
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, feature_extract,
        )
        val_metrics = evaluate(model, val_loader, criterion, device)
        val_loss = val_metrics["loss"]
        val_acc = val_metrics["acc"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_bacc"].append(val_metrics["bacc"])
        history["val_f1_macro"].append(val_metrics["f1_macro"])

        print(f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%")
        print(f"Val   Loss: {val_loss:.4f}  Val   Acc: {val_acc:.2f}%")
        print(f"Val   BACC: {val_metrics['bacc']:.4f}  Val   F1: {val_metrics['f1_macro']:.4f}")

        if use_wandb:
            # --- per-epoch logging with val/ prefix ---
            log_dict = {
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "train/acc": train_acc,
                "val/loss": val_loss,
                "val/acc": val_acc,
                "val/bacc": val_metrics["bacc"],
                "val/f1_macro": val_metrics["f1_macro"],
                "val/f1_weighted": val_metrics["f1_weighted"],
                "val/precision_macro": val_metrics["precision_macro"],
                "val/precision_weighted": val_metrics["precision_weighted"],
                "val/recall_macro": val_metrics["recall_macro"],
                "val/recall_weighted": val_metrics["recall_weighted"],
                "val/sensitivity_macro": val_metrics["sensitivity_macro"],
                "val/specificity_macro": val_metrics["specificity_macro"],
                "val/auc_macro": val_metrics["auc_macro"],
            }
            # per-class metrics
            for i, name in enumerate(classes):
                log_dict[f"val/auc_{name}"] = val_metrics["auc_per_class"][i]
                log_dict[f"val/precision_{name}"] = val_metrics["precision_per_class"][i]
                log_dict[f"val/recall_{name}"] = val_metrics["recall_per_class"][i]
                log_dict[f"val/sensitivity_{name}"] = val_metrics["recall_per_class"][i]
                log_dict[f"val/specificity_{name}"] = val_metrics["specificity_per_class"][i]
                log_dict[f"val/f1_{name}"] = val_metrics["f1_per_class"][i]
            wandb.log(log_dict)

            # --- best tracking keyed to BACC ---
            if val_metrics["bacc"] > best_bacc:
                best_bacc = val_metrics["bacc"]
                best_epoch = epoch + 1
                best_conf_matrix = val_metrics["conf_matrix"]
                wandb.run.summary["best/bacc"] = val_metrics["bacc"]
                wandb.run.summary["best/acc"] = val_acc
                wandb.run.summary["best/f1_macro"] = val_metrics["f1_macro"]
                wandb.run.summary["best/f1_weighted"] = val_metrics["f1_weighted"]
                wandb.run.summary["best/precision_macro"] = val_metrics["precision_macro"]
                wandb.run.summary["best/precision_weighted"] = val_metrics["precision_weighted"]
                wandb.run.summary["best/recall_macro"] = val_metrics["recall_macro"]
                wandb.run.summary["best/recall_weighted"] = val_metrics["recall_weighted"]
                wandb.run.summary["best/sensitivity_macro"] = val_metrics["sensitivity_macro"]
                wandb.run.summary["best/specificity_macro"] = val_metrics["specificity_macro"]
                wandb.run.summary["best/auc_macro"] = val_metrics["auc_macro"]
                wandb.run.summary["best_epoch"] = epoch + 1
                for i, name in enumerate(classes):
                    wandb.run.summary[f"best/auc_{name}"] = val_metrics["auc_per_class"][i]
                    wandb.run.summary[f"best/precision_{name}"] = val_metrics["precision_per_class"][i]
                    wandb.run.summary[f"best/recall_{name}"] = val_metrics["recall_per_class"][i]
                    wandb.run.summary[f"best/sensitivity_{name}"] = val_metrics["recall_per_class"][i]
                    wandb.run.summary[f"best/specificity_{name}"] = val_metrics["specificity_per_class"][i]
                    wandb.run.summary[f"best/f1_{name}"] = val_metrics["f1_per_class"][i]

                # confusion matrix at best epoch
                fig, ax = plt.subplots(figsize=(10, 8))
                ConfusionMatrixDisplay(confusion_matrix=best_conf_matrix,
                                      display_labels=classes).plot(ax=ax, colorbar=False, xticks_rotation=45)
                ax.set_title(f"Best Confusion Matrix — Epoch {best_epoch}")
                wandb.run.summary["confusion_matrix/best"] = wandb.Image(fig)
                plt.close(fig)

    # --- final confusion matrix (last epoch) ---
    if use_wandb:
        fig, ax = plt.subplots(figsize=(10, 8))
        ConfusionMatrixDisplay(confusion_matrix=val_metrics["conf_matrix"],
                              display_labels=classes).plot(ax=ax, colorbar=False, xticks_rotation=45)
        ax.set_title(f"Final Confusion Matrix — Epoch {num_epochs}")
        wandb.run.summary["confusion_matrix/final"] = wandb.Image(fig)
        plt.close(fig)

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

"""
For quick-reference only: 
    
    "architecture": "resnet50",    # "mobilenetv3_small" | "efficientnet_b0" | "resnet50"
        
    "pretrained":   True,           # load ImageNet weights | False = start all weights at random and train from scratch
    "freeze_bb":    "full",         # "full" = freeze all, "partial" = unfreeze layer4, "none" = train everything
    "augmentation": "standard",     # "none" | "geometric" | "color" | "standard"
    
    "num_classes":  8,              # 9: includes 'unk' or images that are none-of-the-known-classes
    "dropout":      0.3,            # 0.0=disabled | E.g.: 0.2=20% RN features dropped b4 classification | to try: 0.05, 0.1, 0,2, 0.3, 0.4, 0.5
    "img_size":     224,
    "batch_size":   32,
    "epochs":       30,
    "lr":           1e-4,           # changed from original 1e-3 (consider 1e-5 for FUF)
"""
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    type=str,   default=Config["data_dir"])
    parser.add_argument("--num_epochs",  type=int,   default=Config["epochs"])
    parser.add_argument("--batch_size",  type=int,   default=Config["batch_size"])
    parser.add_argument("--lr",          type=float, default=Config["lr"])
    parser.add_argument("--num_workers", type=int,   default=Config["num_workers"])
    parser.add_argument("--wandb",       action="store_true")
    # added 4-23:
    parser.add_argument("--freeze_bb",   type=str,   default=Config["freeze_bb"], choices=["full", "partial", "none"])
    parser.add_argument("--dropout",     type=float, default=Config["dropout"])
    parser.add_argument("--arch",        type=str,   default=Config["architecture"], choices=["mobilenetv3_small", "efficientnet_b0", "resnet50"])
    parser.add_argument("--aug",         type=str,   default=Config["augmentation"], choices=["none", "geometric", "color", "standard"])
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
        augmentation=args.aug,
    )

    labels = {"full": "Fully Frozen Backbone", "partial": "Partially Frozen Backbone", "none": "Unfrozen Backbone"}
    print(f"\n{args.arch} — {labels[args.freeze_bb]}")

    model = get_pretrained_model(
        arch=args.arch,
        num_classes=Config["num_classes"],
        dropout_prob=args.dropout,
        feature_extract=args.freeze_bb,
    )

    if args.wandb:
        import wandb
        wandb.init(
            entity="vrb9e-university-of-virginia-school-of-data-science",
            project="ISIC2019",
            name=(
                f"{args.arch}"
                f"_freeze={args.freeze_bb}"
                f"_aug={args.aug}"
                f"_drop={args.dropout}"
                f"_lr={args.lr}"
                f"_ep={args.num_epochs}"
                f"_mini={Config.get('mini-run')}"
            ),
            config={**Config,
                    "architecture": args.arch,
                    "freeze_bb": args.freeze_bb,
                    "dropout": args.dropout,
                    "augmentation": args.aug,
                    "epochs": args.num_epochs,
                    "mode": args.freeze_bb},
        )

    history = train_model(
        device, model, train_loader, val_loader,
        num_epochs=args.num_epochs, lr=args.lr,
        feature_extract=args.freeze_bb,
        use_wandb=args.wandb, base_ds=base_ds,
        arch=args.arch,
    )
    plot_training_history(history, f"{args.arch}_{args.freeze_bb}")

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()