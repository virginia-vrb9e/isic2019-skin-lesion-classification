"""
isic_utils.py
=============
Shared infrastructure for ISIC 2019 runners
In current setup: Imported by 
    - runner_resnet50.py 
    - runner_parallel.py.

Methods:
    - make_transforms()
    - TransformedSubset dataset
    - make_loaders (for now: no more WeightedRandomSampler on train)
    - set_up (seeds, device, train/val split)
    - # EarlyStopping   (not used at present)
    - train_epoch / evaluate
    - plot_training_history
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    balanced_accuracy_score, 
    recall_score, 
    f1_score,
    roc_auc_score, 
    confusion_matrix
)


# -------- Transforms -------- #

def make_transforms():
    """
    Train: Resize(256) -> CenterCrop(224) -> augmentation -> Normalize
    Val:   Resize(256) -> CenterCrop(224) -> Normalize  # no aug here
    ImageNet mean/std normalization throughout!
    """
    # Normalize to ImageNet defaults
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    # resize shortest side of image (ISIC images vary in size) to 256
    # center crop to 224x224
    # augmentations: RandomRotation, ColorJitter - ALL images
    train_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomRotation(degrees=90),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        normalize
    ])
    # no augmentation with val
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    return train_tf, val_tf


# -------- Dataset -------- #


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


# -------- Class weights -------- #

def compute_class_weights(base_ds, train_idx, device):
    """
    Inverse-frequency class weights from training only; used for CrossEntropyLoss(weight=...)
    """
    train_labels = [base_ds.samples[i][1] for i in train_idx]
    counts = np.bincount(train_labels)
    # inverse frequency: rare classes get higher weight
    weights = 1.0 / counts.astype(np.float32)
    weights = weights / weights.sum()  # normalize
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)  # to(device) moves tensor to model location

    return class_weights_tensor


# -------- Loaders -------- #

def make_loaders(base_ds, train_idx, val_idx, batch_size, num_workers, device):
    
    # transforming ds
    train_tf, val_tf = make_transforms()
    train_ds = TransformedSubset(base_ds, train_idx, transform=train_tf)
    val_ds   = TransformedSubset(base_ds, val_idx,   transform=val_tf)

    class_weights = compute_class_weights(base_ds, train_idx, device)

    pin = torch.cuda.is_available()
    
    # shuffle=True
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin
    )
    # shuffle = False
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin
    )
    
    # ---- debugging ---- #
    print(f"[DEBUG] Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    print(f"[DEBUG] class_weights: {class_weights}")
    # --------------------#
    
    return train_loader, val_loader, class_weights


# -------- Setup -------- #

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
    train_size = int(0.8 * n)
    val_size   = n - train_size
    train_sub, val_sub = torch.utils.data.random_split(
        base_ds, [train_size, val_size], generator=g
    )
    
    # For debugging - config in runner
    print(f"[DEBUG] Dataset: {len(base_ds)} images, {len(base_ds.classes)} classes: {base_ds.classes}")
    print(f"[DEBUG] Train: {len(train_sub.indices)} | Val: {len(val_sub.indices)}")

    return device, base_ds, train_sub.indices, val_sub.indices


# ---------------------------------------------------------------#
# Early stopping
# NOT USED at present... not beneficial with FF ResNet50 backbone
# ---------------------------------------------------------------#

class EarlyStopping:
    """
    Monitors val_loss. Saves best model weights to `path`.
    Set min_delta > 0 to require meaningful improvement.
    """
    def __init__(self, patience=5, min_delta=0.0, path='best_model.pt'):
        self.patience  = patience
        self.min_delta = min_delta
        self.path      = path
        self.best_loss = None
        self.counter   = 0
        self.stop      = False

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
            torch.save(model.state_dict(), self.path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True

    def load_best(self, model):
        model.load_state_dict(torch.load(self.path))
        return model



# -------- Training & Evaluation -------- #

def _set_bn_eval(m):
    if isinstance(m, nn.modules.batchnorm._BatchNorm):
        m.eval()


def train_epoch(model, dataloader, criterion, optimizer, device, feature_extract=False):
    model.train()
    if feature_extract:
        model.apply(_set_bn_eval)
    running_loss, correct, total = 0.0, 0, 0  # initialize
    for inputs, labels in tqdm(dataloader, desc="  Train"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total   += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / len(dataloader), 100.0 * correct / total


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0  # initialize
    
    # initilize:
    all_probs  = []
    all_preds  = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="  Val  "):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            # define probs after outputs created
            probs = torch.softmax(outputs, dim=1)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total   += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            # .numpy(): convert tensor output to array for sklearn
            all_probs.append(probs.cpu().numpy())
            all_preds.append(predicted.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_probs = np.concatenate(all_probs)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    # for debugging - config in runner
    print(f"[DEBUG] all_probs shape: {all_probs.shape}")    # should be (n_samples, 8)
    print(f"[DEBUG] all_labels unique: {np.unique(all_labels)}")  # should be 0-7
    
    # macro -- compute for each class, then average equally
    # zero_division: Sets the value to return when there is a zero division, 
    # i.e. when all predictions and labels are negative.
    metrics = {
        'loss': running_loss / len(dataloader),
        'acc': 100.0 * correct / total, 
        'bacc': balanced_accuracy_score(all_labels, all_preds),
        'macro_recall': recall_score(all_labels, all_preds, average='macro', zero_division=0),
        'macro_f1': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'auc_per_class': roc_auc_score(all_labels, all_probs, multi_class='ovr', average=None), 
        'conf_matrix': confusion_matrix(all_labels, all_preds)
    }
        
    return metrics # return dict


# -------- Plotting -------- #

def plot_training_history(history, title="Training History"):
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'],   label='Val Loss')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title(f'{title} - Loss'); ax1.legend(); ax1.grid(True)
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'],   label='Val Acc')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'{title} - Accuracy'); ax2.legend(); ax2.grid(True)
    plt.tight_layout()
    plt.savefig(f"training_history_{title}.png")
    # plt.show()  # no display on Rivanna; substituting: 
    plt.close()
