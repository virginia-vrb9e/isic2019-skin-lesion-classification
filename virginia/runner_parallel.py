"""
runner_parallel.py
==================
Train ResNet50, MobileNetV2, and EfficientNetB3 sequentially on ISIC 2019.
All three models share identical loaders, class weights.
Logs to W&B project: parallel-3-models-15-epochs

Note: "parallel" here means back-to-back in one job, not simultaneous GPU threads.
      On Rivanna, all three run in the same SLURM allocation.

Output: 
    history (dict): 
        'train_loss'
        'train_acc'
        'val_loss'
        'val_acc'
        
Usage (command line):
    python runner_parallel.py                        # local (no W&B)
    python runner_parallel.py --wandb                # with W&B
    python runner_parallel.py --data_dir /path/to/data --wandb
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from TL_resnet50 import get_pretrained_model as get_resnet50
from TL_mobilenetv2 import get_pretrained_model as get_mobilenet
from TL_efficientnetb3 import get_pretrained_model as get_efficientnet
from isic_utils import (
    set_up,
    make_loaders,
    train_epoch, 
    evaluate, 
    plot_training_history
)
import wandb
from sklearn.metrics import ConfusionMatrixDisplay

# ------- Config ------- #

WANDB         = False
SEED          = 42
BATCH_SIZE    = 32
NUM_WORKERS   = 4
NUM_CLASSES   = 8
NUM_EPOCHS    = 15
LR            = 0.001
DROPOUT       = 0.3
DATA_DIR      = os.path.expanduser("~/Downloads/ISIC_2019_mini")
WANDB_PROJECT = "parallel-3-models-15-epochs"

MODELS = {
    "resnet50": get_resnet50,
    "mobilenet_v2": get_mobilenet,
    "efficientnet_b3": get_efficientnet,
}


# ------- Train one model ------- #

def train_one_model(
    name, 
    model, 
    train_loader, 
    val_loader, 
    class_weights,
    device, 
    num_epochs, 
    lr, 
    use_wandb, 
    base_ds) -> dict: 

    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print(f"\n{'='*50}")
    print(f"  Training: {name}")
    print(f"{'='*50}")

    for epoch in range(num_epochs):
        print(f"\n  Epoch {epoch+1}/{num_epochs}")
        print("  " + "-" * 28)

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
        val_acc = val_metrics['acc']

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"  Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%")
        print(f"  Val   Loss: {val_loss:.4f}  Val   Acc: {val_acc:.2f}%")

        if use_wandb:
#            import wandb
            log_dict = {
                "epoch":epoch + 1,
                f"{name}/train_loss":train_loss,
                f"{name}/train_acc":train_acc,
                f"{name}/val_loss":val_loss,
                f"{name}/val_acc": val_acc,
                f"{name}/val_macro_f1": val_metrics['macro_f1'],
                f"{name}/val_bacc": val_metrics['bacc'],
                f"{name}/lr": LR,
            }
            for c, auc in enumerate(val_metrics['auc_per_class']):
                log_dict[f"{name}/val_auc_class_{c}"] = auc
            wandb.log(log_dict)
            
            # confusion matrix            
#            from sklearn.metrics import ConfusionMatrixDisplay
            
            fig, ax = plt.subplots(figsize=(10, 8))
            disp = ConfusionMatrixDisplay(
                confusion_matrix=val_metrics['conf_matrix'],
                display_labels=base_ds.classes
            )
            disp.plot(ax=ax, colorbar=False, xticks_rotation=45)
            ax.set_title(f"Confusion Matrix — Epoch {epoch+1}")
            wandb.log({f"{name}/conf_matrix": wandb.Image(fig)}) # f"{name} to create separate keys -> no overwrite
            plt.close(fig)
            
    print(f"\n {name} Training Finished")
    plot_training_history(history, title=name)
    return history

# ------- Main ------- #

def main(args):
    use_wandb = args.wandb or WANDB

    if use_wandb:
#        import wandb  # in dl-course; not needed here on Rivanna
        wandb.init(project=WANDB_PROJECT, config={
            "models":      list(MODELS.keys()),
            "epochs":      args.num_epochs,
            "batch_size":  args.batch_size,
            "lr":          args.lr,
            "dropout":     DROPOUT,
            "seed":        SEED,
            "feature_extract": True
        })

    # Shared setup — all models see same splits and class weights
    device, base_ds, train_idx, val_idx = set_up(seed=SEED, data_dir=args.data_dir)
    train_loader, val_loader, class_weights = make_loaders(
        base_ds,
        train_idx, 
        val_idx,
        batch_size=args.batch_size,
        num_workers=args.num_workers, 
        device=device
    )

    all_results = {}

    for model_name, loader_fn in MODELS.items():
        model = loader_fn(
        model_name, 
        num_classes=NUM_CLASSES,
        feature_extract=True, 
        dropout_prob=DROPOUT
        )
        
        history = train_one_model(
            model_name, 
            model, 
            train_loader, 
            val_loader, 
            class_weights,
            device, 
            args.num_epochs, 
            args.lr, 
            use_wandb, 
            base_ds
        )
        all_results[model_name] = history['val_acc'][-1]

    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    for name, acc in all_results.items():
        print(f"  {name:20s}  val_acc: {acc:.2f}%")

    if use_wandb:
#        import wandb # in dl-course already; not needed here on Rivanna
        wandb.finish()


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
