"""
identifying-misclassified-lesions.py

Steps: 
    1. Loads trained ResNet-50 checkpoint, 
    2. runs inference on validation split,
    3. prints per-class lists of misclassified and correctly classified 
        image paths (5 per class) for LaTeX figure construction.

Usage:
    # --- (I have to activate an env) ---
    conda activate dl-course
    python identifying-misclassified-lesions.py \
    --weights saved-weights/best_bacc_resnet50_freeze=none_aug=standard_drop=0.3_loss=weighted_ce_lr=0.0001_ep=60_sch=None.pt \
    --data_dir /home/vrb9e/DS6050_Deep-Learning/group-project/data/ \
    --max_examples 1000     # optional (for speed on CPU)
"""

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from collections import defaultdict

# --- importing from base script ---
from rn50_base_va import Config, set_up, TransformedSubset, get_pretrained_model

def main():
    # -- parse arguments ---
    # Config["data_dir"] imported but not working -- must pass a flag
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default=Config["data_dir"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers",type=int, default=0)
    parser.add_argument("--n",type=int, default=5, help="Examples per class")
    parser.add_argument("--max_examples", type=int,default=None, help="Cap val images evaluated")
    args = parser.parse_args()
    
    # --- calling set_up (imported) for data & device ---
    device, base_ds, _, val_idx = set_up(seed=Config["seed"], data_dir=args.data_dir)
    classes = base_ds.classes

    # --- 5/3 CPU limitation!! optionally subsample stratified val set for speed ---
    # Dual check: (1) max_examples exists; (2) max_examples < total val images
    if args.max_examples and args.max_examples < len(val_idx):
        from sklearn.model_selection import train_test_split
        val_targets = [base_ds.samples[i][1] for i in val_idx]
        val_idx, _ = train_test_split(
            val_idx, train_size=args.max_examples,
            stratify=val_targets, random_state=Config["seed"],
        )
        print(f"Subsampled to {len(val_idx)} val images")

    # --- load pre-trainted model ---
    # --- get_pretrained_model imported ---
    model = get_pretrained_model()
    model.load_state_dict(torch.load(args.weights, map_location=device))
    # --- set eval mode ---
    model.to(device).eval()

    # --- Data Loader: val loader (no augmentation) ---
    # validation transforms: resize, to tensor, normalize (no augmentation)
    val_tf = transforms.Compose([
        transforms.Resize((Config["img_size"], Config["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    # --- TransformedSubset imported ---
    # validation subset with transforms
    val_ds = TransformedSubset(
        base_ds, 
        val_idx, 
        transform=val_tf
    )
    # --- DataLoader ---
    val_loader = DataLoader(
        val_ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers
    )

    # --- inference w/ progress bar ---
    all_preds = []
    with torch.no_grad():
        # tqdm: progress bar 
        for inputs, _ in tqdm(val_loader, desc="Inference"):
            preds = model(inputs.to(device)).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())

    # --- convert to numpy arrays ---
    all_preds = np.array(all_preds)
    true_labs = np.array([base_ds.samples[i][1] for i in val_idx])

    # --- collect up to 'limit' per class ---
    correct  = defaultdict(list)
    misclass = defaultdict(list)

    for idx, (tru, pr) in enumerate(zip(true_labs, all_preds)):
        # get filename and classname
        fname = os.path.basename(base_ds.samples[val_idx[idx]][0])
        cname = classes[tru]
        if tru == pr and len(correct[cname]) < args.n:
            correct[cname].append(fname)
        elif tru != pr and len(misclass[cname]) < args.n:
            misclass[cname].append((fname, classes[pr]))

    # --- print results ---
    for label, data in [("CORRECTLY CLASSIFIED", correct),
                        ("MISCLASSIFIED (by true class)", misclass)]:
        print(f"\n{'='*60}\n  {label}\n{'='*60}")
        # print per-class lists
        for cls in classes:
            entries = data.get(cls, [])
            print(f"\n--- {cls} ({len(entries)}) ---")
            for e in entries:
                # tuple: (fname, predicted class)
                if isinstance(e, tuple):
                    print(f"  {e[0]}  -> predicted: {e[1]}")
                else:
                    print(f"  {e}")


if __name__ == "__main__":
    main()
