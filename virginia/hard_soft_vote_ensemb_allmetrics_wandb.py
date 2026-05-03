"""
hard_soft_vote_ensemb_allmetrics_wandb.py

-- Note:--
AUC and MEL sensitivity need probabilities: 
Ensemble AUC uses the averaged softmax probabilities.
hard vote determines BACC/F1.

# --- script path: ---
    * load weights
    * locate ISIC dataset: DATA_DIR
    * run images through each (essentially FF) model to get predictions 
        * using torch.no-grad() so that freeze_bb setting is irrelevant.  
    * vote using majority rule
    
--- !!! -- Future development: --- 
save preds as .npy files to create full arrays that do not need data

# --- MY ENV --
eval "$(conda shell.bash hook)"
conda activate dl-course
python hard_soft_vote_ensemb_allmetrics_wandb.py
OR 
python hard_soft_vote_ensemb_allmetrics_wandb.py 2>&1 | tee ensemble_log.txt

"""

import numpy as np
import torch
from scipy import stats
from tqdm import tqdm
from sklearn.metrics import (balanced_accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix, roc_curve)
                        
# --- import from base architecture --- 
from rn50_base_va import get_pretrained_model, set_up, make_loaders, Config

# --- Edit MODELS & USE_WANDB for each run/combination --- 
""" 
--- !!! -- Future development: --- 
save preds as .npy files to create full arrays that do not need data
"""
# load weights
MODELS = {
#    "resnet50":          "weights/best_bacc_resnet50_freeze=none_aug=standard_drop=0.3_loss=weighted_ce_lr=0.0001_ep=60_sch=None.pt",
    "efficientnet_b0":   "weights/best_bacc_efficientnet_b0.pt",
    "mobilenetv3_small": "weights/best_bacc_mobilenetv3_small.pt",
}

DATA_DIR="/home/vrb9e/DS6050_Deep-Learning/group-project/data/ISIC_2019_Training_Organized"
USE_WANDB = True
# -------------------------------------------------------------

if USE_WANDB:
    import wandb
    wandb.init(
        entity="vrb9e-university-of-virginia-school-of-data-science",
        project="ISIC2019",
        name=f"ensemble_{'_'.join(MODELS.keys())}",
        config={"models": list(MODELS.keys()), "method": "hard_vote"},
    )

# --- set up call --- 
device, base_ds, train_idx, val_idx = set_up(Config["seed"], DATA_DIR)
_, val_loader = make_loaders(base_ds, train_idx, val_idx,
                             seed=Config["seed"], batch_size=Config["batch_size"],
                             num_workers=Config["num_workers"])

classes = Config["classes"]
num_classes = Config["num_classes"]

# ---- LOAD & REMAP MODELS ---- #

# -- remapping of EfficientNetB0 and MobileNetV3-sm --
# they use 'classifier' as the head
# resnet usese 'fc'

models = {}
for arch, wt_filepath in MODELS.items():
    m = get_pretrained_model(arch=arch)
    state = torch.load(wt_filepath, map_location=device)

    # Fix key mismatch for EfficientNet/MobileNet (if trained without Dropout wrapper)
    if arch == "efficientnet_b0":
        fixed = {}
        for k, v in state.items():
            if k == "classifier.1.weight":
                k = "classifier.1.1.weight"
            elif k == "classifier.1.bias":
                k = "classifier.1.1.bias"
            fixed[k] = v
        state = fixed

    elif arch == "mobilenetv3_small":
        fixed = {}
        for k, v in state.items():
            if k == "classifier.3.weight":
                k = "classifier.3.1.weight"
            elif k == "classifier.3.bias":
                k = "classifier.3.1.bias"
            fixed[k] = v
        state = fixed

    m.load_state_dict(state)
    m.to(device).eval()
    models[arch] = m
    print(f"--> {arch} is loaded")

# Run inference — collect preds AND probs for AUC
all_preds = {name: [] for name in models}
all_probs = {name: [] for name in models}
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(val_loader, desc="Running inference"):
        images = images.to(device)
        all_labels.append(labels.numpy())
        for name, model in models.items():
            logits = model(images)
            all_preds[name].append(logits.argmax(1).cpu().numpy())
            all_probs[name].append(torch.softmax(logits, dim=1).cpu().numpy())

all_labels = np.concatenate(all_labels)
for name in models:
    all_preds[name] = np.concatenate(all_preds[name])
    all_probs[name] = np.concatenate(all_probs[name])
    
# -----------------------------------------------------------------------
# MAPPING HS + BL classes

HnB_classes = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
my_classes = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']  # ImageFolder alphabetizing

# FOR PROBS: reorder cols so that col i is prob of my_classes[i]
probs_uni = [HnB_classes.index(c) for c in my_classes]  # [3,2,4,5,0,1,7,6]

# FOR PREDS: map HnB idx -> my idx
pred_map = np.array([my_classes.index(c) for c in HnB_classes])

for name in models: 
    if name in ("efficientnet_b0", "mobilenetv3_small"):
        all_probs[name] = all_probs[name][:, probs_uni]
        all_preds[name] = pred_map[all_preds[name]]
    
# -----------------------------------------------------------------------

# Hard vote (BACC & F1)

pred_matrix = np.stack(list(all_preds.values()), axis=1)
ensemble_preds = stats.mode(pred_matrix, axis=1, keepdims=False).mode

# -----------------------------------------------------------------------

# Weighted averaging for probs (AUC & MEL sens)
# EfficientNet & ResNet are stronger models

model_weights = {
    "resnet50": 0.36, 
    "efficientnet_b0": 0.4, 
    "mobilenetv3_small": 0.24
}

# --- filter for only active models --- #
active_wts = {name: model_weights[name] for name in models}
total = sum(active_wts.values())

# --- normalization --- #
# changed: need to adapt to two models or three --> 1
avg_probs = sum((w / total) * all_probs[name] for name, w in active_wts.items())
# -----Metrics helper f ------------------------------------------------

def compute_metrics(preds, probs, labels, label):
    bacc = balanced_accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")

    auc_per_class = []
    for c in range(num_classes):
        try:
            auc_per_class.append(roc_auc_score((labels == c).astype(int), probs[:, c]))
        except ValueError:
            auc_per_class.append(0.0)

    mel_idx = classes.index("MEL")
    fpr, tpr, _ = roc_curve((labels == mel_idx).astype(int), probs[:, mel_idx])
    spec = 1 - fpr
    idx = np.where(spec >= 0.95)[0]
    mel_sens = tpr[idx[-1]] if len(idx) > 0 else 0.0

    print(f"\n  {label}")
    print(f"    BACC: {bacc:.4f}   Macro F1: {f1:.4f}   MEL sens@95spec: {mel_sens:.4f}")
    for i, name in enumerate(classes):
        print(f"    AUC {name:>4s}: {auc_per_class[i]:.4f}")

    return {"bacc": bacc, "f1_macro": f1, "mel_sens_at_95spec": mel_sens,
            "auc_per_class": auc_per_class}

# --- Print + log ------------------------------------------------------

# Individual models
for name in models:
    compute_metrics(all_preds[name], all_probs[name], all_labels, name)

# Ensemble
m = compute_metrics(ensemble_preds, avg_probs, all_labels,
                    f"ENSEMBLE: {' + '.join(models.keys())}")

if USE_WANDB:
    log = {
        "ensemble/bacc": m["bacc"],
        "ensemble/f1_macro": m["f1_macro"],
        "ensemble/mel_sens_at_95spec": m["mel_sens_at_95spec"],
    }
    for i, name in enumerate(classes):
        log[f"ensemble/auc_{name}"] = m["auc_per_class"][i]
    wandb.log(log)
    wandb.finish()