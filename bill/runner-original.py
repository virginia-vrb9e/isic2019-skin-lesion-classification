# ------------------------------
# IMPORT PACKAGES
# ------------------------------

import os
import zipfile
import urllib.request
import random
import time

import numpy as np
import pandas as pd
from PIL import Image

import matplotlib.pyplot as plt
from importlib import import_module

import torch
import torch.nn as nn
import torch.optim as optim

# fix broken cuDNN convolution engine
torch.backends.cudnn.enabled = False

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from torchvision import transforms, models

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, accuracy_score

# ------------------------------
# SEED CONTROL
# ------------------------------

def set_seeds_to(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ------------------------------
# DOWNLOAD ISIC DATASET
# ------------------------------

def download_dataset():

    os.makedirs("isic2019", exist_ok=True)

    files = {
        "train_images": "https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Training_Input.zip",
        "train_metadata": "https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Training_Metadata.csv",
        "train_gt": "https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Training_GroundTruth.csv",
        "test_images": "https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Test_Input.zip",
        "test_metadata": "https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Test_Metadata.csv",
        "test_gt": "https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Test_GroundTruth.csv",
    }

    for name, url in files.items():

        fname = os.path.join("isic2019", url.split("/")[-1])

        # establish dataset directory if it doesn't exist
        if not os.path.exists(fname):

            print(f"Downloading {name}...")
            urllib.request.urlretrieve(url, fname)

            if fname.endswith(".zip"):

                print(f"Extracting {fname}...")
                with zipfile.ZipFile(fname, 'r') as z:
                    z.extractall("isic2019")

    print("Dataset ready.")

# ------------------------------
# DATASET CLASS
# ------------------------------

class ISIC2019Dataset(Dataset):

    # Code	Meaning
    # MEL	Melanoma
    # NV	Nevus
    # BCC	Basal Cell Carcinoma
    # BKL	Benign Keratosis
    # AK	Actinic Keratosis
    # SCC	Squamous Cell Carcinoma
    # VASC	Vascular lesion
    # DF	Dermatofibroma
    # ###########################
    # UNK   *** IS EXCLUDED *** #
    # ###########################
   
    CLASS_NAMES = ["MEL", "NV", "BCC", "BKL", "AK", "SCC", "VASC", "DF"]

    def __init__(self, df, img_dir, transform=None):
        # pandas dataframe containing labels and image IDs
        self.df = df
        # folder where images are stored
        self.img_dir = img_dir
        # image preprocessing (resize, normalization, augmentation)
        self.transform = transform

        # convert string labels to numeric labels
        # Label	Index
        # MEL	0
        # NV	1
        # BCC	2
        # BKL	3
        # AK	4
        # SCC	5
        # VASC	6
        # DF	7
        self.label_map = {c:i for i,c in enumerate(self.CLASS_NAMES)}

        # extract columns from dataframe
        self.labels = df["label"].values
        self.image_ids = df["image"].values  # or the column in your CSV that has image filenames

    # number of samples
    def __len__(self):
        return len(self.df)

    # load one training example - idx is sample index
    def __getitem__(self, idx):

        # get row; extract image ID
        row = self.df.iloc[idx]
        img_id = row["image"]
        
        # convert label to number
        # label_map = {c:i for i,c in enumerate(self.CLASS_NAMES)}
        label = self.label_map[row["label"]]

        # construct image path; load image
        # convert("RGB") ensures: 3 color channels, consistent input format
        path = os.path.join(self.img_dir, img_id + ".jpg")
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label
    
# ------------------------------
# TRANSFORMS
# ------------------------------

def make_transforms(image_size=224):

    # standardize pixel values
    # for For each RGB channel: xnorm​=(x−mean)/std
    # The numbers are ImageNet dataset statistics, used because many pretrained models were trained on ImageNet.
    normalize = transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )

    # Compose chains multiple transforms in order
    #   randomly: crops a region of the image
    #   resize to image_size
    #   crop (scale) to 70%–100% of original area
    #   randomly flip image left ↔ right
    #   randomly flip top ↔ bottom.Rotates image randomly between: -30 to +30 degree
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.7,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),

        # randomly changes color properties
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.02
        ),

        transforms.ToTensor(),
        normalize
    ])

    # Resize image to 224 × 224
    val_tf = transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(),
        normalize
    ])

    return train_tf, val_tf

# ------------------------------
# DATASET VERIFICATION
# ------------------------------

def report_distribution(df, title):

    print("\n",title)
    print("-"*40)

    # df["label"] selects the label column from the DataFrame
    # .value_counts() count how many times each label appears
    # normalize=True return proportions
    counts = df["label"].value_counts(normalize=True)*100

    # loop through all known classes
    # retrieves the percentage for class c
    # if the class is missing in dataframe, return 0
    for c in ISIC2019Dataset.CLASS_NAMES:
        pct = counts.get(c,0)
        print(f"{c:5s}: {pct:.2f}%")


def build_dataframe(meta_path, gt_path):
    meta = pd.read_csv(meta_path)
    gt = pd.read_csv(gt_path)

    # Merge on image column
    df = meta.merge(gt, on="image")

    # Convert one-hot to class label
    class_cols = ["MEL","NV","BCC","BKL","AK","SCC","VASC","DF"]
    df["label"] = df[class_cols].idxmax(axis=1)

    # Keep only columns that exist
    cols = ["image", "label"]

    if "patient_id" in df.columns:
        cols.append("patient_id")
    if "lesion_id" in df.columns:
        cols.append("lesion_id")

    df = df[cols]

    return df

# ------------------------------
# STRATIFIED PATIENT-GROUPED SPLIT (80/20)
# ------------------------------

def stratified_patient_split(df, seed=42, val_ratio=0.2):

    # Determine grouping column - related samples stay together
    # If patient_id exists → group by patient
    # Else if lesion_id exists → group by lesion
    # Otherwise → each image is its own group
    if "patient_id" in df.columns:
        groups = df["patient_id"].fillna("missing").astype(str)
    elif "lesion_id" in df.columns:
        groups = df["lesion_id"].fillna("missing").astype(str)
    else:
        groups = df["image"]

    # class labels used for stratification
    y = df["label"]

    # Create StratifiedGroupKFold splitter (from sklearn)
    # val_ratio = 0.2...asciiso 1 / 0.2 = 5 (folds)
    sgkf = StratifiedGroupKFold(
        n_splits=int(1/val_ratio),
        shuffle=True,
        random_state=seed
    )

    # returns indices for train and validation
    train_idx, val_idx = next(sgkf.split(df, y, groups))

    # .iloc[] selects rows by index
    # .reset_index(drop=True) cleans the index
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    return train_df, val_df

# ------------------------------
# CLASS WEIGHTS
# ------------------------------

def compute_class_weights(labels, num_classes=8):

    # convert class names into numeric indices
    label_map = {c:i for i,c in enumerate(ISIC2019Dataset.CLASS_NAMES)}
    # convert string labels to numbers
    labels = np.array([label_map[l] for l in labels])

    # count how many samples exist in each class
    counts = np.bincount(labels, minlength=num_classes)
    # prevent zerodivide
    counts[counts == 0] = 1

    N = counts.sum()
    # inverse frequency weighting formula
    weights = N/(num_classes*counts)

    return torch.tensor(weights, dtype=torch.float)
    
# ------------------------------
# DATALOADERS WITH OVERSAMPLING
# ------------------------------

def make_loaders(train_ds, val_ds, batch_size=32, num_workers=1):   # changed from 4

    # convert class names into integer IDs
    label_map = {c:i for i,c in enumerate(ISIC2019Dataset.CLASS_NAMES)}
    # convert dataset labels to numeric
    labels = np.array([label_map[l] for l in train_ds.labels])

    class_counts = np.bincount(labels, minlength=len(ISIC2019Dataset.CLASS_NAMES))
    # prevent zerodivide
    class_counts[class_counts == 0] = 1
    # ADDRESS CLASS IMBALANCE - rare classes get much larger weight.
    weights = 1.0 / class_counts

    # ADDRESS CLASS IMBALANCE - rare-class samples receive higher probability
    # rare classes → sampled more often, common classes → sampled less often
    sample_weights = weights[labels]
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # num_workers = parallel loading threads
    # pin_memory=True - faster GPU transfer
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader

# ------------------------------
# EVALUATION METRICS
# ------------------------------

def evaluate(model, loader, device):

    # switches model to inference mode
    model.eval()

    # lists to store results
    all_preds=[]
    all_labels=[]
    all_probs=[]

    # do not compute gradients - reduces memory usage and speeds up evaluation
    with torch.no_grad():

        # x = input images/features; y = true labels
        for x,y in loader:

            # use GPU if device = cuda
            x=x.to(device)
            y=y.to(device)

            # raw scores before probabilities
            logits=model(x)

            # convert logits to class probabilities (dim=1 is across channels)
            probs=torch.softmax(logits,dim=1)

            # selecrt highest probability class
            preds=torch.argmax(probs,dim=1)

            # move tensors to CPU; convert to NumPy arrays; append to the lists
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # correct_predictions / total_samples
    acc = accuracy_score(all_labels, all_preds)

    # average(recall_per_class)
    bacc = balanced_accuracy_score(all_labels,all_preds)

    # AUC measures ranking quality of predictions.
    # multi_class="ovr" is One-vs-Rest strategy
    # For each class: class_i vs all_other_classes
    auc = roc_auc_score(
        all_labels,
        np.array(all_probs),
        multi_class="ovr"
    )

    # AUC for each class individually.
    auc_per_class = roc_auc_score(
        all_labels,
        np.array(all_probs),
        multi_class="ovr",
        average=None
    )

    return acc, bacc, auc, auc_per_class

# ------------------------------
# TRAINING LOOP
# ------------------------------

def train(model, train_loader, val_loader, criterion, optimizer, device, epochs=10):

    history = {
        "val_loss": [],
        "val_acc": [],
        "val_bacc": [],
        "val_auc": []
    }

    for epoch in range(epochs):

        model.train()
        total_loss = 0

        for x, y in train_loader:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            out = model(x)
            loss = criterion(out, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # ---- Validation Loss ----
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        acc, bacc, auc, auc_pc = evaluate(model, val_loader, device)

        history["val_loss"].append(val_loss)
        history["val_acc"].append(acc)
        history["val_bacc"].append(bacc)
        history["val_auc"].append(auc)

        print(
            f"Epoch {epoch+1} | Train Loss {total_loss:.3f} | "
            f"Val Loss {val_loss:.4f} | "
            f"ACC {acc:.4f} | BACC {bacc:.4f} | AUC {auc:.4f}"
        )

        print("Per-class AUC:")
        for i, c in enumerate(ISIC2019Dataset.CLASS_NAMES):
            print(f"{c}: {auc_pc[i]:.4f}")

    return history

def count_parameters(model):
    """Count total and trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_model_size_mb(model):
    """Calculate model size in MB (assuming float32 weights)."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / (1024 ** 2)
    return size_all_mb


def measure_inference_time(model, input_shape=(1, 3, 224, 224), num_runs=100):
    """Measure average inference time in milliseconds."""
    model.eval()
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_shape).to(device)

    # Warm up (important for accurate timing)
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)

    # Synchronize if using CUDA
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Time the inference
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.time()
    avg_time = (end_time - start_time) / num_runs * 1000.0  # Convert to ms
    return avg_time


def estimate_flops(
    model: nn.Module,
    input_shape: tuple[int, int, int, int] = (1, 3, 224, 224),
    device: torch.device | None = None,
) -> int:
    """
    Estimate FLOPs (multiply-adds counted as 2 FLOPs) for a single forward pass.

    Counts:
      - nn.Conv2d: 2 * N * Hout * Wout * Cout * (Cin/groups) * Kh * Kw
      - nn.Linear: 2 * N * in_features * out_features

    Ignores (0 FLOPs):
      - activations, pooling, batchnorm, adds, etc. (usually small vs convs)
    """
    model_was_training = model.training
    model.eval()

    # Pick device
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    flops_total = 0
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def conv_hook(m: nn.Conv2d, inp, out):
        nonlocal flops_total
        # inp[0]: (N, Cin, Hin, Win)
        x = inp[0]
        N = x.shape[0]
        # out: (N, Cout, Hout, Wout)
        Hout, Wout = out.shape[-2], out.shape[-1]

        Cin = m.in_channels
        Cout = m.out_channels
        Kh, Kw = m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size, m.kernel_size)
        groups = m.groups

        # Multiply-adds per output element:
        # (Cin/groups) * Kh * Kw multiplies + same number adds => *2
        flops_per_out_elem = 2 * (Cin // groups) * Kh * Kw

        out_elems = N * Cout * Hout * Wout
        flops = out_elems * flops_per_out_elem

        # Optional: count bias adds (1 add per output element)
        if m.bias is not None:
            flops += out_elems

        flops_total += int(flops)

    def linear_hook(m: nn.Linear, inp, out):
        nonlocal flops_total
        x = inp[0]
        # x can be (N, in_features) or (..., in_features); flatten leading dims into batch
        in_features = m.in_features
        out_features = m.out_features
        batch = int(x.numel() // in_features)

        flops = 2 * batch * in_features * out_features
        if m.bias is not None:
            flops += batch * out_features  # bias adds
        flops_total += int(flops)

    # Register hooks
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))

    # Run a dummy forward
    dummy = torch.randn(*input_shape, device=device)
    with torch.no_grad():
        _ = model.to(device)(dummy)

    # Cleanup
    for h in handles:
        h.remove()

    # Restore mode
    model.train(model_was_training)

    return flops_total


def plot_model_comparison(models_dict, device, input_shape = (1, 3, 224, 224)):
    """Compare efficiency metrics of different models."""
    _, axes = plt.subplots(2, 3, figsize=(18, 10))

    model_names = list(models_dict.keys())
    params_list = []
    size_list = []
    time_list = []
    flops_list = []

    for name, model in models_dict.items():
        total_params, _ = count_parameters(model)
        params_list.append(total_params / 1e6)  # Convert to millions
        size_list.append(get_model_size_mb(model))

        model = model.to(device)
        flops = estimate_flops(model, input_shape = input_shape, device = device)
        flops_list.append(flops / 1e9)

        time_list.append(measure_inference_time(model, input_shape = input_shape))

    # Plot 1: Parameters
    axes[0, 0].bar(model_names, params_list, color='blue', alpha=0.7)
    axes[0, 0].set_ylabel('Parameters (Millions)')
    axes[0, 0].set_title('Model Parameters Comparison')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, axis='y')

    # Plot 2: Model Size
    axes[0, 1].bar(model_names, size_list, color='green', alpha=0.7)
    axes[0, 1].set_ylabel('Size (MB)')
    axes[0, 1].set_title('Model Size on Disk')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, axis='y')

    # Plot 3: Inference Time
    axes[1, 0].bar(model_names, time_list, color='red', alpha=0.7)
    axes[1, 0].set_ylabel('Time (ms)')
    axes[1, 0].set_title('Inference Time (Lower is Better)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, axis='y')

    # Plot 4: FLOPs (replaces your scatter)
    axes[1, 1].bar(model_names, flops_list, alpha = 0.7)
    axes[1, 1].set_ylabel("FLOPs (GFLOPs)")
    axes[1, 1].set_title("Compute Cost per Forward (Lower is Better)")
    axes[1, 1].tick_params(axis = 'x', rotation = 45)
    axes[1, 1].grid(True, axis = 'y')

    # Plot 5: Efficiency Score
    axes[1, 2].scatter(params_list, time_list, s=100, alpha=0.7)
    for i, name in enumerate(model_names):
        axes[1, 2].annotate(name, (params_list[i], time_list[i]), fontsize=8, ha='right')
    axes[1, 2].set_xlabel('Parameters (Millions)')
    axes[1, 2].set_ylabel('Inference Time (ms)')
    axes[1, 2].set_title('Efficiency Trade-off (Lower-left is Better)')
    axes[1, 2].grid(True)

    axes[0, 2].axis("off")

    plt.tight_layout()
    plt.savefig("model_comparison.png")
    plt.show()


def test_assignment_extension(MobileNet, EfficientNet, ResNet, accelerator, train_loader, val_loader, num_classes):
    print("="*80)
    print("Testing Your Efficient Architecture Implementation")
    print("="*80)

    # Test your implementations
    try:
        # Test DepthwiseSeparableConv
        print("\n1. Testing DepthwiseSeparableConv...")
        dw_conv = MobileNet.DepthwiseSeparableConv(32, 64).to(accelerator)
        test_input = torch.randn(1, 32, 56, 56, device = accelerator)
        output_of_DepthwiseSeparableConv = dw_conv(test_input)
        numpy_array = output_of_DepthwiseSeparableConv.cpu().detach().numpy()
        # np.save("output_of_DepthwiseSeparableConv.npy", numpy_array)
        print(f"   Input shape: {test_input.shape}")
        print(f"   Output shape: {output_of_DepthwiseSeparableConv.shape}")
        print(f"   ✓ DepthwiseSeparableConv working!")
    except Exception as e:
        print(f"   ✗ Error in DepthwiseSeparableConv: {e}")
        numpy_array = None

    try:
        # Test InvertedResidual
        print("\n2. Testing InvertedResidual...")

        inv_res = MobileNet.InvertedResidual(32, 32, stride=1, expand_ratio=6, kernel_size=3
        ).to(accelerator)
        test_input = torch.randn(1, 32, 56, 56, device=accelerator)
        output = inv_res(test_input)

        print(f"   Input shape: {test_input.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   ✓ InvertedResidual working!")
    except Exception as e:
        print(f"   ✗ Error in InvertedResidual: {e}")

    try:
        # Test MobileNet
        print("\n3. Testing MobileNet...")
        mobilenet = MobileNet.MobileNet(num_classes=num_classes).to(accelerator)
        test_input = torch.randn(1, 3, 224, 224, device = accelerator)
        output = mobilenet(test_input)
        print(f"   Input shape: {test_input.shape}")
        print(f"   Output shape: {output.shape}")

        # Analyze model
        total_params, trainable_params = count_parameters(mobilenet)
        model_size = get_model_size_mb(mobilenet)
        inf_ms = measure_inference_time(mobilenet.to(accelerator), input_shape = (1, 3, 224, 224))
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Model size: {model_size:.2f} MB")
        print(f"    inference (ms): {inf_ms:.2f}")
        print(f"   ✓ MobileNet working!")

    except Exception as e:
        print(f"   ✗ Error in MobileNet: {e}")

    # Compare with other models
    print("\n" + "="*80)
    print("Model Comparison")
    print("="*80)

    try:
        # Create models for comparison
        models_to_compare = {
            "EfficientNet-B0": EfficientNet.EfficientNet(num_classes=num_classes),
            "MobileNetV3-Small": MobileNet.MobileNet(num_classes=num_classes),
            "ResNet-50": ResNet.ResNet50(num_classes=num_classes)
        }

        # Compare models
        for name, model in models_to_compare.items():
            model = model.to(accelerator)
            total_params, _ = count_parameters(model)
            size_mb = get_model_size_mb(model)
            flops = estimate_flops(model, input_shape = (1, 3, 224, 224), device = accelerator)
            print(f"{name:20s}: {total_params/1e6:.2f}M params, {size_mb:.2f} MB, {flops/1e9:.2f} GFLOPs")

        # Visualize comparison
        plot_model_comparison(models_to_compare, accelerator)

    except Exception as e:
        print(f"Error in model comparison: {e}")

    # Train your model (optional - takes time)
    print("\n" + "="*80)
    print("Training MobileNetV3-Small")
    print("="*80)

    try:
        model = MobileNet.MobileNet(num_classes=num_classes, dropout_prob=0.2).to(accelerator)

        criterion = nn.CrossEntropyLoss()
        # change lr from 0.001 to 3e-4  Internet suggestion
        optimizer = optim.Adam(model.parameters(), lr = 3e-4)

        history = train(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            accelerator,
            epochs=10
        )
        
        print(f"\nFinal Validation Accuracy: {history['val_acc'][-1]:.2f}%")
        if history['val_acc'][-1] > 80:
            print("✓ Great job! Your model achieves good accuracy while being efficient!")
        elif history['val_acc'][-1] > 70:
            print("✓ Good start! Try fine-tuning hyperparameters or training longer.")
        else:
            print("Keep working! Check your implementation and try different settings.")

    except Exception as e:
        print(f"Error during training: {e}")
        history = {}

    return numpy_array, history

# ------------------------------
# MAIN PIPELINE
# ------------------------------

def main():
    seed = 42
    batch_size = 32
    num_workers = 1
    num_classes = 8

    results = {}

    set_seeds_to(seed)

    # ------------------------------
    # DOWNLOAD DATASET
    # ------------------------------

    download_dataset()

    train_img_dir = "isic2019/ISIC_2019_Training_Input"
    train_meta = "isic2019/ISIC_2019_Training_Metadata.csv"
    train_gt   = "isic2019/ISIC_2019_Training_GroundTruth.csv"

    df = build_dataframe(train_meta, train_gt)

    # ------------------------------
    # TEST DATASET (not used for training)
    # ------------------------------

    test_meta = "isic2019/ISIC_2019_Test_Metadata.csv"
    test_gt   = "isic2019/ISIC_2019_Test_GroundTruth.csv"
    test_df = build_dataframe(test_meta, test_gt)

    print("\nDataset verification")
    print("-----------------------------")
    print("Total dataset:", len(df) + len(test_df))
    print("Training images:", len(df))
    print("Test images:", len(test_df))

    report_distribution(df, "Training distribution")
    report_distribution(test_df, "Test distribution")

    # ------------------------------
    # TRANSFORMS
    # ------------------------------

    train_transform, val_transform = make_transforms(224)

    # ------------------------------
    # TRAIN / VALIDATION SPLIT
    # ------------------------------

    train_df, val_df = stratified_patient_split(df, seed=seed)

    print("\nSplit verification")
    print("-----------------------------")
    print("Train size:", len(train_df))
    print("Val size:", len(val_df))

    report_distribution(train_df, "Train distribution")
    report_distribution(val_df, "Validation distribution")

    # ------------------------------
    # DATASETS
    # ------------------------------

    train_ds = ISIC2019Dataset(
        train_df,
        img_dir=train_img_dir,
        transform=train_transform
    )

    val_ds = ISIC2019Dataset(
        val_df,
        img_dir=train_img_dir,
        transform=val_transform
    )

    # ------------------------------
    # DATALOADERS
    # ------------------------------

    train_loader, val_loader = make_loaders(
        train_ds,
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers
    )

    # ------------------------------
    # DEVICE
    # ------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ------------------------------
    # IMPORT MODELS
    # ------------------------------

    EfficientNet = import_module("EfficientNet")
    MobileNet = import_module("MobileNet")
    ResNet = import_module("ResNet")

    # ============================================================
    # TRAIN EFFICIENTNET
    # ============================================================

    print("\n" + "="*80)
    print("Training EfficientNet-B0")
    print("="*80)

    try:

        efficientnet_model = EfficientNet.EfficientNet(
            num_classes=num_classes,
            dropout_prob=0.2
        ).to(device)

        total_params, trainable_params = count_parameters(efficientnet_model)
        model_size = get_model_size_mb(efficientnet_model)
        flops = estimate_flops(efficientnet_model, device=device)

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model size: {model_size:.2f} MB")
        print(f"FLOPs: {flops/1e9:.2f} GFLOPs")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(efficientnet_model.parameters(), lr = 3e-4)

        history_eff = train(
            efficientnet_model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            device,
            epochs=10
        )

        final_acc = history_eff["val_acc"][-1] * 100
        final_acc = history_eff["val_acc"][-1] * 100
        final_bacc = history_eff["val_bacc"][-1] * 100
        final_auc = history_eff["val_auc"][-1] * 100
        
        print("Final EfficientNet Accuracy:", final_acc)
        
        results["EfficientNet-B0"] = {
            "loss": history_eff["val_loss"][-1],
            "accuracy": final_acc,
            "balanced_accuracy": final_bacc,
            "auc": final_auc
        }

    except Exception as e:
        print("Error training EfficientNet:", e)

    # ============================================================
    # TEST MOBILENET EXTENSION
    # ============================================================

    print("\n" + "="*80)
    print("Testing MobileNet Extension")
    print("="*80)

    numpy_array, history = test_assignment_extension(
        MobileNet,
        EfficientNet,
        ResNet,
        device,
        train_loader,
        val_loader,
        num_classes
    )

    if "val_acc" in history and len(history["val_acc"]) > 0:
        results["MobileNetV3-Small"] = {
            "loss": history["val_loss"][-1],
            "accuracy": history["val_acc"][-1] * 100,
            "balanced_accuracy": history["val_bacc"][-1] * 100,
            "auc": history["val_auc"][-1] * 100
        }


    # ============================================================
    # TRAIN RESNET
    # ============================================================

    print("\n" + "="*80)
    print("Training ResNet-50")
    print("="*80)

    try:

        resnet_model = ResNet.ResNet50(num_classes=num_classes).to(device)

        total_params, trainable_params = count_parameters(resnet_model)
        model_size = get_model_size_mb(resnet_model)
        flops = estimate_flops(resnet_model, device=device)

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model size: {model_size:.2f} MB")
        print(f"FLOPs: {flops/1e9:.2f} GFLOPs")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(resnet_model.parameters(), lr = 3e-4)

        history_resnet = train(
            resnet_model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            device,
            epochs=10
        )

        final_acc = history_resnet["val_acc"][-1] * 100
        final_bacc = history_resnet["val_bacc"][-1] * 100
        final_auc = history_resnet["val_auc"][-1] * 100
        
        print("Final ResNet Accuracy:", final_acc)
        
        results["ResNet-50"] = {
            "loss": history_resnet["val_loss"][-1],
            "accuracy": final_acc,
            "balanced_accuracy": final_bacc,
            "auc": final_auc
        }
    
    except Exception as e:
        print("Error training ResNet:", e)

    print("\nExecution complete.")
    

    print("\nExecution complete.")

    # ------------------------------
    # RESULTS SUMMARY
    # ------------------------------

    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    for model_name, metrics in results.items():
         print(
            f"{model_name:20s} | "
            f"Loss: {metrics['loss']:.4f} | "
            f"ACC: {metrics['accuracy']:.2f}% | "
            f"BACC: {metrics['balanced_accuracy']:.2f}% | "
            f"AUC: {metrics['auc']:.2f}%"
        )

    best_model = max(results, key=lambda k: results[k]["accuracy"])
    best_accuracy = results[best_model]["accuracy"]
    
    print(f"\nBest Model: {best_model} with {best_accuracy:.2f}% validation accuracy")
    
    if best_accuracy > 94:
        print("\n🎉 Accuracy > 94%")
    else:
        print(f"\nCurrent best: {best_accuracy:.2f}% (Target: 94%)")
    
    print("\nFull results dictionary:")
    print(results)

    # ------------------------------
    # CREATE RESULTS TABLE
    # ------------------------------
    
    results_table = pd.DataFrame(results).T
    results_table.columns = ["Loss", "Accuracy (%)", "Balanced Accuracy (%)", "AUC (%)"]

    print("\nResults Table:")
    print(results_table)

    results_table.to_csv("model_results.csv")

    # Save Table as Image
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")
    
    table = ax.table(
        cellText=results_table.round(2).values,
        colLabels=results_table.columns,
        rowLabels=results_table.index,
        loc="center"
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    plt.title("Model Performance Comparison", pad=20)
    plt.savefig("model_results_table.png", bbox_inches="tight", dpi=300)
    plt.show()

    # bar chart comparing the metrics
    results_table.plot(kind="bar", figsize=(10,5))
    plt.title("Model Performance Comparison")
    plt.ylabel("Score (%)")
    plt.xticks(rotation=0)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig("model_performance_comparison.png", dpi=300)
    plt.show()
    
    return results
    
if __name__=="__main__":
    main()

'''
def main() -> None:
    batch_size = 32
    data_dir = "dataset/"
    num_classes = 18
    num_workers = 0
    seed = 42
    accelerator, base_ds, train_idx, val_idx = set_up(seed = seed, data_dir = data_dir)
    train_loader_150, val_loader_150 = make_loaders(
        base_ds, train_idx, val_idx, seed = seed, batch_size = batch_size,
        num_workers = num_workers, image_size = (150, 150), train_aug = "resize_flip"
    )
    VGGNet = import_module("VGGNet")
    NiN = import_module("NiN")
    GoogLeNet = import_module("GoogLeNet")
    ResNet = import_module("ResNet")
    transfer_learning = import_module("transfer_learning")
    test_VGGNet(VGGNet.VGGNet, accelerator, train_loader_150, val_loader_150, num_classes)
    test_NiN(NiN.NiN, accelerator, train_loader_150, val_loader_150, num_classes)
    test_GoogLeNet(GoogLeNet.GoogLeNet, accelerator, train_loader_150, val_loader_150, num_classes)
    test_ResNet(ResNet.ResNet50, accelerator, train_loader_150, val_loader_150, num_classes)
    test_transfer_learning(transfer_learning, accelerator, train_loader_150, val_loader_150, num_classes)
    train_loader_224, val_loader_224 = make_loaders(base_ds, train_idx, val_idx, seed = seed, batch_size = batch_size, num_workers = num_workers, image_size = 224, train_aug = "random_resized_crop")
    MobileNet = import_module("MobileNet")
    test_assignment_extension(MobileNet, accelerator, train_loader_224, val_loader_224, num_classes)


if __name__ == '__main__':
    main()
'''