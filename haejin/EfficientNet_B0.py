{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "060295aa-4eb6-44ce-a2e9-54f16a44fe21",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[34m\u001b[1mwandb\u001b[0m: \u001b[32m\u001b[41mERROR\u001b[0m Failed to detect the name of this notebook. You can set it manually with the WANDB_NOTEBOOK_NAME environment variable to enable code saving.\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: [wandb.login()] Loaded credentials for https://api.wandb.ai from /home/umw7eg/.netrc.\n",
      "\u001b[34m\u001b[1mwandb\u001b[0m: Currently logged in as: \u001b[33mumw7eg\u001b[0m (\u001b[33mvrb9e-university-of-virginia-school-of-data-science\u001b[0m) to \u001b[32mhttps://api.wandb.ai\u001b[0m. Use \u001b[1m`wandb login --relogin`\u001b[0m to force relogin\n"
     ]
    },
    {
     "data": {
      "text/html": [],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "Tracking run with wandb version 0.25.1"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "Run data is saved locally in <code>/sfs/gpfs/tardis/home/umw7eg/isic2019-skin-lesion-classification/haejin/wandb/run-20260415_232954-vx07c5kq</code>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "Syncing run <strong><a href='https://wandb.ai/vrb9e-university-of-virginia-school-of-data-science/isic2019-skin-lesion-classification-haejin/runs/vx07c5kq' target=\"_blank\">laced-totem-1</a></strong> to <a href='https://wandb.ai/vrb9e-university-of-virginia-school-of-data-science/isic2019-skin-lesion-classification-haejin' target=\"_blank\">Weights & Biases</a> (<a href='https://wandb.me/developer-guide' target=\"_blank\">docs</a>)<br>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       " View project at <a href='https://wandb.ai/vrb9e-university-of-virginia-school-of-data-science/isic2019-skin-lesion-classification-haejin' target=\"_blank\">https://wandb.ai/vrb9e-university-of-virginia-school-of-data-science/isic2019-skin-lesion-classification-haejin</a>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       " View run at <a href='https://wandb.ai/vrb9e-university-of-virginia-school-of-data-science/isic2019-skin-lesion-classification-haejin/runs/vx07c5kq' target=\"_blank\">https://wandb.ai/vrb9e-university-of-virginia-school-of-data-science/isic2019-skin-lesion-classification-haejin/runs/vx07c5kq</a>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Traceback (most recent call last):\n",
      "  File \"/home/umw7eg/.local/lib/python3.12/site-packages/wandb/sdk/wandb_config.py\", line 166, in __getattr__\n",
      "    return self.__getitem__(key)\n",
      "           ^^^^^^^^^^^^^^^^^^^^^\n",
      "  File \"/home/umw7eg/.local/lib/python3.12/site-packages/wandb/sdk/wandb_config.py\", line 131, in __getitem__\n",
      "    return self._items[key]\n",
      "           ~~~~~~~~~~~^^^^^\n",
      "KeyError: 'loss_fn'\n",
      "\n",
      "The above exception was the direct cause of the following exception:\n",
      "\n",
      "Traceback (most recent call last):\n",
      "  File \"/tmp/ipykernel_635592/2790357207.py\", line 185, in train\n",
      "    Config[\"loss_fn\"] = cfg.loss_fn\n",
      "                        ^^^^^^^^^^^\n",
      "  File \"/home/umw7eg/.local/lib/python3.12/site-packages/wandb/sdk/wandb_config.py\", line 168, in __getattr__\n",
      "    raise AttributeError(\n",
      "AttributeError: <class 'wandb.sdk.wandb_config.Config'> object has no attribute 'loss_fn'\n"
     ]
    },
    {
     "data": {
      "text/html": [],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       " View run <strong style=\"color:#cdcd00\">laced-totem-1</strong> at: <a href='https://wandb.ai/vrb9e-university-of-virginia-school-of-data-science/isic2019-skin-lesion-classification-haejin/runs/vx07c5kq' target=\"_blank\">https://wandb.ai/vrb9e-university-of-virginia-school-of-data-science/isic2019-skin-lesion-classification-haejin/runs/vx07c5kq</a><br> View project at: <a href='https://wandb.ai/vrb9e-university-of-virginia-school-of-data-science/isic2019-skin-lesion-classification-haejin' target=\"_blank\">https://wandb.ai/vrb9e-university-of-virginia-school-of-data-science/isic2019-skin-lesion-classification-haejin</a><br>Synced 4 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "Find logs at: <code>./wandb/run-20260415_232954-vx07c5kq/logs</code>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "ename": "AttributeError",
     "evalue": "<class 'wandb.sdk.wandb_config.Config'> object has no attribute 'loss_fn'",
     "output_type": "error",
     "traceback": [
      "\u001b[31m---------------------------------------------------------------------------\u001b[39m",
      "\u001b[31mKeyError\u001b[39m                                  Traceback (most recent call last)",
      "\u001b[36mFile \u001b[39m\u001b[32m~/.local/lib/python3.12/site-packages/wandb/sdk/wandb_config.py:166\u001b[39m, in \u001b[36mConfig.__getattr__\u001b[39m\u001b[34m(self, key)\u001b[39m\n\u001b[32m    165\u001b[39m \u001b[38;5;28;01mtry\u001b[39;00m:\n\u001b[32m--> \u001b[39m\u001b[32m166\u001b[39m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28;43mself\u001b[39;49m\u001b[43m.\u001b[49m\u001b[34;43m__getitem__\u001b[39;49m\u001b[43m(\u001b[49m\u001b[43mkey\u001b[49m\u001b[43m)\u001b[49m\n\u001b[32m    167\u001b[39m \u001b[38;5;28;01mexcept\u001b[39;00m \u001b[38;5;167;01mKeyError\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m ke:\n",
      "\u001b[36mFile \u001b[39m\u001b[32m~/.local/lib/python3.12/site-packages/wandb/sdk/wandb_config.py:131\u001b[39m, in \u001b[36mConfig.__getitem__\u001b[39m\u001b[34m(self, key)\u001b[39m\n\u001b[32m    130\u001b[39m \u001b[38;5;28;01mdef\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[34m__getitem__\u001b[39m(\u001b[38;5;28mself\u001b[39m, key):\n\u001b[32m--> \u001b[39m\u001b[32m131\u001b[39m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28;43mself\u001b[39;49m\u001b[43m.\u001b[49m\u001b[43m_items\u001b[49m\u001b[43m[\u001b[49m\u001b[43mkey\u001b[49m\u001b[43m]\u001b[49m\n",
      "\u001b[31mKeyError\u001b[39m: 'loss_fn'",
      "\nThe above exception was the direct cause of the following exception:\n",
      "\u001b[31mAttributeError\u001b[39m                            Traceback (most recent call last)",
      "\u001b[36mCell\u001b[39m\u001b[36m \u001b[39m\u001b[32mIn[1]\u001b[39m\u001b[32m, line 247\u001b[39m\n\u001b[32m    243\u001b[39m             scheduler.step()\n\u001b[32m    246\u001b[39m \u001b[38;5;28;01mif\u001b[39;00m \u001b[34m__name__\u001b[39m == \u001b[33m\"\u001b[39m\u001b[33m__main__\u001b[39m\u001b[33m\"\u001b[39m:\n\u001b[32m--> \u001b[39m\u001b[32m247\u001b[39m     \u001b[43mtrain\u001b[49m\u001b[43m(\u001b[49m\u001b[43m)\u001b[49m\n",
      "\u001b[36mCell\u001b[39m\u001b[36m \u001b[39m\u001b[32mIn[1]\u001b[39m\u001b[32m, line 185\u001b[39m, in \u001b[36mtrain\u001b[39m\u001b[34m()\u001b[39m\n\u001b[32m    182\u001b[39m \u001b[38;5;28;01mwith\u001b[39;00m wandb.init(settings=wandb.Settings(console=\u001b[33m\"\u001b[39m\u001b[33moff\u001b[39m\u001b[33m\"\u001b[39m)) \u001b[38;5;28;01mas\u001b[39;00m run:\n\u001b[32m    184\u001b[39m     cfg = wandb.config\n\u001b[32m--> \u001b[39m\u001b[32m185\u001b[39m     Config[\u001b[33m\"\u001b[39m\u001b[33mloss_fn\u001b[39m\u001b[33m\"\u001b[39m] = \u001b[43mcfg\u001b[49m\u001b[43m.\u001b[49m\u001b[43mloss_fn\u001b[49m\n\u001b[32m    186\u001b[39m     Config[\u001b[33m\"\u001b[39m\u001b[33mfreeze_bb\u001b[39m\u001b[33m\"\u001b[39m] = cfg.freeze_bb\n\u001b[32m    188\u001b[39m     set_seed(Config[\u001b[33m\"\u001b[39m\u001b[33mseed\u001b[39m\u001b[33m\"\u001b[39m])\n",
      "\u001b[36mFile \u001b[39m\u001b[32m~/.local/lib/python3.12/site-packages/wandb/sdk/wandb_config.py:168\u001b[39m, in \u001b[36mConfig.__getattr__\u001b[39m\u001b[34m(self, key)\u001b[39m\n\u001b[32m    166\u001b[39m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m.\u001b[34m__getitem__\u001b[39m(key)\n\u001b[32m    167\u001b[39m \u001b[38;5;28;01mexcept\u001b[39;00m \u001b[38;5;167;01mKeyError\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m ke:\n\u001b[32m--> \u001b[39m\u001b[32m168\u001b[39m     \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mAttributeError\u001b[39;00m(\n\u001b[32m    169\u001b[39m         \u001b[33mf\u001b[39m\u001b[33m\"\u001b[39m\u001b[38;5;132;01m{\u001b[39;00m\u001b[38;5;28mself\u001b[39m.\u001b[34m__class__\u001b[39m\u001b[38;5;132;01m!r}\u001b[39;00m\u001b[33m object has no attribute \u001b[39m\u001b[38;5;132;01m{\u001b[39;00mkey\u001b[38;5;132;01m!r}\u001b[39;00m\u001b[33m\"\u001b[39m\n\u001b[32m    170\u001b[39m     ) \u001b[38;5;28;01mfrom\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[34;01mke\u001b[39;00m\n",
      "\u001b[31mAttributeError\u001b[39m: <class 'wandb.sdk.wandb_config.Config'> object has no attribute 'loss_fn'"
     ]
    }
   ],
   "source": [
    "import os, time, random, warnings\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "from PIL import Image\n",
    "\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F\n",
    "from torch.utils.data import Dataset, DataLoader\n",
    "from torch.optim.lr_scheduler import CosineAnnealingLR\n",
    "from torchvision import transforms\n",
    "\n",
    "import timm\n",
    "import wandb\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import (\n",
    "    accuracy_score, balanced_accuracy_score, \n",
    "    confusion_matrix, roc_auc_score,\n",
    "    precision_recall_fscore_support)\n",
    "\n",
    "warnings.filterwarnings(\"ignore\")\n",
    "\n",
    "# =========================\n",
    "# Config\n",
    "# =========================\n",
    "Config = {\n",
    "    \"architecture\": \"efficientnet_b0\",\n",
    "    \"pretrained\": True,\n",
    "    \"freeze_bb\": False,\n",
    "    \"loss_fn\": \"weighted_ce\",\n",
    "    \"img_size\": 224,\n",
    "    \"batch_size\": 32,\n",
    "    \"epochs\": 30,\n",
    "    \"lr\": 1e-4,\n",
    "    \"val_split\": 0.20,\n",
    "    \"seed\": 42,\n",
    "    \"classes\": [\"MEL\", \"NV\", \"BCC\", \"AK\", \"BKL\", \"DF\", \"VASC\", \"SCC\"],\n",
    "}\n",
    "\n",
    "Config[\"num_classes\"] = len(Config[\"classes\"])\n",
    "\n",
    "DATA_ROOT = \"/scratch/umw7eg/isic2019\"\n",
    "TRAIN_DIR = os.path.join(DATA_ROOT, \"ISIC_2019_Training_Input\")\n",
    "TRAIN_CSV = os.path.join(DATA_ROOT, \"train_gt.csv\")\n",
    "\n",
    "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "\n",
    "\n",
    "# =========================\n",
    "# Seed\n",
    "# =========================\n",
    "def set_seed(seed):\n",
    "    random.seed(seed)\n",
    "    np.random.seed(seed)\n",
    "    torch.manual_seed(seed)\n",
    "    torch.cuda.manual_seed_all(seed)\n",
    "\n",
    "\n",
    "# =========================\n",
    "# Dataset\n",
    "# =========================\n",
    "class ISICDataset(Dataset):\n",
    "    def __init__(self, df, img_dir, transform=None):\n",
    "        self.df = df.reset_index(drop=True)\n",
    "        self.img_dir = img_dir\n",
    "        self.transform = transform\n",
    "\n",
    "    def __len__(self):\n",
    "        return len(self.df)\n",
    "\n",
    "    def __getitem__(self, idx):\n",
    "        row = self.df.iloc[idx]\n",
    "        img = Image.open(f\"{self.img_dir}/{row['image']}.jpg\").convert(\"RGB\")\n",
    "        label = int(row[\"label\"])\n",
    "\n",
    "        if self.transform:\n",
    "            img = self.transform(img)\n",
    "\n",
    "        return img, label\n",
    "\n",
    "\n",
    "def get_transforms(split):\n",
    "    if split == \"train\":\n",
    "        return transforms.Compose([\n",
    "            transforms.Resize((Config[\"img_size\"], Config[\"img_size\"])),\n",
    "            transforms.RandomHorizontalFlip(),\n",
    "            transforms.ToTensor(),\n",
    "            transforms.Normalize([0.485, 0.456, 0.406],\n",
    "                                 [0.229, 0.224, 0.225]),\n",
    "        ])\n",
    "    else:\n",
    "        return transforms.Compose([\n",
    "            transforms.Resize((Config[\"img_size\"], Config[\"img_size\"])),\n",
    "            transforms.ToTensor(),\n",
    "            transforms.Normalize([0.485, 0.456, 0.406],\n",
    "                                 [0.229, 0.224, 0.225]),\n",
    "        ])\n",
    "\n",
    "\n",
    "# =========================\n",
    "# Model\n",
    "# =========================\n",
    "def build_model():\n",
    "    model = timm.create_model(\n",
    "        \"efficientnet_b0\",\n",
    "        pretrained=Config[\"pretrained\"],\n",
    "        num_classes=Config[\"num_classes\"]\n",
    "    )\n",
    "    return model\n",
    "\n",
    "\n",
    "# =========================\n",
    "# Loss\n",
    "# =========================\n",
    "class FocalLoss(nn.Module):\n",
    "    def __init__(self, gamma=2, weight=None):\n",
    "        super().__init__()\n",
    "        self.gamma = gamma\n",
    "        self.weight = weight\n",
    "\n",
    "    def forward(self, inputs, targets):\n",
    "        ce = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')\n",
    "        pt = torch.exp(-ce)\n",
    "        return ((1 - pt) ** self.gamma * ce).mean()\n",
    "\n",
    "\n",
    "def get_loss_fn(name, class_weights):\n",
    "    if name == \"ce\":\n",
    "        return nn.CrossEntropyLoss()\n",
    "    elif name == \"weighted_ce\":\n",
    "        return nn.CrossEntropyLoss(weight=class_weights)\n",
    "    elif name == \"focal\":\n",
    "        return FocalLoss(weight=class_weights)\n",
    "    else:\n",
    "        raise ValueError(\"Invalid loss_fn\")\n",
    "\n",
    "\n",
    "# =========================\n",
    "# Train / Eval\n",
    "# =========================\n",
    "def train_one_epoch(model, loader, optimizer, criterion):\n",
    "    model.train()\n",
    "    total_loss = 0\n",
    "\n",
    "    for x, y in loader:\n",
    "        x, y = x.to(device), y.to(device)\n",
    "\n",
    "        optimizer.zero_grad()\n",
    "        out = model(x)\n",
    "        loss = criterion(out, y)\n",
    "        loss.backward()\n",
    "        optimizer.step()\n",
    "\n",
    "        total_loss += loss.item()\n",
    "\n",
    "    return total_loss / len(loader)\n",
    "\n",
    "\n",
    "def evaluate(model, loader):\n",
    "    model.eval()\n",
    "    preds, labels = [], []\n",
    "\n",
    "    with torch.no_grad():\n",
    "        for x, y in loader:\n",
    "            x, y = x.to(device), y.to(device)\n",
    "            out = model(x)\n",
    "\n",
    "            preds.extend(out.argmax(1).cpu().numpy())\n",
    "            labels.extend(y.cpu().numpy())\n",
    "\n",
    "    preds = np.array(preds)\n",
    "    labels = np.array(labels)\n",
    "\n",
    "    acc = (preds == labels).mean()\n",
    "    bacc = balanced_accuracy_score(labels, preds)\n",
    "\n",
    "    return acc, bacc\n",
    "\n",
    "\n",
    "# =========================\n",
    "# Main train()\n",
    "# =========================\n",
    "def train():\n",
    "    with wandb.init(settings=wandb.Settings(console=\"off\")) as run:\n",
    "\n",
    "        cfg = wandb.config\n",
    "        Config[\"loss_fn\"] = cfg.loss_fn\n",
    "        Config[\"freeze_bb\"] = cfg.freeze_bb\n",
    "\n",
    "        set_seed(Config[\"seed\"])\n",
    "\n",
    "        # Data\n",
    "        df = pd.read_csv(TRAIN_CSV)\n",
    "        df[\"label\"] = df[Config[\"classes\"]].values.argmax(axis=1)\n",
    "\n",
    "        train_df, val_df = train_test_split(\n",
    "            df,\n",
    "            test_size=Config[\"val_split\"],\n",
    "            stratify=df[\"label\"],\n",
    "            random_state=Config[\"seed\"]\n",
    "        )\n",
    "\n",
    "        train_loader = DataLoader(\n",
    "            ISICDataset(train_df, TRAIN_DIR, get_transforms(\"train\")),\n",
    "            batch_size=Config[\"batch_size\"], shuffle=True, num_workers=4\n",
    "        )\n",
    "\n",
    "        val_loader = DataLoader(\n",
    "            ISICDataset(val_df, TRAIN_DIR, get_transforms(\"val\")),\n",
    "            batch_size=Config[\"batch_size\"], shuffle=False, num_workers=4\n",
    "        )\n",
    "\n",
    "        # Class weights\n",
    "        counts = train_df[\"label\"].value_counts().sort_index()\n",
    "        weights = (1.0 / counts).values\n",
    "        weights = weights / weights.sum() * len(weights)\n",
    "        class_weights = torch.tensor(weights, dtype=torch.float32).to(device)\n",
    "\n",
    "        # Model\n",
    "        model = build_model().to(device)\n",
    "\n",
    "        if Config[\"freeze_bb\"]:\n",
    "            for param in model.parameters():\n",
    "                param.requires_grad = False\n",
    "            for param in model.get_classifier().parameters():\n",
    "                param.requires_grad = True\n",
    "\n",
    "        # Loss / Optim\n",
    "        criterion = get_loss_fn(Config[\"loss_fn\"], class_weights)\n",
    "        optimizer = torch.optim.AdamW(model.parameters(), lr=Config[\"lr\"])\n",
    "        scheduler = CosineAnnealingLR(optimizer, T_max=Config[\"epochs\"])\n",
    "\n",
    "        # Training loop\n",
    "        for epoch in range(Config[\"epochs\"]):\n",
    "            train_loss = train_one_epoch(model, train_loader, optimizer, criterion)\n",
    "            acc, bacc = evaluate(model, val_loader)\n",
    "\n",
    "            wandb.log({\n",
    "                \"epoch\": epoch,\n",
    "                \"train_loss\": train_loss,\n",
    "                \"val_acc\": acc,\n",
    "                \"val_bacc\": bacc,\n",
    "            })\n",
    "\n",
    "            scheduler.step()\n",
    "\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    train()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "84f2beb0-1afb-495f-a921-b138e7ae0e16",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
