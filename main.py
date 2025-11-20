#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CV8502 Failure Analysis — Starter (Classification)
- Baseline: DenseNet121 (multi-label) with BCEWithLogitsLoss
- Metrics: AUROC, AUPRC, Sens@95%Spec, F1
- Stress tests: noise/blur/jpeg/brightness-contrast (3 severities)
- Calibration: ECE, reliability diagram, temperature scaling; MC-Dropout, TTA variance
- Selective prediction: risk–coverage curve (micro, flattened over classes)
- Slicing: report metrics per group column (e.g., site/scanner)

Usage examples: see README.md
"""
import os, sys, json, time, math, argparse, random, warnings
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

try:
    from torchvision.models import densenet121, DenseNet121_Weights
    _HAS_WEIGHTS = True
except Exception:
    from torchvision.models import densenet121
    _HAS_WEIGHTS = False

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve, precision_recall_curve, accuracy_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Lazy import for attribution
try:
    from captum.attr import IntegratedGradients
    _HAS_CAPTUM = True
except Exception:
    _HAS_CAPTUM = False

# --------------------------
# Utils
# --------------------------

def seed_everything(seed: int = 1337):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_json(obj, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

# --------------------------
# Data
# --------------------------

class MultiLabelImageDataset(Dataset):
    def __init__(self, csv_path: str, img_root: str, labels: List[str],
                 split: Optional[str] = None, transform=None, group_col: Optional[str] = None):
        self.df = pd.read_csv(csv_path)
        self.img_root = img_root
        self.labels = labels
        if split is not None:
            if "split" not in self.df.columns:
                raise ValueError("CSV has no 'split' column; either add it or use --auto-split during training.")
            self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        # Basic checks
        for c in labels:
            if c not in self.df.columns:
                raise ValueError(f"Label column '{c}' not found in CSV.")
        self.transform = transform
        self.group_col = group_col if (group_col in self.df.columns) else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        if not os.path.isabs(img_path):
            img_path = os.path.join(self.img_root, img_path)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Could not read image at: {img_path}")
        if len(img.shape) == 2:  # grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        labels = row[self.labels].values.astype(np.float32)
        if self.transform is not None:
            out = self.transform(image=img)
            img_t = out["image"]
        else:
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
            img_t = torch.from_numpy(img.transpose(2,0,1)).float() / 255.0
        sample = {"image": img_t, "labels": torch.from_numpy(labels)}
        if self.group_col is not None:
            sample["group"] = row[self.group_col]
        return sample

def auto_split_csv(csv_path: str, train: float, val: float, test: float, seed: int = 1337) -> str:
    """Adds a 'split' column to CSV if absent, returns path to new CSV (in-place overwrite)."""
    df = pd.read_csv(csv_path).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df)
    n_train = int(n * train); n_val = int(n * val)
    splits = ["train"] * n_train + ["val"] * n_val + ["test"] * (n - n_train - n_val)
    df["split"] = splits
    df.to_csv(csv_path, index=False)
    return csv_path

def get_transforms(image_size: int = 224, split: str = "train"):
    if split == "train":
        return A.Compose([
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(image_size, image_size, border_mode=cv2.BORDER_CONSTANT),
            A.RandomResizedCrop(image_size, image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.8),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            A.GaussianBlur(blur_limit=(3,5), p=0.2),
            A.Normalize(),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(image_size, image_size, border_mode=cv2.BORDER_CONSTANT),
            A.CenterCrop(image_size, image_size),
            A.Normalize(),
            ToTensorV2(),
        ])

# --------------------------
# Model
# --------------------------

def build_model(num_classes: int, drop_rate: float = 0.2, pretrained: bool = True) -> nn.Module:
    if _HAS_WEIGHTS and pretrained:
        model = densenet121(weights=DenseNet121_Weights.DEFAULT, drop_rate=drop_rate)
    else:
        model = densenet121(drop_rate=drop_rate)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model

# --------------------------
# Training / Evaluation
# --------------------------

def compute_pos_weight(train_df: pd.DataFrame, labels: List[str]) -> torch.Tensor:
    # pos_weight = (N - P) / P  (balances BCE)
    pos_weights = []
    N = len(train_df)
    for c in labels:
        P = train_df[c].sum()
        pw = (N - P) / max(P, 1.0)
        pos_weights.append(pw)
    return torch.tensor(pos_weights, dtype=torch.float32)

@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, Optional[List]]:
    model.eval()
    logits_all, labels_all, groups = [], [], []
    for batch in tqdm(loader, desc="Predict", leave=False):
        images = batch["image"].to(device)
        labels = batch["labels"].numpy()
        out = model(images)
        logits_all.append(out.detach().cpu().numpy())
        labels_all.append(labels)
        if "group" in batch:
            groups.extend(list(batch["group"]))
    logits_all = np.concatenate(logits_all, axis=0)
    labels_all = np.concatenate(labels_all, axis=0)
    return logits_all, labels_all, groups if len(groups)>0 else None

def metrics_from_logits(logits: np.ndarray, labels: np.ndarray, label_names: List[str], threshold: float = 0.5) -> Dict:
    probs = 1 / (1 + np.exp(-logits))
    y_true = labels.copy()
    y_pred = (probs >= threshold).astype(int)
    res = {"per_class": {}, "macro": {}, "micro": {}}
    # Per-class metrics
    aurocs, auprcs, f1s, sens95 = [], [], [], []
    for j, name in enumerate(label_names):
        yj_true = y_true[:, j]; yj_prob = probs[:, j]; yj_pred = y_pred[:, j]
        m = {}
        try:
            m["auroc"] = float(roc_auc_score(yj_true, yj_prob))
        except Exception:
            m["auroc"] = float("nan")
        try:
            m["auprc"] = float(average_precision_score(yj_true, yj_prob))
        except Exception:
            m["auprc"] = float("nan")
        m["f1@0.5"] = float(f1_score(yj_true, yj_pred, zero_division=0))
        m["sens@95spec"] = float(sensitivity_at_specificity(yj_true, yj_prob, spec_target=0.95))
        res["per_class"][name] = m
        aurocs.append(m["auroc"]); auprcs.append(m["auprc"]); f1s.append(m["f1@0.5"]); sens95.append(m["sens@95spec"])
    # Macro
    def _nanmean(x): return float(np.nanmean(np.array(x, dtype=np.float32)))
    res["macro"] = {
        "auroc": _nanmean(aurocs),
        "auprc": _nanmean(auprcs),
        "f1@0.5": _nanmean(f1s),
        "sens@95spec": _nanmean(sens95),
    }
    # Micro via flattening
    res["micro"] = micro_metrics(y_true, probs, threshold)
    return res

def micro_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> Dict:
    y_pred = (probs >= threshold).astype(int)
    try:
        auroc = roc_auc_score(y_true.ravel(), probs.ravel())
    except Exception:
        auroc = float("nan")
    try:
        auprc = average_precision_score(y_true.ravel(), probs.ravel())
    except Exception:
        auprc = float("nan")
    f1 = f1_score(y_true.ravel(), y_pred.ravel(), zero_division=0)
    sens95 = sensitivity_at_specificity(y_true.ravel(), probs.ravel(), spec_target=0.95)
    return {"auroc": float(auroc), "auprc": float(auprc), "f1@0.5": float(f1), "sens@95spec": float(sens95)}

def sensitivity_at_specificity(y_true: np.ndarray, y_score: np.ndarray, spec_target: float = 0.95) -> float:
    fpr, tpr, thr = roc_curve(y_true, y_score)
    spec = 1 - fpr
    idx = (np.abs(spec - spec_target)).argmin()
    return float(tpr[idx])

def _group_sample_weights(df: pd.DataFrame, group_col: str) -> np.ndarray:
    """Inverse-frequency weights normalized around 1.0 for WeightedRandomSampler."""
    counts = df[group_col].value_counts()
    w = df[group_col].map(lambda g: 1.0 / counts[g])
    w = w / w.mean()
    return w.astype(np.float32).to_numpy()

def train(args):
    device = get_device()
    seed_everything(args.seed)
    ensure_dir(args.outdir)

    # Auto split if needed
    if args.auto_split is not None:
        if not os.path.exists(args.csv):
            raise FileNotFoundError(args.csv)
        train_p, val_p, test_p = args.auto_split
        auto_split_csv(args.csv, train_p, val_p, test_p, seed=args.seed)

    # Datasets
    train_tf = get_transforms(args.image_size, "train")
    val_tf = get_transforms(args.image_size, "val")

    df = pd.read_csv(args.csv)
    if "split" not in df.columns:
        raise ValueError("CSV must contain a 'split' column or use --auto-split to add one.")

    train_ds = MultiLabelImageDataset(args.csv, args.img_root, args.labels, split="train", transform=train_tf, group_col=args.group_col)
    val_ds   = MultiLabelImageDataset(args.csv, args.img_root, args.labels, split="val", transform=val_tf, group_col=args.group_col)

    if args.group_dro and args.group_col is None:
        raise ValueError("GroupDRO requires --group-col to be set.")
    if args.reweight_by_group and args.group_col is None:
        raise ValueError("Reweighting requires --group-col to be set.")

    sampler = None
    if args.reweight_by_group:
        weights = _group_sample_weights(train_ds.df, args.group_col)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
    )
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = build_model(num_classes=len(args.labels), drop_rate=args.drop_rate, pretrained=True).to(device)

    # Loss with pos_weight
    pos_weight = compute_pos_weight(pd.read_csv(args.csv).query("split=='train'"), args.labels).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_auroc = -np.inf
    history = []
    start = time.time()
    for epoch in range(1, args.epochs+1):
        model.train()
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Train epoch {epoch}/{args.epochs}"):
            images = batch["image"].to(device)
            labels = batch["labels"].to(device)
            logits = model(images)
            if args.group_dro:
                if "group" not in batch:
                    raise ValueError("Group information missing in batch while --group-dro is enabled.")
                # Compute per-group loss and take worst-case
                group_vals = batch["group"]
                uniq = pd.unique(np.array(group_vals))
                group_losses = []
                for g in uniq:
                    mask = torch.tensor([gv == g for gv in group_vals], device=device, dtype=torch.bool)
                    if mask.sum() == 0:
                        continue
                    group_losses.append(criterion(logits[mask], labels[mask]))
                loss = torch.stack(group_losses).max()
            else:
                loss = criterion(logits, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            epoch_loss += loss.item() * images.size(0)
        epoch_loss /= len(train_loader.dataset)

        # Validation
        with torch.no_grad():
            val_logits, val_labels, _ = predict(model, val_loader, device)
            val_metrics = metrics_from_logits(val_logits, val_labels, args.labels, threshold=0.5)
        mean_auroc = val_metrics["macro"]["auroc"]
        history.append({"epoch": epoch, "train_loss": epoch_loss, "val_macro_auroc": mean_auroc})
        save_json({"epoch": epoch, "train_loss": epoch_loss, "val": val_metrics}, os.path.join(args.outdir, f"epoch_{epoch:03d}.json"))

        if mean_auroc > best_auroc:
            best_auroc = mean_auroc
            torch.save(model.state_dict(), os.path.join(args.outdir, "best.pt"))
        scheduler.step()

    elapsed = time.time() - start
    save_json({"history": history, "best_val_macro_auroc": best_auroc, "elapsed_sec": elapsed}, os.path.join(args.outdir, "train_summary.json"))
    print(f"Training done. Best val macro AUROC: {best_auroc:.4f}. Elapsed: {elapsed/60:.1f} min.")

def evaluate(args):
    device = get_device(); seed_everything(args.seed); ensure_dir(args.outdir)
    tf = get_transforms(args.image_size, "val")
    ds = MultiLabelImageDataset(args.csv, args.img_root, args.labels, split=args.split, transform=tf, group_col=args.group_col)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    model = build_model(num_classes=len(args.labels), drop_rate=args.drop_rate, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    logits, labels, groups = predict(model, loader, device)
    metrics = metrics_from_logits(logits, labels, args.labels, threshold=0.5)
    save_json({"split": args.split, "metrics": metrics}, os.path.join(args.outdir, f"metrics_{args.split}.json"))
    # Save predictions for fairness analysis
    probs = 1 / (1 + np.exp(-logits))
    df_out = pd.DataFrame({"image_path": ds.df["image_path"].values})
    df_out["split"] = args.split
    for j, name in enumerate(args.labels):
        df_out[name] = labels[:, j]
        df_out[f"prob_{name}"] = probs[:, j]
    if args.group_col is not None and args.group_col in ds.df.columns:
        df_out[args.group_col] = ds.df[args.group_col].values
    df_out.to_csv(os.path.join(args.outdir, f"preds_{args.split}.csv"), index=False)
    print("Clean evaluation:", json.dumps(metrics["macro"], indent=2))

    # Optional group-wise slicing
    if args.group_col is not None and groups is not None:
        group_vals = ds.df[args.group_col].values
        uniq = sorted(pd.unique(group_vals))
        by_group = {}
        for g in uniq:
            idx = np.where(group_vals == g)[0]
            m = metrics_from_logits(logits[idx], labels[idx], args.labels, threshold=0.5)
            by_group[str(g)] = m["macro"]
        save_json({"group_col": args.group_col, "by_group": by_group}, os.path.join(args.outdir, f"group_metrics_{args.split}.json"))

# --------------------------
# Corruptions / Stress tests
# --------------------------

def corruption_transform(kind: str, severity: int, image_size: int = 224):
    assert severity in [1,2,3]
    # baselines for intensities
    if kind == "gaussian_noise":
        var = {1: 10.0, 2: 25.0, 3: 50.0}[severity]
        return A.Compose([
            A.LongestMaxSize(max_size=image_size), A.PadIfNeeded(image_size, image_size),
            A.GaussNoise(var_limit=(var, var), always_apply=True),
            A.CenterCrop(image_size, image_size), A.Normalize(), ToTensorV2()
        ])
    if kind == "gaussian_blur":
        bl = {1: (3,3), 2: (5,5), 3: (9,9)}[severity]
        return A.Compose([
            A.LongestMaxSize(max_size=image_size), A.PadIfNeeded(image_size, image_size),
            A.GaussianBlur(blur_limit=bl, sigma_limit=0, always_apply=True),
            A.CenterCrop(image_size, image_size), A.Normalize(), ToTensorV2()
        ])
    if kind == "jpeg":
        q = {1: (70,70), 2: (50,50), 3: (30,30)}[severity]
        return A.Compose([
            A.LongestMaxSize(max_size=image_size), A.PadIfNeeded(image_size, image_size),
            A.ImageCompression(quality_lower=q[0], quality_upper=q[1], always_apply=True),
            A.CenterCrop(image_size, image_size), A.Normalize(), ToTensorV2()
        ])
    if kind == "brightness_contrast":
        lim = {1: 0.15, 2: 0.3, 3: 0.45}[severity]
        return A.Compose([
            A.LongestMaxSize(max_size=image_size), A.PadIfNeeded(image_size, image_size),
            A.RandomBrightnessContrast(brightness_limit=lim, contrast_limit=lim, p=1.0),
            A.CenterCrop(image_size, image_size), A.Normalize(), ToTensorV2()
        ])
    raise ValueError(f"Unknown corruption kind: {kind}")

def stress(args):
    device = get_device(); seed_everything(args.seed); ensure_dir(args.outdir)
    kinds = ["gaussian_noise", "gaussian_blur", "jpeg", "brightness_contrast"]
    results = []
    # Clean baseline first
    clean_tf = get_transforms(args.image_size, "val")
    clean_ds = MultiLabelImageDataset(args.csv, args.img_root, args.labels, split=args.split, transform=clean_tf, group_col=args.group_col)
    clean_loader = DataLoader(clean_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = build_model(num_classes=len(args.labels), drop_rate=args.drop_rate, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))

    clean_logits, clean_labels, _ = predict(model, clean_loader, device)
    clean_metrics = metrics_from_logits(clean_logits, clean_labels, args.labels, threshold=0.5)
    base_macro = clean_metrics["macro"]
    save_json({"clean": clean_metrics}, os.path.join(args.outdir, "clean_metrics.json"))

    for kind in kinds:
        for sev in [1,2,3]:
            tf = corruption_transform(kind, sev, args.image_size)
            ds = MultiLabelImageDataset(args.csv, args.img_root, args.labels, split=args.split, transform=tf, group_col=args.group_col)
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
            logits, labels, _ = predict(model, loader, device)
            m = metrics_from_logits(logits, labels, args.labels, threshold=0.5)
            row = {"corruption": kind, "severity": sev,
                   "macro_auroc": m["macro"]["auroc"], "macro_auprc": m["macro"]["auprc"],
                   "macro_f1@0.5": m["macro"]["f1@0.5"], "macro_sens@95spec": m["macro"]["sens@95spec"],
                   "delta_auroc": m["macro"]["auroc"] - base_macro["auroc"],
                   "delta_auprc": m["macro"]["auprc"] - base_macro["auprc"]}
            results.append(row)
            save_json(m, os.path.join(args.outdir, f"{kind}_sev{sev}.json"))
    # Save table
    pd.DataFrame(results).to_csv(os.path.join(args.outdir, "stress_summary.csv"), index=False)
    print("Wrote stress tests to:", args.outdir)

# --------------------------
# Calibration & Uncertainty
# --------------------------

def ece_score(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> Tuple[float, Dict]:
    """ECE over flattened predictions; bins by confidence of predicted class in binary setting."""
    y_true = y_true.ravel().astype(int)
    probs = probs.ravel()
    conf = np.where(probs >= 0.5, probs, 1 - probs)  # confidence of predicted label
    preds = (probs >= 0.5).astype(int)
    bins = np.linspace(0.0, 1.0, n_bins+1)
    idxs = np.digitize(conf, bins) - 1
    ece = 0.0; bin_stats = []
    for b in range(n_bins):
        I = np.where(idxs == b)[0]
        if len(I) == 0:
            bin_stats.append({"bin": b, "conf": None, "acc": None, "count": 0}); continue
        acc = (preds[I] == y_true[I]).mean()
        c = conf[I].mean()
        ece += (len(I)/len(conf)) * abs(acc - c)
        bin_stats.append({"bin": b, "conf": float(c), "acc": float(acc), "count": int(len(I))})
    return float(ece), {"bins": bin_stats}

def plot_reliability(bin_stats: Dict, out_path: str, title: str = "Reliability diagram"):
    xs = []; accs = []; confs = []; sizes = []
    for b in bin_stats["bins"]:
        if b["count"] > 0:
            confs.append(b["conf"]); accs.append(b["acc"]); sizes.append(b["count"])
    plt.figure(figsize=(4,4))
    plt.plot([0,1],[0,1], linestyle="--")
    plt.scatter(confs, accs, s=np.array(sizes)/np.max(sizes)*200.0, alpha=0.7)
    plt.xlabel("Confidence"); plt.ylabel("Accuracy"); plt.title(title)
    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()

class TemperatureScaler(nn.Module):
    def __init__(self, init_T: float = 1.0):
        super().__init__()
        self.log_T = nn.Parameter(torch.tensor(math.log(init_T), dtype=torch.float32))
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        T = torch.exp(self.log_T)
        return logits / T

def fit_temperature(model: nn.Module, loader: DataLoader, device: torch.device, max_iter: int = 500, lr: float = 0.01) -> float:
    model.eval()
    scaler = TemperatureScaler().to(device)
    opt = torch.optim.LBFGS(scaler.parameters(), lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe")
    criterion = nn.BCEWithLogitsLoss()
    logits_list = []; labels_list = []
    with torch.no_grad():
        for b in loader:
            x = b["image"].to(device); y = b["labels"].to(device)
            logits = model(x)
            logits_list.append(logits); labels_list.append(y)
    logits = torch.cat(logits_list); labels = torch.cat(labels_list)

    def closure():
        opt.zero_grad()
        out = scaler(logits)
        loss = criterion(out, labels)
        loss.backward()
        return loss
    opt.step(closure)
    T = float(torch.exp(scaler.log_T).item())
    return T

def enable_mc_dropout(model: nn.Module):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

@torch.no_grad()
def mc_dropout_predict(model: nn.Module, loader: DataLoader, device: torch.device, n_samples: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    enable_mc_dropout(model)
    preds = []
    for _ in range(n_samples):
        logits, labels, _ = predict(model, loader, device)
        preds.append(1 / (1 + np.exp(-logits)))
    probs = np.stack(preds, axis=0)  # [S, N, C]
    mean_p = probs.mean(axis=0)
    var_p = probs.var(axis=0)
    return mean_p, var_p

@torch.no_grad()
def tta_predict(model: nn.Module, ds: Dataset, device: torch.device, n_samples: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    aug = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.05, rotate_limit=5, border_mode=cv2.BORDER_REFLECT_101, p=1.0),
        A.GaussianBlur(blur_limit=(3,5), p=0.2),
        A.Normalize(), ToTensorV2()
    ])
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    probs_list, vars_list = [], []
    for batch in tqdm(loader, desc="TTA", leave=False):
        img = batch["image"][0].permute(1,2,0).numpy()
        ys = []
        for _ in range(n_samples):
            out = aug(image=(img*255).astype(np.uint8))
            x = out["image"].unsqueeze(0).to(device)
            logits = model(x)
            ys.append(torch.sigmoid(logits).cpu().numpy())
        ys = np.concatenate(ys, axis=0)
        probs_list.append(ys.mean(axis=0))
        vars_list.append(ys.var(axis=0))
    return np.vstack(probs_list), np.vstack(vars_list)

def calibrate(args):
    device = get_device(); seed_everything(args.seed); ensure_dir(args.outdir)
    # Data
    val_tf = get_transforms(args.image_size, "val")
    test_tf = get_transforms(args.image_size, "val")
    val_ds  = MultiLabelImageDataset(args.csv, args.img_root, args.labels, split=args.val_split, transform=val_tf, group_col=args.group_col)
    test_ds = MultiLabelImageDataset(args.csv, args.img_root, args.labels, split=args.test_split, transform=test_tf, group_col=args.group_col)
    val_loader  = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = build_model(num_classes=len(args.labels), drop_rate=args.drop_rate, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))

    # Baseline (uncalibrated)
    logits, labels, _ = predict(model, test_loader, device)
    probs = 1/(1+np.exp(-logits))
    ece0, bins0 = ece_score(labels, probs, n_bins=15)
    plot_reliability(bins0, os.path.join(args.outdir, "reliability_uncal.png"), "Reliability (uncalibrated)")
    base_metrics = metrics_from_logits(logits, labels, args.labels, threshold=0.5)

    # Temperature scaling (fit on validation)
    T = fit_temperature(model, val_loader, device)
    logits_T = logits / T
    probs_T = 1/(1+np.exp(-logits_T))
    eceT, binsT = ece_score(labels, probs_T, n_bins=15)
    plot_reliability(binsT, os.path.join(args.outdir, "reliability_temp_scaled.png"), f"Reliability (Temp scaling T={T:.2f})")
    temp_metrics = metrics_from_logits(logits_T, labels, args.labels, threshold=0.5)

    # MC-Dropout (uncertainty)
    mean_p, var_p = mc_dropout_predict(model, test_loader, device, n_samples=20)
    ece_mc, bins_mc = ece_score(labels, mean_p, n_bins=15)
    plot_reliability(bins_mc, os.path.join(args.outdir, "reliability_mc_dropout.png"), "Reliability (MC-Dropout)")

    # Save summary
    summary = {
        "baseline": {"ece": ece0, "macro": base_metrics["macro"]},
        "temp_scaling": {"T": T, "ece": eceT, "macro": temp_metrics["macro"]},
        "mc_dropout": {"ece": ece_mc, "uncertainty_mean_var": float(var_p.mean())}
    }
    save_json(summary, os.path.join(args.outdir, "calibration_summary.json"))
    print("Calibration summary:", json.dumps(summary, indent=2))

# --------------------------
# Selective prediction
# --------------------------

def risk_coverage(y_true: np.ndarray, probs: np.ndarray, n_points: int = 30) -> pd.DataFrame:
    """Binary micro risk–coverage: flatten over classes, confidence = p if yhat=1 else 1-p; risk=1-accuracy on covered set."""
    y_true = y_true.ravel().astype(int)
    probs = probs.ravel()
    yhat = (probs >= 0.5).astype(int)
    conf = np.where(yhat == 1, probs, 1 - probs)
    order = np.argsort(-conf)  # high to low
    y_true = y_true[order]; yhat = yhat[order]; conf = conf[order]
    N = len(conf)
    ks = np.linspace(int(N/n_points), N, n_points, dtype=int)
    covs, risks = [], []
    for k in ks:
        if k <= 0: continue
        acc = (yhat[:k] == y_true[:k]).mean()
        covs.append(k / N)
        risks.append(1 - acc)
    return pd.DataFrame({"coverage": covs, "risk": risks})

def selective(args):
    device = get_device(); seed_everything(args.seed); ensure_dir(args.outdir)
    tf = get_transforms(args.image_size, "val")
    ds = MultiLabelImageDataset(args.csv, args.img_root, args.labels, split=args.split, transform=tf, group_col=args.group_col)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    model = build_model(num_classes=len(args.labels), drop_rate=args.drop_rate, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    logits, labels, _ = predict(model, loader, device)
    probs = 1/(1+np.exp(-logits))
    df = risk_coverage(labels, probs, n_points=40)
    df.to_csv(os.path.join(args.outdir, f"risk_coverage_{args.split}.csv"), index=False)
    # Also save plot
    plt.figure(figsize=(4,4))
    plt.plot(df["coverage"], df["risk"])
    plt.xlabel("Coverage"); plt.ylabel("Risk (1-acc)"); plt.title("Risk–Coverage (micro)")
    plt.tight_layout(); plt.savefig(os.path.join(args.outdir, f"risk_coverage_{args.split}.png"), dpi=200); plt.close()
    print("Selective prediction written to:", args.outdir)

# --------------------------
# Explainability (Grad-CAM + IG)
# --------------------------

def _tensor_to_uint8(img_t: torch.Tensor) -> np.ndarray:
    """Map CHW float tensor in [0,1] (roughly) to uint8 RGB."""
    img = img_t.detach().cpu().permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)

def _overlay_heatmap(img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    heat = np.uint8(255 * heatmap)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    return np.clip(alpha * img + (1 - alpha) * heat_color, 0, 255).astype(np.uint8)

def _grad_cam(model: nn.Module, images: torch.Tensor, target_idx: int, target_layer: nn.Module) -> np.ndarray:
    activations = []
    gradients = []
    def fwd_hook(_m, _i, o): activations.append(o)
    def bwd_hook(_m, _gi, go): gradients.append(go[0])
    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    logits = model(images)
    target = logits[:, target_idx].sum()
    model.zero_grad()
    target.backward(retain_graph=True)

    acts = activations[-1]          # [B, C, H, W]
    grads = gradients[-1]           # [B, C, H, W]
    weights = grads.mean(dim=(2,3), keepdim=True)
    cam = F.relu((weights * acts).sum(dim=1, keepdim=True))  # [B,1,H,W]
    cam = F.interpolate(cam, size=images.shape[2:], mode="bilinear", align_corners=False)
    cam = cam.squeeze(1)
    cam_min = cam.view(cam.size(0), -1).min(dim=1)[0].view(-1, 1, 1)
    cam_max = cam.view(cam.size(0), -1).max(dim=1)[0].view(-1, 1, 1) + 1e-6
    cam = (cam - cam_min) / cam_max

    h1.remove(); h2.remove()
    return cam.detach().cpu().numpy()

def explain(args):
    device = get_device(); seed_everything(args.seed); ensure_dir(args.outdir)
    tf = get_transforms(args.image_size, "val")
    ds = MultiLabelImageDataset(args.csv, args.img_root, args.labels, split=args.split, transform=tf, group_col=args.group_col)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    target_label = args.target_label or args.labels[0]
    if target_label not in args.labels:
        raise ValueError(f"--target-label must be one of {args.labels}")
    target_idx = args.labels.index(target_label)

    model = build_model(num_classes=len(args.labels), drop_rate=args.drop_rate, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()
    target_layer = model.features

    ig = None
    if _HAS_CAPTUM:
        ig = IntegratedGradients(model)
    else:
        warnings.warn("captum not installed; Integrated Gradients will be skipped.")

    saved = 0
    summary_rows = []
    for batch in tqdm(loader, desc="Explain", leave=False):
        images = batch["image"].to(device)
        logits = model(images)
        probs = torch.sigmoid(logits)
        cams = _grad_cam(model, images, target_idx, target_layer)
        ig_maps = None
        if ig is not None:
            attributions = ig.attribute(images, target=target_idx, n_steps=args.ig_steps, internal_batch_size=None)
            ig_maps = attributions.abs().mean(dim=1).detach().cpu().numpy()
            # normalize per-sample
            ig_maps = ig_maps / (ig_maps.reshape(ig_maps.shape[0], -1).max(axis=1, keepdims=True)[:, None, None] + 1e-6)

        for i in range(images.size(0)):
            if saved >= args.samples:
                break
            img_np = _tensor_to_uint8(batch["image"][i])
            cam_overlay = _overlay_heatmap(img_np, cams[i])
            to_concat = [cam_overlay]
            names = [f"cam"]
            if ig_maps is not None:
                ig_overlay = _overlay_heatmap(img_np, ig_maps[i])
                to_concat.append(ig_overlay)
                names.append("ig")
            combined = np.concatenate(to_concat, axis=1)
            fname = os.path.join(args.outdir, f"exp_{args.split}_{saved:04d}.png")
            cv2.imwrite(fname, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
            summary_rows.append({
                "image_path": ds.df.iloc[saved]["image_path"] if saved < len(ds.df) else "",
                "prob": float(probs[i, target_idx].item()),
                "logit": float(logits[i, target_idx].item()),
                "file": fname,
                "maps": "+".join(names),
            })
            saved += 1
        if saved >= args.samples:
            break
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(os.path.join(args.outdir, "explain_summary.csv"), index=False)
    print(f"Saved {saved} explanation visualizations to {args.outdir}")

# --------------------------
# Validation tool (bonus)
# --------------------------

@torch.no_grad()
def valtool(args):
    """Run model on a new CSV/folder. If labels are present, compute metrics; else output predictions only."""
    device = get_device(); seed_everything(args.seed); ensure_dir(args.outdir)
    tf = get_transforms(args.image_size, "val")
    ds = MultiLabelImageDataset(args.csv, args.img_root, args.labels, split=args.split, transform=tf, group_col=args.group_col) if args.split else \
         MultiLabelImageDataset(args.csv, args.img_root, args.labels, split=None, transform=tf, group_col=args.group_col)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    model = build_model(num_classes=len(args.labels), drop_rate=args.drop_rate, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    logits, labels, groups = predict(model, loader, device)
    probs = 1/(1+np.exp(-logits))

    # Save predictions
    df = pd.DataFrame(probs, columns=[f"prob_{c}" for c in args.labels])
    df.insert(0, "image_path", ds.df["image_path"].values)
    if groups is not None:
        df["group"] = ds.df[args.group_col].values
    df.to_csv(os.path.join(args.outdir, "predictions.csv"), index=False)

    # Optional metrics if labels exist
    has_labels = all(c in ds.df.columns for c in args.labels)
    if has_labels and (args.split is not None):
        m = metrics_from_logits(logits, labels, args.labels, threshold=0.5)
        save_json(m, os.path.join(args.outdir, "valtool_metrics.json"))
        print("Valtool metrics (macro):", json.dumps(m["macro"], indent=2))
    else:
        print("Predictions saved (no labels provided, metrics skipped).")

# --------------------------
# Argparse
# --------------------------

def build_parser():
    p = argparse.ArgumentParser(description="CV8502 Failure Analysis Starter (Classification)")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Shared
    def add_shared(sp):
        sp.add_argument("--csv", type=str, required=True, help="CSV with image_path, split, and label columns.")
        sp.add_argument("--img-root", type=str, required=True, help="Root dir for images (prefix to image_path if relative).")
        sp.add_argument("--labels", type=str, nargs="+", required=True, help="Label column names (2–4 pathologies).")
        sp.add_argument("--group-col", type=str, default=None, help="Optional CSV column for slicing (e.g., site/scanner).")
        sp.add_argument("--image-size", type=int, default=224)
        sp.add_argument("--batch-size", type=int, default=16)
        sp.add_argument("--workers", type=int, default=2)
        sp.add_argument("--seed", type=int, default=1337)
        sp.add_argument("--drop-rate", type=float, default=0.2)
        sp.add_argument("--outdir", type=str, required=True)

    # Train
    sp_tr = sub.add_parser("train")
    add_shared(sp_tr)
    sp_tr.add_argument("--epochs", type=int, default=10)
    sp_tr.add_argument("--lr", type=float, default=1e-4)
    sp_tr.add_argument("--auto-split", type=float, nargs=3, default=None, metavar=("TRAIN","VAL","TEST"),
                       help="If provided and CSV lacks 'split', add it with given ratios (e.g., 0.7 0.1 0.2).")
    sp_tr.add_argument("--reweight-by-group", action="store_true", help="Use inverse-frequency group weights in a sampler.")
    sp_tr.add_argument("--group-dro", action="store_true", help="GroupDRO: optimize worst-group loss per batch.")

    # Eval
    sp_ev = sub.add_parser("eval")
    add_shared(sp_ev)
    sp_ev.add_argument("--split", type=str, default="test", choices=["train","val","test"])
    sp_ev.add_argument("--weights", type=str, required=True)

    # Stress
    sp_st = sub.add_parser("stress")
    add_shared(sp_st)
    sp_st.add_argument("--split", type=str, default="test", choices=["train","val","test"])
    sp_st.add_argument("--weights", type=str, required=True)

    # Calibrate
    sp_ca = sub.add_parser("calibrate")
    add_shared(sp_ca)
    sp_ca.add_argument("--val-split", type=str, default="val", choices=["train","val","test"])
    sp_ca.add_argument("--test-split", type=str, default="test", choices=["train","val","test"])
    sp_ca.add_argument("--weights", type=str, required=True)
    sp_ca.add_argument("--batch-n", type=int, default=20)

    # Selective
    sp_sel = sub.add_parser("selective")
    add_shared(sp_sel)
    sp_sel.add_argument("--split", type=str, default="test", choices=["train","val","test"])
    sp_sel.add_argument("--weights", type=str, required=True)

    # Explainability
    sp_exp = sub.add_parser("explain")
    add_shared(sp_exp)
    sp_exp.add_argument("--split", type=str, default="test", choices=["train","val","test"])
    sp_exp.add_argument("--weights", type=str, required=True)
    sp_exp.add_argument("--target-label", type=str, default=None, help="Label to visualize; defaults to first label.")
    sp_exp.add_argument("--samples", type=int, default=16, help="Number of images to visualize.")
    sp_exp.add_argument("--ig-steps", type=int, default=16, help="Integrated Gradients steps.")

    # Valtool
    sp_vt = sub.add_parser("valtool")
    add_shared(sp_vt)
    sp_vt.add_argument("--weights", type=str, required=True)
    sp_vt.add_argument("--split", type=str, default=None, help="If provided, will filter by split; else run all rows.")

    return p

def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "train":
        train(args)
    elif args.cmd == "eval":
        evaluate(args)
    elif args.cmd == "stress":
        stress(args)
    elif args.cmd == "calibrate":
        calibrate(args)
    elif args.cmd == "selective":
        selective(args)
    elif args.cmd == "explain":
        explain(args)
    elif args.cmd == "valtool":
        valtool(args)
    else:
        raise ValueError(args.cmd)

if __name__ == "__main__":
    main()
