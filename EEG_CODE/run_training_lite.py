"""
Run Improved Trimodal Training with V4-LITE Model
==================================================
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import logging
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader, Subset

# Import config
from config import Config, set_seed

# Import V4 enhancements
from crossmodal_v4_enhancements import (
    EnhancedTriModalFusionNetV4Lite,
    BalancedTriModalDataset,
    CosineAnnealingWarmup,
    EarlyStopping,
    LabelSmoothingCrossEntropy,
    get_lite_fusion_weights,
)

# Set seeds
set_seed(42)

print("="*70)
print("IMPROVED TRIMODAL TRAINING WITH V4-LITE")
print("="*70)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def normalize_modality(feat, eps=1e-8):
    mean = feat.mean()
    std = feat.std() + eps
    return (feat - mean) / std

def vec_upper_triangle(mat):
    n = mat.shape[0]
    idx = np.triu_indices(n, k=1)
    return mat[idx]

# ============================================================================
# DATA LOADERS
# ============================================================================

class EEGDatasetCONN(Dataset):
    def __init__(self, subj_list, band_list, cond_list, conn_dir, labels=None, verbose=False):
        import glob
        import h5py
        import scipy.io

        self.samples = []
        conn_dir = Path(conn_dir)

        for subj in subj_list:
            subj_str = f"{subj:03d}"
            for band in band_list.keys():
                for cond in cond_list:
                    pattern = str(conn_dir / f'*sub{subj_str}*{band}*{cond}*.mat')
                    files = glob.glob(pattern)
                    if not files:
                        pattern = str(conn_dir / f'*{subj_str}*{band}*.mat')
                        files = glob.glob(pattern)

                    for fpath in files:
                        try:
                            # Load connectivity
                            try:
                                with h5py.File(fpath, 'r') as f:
                                    for key in ['conn', 'connectivity', 'data']:
                                        if key in f:
                                            data = np.array(f[key])
                                            break
                                    else:
                                        key = list(f.keys())[0]
                                        data = np.array(f[key])
                            except:
                                mat = scipy.io.loadmat(fpath)
                                for key in ['conn', 'connectivity', 'data']:
                                    if key in mat:
                                        data = mat[key]
                                        break
                                else:
                                    key = [k for k in mat.keys() if not k.startswith('_')][0]
                                    data = mat[key]

                            if data.ndim == 2:
                                conn_flat = vec_upper_triangle(data)
                            else:
                                conn_flat = data.flatten()

                            conn_flat = normalize_modality(conn_flat)
                            y = labels.get(subj, 0) if labels else 0
                            self.samples.append((conn_flat.astype(np.float32), subj, band, cond, y))
                        except Exception as e:
                            if verbose:
                                print(f"Failed to load {fpath}: {e}")

        print(f"  Loaded {len(self.samples)} CONN samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        conn, subj, band, cond, y = self.samples[idx]
        return torch.tensor(conn, dtype=torch.float32), subj, band, cond, y


class EEGDatasetPW(Dataset):
    def __init__(self, subj_list, band_list, freq_list, pw_dir, labels=None, verbose=False):
        import glob
        import h5py
        import scipy.io

        self.samples = []
        pw_dir = Path(pw_dir)

        for subj in subj_list:
            subj_str = f"{subj:03d}"
            for band in band_list.keys():
                for freq in freq_list:
                    pattern = str(pw_dir / f'*sub{subj_str}*{band}*{freq}*.mat')
                    files = glob.glob(pattern)

                    for fpath in files:
                        try:
                            try:
                                with h5py.File(fpath, 'r') as f:
                                    for key in ['powspctrm', 'pw', 'power', 'data']:
                                        if key in f:
                                            pw = np.array(f[key]).T.astype(np.float32)
                                            break
                                    else:
                                        key = list(f.keys())[0]
                                        pw = np.array(f[key]).T.astype(np.float32)
                            except:
                                mat = scipy.io.loadmat(fpath)
                                for key in ['powspctrm', 'pw', 'power', 'data']:
                                    if key in mat:
                                        pw = mat[key].astype(np.float32)
                                        break
                                else:
                                    key = [k for k in mat.keys() if not k.startswith('_')][0]
                                    pw = mat[key].astype(np.float32)

                            pw = normalize_modality(pw)
                            y = labels.get(subj, 0) if labels else 0
                            self.samples.append((pw, subj, band, freq, y))
                        except Exception as e:
                            if verbose:
                                print(f"Failed to load {fpath}: {e}")

        print(f"  Loaded {len(self.samples)} PW samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pw, subj, band, freq, y = self.samples[idx]
        return torch.tensor(pw, dtype=torch.float32), subj, band, freq, y


class EEGDatasetERP(Dataset):
    def __init__(self, subj_list, band_list, freq_list, erp_dir, labels=None, verbose=False):
        import glob
        import h5py
        import scipy.io

        self.samples = []
        erp_dir = Path(erp_dir)

        for subj in subj_list:
            subj_str = f"{subj:03d}"
            for band in band_list.keys():
                for freq in freq_list:
                    pattern = str(erp_dir / f'*sub{subj_str}*{band}*{freq}*.mat')
                    files = glob.glob(pattern)

                    for fpath in files:
                        try:
                            try:
                                with h5py.File(fpath, 'r') as f:
                                    for key in ['ERP', 'erp', 'data']:
                                        if key in f:
                                            erp = np.array(f[key]).T.astype(np.float32)
                                            break
                                    else:
                                        key = list(f.keys())[0]
                                        erp = np.array(f[key]).T.astype(np.float32)
                            except:
                                mat = scipy.io.loadmat(fpath)
                                for key in ['ERP', 'erp', 'data']:
                                    if key in mat:
                                        erp = mat[key].astype(np.float32)
                                        break
                                else:
                                    key = [k for k in mat.keys() if not k.startswith('_')][0]
                                    erp = mat[key].astype(np.float32)

                            erp = normalize_modality(erp)
                            y = labels.get(subj, 0) if labels else 0
                            self.samples.append((erp, subj, band, freq, y))
                        except Exception as e:
                            if verbose:
                                print(f"Failed to load {fpath}: {e}")

        print(f"  Loaded {len(self.samples)} ERP samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        erp, subj, band, freq, y = self.samples[idx]
        return torch.tensor(erp, dtype=torch.float32), subj, band, freq, y


def aggregate_features(dataset, name="data"):
    """Aggregate features by subject."""
    print(f"Aggregating {len(dataset)} {name} samples...")

    subj_features = defaultdict(list)
    subj_labels = {}

    for i in range(len(dataset)):
        sample = dataset[i]
        feat = sample[0]
        subj = sample[1]
        y = sample[-1]

        if isinstance(feat, torch.Tensor):
            feat = feat.numpy()
        subj_features[subj].append(feat)
        subj_labels[subj] = y

    aggregated = {}
    for subj, feats in subj_features.items():
        stacked = np.stack(feats, axis=0)
        agg_feat = np.mean(stacked, axis=0)
        aggregated[subj] = torch.tensor(agg_feat, dtype=torch.float32)

    print(f"  Aggregated to {len(aggregated)} subjects")
    return aggregated, subj_labels


def load_labels(label_path, binary=True):
    """Load labels from CSV."""
    import pandas as pd

    df = pd.read_csv(label_path)

    # Find columns
    subj_col = None
    label_col = None

    for col in ['subject', 'Subject', 'subj', 'ID', 'id', 'SubjectID']:
        if col in df.columns:
            subj_col = col
            break

    for col in ['label', 'Label', 'class', 'Class', 'score', 'Score', 'y', 'target']:
        if col in df.columns:
            label_col = col
            break

    if subj_col is None:
        subj_col = df.columns[0]
    if label_col is None:
        label_col = df.columns[1]

    labels = {}
    for _, row in df.iterrows():
        subj = int(row[subj_col])
        label = int(row[label_col])
        if binary:
            label = 0 if label <= 1 else 1
        labels[subj] = label

    print(f"Loaded {len(labels)} labels")
    return labels


# ============================================================================
# MODEL WRAPPER
# ============================================================================

class ImprovedTriModalFusionNetLite(nn.Module):
    def __init__(self, in_pw_dim, in_erp_dim, in_conn_dim,
                 fusion_dim=96, num_classes=2, dropout=0.4, conn_boost=1.3):
        super().__init__()

        self.model = EnhancedTriModalFusionNetV4Lite(
            erp_channels=in_erp_dim,
            pw_channels=in_pw_dim,
            conn_features=in_conn_dim,
            hidden_dim=fusion_dim,
            num_classes=num_classes,
            dropout=dropout,
            conn_boost=conn_boost
        )
        self.fusion_weight_history = []

    def forward(self, pw, erp, conn):
        logits, weights = self.model(erp, pw, conn, return_fusion_weights=True)
        return logits

    def get_fusion_weights(self):
        return get_lite_fusion_weights(self.model)

    def track_fusion_weights(self):
        weights = self.get_fusion_weights()
        if weights:
            self.fusion_weight_history.append(weights)


def collate_balanced(batch):
    """Collate for BalancedTriModalDataset."""
    erps, pws, conns, labels, subjs = [], [], [], [], []

    for sample in batch:
        if isinstance(sample, dict):
            erps.append(sample['erp'])
            pws.append(sample['pw'])
            conns.append(sample['conn'])
            labels.append(sample['label'])
            subjs.append(sample['subject'])
        else:
            erps.append(sample[0])
            pws.append(sample[1])
            conns.append(sample[2])
            labels.append(sample[3])
            subjs.append(sample[4])

    return (torch.stack(erps), torch.stack(pws), torch.stack(conns),
            torch.tensor(labels, dtype=torch.long), subjs)


# ============================================================================
# MAIN TRAINING
# ============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load config
    print("\nLoading configuration...")
    config = Config(None)

    # Load labels
    print("\nLoading labels...")
    label_dict = load_labels(config.label_path, binary=True)

    # Load datasets
    print("\nLoading ERP data...")
    erp_dataset = EEGDatasetERP(
        config.subject_list, config.bands, config.freq_bands,
        config.eeg_path_erp, labels=label_dict
    )

    print("\nLoading PW data...")
    pw_dataset = EEGDatasetPW(
        config.subject_list, config.bands, config.freq_bands,
        config.eeg_path_pw, labels=label_dict
    )

    print("\nLoading CONN data...")
    conn_dataset = EEGDatasetCONN(
        config.subject_list, config.bands, config.func_segments,
        config.eeg_path_conn, labels=label_dict
    )

    # Check if data loaded
    if len(erp_dataset) == 0 or len(pw_dataset) == 0 or len(conn_dataset) == 0:
        print("ERROR: No data loaded!")
        return

    # Aggregate
    print("\nAggregating features per subject...")
    erp_agg, erp_labels = aggregate_features(erp_dataset, "ERP")
    pw_agg, pw_labels = aggregate_features(pw_dataset, "PW")
    conn_agg, conn_labels = aggregate_features(conn_dataset, "CONN")

    # Create balanced dataset
    print("\nCreating balanced trimodal dataset...")
    dataset = BalancedTriModalDataset(
        erp_agg, pw_agg, conn_agg, label_dict, agg_method='mean'
    )

    if len(dataset) == 0:
        print("ERROR: No samples in balanced dataset!")
        return

    # Get arrays
    labels = [s['label'] for s in dataset.samples]
    subjects = [s['subject'] for s in dataset.samples]
    label_array = np.array(labels)
    subj_array = np.array(subjects)

    n_classes = len(np.unique(label_array))
    print(f"\nDataset: {len(dataset)} samples, {n_classes} classes")
    print(f"Class distribution: {dict(zip(*np.unique(label_array, return_counts=True)))}")

    # Get dimensions
    sample = dataset[0]
    erp_ch = sample['erp'].shape[0]
    pw_ch = sample['pw'].shape[0]
    conn_dim = sample['conn'].numel()
    print(f"Dimensions: ERP={erp_ch}, PW={pw_ch}, CONN={conn_dim}")

    # Results
    results = []
    fusion_weights_all = []

    # CV
    n_splits = min(config.n_splits, len(np.unique(subj_array)))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = list(sgkf.split(np.zeros(len(dataset)), label_array, groups=subj_array))

    print(f"\nRunning {n_splits}-fold cross-validation...")
    print("="*70)

    for fold_idx, (train_idx, test_idx) in enumerate(splits, 1):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx}/{n_splits}")
        print(f"{'='*60}")
        print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")

        # Loaders
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=config.batch_size,
                                  shuffle=True, collate_fn=collate_balanced)
        test_loader = DataLoader(Subset(dataset, test_idx), batch_size=config.batch_size,
                                 shuffle=False, collate_fn=collate_balanced)

        # Class weights
        y_train = label_array[train_idx]
        weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

        # Model
        model = ImprovedTriModalFusionNetLite(
            in_pw_dim=pw_ch, in_erp_dim=erp_ch, in_conn_dim=conn_dim,
            fusion_dim=96, num_classes=n_classes, dropout=0.4, conn_boost=1.3
        ).to(device)

        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Parameters: {param_count:,}")

        # Training setup
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
        scheduler = CosineAnnealingWarmup(optimizer, warmup_epochs=3, total_epochs=config.epochs)
        early_stopper = EarlyStopping(patience=15, mode='max')

        best_f1 = 0.0
        best_state = None

        # Training loop
        for epoch in range(1, config.epochs + 1):
            model.train()
            total_loss = 0.0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
                erp, pw, conn, y, _ = batch
                erp, pw, conn = erp.to(device), pw.to(device), conn.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                logits = model(pw, erp, conn)
                loss = criterion(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()

            scheduler.step()

            # Eval
            model.eval()
            all_preds, all_targets = [], []
            with torch.no_grad():
                for batch in test_loader:
                    erp, pw, conn, y, _ = batch
                    erp, pw, conn = erp.to(device), pw.to(device), conn.to(device)
                    logits = model(pw, erp, conn)
                    all_preds.extend(logits.argmax(dim=1).cpu().numpy())
                    all_targets.extend(y.numpy())

            f1 = f1_score(all_targets, all_preds, average='weighted')
            acc = accuracy_score(all_targets, all_preds)

            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch:2d} | Loss: {total_loss/len(train_loader):.4f} | F1: {f1:.4f} | Acc: {acc:.4f}")

            if f1 > best_f1:
                best_f1 = f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if early_stopper(f1):
                print(f"  Early stopping at epoch {epoch}")
                break

        # Final eval
        if best_state:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in test_loader:
                erp, pw, conn, y, _ = batch
                erp, pw, conn = erp.to(device), pw.to(device), conn.to(device)
                logits = model(pw, erp, conn)
                all_preds.extend(logits.argmax(dim=1).cpu().numpy())
                all_targets.extend(y.numpy())

        metrics = {
            "Accuracy": accuracy_score(all_targets, all_preds),
            "F1": f1_score(all_targets, all_preds, average='weighted'),
            "Precision": precision_score(all_targets, all_preds, average='weighted', zero_division=0),
            "Recall": recall_score(all_targets, all_preds, average='weighted', zero_division=0)
        }
        results.append(metrics)

        # Fusion weights
        fw = model.get_fusion_weights()
        if fw:
            fusion_weights_all.append(fw)
            print(f"\n  Fusion: ERP={fw.get('erp_weight', 0):.3f}, PW={fw.get('pw_weight', 0):.3f}, CONN={fw.get('conn_weight', 0):.3f}")

        print(f"\n  FOLD {fold_idx}: Acc={metrics['Accuracy']:.4f}, F1={metrics['F1']:.4f}")

    # Summary
    print("\n" + "="*70)
    print("FINAL RESULTS - TRIMODAL-LITE")
    print("="*70)

    accs = [r['Accuracy'] for r in results]
    f1s = [r['F1'] for r in results]

    print(f"\n  Accuracy:  {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    print(f"  F1:        {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")

    if fusion_weights_all:
        print(f"\nAverage Fusion Weights:")
        print(f"  ERP:  {np.mean([fw.get('erp_weight', 0) for fw in fusion_weights_all]):.3f}")
        print(f"  PW:   {np.mean([fw.get('pw_weight', 0) for fw in fusion_weights_all]):.3f}")
        print(f"  CONN: {np.mean([fw.get('conn_weight', 0) for fw in fusion_weights_all]):.3f}")

    print("\n" + "="*70)
    print("COMPARISON WITH PREVIOUS")
    print("="*70)
    print("\nPrevious (V4 Full):")
    print("  TRIMODAL:  0.5668 +/- 0.1375")
    print("  PWONLY:    0.5973 +/- 0.0574")
    print(f"\nNew (V4-LITE):")
    print(f"  TRIMODAL:  {np.mean(accs):.4f} +/- {np.std(accs):.4f}")

    improvement = np.mean(accs) - 0.5668
    print(f"\nImprovement: {improvement:+.4f} ({improvement/0.5668*100:+.1f}%)")
    print("="*70)


if __name__ == "__main__":
    main()
