# CrossModal_fmri_v11 - Executable Script
# Generated from CrossModal_fmri_v11.ipynb

# Libraries imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import copy
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                            roc_auc_score, mean_squared_error, mean_absolute_error, r2_score)
from sklearn.utils.class_weight import compute_class_weight

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

# Plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')

print("fMRI Pipeline imports loaded successfully")


# CONFIGURATION
class fMRIConfig:
    def __init__(self, base_path: str = 'E:\Intermediate\BACON_ERIC\Head_neck\fMRI\Neck-Tumor_data\PATIENTS'):
        self.base_path = Path(base_path)
        self.data_dir = self.base_path
        self.label_path = self.base_path / 'DATA' / 'labels'
        self.subject_list = list(range(1, 33))
        self.activation_types = ['sensory', 'AN', 'LN', 'cognitive', 'DMN']
        self.connectivity_types = ['DMN']
        self.agg_method = 'both'

        self.hidden_dim = 64
        self.fusion_dim = 128
        self.dropout = 0.4
        self.num_classes = 2

        self.batch_size = 8
        self.num_epochs = 100
        self.learning_rate = 1e-4
        self.weight_decay = 1e-4
        self.patience = 15
        self.n_splits = 5
        self.val_ratio = 0.15  # Validation split ratio from training set
        self.grad_clip = 1.0

        self.output_dir = Path('./results_fmri')
        self.checkpoint_dir = Path('./checkpoints_fmri')
        self.log_dir = Path('./logs_fmri')
        for dir_path in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def __repr__(self):
        return (f"fMRIConfig(subjects={len(self.subject_list)}, "
                f"activation={self.activation_types}, "
                f"connectivity={self.connectivity_types}, "
                f"agg={self.agg_method}, val_ratio={self.val_ratio})")


# DATA LOADING
def load_activation_features(data_dir: Path, subject_list: List[int], activation_types: List[str],
                             agg_method: str = 'mean') -> Dict[int, torch.Tensor]:
    features = {}
    missing_files = []
    for subj in tqdm(subject_list, desc="Loading activation features"):
        subj_features = []
        subj_dir = data_dir / f'sub-{subj}'
        for act_type in activation_types:
            filepath = subj_dir / f"subject_{subj}_activation_{act_type}.csv"
            if not filepath.exists():
                missing_files.append(str(filepath))
                continue
            try:
                df = pd.read_csv(filepath)
                if 'Subject' in df.columns:
                    df = df.drop('Subject', axis=1)
                data = df.values.astype(np.float32)
                data = np.nan_to_num(data, nan=0.0)
                if agg_method == 'mean':
                    agg_data = np.mean(data, axis=0)
                elif agg_method == 'std':
                    agg_data = np.std(data, axis=0)
                elif agg_method == 'both':
                    agg_data = np.concatenate([np.mean(data, axis=0), np.std(data, axis=0)])
                else:
                    raise ValueError(f"Unknown aggregation method: {agg_method}")
                subj_features.append(agg_data)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                continue
        if subj_features:
            concat_features = np.concatenate(subj_features)
            features[subj] = torch.tensor(concat_features, dtype=torch.float32)
    print(f"\nActivation Features Summary:")
    print(f"  Loaded: {len(features)}/{len(subject_list)} subjects")
    if features:
        sample_feat = list(features.values())[0]
        print(f"  Feature dimension: {sample_feat.shape[0]}")
    if missing_files:
        print(f"  Missing files: {len(missing_files)}")
    return features


def load_connectivity_features(data_dir: Path, subject_list: List[int], connectivity_types: List[str]) -> Dict[int, torch.Tensor]:
    features = {}
    missing_files = []
    for subj in tqdm(subject_list, desc="Loading connectivity features"):
        subj_features = []
        subj_dir = data_dir / f'sub-{subj}'
        for conn_type in connectivity_types:
            filepath = subj_dir / f"subject_{subj}_fdr_PPI_Connectivity_{conn_type}.csv"
            if not filepath.exists():
                missing_files.append(str(filepath))
                continue
            try:
                df = pd.read_csv(filepath)
                if 'Subject' in df.columns:
                    df = df.drop('Subject', axis=1)
                data = df.values.astype(np.float32).flatten()
                data = np.nan_to_num(data, nan=0.0)
                subj_features.append(data)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                continue
        if subj_features:
            concat_features = np.concatenate(subj_features)
            features[subj] = torch.tensor(concat_features, dtype=torch.float32)
    print(f"\nConnectivity Features Summary:")
    print(f"  Loaded: {len(features)}/{len(subject_list)} subjects")
    if features:
        sample_feat = list(features.values())[0]
        print(f"  Feature dimension: {sample_feat.shape[0]}")
    if missing_files:
        print(f"  Missing files: {len(missing_files)}")
    return features


def load_labels(label_path: Path, subject_list: List[int], binary: bool = True) -> Tuple[Dict[int, int], Optional[Dict[int, float]]]:
    class_labels = {}
    reg_labels = {}
    label_files = [label_path / 'labels.csv', label_path / 'outcomes.csv',
                   label_path / 'subjects_labels.csv', label_path.parent / 'labels.csv']
    label_file = None
    for lf in label_files:
        if lf.exists():
            label_file = lf
            break
    if label_file is None:
        print("\n  WARNING: No label file found. Using dummy labels for testing.")
        for subj in subject_list:
            class_labels[subj] = np.random.randint(0, 2)
            reg_labels[subj] = np.random.randn()
        return class_labels, reg_labels
    df = pd.read_csv(label_file)
    print(f"\nLabels loaded from: {label_file}")
    print(f"  Columns: {df.columns.tolist()}")
    subj_col = None
    for col in ['Subject', 'subject', 'SubjectID', 'subject_id', 'ID', 'id']:
        if col in df.columns:
            subj_col = col
            break
    label_col = None
    for col in ['Label', 'label', 'Outcome', 'outcome', 'Class', 'class', 'Group', 'group']:
        if col in df.columns:
            label_col = col
            break
    reg_col = None
    for col in ['Score', 'score', 'Value', 'value', 'Continuous', 'continuous']:
        if col in df.columns:
            reg_col = col
            break
    if not subj_col or not label_col:
        raise ValueError(f"Could not identify subject or label columns in {label_file}")
    for _, row in df.iterrows():
        subj = int(row[subj_col])
        if subj not in subject_list:
            continue
        label = row[label_col]
        if binary:
            if isinstance(label, str):
                label = 1 if label.lower() in ['good', 'positive', 'yes', '1'] else 0
            else:
                label = int(label)
        class_labels[subj] = label
        if reg_col and reg_col in row:
            reg_labels[subj] = float(row[reg_col])
    print(f"\nLabel Summary:")
    print(f"  Classification labels: {len(class_labels)}")
    print(f"  Unique classes: {set(class_labels.values())}")
    if reg_labels:
        print(f"  Regression labels: {len(reg_labels)}")
    return class_labels, reg_labels if reg_labels else None


# DATASET
class fMRIDataset(Dataset):
    def __init__(self, activation_features: Dict[int, torch.Tensor], connectivity_features: Dict[int, torch.Tensor],
                 class_labels: Dict[int, int], reg_labels: Optional[Dict[int, float]] = None, transform=None):
        self.transform = transform
        self.samples = []
        act_subjects = set(activation_features.keys())
        conn_subjects = set(connectivity_features.keys())
        label_subjects = set(class_labels.keys())
        common_subjects = act_subjects & conn_subjects & label_subjects

        print(f"\nDataset Construction:")
        print(f"  Activation subjects: {len(act_subjects)}")
        print(f"  Connectivity subjects: {len(conn_subjects)}")
        print(f"  Labeled subjects: {len(label_subjects)}")
        print(f"  Complete data: {len(common_subjects)}")

        for subj in sorted(common_subjects):
            sample = {
                'activation': activation_features[subj],
                'connectivity': connectivity_features[subj],
                'class_label': class_labels[subj],
                'reg_label': reg_labels[subj] if reg_labels and subj in reg_labels else 0.0,
                'subject': subj
            }
            self.samples.append(sample)
        print(f"  Final dataset size: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        activation = sample['activation']
        connectivity = sample['connectivity']
        class_label = sample['class_label']
        reg_label = sample['reg_label']
        subject = sample['subject']
        if self.transform:
            activation = self.transform(activation)
            connectivity = self.transform(connectivity)
        return activation, connectivity, class_label, reg_label, subject


def collate_fmri(batch):
    """Collate function for fMRI DataLoader."""
    activations = torch.stack([item[0] for item in batch])
    connectivities = torch.stack([item[1] for item in batch])
    class_labels = torch.tensor([item[2] for item in batch], dtype=torch.long)
    reg_labels = torch.tensor([item[3] for item in batch], dtype=torch.float32)
    subjects = [item[4] for item in batch]
    return activations, connectivities, class_labels, reg_labels, subjects

print("Dataset classes defined")


# MODEL ENCODER & COMPONENTS
class ActivationEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.encoder(x)


class ConnectivityEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.encoder(x)

print("Encoder components defined")


# UNIMODAL MODELS
class fMRIActivationOnly(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, num_classes: int = 2,
                 dropout: float = 0.4, task: str = 'classification'):
        super().__init__()
        self.task = task
        self.encoder = ActivationEncoder(in_dim, hidden_dim, dropout)
        if task == 'classification':
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_classes)
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            )

    def forward(self, activation, connectivity=None):
        feat = self.encoder(activation)
        output = self.head(feat)
        if self.task == 'regression':
            output = output.squeeze(-1)
        return output


class fMRIConnectivityOnly(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, num_classes: int = 2,
                 dropout: float = 0.4, task: str = 'classification'):
        super().__init__()
        self.task = task
        self.encoder = ConnectivityEncoder(in_dim, hidden_dim, dropout)
        if task == 'classification':
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_classes)
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            )

    def forward(self, activation=None, connectivity=None):
        feat = self.encoder(connectivity)
        output = self.head(feat)
        if self.task == 'regression':
            output = output.squeeze(-1)
        return output

print("Unimodal models defined")


# FUSION MODEL
class fMRIFusionNet(nn.Module):
    def __init__(self, activation_dim: int, connectivity_dim: int, hidden_dim: int = 64,
                 num_classes: int = 2, dropout: float = 0.4, task: str = 'classification'):
        super().__init__()
        self.task = task
        self.activation_encoder = ActivationEncoder(activation_dim, hidden_dim, dropout)
        self.connectivity_encoder = ConnectivityEncoder(connectivity_dim, hidden_dim, dropout)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.activation_weight = nn.Parameter(torch.ones(1) * 0.5)
        self.connectivity_weight = nn.Parameter(torch.ones(1) * 0.5)

        if task == 'classification':
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_classes)
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            )

    def forward(self, activation, connectivity, return_features=False):
        act_feat = self.activation_encoder(activation)
        conn_feat = self.connectivity_encoder(connectivity)
        weights = F.softmax(torch.stack([self.activation_weight, self.connectivity_weight]), dim=0)
        weighted_act = act_feat * weights[0]
        weighted_conn = conn_feat * weights[1]
        combined = torch.cat([weighted_act, weighted_conn], dim=1)
        fused = self.fusion(combined)
        output = self.head(fused)
        if self.task == 'regression':
            output = output.squeeze(-1)
        if return_features:
            return output, fused
        return output

    def get_fusion_weights(self):
        with torch.no_grad():
            weights = F.softmax(torch.stack([self.activation_weight, self.connectivity_weight]), dim=0)
            return {
                'activation': weights[0].item(),
                'connectivity': weights[1].item()
            }

print("Fusion model defined")


# TRAINING FUNCTIONS
def train_epoch(model, train_loader, optimizer, criterion, device, task='classification', grad_clip=1.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        activation, connectivity, class_labels, reg_labels, _ = batch
        activation = activation.to(device)
        connectivity = connectivity.to(device)
        if task == 'classification':
            labels = class_labels.to(device)
        else:
            labels = reg_labels.to(device)
        optimizer.zero_grad()
        outputs = model(activation, connectivity)
        loss = criterion(outputs, labels)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def evaluate(model, data_loader, device, task='classification', num_classes=2):
    """Evaluate model on given data loader."""
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch in data_loader:
            activation, connectivity, class_labels, reg_labels, _ = batch
            activation = activation.to(device)
            connectivity = connectivity.to(device)
            outputs = model(activation, connectivity)

            if task == 'classification':
                probs = F.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)
                targets = class_labels
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.numpy())
                all_probs.extend(probs.cpu().numpy())
            else:
                preds = outputs
                targets = reg_labels
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    if task == 'classification':
        all_probs = np.array(all_probs)
        metrics = {
            'Accuracy': accuracy_score(all_targets, all_preds),
            'F1': f1_score(all_targets, all_preds, average='weighted', zero_division=0),
            'Precision': precision_score(all_targets, all_preds, average='weighted', zero_division=0),
            'Recall': recall_score(all_targets, all_preds, average='weighted', zero_division=0)
        }
        if num_classes == 2:
            try:
                metrics['AUC'] = roc_auc_score(all_targets, all_probs[:, 1])
            except:
                metrics['AUC'] = 0.5
        return metrics, all_targets, all_probs
    else:
        metrics = {
            'MSE': mean_squared_error(all_targets, all_preds),
            'RMSE': np.sqrt(mean_squared_error(all_targets, all_preds)),
            'MAE': mean_absolute_error(all_targets, all_preds),
            'R2': r2_score(all_targets, all_preds)
        }
        return metrics, all_targets, all_preds

print("Training functions defined")


# VISUALIZATION & RESULTS FUNCTIONS
def create_results_dataframe(results: Dict, task: str, fusion_weights: List[Dict] = None) -> pd.DataFrame:
    """Create a comprehensive DataFrame from experiment results."""
    rows = []

    for model_name, model_results in results.items():
        for fold_idx, fold_metrics in enumerate(model_results, 1):
            row = {
                'Model': model_name.replace('_', ' ').title(),
                'Fold': fold_idx,
            }
            row.update(fold_metrics)
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def create_summary_dataframe(results: Dict, task: str) -> pd.DataFrame:
    """Create summary statistics DataFrame (mean +/- std)."""
    summary_rows = []

    if task == 'classification':
        metrics = ['Accuracy', 'F1', 'Precision', 'Recall', 'AUC']
    else:
        metrics = ['R2', 'RMSE', 'MAE', 'MSE']

    for model_name, model_results in results.items():
        if not model_results:
            continue
        row = {'Model': model_name.replace('_', ' ').title()}
        for metric in metrics:
            if metric in model_results[0]:
                values = [r[metric] for r in model_results]
                row[f'{metric}_mean'] = np.mean(values)
                row[f'{metric}_std'] = np.std(values)
                row[f'{metric}'] = f"{np.mean(values):.4f} +/- {np.std(values):.4f}"
        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def plot_model_comparison(results: Dict, task: str, output_dir: Path, timestamp: str):
    """Plot bar charts comparing models across metrics."""
    if task == 'classification':
        metrics = ['Accuracy', 'F1', 'Precision', 'Recall', 'AUC']
    else:
        metrics = ['R2', 'RMSE', 'MAE']

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    colors = sns.color_palette('husl', len(results))
    model_names = [name.replace('_', ' ').title() for name in results.keys()]

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        means = []
        stds = []

        for model_name, model_results in results.items():
            if model_results and metric in model_results[0]:
                values = [r[metric] for r in model_results]
                means.append(np.mean(values))
                stds.append(np.std(values))
            else:
                means.append(0)
                stds.append(0)

        x = np.arange(len(model_names))
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor='black', alpha=0.8)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f'{metric}', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=10)
        ax.set_ylim(0, max(means) * 1.3 if max(means) > 0 else 1)

        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                   f'{mean:.3f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle(f'Model Comparison - {task.title()}', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save figure
    save_path = output_dir / f'model_comparison_{task}_{timestamp}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.close()

    return fig


def plot_fold_performance(df: pd.DataFrame, task: str, output_dir: Path, timestamp: str):
    """Plot performance across folds for each model."""
    if task == 'classification':
        primary_metric = 'F1'
    else:
        primary_metric = 'R2'

    fig, ax = plt.subplots(figsize=(10, 6))

    models = df['Model'].unique()
    colors = sns.color_palette('husl', len(models))

    for i, model in enumerate(models):
        model_data = df[df['Model'] == model]
        ax.plot(model_data['Fold'], model_data[primary_metric],
                marker='o', linewidth=2, markersize=8,
                label=model, color=colors[i])

    ax.set_xlabel('Fold', fontsize=12)
    ax.set_ylabel(primary_metric, fontsize=12)
    ax.set_title(f'{primary_metric} Across Folds - {task.title()}', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.set_xticks(df['Fold'].unique())
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = output_dir / f'fold_performance_{task}_{timestamp}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.close()

    return fig


def plot_fusion_weights(fusion_weights: List[Dict], output_dir: Path, timestamp: str):
    """Plot fusion weights across folds."""
    if not fusion_weights:
        print("No fusion weights to plot")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    folds = list(range(1, len(fusion_weights) + 1))
    act_weights = [fw['activation'] for fw in fusion_weights]
    conn_weights = [fw['connectivity'] for fw in fusion_weights]

    # Line plot
    ax1 = axes[0]
    ax1.plot(folds, act_weights, marker='o', linewidth=2, label='Activation', color='#2ecc71')
    ax1.plot(folds, conn_weights, marker='s', linewidth=2, label='Connectivity', color='#e74c3c')
    ax1.set_xlabel('Fold', fontsize=12)
    ax1.set_ylabel('Weight', fontsize=12)
    ax1.set_title('Fusion Weights Across Folds', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.set_xticks(folds)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    # Bar plot (average)
    ax2 = axes[1]
    x = ['Activation', 'Connectivity']
    means = [np.mean(act_weights), np.mean(conn_weights)]
    stds = [np.std(act_weights), np.std(conn_weights)]
    colors = ['#2ecc71', '#e74c3c']

    bars = ax2.bar(x, means, yerr=stds, capsize=10, color=colors, edgecolor='black', alpha=0.8)
    ax2.set_ylabel('Average Weight', fontsize=12)
    ax2.set_title('Average Fusion Weights', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 1)

    for bar, mean, std in zip(bars, means, stds):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()

    save_path = output_dir / f'fusion_weights_{timestamp}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.close()

    return fig


def save_results(results_df: pd.DataFrame, summary_df: pd.DataFrame,
                 fusion_weights: List[Dict], output_dir: Path, task: str, timestamp: str):
    """Save all results to CSV files."""
    # Save detailed results
    detailed_path = output_dir / f'detailed_results_{task}_{timestamp}.csv'
    results_df.to_csv(detailed_path, index=False)
    print(f"Saved: {detailed_path}")

    # Save summary
    summary_path = output_dir / f'summary_results_{task}_{timestamp}.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")

    # Save fusion weights
    if fusion_weights:
        fw_df = pd.DataFrame(fusion_weights)
        fw_df['Fold'] = range(1, len(fusion_weights) + 1)
        fw_path = output_dir / f'fusion_weights_{timestamp}.csv'
        fw_df.to_csv(fw_path, index=False)
        print(f"Saved: {fw_path}")

print("Visualization and results functions defined")


# MAIN EXPERIMENT PIPELINE (with proper train/val/test split)
def run_experiment(dataset, config, task='classification'):
    """Run experiment with proper train/validation/test split.

    CHANGE v11: Fixed data leakage issue by splitting training data into
    train and validation sets. Early stopping and LR scheduling now use
    validation set instead of test set.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {task.upper()}")
    print(f"Device: {device}")
    print(f"Validation ratio: {config.val_ratio}")
    print(f"{'='*70}\n")

    if task == 'classification':
        labels = np.array([s['class_label'] for s in dataset.samples])
        num_classes = len(np.unique(labels))
    else:
        labels = np.array([s['reg_label'] for s in dataset.samples])
        num_classes = 1

    sample = dataset[0]
    activation_dim = sample[0].shape[0]
    connectivity_dim = sample[1].shape[0]

    print(f"Data Summary:")
    print(f"  Activation dim: {activation_dim}")
    print(f"  Connectivity dim: {connectivity_dim}")
    print(f"  Samples: {len(dataset)}")
    if task == 'classification':
        print(f"  Classes: {num_classes}")
        print(f"  Class distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
    print()

    # Setup cross-validation
    if task == 'classification':
        cv = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=SEED)
        splits = list(cv.split(np.zeros(len(dataset)), labels))
    else:
        cv = KFold(n_splits=config.n_splits, shuffle=True, random_state=SEED)
        splits = list(cv.split(np.zeros(len(dataset))))

    results = {'fusion': [], 'activation_only': [], 'connectivity_only': []}
    fusion_weights_all = []

    # Cross-validation loop
    for fold_idx, (train_val_idx, test_idx) in enumerate(splits, 1):
        print(f"\n{'='*70}")
        print(f"FOLD {fold_idx}/{config.n_splits}")
        print(f"{'='*70}")

        # Split train_val into train and validation
        train_val_labels = labels[train_val_idx]
        if task == 'classification':
            train_idx, val_idx = train_test_split(
                np.arange(len(train_val_idx)),
                test_size=config.val_ratio,
                stratify=train_val_labels,
                random_state=SEED + fold_idx
            )
        else:
            train_idx, val_idx = train_test_split(
                np.arange(len(train_val_idx)),
                test_size=config.val_ratio,
                random_state=SEED + fold_idx
            )

        # Map back to original indices
        actual_train_idx = train_val_idx[train_idx]
        actual_val_idx = train_val_idx[val_idx]

        print(f"  Train: {len(actual_train_idx)}, Val: {len(actual_val_idx)}, Test: {len(test_idx)}")

        # Create data loaders
        train_subset = Subset(dataset, actual_train_idx)
        val_subset = Subset(dataset, actual_val_idx)
        test_subset = Subset(dataset, test_idx)

        train_loader = DataLoader(train_subset, batch_size=config.batch_size,
                                  shuffle=True, collate_fn=collate_fmri)
        val_loader = DataLoader(val_subset, batch_size=config.batch_size,
                                shuffle=False, collate_fn=collate_fmri)
        test_loader = DataLoader(test_subset, batch_size=config.batch_size,
                                 shuffle=False, collate_fn=collate_fmri)

        # Setup loss with class weights
        if task == 'classification':
            train_labels = labels[actual_train_idx]
            class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
            class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        else:
            criterion = nn.MSELoss()

        # Train each model
        for model_name in ['fusion', 'activation_only', 'connectivity_only']:
            print(f"\n--- Training {model_name.upper().replace('_', ' ')} ---")

            # Initialize model
            if model_name == 'fusion':
                model = fMRIFusionNet(
                    activation_dim=activation_dim, connectivity_dim=connectivity_dim,
                    hidden_dim=config.hidden_dim, num_classes=num_classes,
                    dropout=config.dropout, task=task
                ).to(device)
            elif model_name == 'activation_only':
                model = fMRIActivationOnly(
                    in_dim=activation_dim, hidden_dim=config.hidden_dim,
                    num_classes=num_classes, dropout=config.dropout, task=task
                ).to(device)
            else:
                model = fMRIConnectivityOnly(
                    in_dim=connectivity_dim, hidden_dim=config.hidden_dim,
                    num_classes=num_classes, dropout=config.dropout, task=task
                ).to(device)

            optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate,
                                    weight_decay=config.weight_decay)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                              factor=0.5, patience=5)

            best_val_metric = -np.inf if task == 'classification' else np.inf
            best_state = None
            patience_counter = 0

            for epoch in range(1, config.num_epochs + 1):
                # Train
                train_loss = train_epoch(model, train_loader, optimizer, criterion,
                                        device, task, config.grad_clip)

                # Evaluate on VALIDATION set for early stopping
                val_metrics, _, _ = evaluate(model, val_loader, device, task, num_classes)

                # Scheduler step on validation performance
                if task == 'classification':
                    scheduler.step(1 - val_metrics['F1'])
                    current_val_metric = val_metrics['F1']
                    is_better = current_val_metric > best_val_metric
                else:
                    scheduler.step(-val_metrics['R2'])
                    current_val_metric = val_metrics['R2']
                    is_better = current_val_metric > best_val_metric

                # Logging
                if epoch % 10 == 0:
                    if task == 'classification':
                        print(f"  Epoch {epoch:3d}: Loss={train_loss:.4f}, "
                              f"Val_F1={val_metrics['F1']:.4f}, Val_Acc={val_metrics['Accuracy']:.4f}")
                    else:
                        print(f"  Epoch {epoch:3d}: Loss={train_loss:.4f}, "
                              f"Val_R2={val_metrics['R2']:.4f}, Val_RMSE={val_metrics['RMSE']:.4f}")

                # Early stopping based on validation
                if is_better:
                    best_val_metric = current_val_metric
                    best_state = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= config.patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break

            # Load best model and evaluate on TEST set (held out)
            if best_state:
                model.load_state_dict(best_state)

            # Final evaluation on TEST set only (no data leakage)
            test_metrics, _, _ = evaluate(model, test_loader, device, task, num_classes)
            results[model_name].append(test_metrics)

            # Store fusion weights
            if model_name == 'fusion' and hasattr(model, 'get_fusion_weights'):
                fw = model.get_fusion_weights()
                fusion_weights_all.append(fw)
                print(f"  Fusion weights: Activation={fw['activation']:.3f}, "
                      f"Connectivity={fw['connectivity']:.3f}")

            # Print test metrics
            if task == 'classification':
                print(f"  Test: Acc={test_metrics['Accuracy']:.4f}, "
                      f"F1={test_metrics['F1']:.4f}, "
                      f"Precision={test_metrics['Precision']:.4f}")
            else:
                print(f"  Test: R2={test_metrics['R2']:.4f}, "
                      f"RMSE={test_metrics['RMSE']:.4f}, "
                      f"MAE={test_metrics['MAE']:.4f}")

    # Print summary
    print(f"\n{'='*70}")
    print("FINAL RESULTS SUMMARY (Test Set)")
    print(f"{'='*70}\n")

    for model_name, model_results in results.items():
        if model_results:
            print(f"{model_name.upper().replace('_', ' ')}:")
            if task == 'classification':
                for metric in ['Accuracy', 'F1', 'Precision', 'Recall', 'AUC']:
                    if metric in model_results[0]:
                        values = [r[metric] for r in model_results]
                        print(f"  {metric:12s}: {np.mean(values):.4f} +/- {np.std(values):.4f}")
            else:
                for metric in ['R2', 'RMSE', 'MAE']:
                    values = [r[metric] for r in model_results]
                    print(f"  {metric:12s}: {np.mean(values):.4f} +/- {np.std(values):.4f}")
            print()

    # Fusion weights summary
    if fusion_weights_all:
        print("FUSION WEIGHTS SUMMARY:")
        act_weights = [fw['activation'] for fw in fusion_weights_all]
        conn_weights = [fw['connectivity'] for fw in fusion_weights_all]
        print(f"  Activation:   {np.mean(act_weights):.4f} +/- {np.std(act_weights):.4f}")
        print(f"  Connectivity: {np.mean(conn_weights):.4f} +/- {np.std(conn_weights):.4f}")

    return results, fusion_weights_all


# MAIN EXECUTION
def main():
    """Main execution function with visualization and results saving."""
    config = fMRIConfig()
    print(config)
    print()

    # Generate timestamp for output files
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Load data
    print("Loading fMRI features...")
    activation_feats = load_activation_features(
        config.data_dir, config.subject_list,
        activation_types=config.activation_types, agg_method=config.agg_method
    )
    connectivity_feats = load_connectivity_features(
        config.data_dir, config.subject_list,
        connectivity_types=config.connectivity_types
    )
    class_labels, reg_labels = load_labels(config.label_path, config.subject_list)

    dataset = fMRIDataset(activation_feats, connectivity_feats, class_labels, reg_labels)

    # Run classification experiment
    print("\n" + "="*70)
    print("RUNNING CLASSIFICATION EXPERIMENT")
    print("="*70)
    clf_results, clf_fusion_weights = run_experiment(dataset, config, task='classification')

    # Create DataFrames for classification
    clf_detailed_df = create_results_dataframe(clf_results, 'classification')
    clf_summary_df = create_summary_dataframe(clf_results, 'classification')

    print("\n" + "="*70)
    print("CLASSIFICATION RESULTS DATAFRAME")
    print("="*70)
    print("\nDetailed Results (per fold):")
    print(clf_detailed_df.to_string(index=False))
    print("\nSummary Statistics:")
    print(clf_summary_df[['Model', 'Accuracy', 'F1', 'Precision', 'Recall']].to_string(index=False))

    # Save classification results
    save_results(clf_detailed_df, clf_summary_df, clf_fusion_weights,
                 config.output_dir, 'classification', timestamp)

    # Plot classification results
    print("\n" + "="*70)
    print("GENERATING CLASSIFICATION PLOTS")
    print("="*70)
    plot_model_comparison(clf_results, 'classification', config.output_dir, timestamp)
    plot_fold_performance(clf_detailed_df, 'classification', config.output_dir, timestamp)
    if clf_fusion_weights:
        plot_fusion_weights(clf_fusion_weights, config.output_dir, timestamp)

    # Run regression experiment if applicable
    if reg_labels and len(set(reg_labels.values())) > 1:
        print("\n" + "="*70)
        print("RUNNING REGRESSION EXPERIMENT")
        print("="*70)
        reg_results, reg_fusion_weights = run_experiment(dataset, config, task='regression')

        # Create DataFrames for regression
        reg_detailed_df = create_results_dataframe(reg_results, 'regression')
        reg_summary_df = create_summary_dataframe(reg_results, 'regression')

        print("\n" + "="*70)
        print("REGRESSION RESULTS DATAFRAME")
        print("="*70)
        print("\nDetailed Results (per fold):")
        print(reg_detailed_df.to_string(index=False))
        print("\nSummary Statistics:")
        print(reg_summary_df[['Model', 'R2', 'RMSE', 'MAE']].to_string(index=False))

        # Save regression results
        save_results(reg_detailed_df, reg_summary_df, reg_fusion_weights,
                     config.output_dir, 'regression', timestamp)

        # Plot regression results
        print("\n" + "="*70)
        print("GENERATING REGRESSION PLOTS")
        print("="*70)
        plot_model_comparison(reg_results, 'regression', config.output_dir, timestamp)
        plot_fold_performance(reg_detailed_df, 'regression', config.output_dir, timestamp)
    else:
        print("\nSkipping regression experiment (no valid regression labels)")

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {config.output_dir}")
    print(f"Timestamp: {timestamp}")


if __name__ == "__main__":
    main()
