"""fMRI utility models and data loaders for the bridge pipeline.

Extracted from run_fmri_v11.py (models) and bridge notebook (data loaders).
Does NOT modify run_fmri_v11.py — this is a parallel importable module.
"""

import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Models (from run_fmri_v11.py lines 272-425)
# ---------------------------------------------------------------------------

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


class fMRIFusionNet(nn.Module):
    """fMRI fusion model matching checkpoint structure."""
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
            return {'activation': weights[0].item(), 'connectivity': weights[1].item()}


# ---------------------------------------------------------------------------
# Data loaders (adapted from bridge notebook cells 6)
# ---------------------------------------------------------------------------

def load_activation_features(data_dir, subject_list, activation_types, agg_method='both'):
    """Load fMRI activation features.

    Args:
        data_dir: Path to fMRI data directory containing sub-N folders.
        subject_list: List of subject IDs (ints).
        activation_types: List of activation type strings.
        agg_method: 'mean', 'std', or 'both'.

    Returns:
        Dict mapping subject ID to feature tensor.
    """
    data_dir = Path(data_dir)
    features = {}
    for subj in tqdm(subject_list, desc='Loading fMRI activations'):
        subj_features = []
        subj_dir = data_dir / f'sub-{subj}'
        for act_type in activation_types:
            filepath = subj_dir / f'subject_{subj}_activation_{act_type}.csv'
            if not filepath.exists():
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
                    raise ValueError(f'Unknown agg method: {agg_method}')
                subj_features.append(agg_data)
            except Exception as e:
                logger.warning(f'Error loading {filepath}: {e}')
        if subj_features:
            features[subj] = torch.tensor(np.concatenate(subj_features), dtype=torch.float32)
    logger.info(f'fMRI activation features: {len(features)}/{len(subject_list)} subjects')
    if features:
        sample = list(features.values())[0]
        logger.info(f'  Activation feature dim: {sample.shape[0]}')
    return features


def load_connectivity_features(data_dir, subject_list, connectivity_types):
    """Load fMRI connectivity features.

    Args:
        data_dir: Path to fMRI data directory containing sub-N folders.
        subject_list: List of subject IDs (ints).
        connectivity_types: List of connectivity type strings.

    Returns:
        Dict mapping subject ID to feature tensor.
    """
    data_dir = Path(data_dir)
    features = {}
    for subj in tqdm(subject_list, desc='Loading fMRI connectivity'):
        subj_features = []
        subj_dir = data_dir / f'sub-{subj}'
        for conn_type in connectivity_types:
            filepath = subj_dir / f'subject_{subj}_fdr_PPI_Connectivity_{conn_type}.csv'
            if not filepath.exists():
                continue
            try:
                df = pd.read_csv(filepath)
                if 'Subject' in df.columns:
                    df = df.drop('Subject', axis=1)
                data = df.values.astype(np.float32).flatten()
                data = np.nan_to_num(data, nan=0.0)
                subj_features.append(data)
            except Exception as e:
                logger.warning(f'Error loading {filepath}: {e}')
        if subj_features:
            features[subj] = torch.tensor(np.concatenate(subj_features), dtype=torch.float32)
    logger.info(f'fMRI connectivity features: {len(features)}/{len(subject_list)} subjects')
    if features:
        sample = list(features.values())[0]
        logger.info(f'  Connectivity feature dim: {sample.shape[0]}')
    return features


def load_fmri_labels(label_path, subject_list):
    """Load fMRI classification labels.

    Args:
        label_path: Path to labels directory.
        subject_list: List of subject IDs (ints).

    Returns:
        Dict[int, int] mapping subject ID to class label.
    """
    label_path = Path(label_path)
    label_files = [label_path / 'labels.csv', label_path / 'outcomes.csv',
                   label_path / 'subjects_labels.csv', label_path.parent / 'labels.csv']
    label_file = None
    for lf in label_files:
        if lf.exists():
            label_file = lf
            break
    if label_file is None:
        logger.warning('No fMRI label file found. Using dummy labels.')
        return {subj: np.random.randint(0, 2) for subj in subject_list}

    df = pd.read_csv(label_file)
    subj_col = next((c for c in ['Subject', 'subject', 'SubjectID', 'ID', 'id'] if c in df.columns), None)
    label_col = next((c for c in ['Label', 'label', 'Outcome', 'outcome', 'Class', 'class', 'Group', 'group'] if c in df.columns), None)
    if not subj_col or not label_col:
        raise ValueError(f'Cannot identify columns in {label_file}: {df.columns.tolist()}')

    class_labels = {}
    for _, row in df.iterrows():
        subj = int(row[subj_col])
        if subj not in subject_list:
            continue
        label = row[label_col]
        if isinstance(label, str):
            label = 1 if label.lower() in ['good', 'positive', 'yes', '1'] else 0
        else:
            label = int(label)
        class_labels[subj] = label
    logger.info(f'fMRI labels: {len(class_labels)} subjects, classes={set(class_labels.values())}')
    return class_labels
