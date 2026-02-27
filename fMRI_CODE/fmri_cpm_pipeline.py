"""fMRI CPM pipeline — matrix builders and biomarker logging.

Reuses the core CPM algorithm from eeg_cpm_pipeline (build_cpm_mask,
compute_css). Adds fMRI-specific matrix construction and network mapping.

Usage:
    Imported by fmri_cpm_loso.py and fmri_cpm_fusion.py.
"""

import sys
import logging
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# Import shared CPM engine
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from EEG_CODE.eeg_cpm_pipeline import BiomarkerLog

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# fMRI network mapping
# ---------------------------------------------------------------------------

FMRI_NETWORK_MAP = {
    # Activation type → canonical network label
    'DMN': 'Default Mode',
    'sensory': 'Sensory',
    'AN': 'Auditory',
    'LN': 'Language',
    'cognitive': 'Cognitive Control',
}

# Default activation and connectivity types
DEFAULT_ACTIVATION_TYPES = ['sensory', 'AN', 'LN', 'cognitive', 'DMN']
DEFAULT_CONNECTIVITY_TYPES = ['DMN']


# ---------------------------------------------------------------------------
# Matrix builders
# ---------------------------------------------------------------------------

def build_activation_matrix(act_feats, subject_list, labels):
    """Convert fMRI activation features (torch tensors) to a subject matrix.

    Args:
        act_feats: Dict[int, torch.Tensor] from load_activation_features.
        subject_list: List[int] of subject IDs.
        labels: Dict[int, int] subject → label.

    Returns:
        matrix_df: DataFrame (N_valid, n_features) with ROI-named columns.
        labels_arr: ndarray (N_valid,).
        valid_subjects: List[int].
    """
    valid_subjects = []
    vectors = []

    for subj in sorted(subject_list):
        if subj not in act_feats or subj not in labels:
            continue
        feat = act_feats[subj]
        if isinstance(feat, torch.Tensor):
            feat = feat.numpy()
        vectors.append(feat.flatten().astype(np.float32))
        valid_subjects.append(subj)

    if not vectors:
        logger.warning('No valid subjects for activation matrix.')
        return pd.DataFrame(), np.array([]), []

    matrix = np.stack(vectors)
    n_feat = matrix.shape[1]

    # Build ROI-based column names
    # With 'both' aggregation: [mean_sensory..., std_sensory..., mean_AN..., ...]
    col_names = [f'fMRI_ACT_{i}' for i in range(n_feat)]

    labels_arr = np.array([labels[s] for s in valid_subjects])
    matrix_df = pd.DataFrame(matrix, columns=col_names, index=valid_subjects)

    logger.info(f'Activation matrix: {matrix_df.shape[0]} subjects, '
                f'{matrix_df.shape[1]} features')
    return matrix_df, labels_arr, valid_subjects


def build_connectivity_matrix(conn_feats, subject_list, labels,
                               roi_names=None):
    """Convert fMRI connectivity features to a subject matrix.

    Args:
        conn_feats: Dict[int, torch.Tensor] from load_connectivity_features.
        subject_list: List[int] of subject IDs.
        labels: Dict[int, int] subject → label.
        roi_names: optional list of ROI names for column naming.

    Returns:
        matrix_df: DataFrame (N_valid, n_features) with edge columns.
        labels_arr: ndarray (N_valid,).
        valid_subjects: List[int].
    """
    valid_subjects = []
    vectors = []

    for subj in sorted(subject_list):
        if subj not in conn_feats or subj not in labels:
            continue
        feat = conn_feats[subj]
        if isinstance(feat, torch.Tensor):
            feat = feat.numpy()
        vectors.append(feat.flatten().astype(np.float32))
        valid_subjects.append(subj)

    if not vectors:
        logger.warning('No valid subjects for connectivity matrix.')
        return pd.DataFrame(), np.array([]), []

    matrix = np.stack(vectors)
    n_feat = matrix.shape[1]

    # Build edge column names
    if roi_names is not None:
        n_rois = len(roi_names)
        col_names = []
        idx = 0
        for i in range(n_rois):
            for j in range(n_rois):
                if idx < n_feat:
                    col_names.append(f'fMRI_CONN_{roi_names[i]}--{roi_names[j]}')
                    idx += 1
        # Pad if needed
        while len(col_names) < n_feat:
            col_names.append(f'fMRI_CONN_{len(col_names)}')
    else:
        col_names = [f'fMRI_CONN_{i}' for i in range(n_feat)]

    labels_arr = np.array([labels[s] for s in valid_subjects])
    matrix_df = pd.DataFrame(matrix, columns=col_names[:n_feat],
                             index=valid_subjects)

    logger.info(f'Connectivity matrix: {matrix_df.shape[0]} subjects, '
                f'{matrix_df.shape[1]} features')
    return matrix_df, labels_arr, valid_subjects


# ---------------------------------------------------------------------------
# fMRI-specific BiomarkerLog
# ---------------------------------------------------------------------------

class fMRIBiomarkerLog(BiomarkerLog):
    """Extended BiomarkerLog with fMRI network-level summaries."""

    def __init__(self, n_features, feature_names=None,
                 network_map=None):
        super().__init__(n_features, feature_names)
        self.network_map = network_map or FMRI_NETWORK_MAP

    def get_roi_degree(self, roi_names=None):
        """Compute per-ROI degree from connectivity features.

        Args:
            roi_names: list of ROI name strings.

        Returns:
            DataFrame with columns: roi, degree_pos, degree_neg.
        """
        if roi_names is None:
            # Extract unique ROIs from feature names
            roi_set = set()
            for name in self.feature_names:
                if '--' in name:
                    parts = name.replace('fMRI_CONN_', '').split('--')
                    roi_set.update(parts)
            roi_names = sorted(roi_set) if roi_set else []

        if not roi_names:
            return pd.DataFrame(columns=['roi', 'degree_pos', 'degree_neg'])

        degree_pos = {r: 0.0 for r in roi_names}
        degree_neg = {r: 0.0 for r in roi_names}

        stability = self.compute_stability()
        for _, row in stability.iterrows():
            name = row['feature_name']
            if '--' not in name:
                continue
            clean = name.replace('fMRI_CONN_', '')
            parts = clean.split('--')
            if len(parts) != 2:
                continue
            roi_a, roi_b = parts
            if roi_a in degree_pos:
                degree_pos[roi_a] += row['stability_pos']
                degree_neg[roi_a] += row['stability_neg']
            if roi_b in degree_pos:
                degree_pos[roi_b] += row['stability_pos']
                degree_neg[roi_b] += row['stability_neg']

        return pd.DataFrame({
            'roi': roi_names,
            'degree_pos': [degree_pos[r] for r in roi_names],
            'degree_neg': [degree_neg[r] for r in roi_names],
        })

    def get_network_summary(self):
        """Aggregate stability by fMRI network.

        Returns:
            DataFrame with columns: network, mean_stability_pos, mean_stability_neg.
        """
        stability = self.compute_stability()

        # Map features to networks
        network_stab = {}
        for _, row in stability.iterrows():
            name = row['feature_name']
            # Try to identify network from feature name
            matched_network = None
            for net_key, net_label in self.network_map.items():
                if net_key.lower() in name.lower():
                    matched_network = net_label
                    break
            if matched_network is None:
                matched_network = 'Other'

            if matched_network not in network_stab:
                network_stab[matched_network] = {'pos': [], 'neg': []}
            network_stab[matched_network]['pos'].append(row['stability_pos'])
            network_stab[matched_network]['neg'].append(row['stability_neg'])

        rows = []
        for net, vals in sorted(network_stab.items()):
            rows.append({
                'network': net,
                'mean_stability_pos': np.mean(vals['pos']),
                'mean_stability_neg': np.mean(vals['neg']),
                'n_features': len(vals['pos']),
            })

        return pd.DataFrame(rows)
