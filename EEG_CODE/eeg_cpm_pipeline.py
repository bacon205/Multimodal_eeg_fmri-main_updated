"""Core Connectome-Based Predictive Modeling (CPM) engine.

Implements the CPM algorithm (Shen et al., Nature Protocols, 2017) for
small-sample neuroimaging studies. Shared by EEG and fMRI stages.

Functions:
    build_cpm_mask  — feature selection via univariate correlation
    compute_css     — summary score for a single subject
    compute_css_batch — vectorized summary scores for N subjects

Class:
    BiomarkerLog    — tracks edge masks across CV folds for stability analysis
"""

import logging
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CPM core functions
# ---------------------------------------------------------------------------

def build_cpm_mask(train_matrix, train_labels, p_threshold=0.01):
    """Select features correlated with the outcome in the training set.

    For each feature j, compute Pearson r with train_labels.
    Positive mask: r > 0 AND p < p_threshold
    Negative mask: r < 0 AND p < p_threshold

    Args:
        train_matrix: ndarray of shape (N_train, n_feat).
        train_labels: ndarray of shape (N_train,).
        p_threshold: significance threshold (default 0.01).

    Returns:
        pos_mask: boolean array (n_feat,) — positively correlated features.
        neg_mask: boolean array (n_feat,) — negatively correlated features.
        edge_stats: DataFrame with columns [feature_idx, r, p, direction].
    """
    n_feat = train_matrix.shape[1]
    r_values = np.zeros(n_feat)
    p_values = np.ones(n_feat)

    for j in range(n_feat):
        col = train_matrix[:, j]
        # Handle constant features (zero variance)
        if np.std(col) < 1e-12:
            r_values[j] = 0.0
            p_values[j] = 1.0
            continue
        r, p = pearsonr(col, train_labels)
        if np.isnan(r):
            r_values[j] = 0.0
            p_values[j] = 1.0
        else:
            r_values[j] = r
            p_values[j] = p

    pos_mask = (r_values > 0) & (p_values < p_threshold)
    neg_mask = (r_values < 0) & (p_values < p_threshold)

    edge_stats = pd.DataFrame({
        'feature_idx': np.arange(n_feat),
        'r': r_values,
        'p': p_values,
        'direction': np.where(pos_mask, 'pos', np.where(neg_mask, 'neg', 'ns'))
    })

    logger.debug(
        f'CPM mask: {pos_mask.sum()} pos, {neg_mask.sum()} neg '
        f'out of {n_feat} features (p<{p_threshold})'
    )
    return pos_mask, neg_mask, edge_stats


def compute_css(feature_vector, pos_mask, neg_mask):
    """Compute condensed summary scores for one subject.

    Args:
        feature_vector: ndarray of shape (n_feat,).
        pos_mask: boolean array (n_feat,).
        neg_mask: boolean array (n_feat,).

    Returns:
        css_pos: float — sum of positively-masked features.
        css_neg: float — sum of negatively-masked features.
    """
    css_pos = float(np.sum(feature_vector[pos_mask])) if pos_mask.any() else 0.0
    css_neg = float(np.sum(feature_vector[neg_mask])) if neg_mask.any() else 0.0
    return css_pos, css_neg


def compute_css_batch(feature_matrix, pos_mask, neg_mask):
    """Vectorized CSS computation for N subjects.

    Args:
        feature_matrix: ndarray of shape (N, n_feat).
        pos_mask: boolean array (n_feat,).
        neg_mask: boolean array (n_feat,).

    Returns:
        ndarray of shape (N, 2) — columns [css_pos, css_neg].
    """
    css_pos = feature_matrix[:, pos_mask].sum(axis=1) if pos_mask.any() else np.zeros(feature_matrix.shape[0])
    css_neg = feature_matrix[:, neg_mask].sum(axis=1) if neg_mask.any() else np.zeros(feature_matrix.shape[0])
    return np.column_stack([css_pos, css_neg])


# ---------------------------------------------------------------------------
# BiomarkerLog — track feature masks across CV folds
# ---------------------------------------------------------------------------

class BiomarkerLog:
    """Records CPM masks across cross-validation folds for stability analysis.

    Usage:
        log = BiomarkerLog(n_features=100, feature_names=['f0', ...])
        for fold in folds:
            pos_mask, neg_mask, stats = build_cpm_mask(X_train, y_train)
            log.record_fold(pos_mask, neg_mask, stats)
        stability_df = log.compute_stability()
    """

    def __init__(self, n_features, feature_names=None):
        """
        Args:
            n_features: total number of features.
            feature_names: optional list of feature name strings.
        """
        self.n_features = n_features
        self.feature_names = feature_names or [f'feat_{i}' for i in range(n_features)]
        self._pos_counts = np.zeros(n_features, dtype=int)
        self._neg_counts = np.zeros(n_features, dtype=int)
        self._r_sum = np.zeros(n_features, dtype=float)
        self._n_folds = 0

    def record_fold(self, pos_mask, neg_mask, edge_stats):
        """Store one fold's masks and statistics.

        Args:
            pos_mask: boolean array (n_features,).
            neg_mask: boolean array (n_features,).
            edge_stats: DataFrame from build_cpm_mask.
        """
        self._pos_counts += pos_mask.astype(int)
        self._neg_counts += neg_mask.astype(int)
        self._r_sum += edge_stats['r'].values
        self._n_folds += 1

    def compute_stability(self):
        """Compute per-feature stability across folds.

        Returns:
            DataFrame with columns:
                feature_name, stability_pos, stability_neg, mean_r
            where stability = fraction of folds in which the feature was selected.
        """
        if self._n_folds == 0:
            raise ValueError('No folds recorded.')
        return pd.DataFrame({
            'feature_name': self.feature_names,
            'stability_pos': self._pos_counts / self._n_folds,
            'stability_neg': self._neg_counts / self._n_folds,
            'mean_r': self._r_sum / self._n_folds,
        })

    def get_channel_degree(self, channel_names):
        """Compute per-channel degree from connectivity feature names.

        Expects feature names like 'CONN_ChA--ChB' (connectivity edges).
        For non-connectivity modalities, returns empty DataFrame.

        Args:
            channel_names: list of channel name strings.

        Returns:
            DataFrame with columns: channel, degree_pos, degree_neg.
        """
        degree_pos = {ch: 0.0 for ch in channel_names}
        degree_neg = {ch: 0.0 for ch in channel_names}

        stability = self.compute_stability()
        for _, row in stability.iterrows():
            name = row['feature_name']
            if '--' not in name:
                continue
            # Parse 'PREFIX_ChA--ChB' or 'ChA--ChB'
            edge_part = name.split('_', 1)[-1] if '_' in name else name
            parts = edge_part.split('--')
            if len(parts) != 2:
                continue
            ch_a, ch_b = parts[0].strip(), parts[1].strip()
            if ch_a in degree_pos:
                degree_pos[ch_a] += row['stability_pos']
                degree_neg[ch_a] += row['stability_neg']
            if ch_b in degree_pos:
                degree_pos[ch_b] += row['stability_pos']
                degree_neg[ch_b] += row['stability_neg']

        return pd.DataFrame({
            'channel': channel_names,
            'degree_pos': [degree_pos[ch] for ch in channel_names],
            'degree_neg': [degree_neg[ch] for ch in channel_names],
        })

    def to_csv(self, path):
        """Export stability table to CSV.

        Args:
            path: output file path.
        """
        df = self.compute_stability()
        df.to_csv(path, index=False)
        logger.info(f'Stability table saved: {path} ({len(df)} features)')
