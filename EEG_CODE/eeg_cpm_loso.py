"""EEG CPM Leave-One-Subject-Out pipeline.

Runs CPM feature selection + CSS computation for each EEG modality
(ERP, Power Spectrum, Connectivity) using LOSO cross-validation.

Outputs:
    css_matrix_eeg.csv — per-subject CSS values for all 3 modalities
    *_edge_stability.csv — per-feature stability across folds

Usage:
    python -m EEG_CODE.eeg_cpm_loso
"""

import os
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from .eeg_cpm_pipeline import build_cpm_mask, compute_css, BiomarkerLog
from .eeg_data_utils import (
    load_eeg_labels, load_eeg_erp_features,
    load_eeg_pw_features, load_eeg_conn_features,
    build_subject_matrix,
)
from .config import Config

logger = logging.getLogger(__name__)


def run_single_modality_cpm_loso(feature_matrix, labels, subject_ids,
                                  p_threshold=0.01, feature_names=None):
    """Run LOSO CPM for a single modality.

    For each fold, hold out one subject, build CPM masks on the remaining
    training subjects, and compute CSS for the held-out subject.

    Args:
        feature_matrix: ndarray of shape (N, n_feat).
        labels: ndarray of shape (N,).
        subject_ids: list of N subject IDs (aligned with rows).
        p_threshold: CPM significance threshold.
        feature_names: optional list of feature name strings.

    Returns:
        css_matrix: ndarray of shape (N, 2) — [css_pos, css_neg] per subject.
        biomarker_log: BiomarkerLog with per-fold mask records.
    """
    N, n_feat = feature_matrix.shape
    css_matrix = np.zeros((N, 2))
    bio_log = BiomarkerLog(n_feat, feature_names)

    for i in range(N):
        # Train on all except subject i
        train_mask = np.ones(N, dtype=bool)
        train_mask[i] = False
        X_train = feature_matrix[train_mask]
        y_train = labels[train_mask]
        X_test = feature_matrix[i]

        pos_mask, neg_mask, edge_stats = build_cpm_mask(
            X_train, y_train, p_threshold
        )
        bio_log.record_fold(pos_mask, neg_mask, edge_stats)

        css_pos, css_neg = compute_css(X_test, pos_mask, neg_mask)
        css_matrix[i] = [css_pos, css_neg]

    logger.info(
        f'LOSO complete: {N} folds, '
        f'CSS_pos range [{css_matrix[:,0].min():.2f}, {css_matrix[:,0].max():.2f}], '
        f'CSS_neg range [{css_matrix[:,1].min():.2f}, {css_matrix[:,1].max():.2f}]'
    )
    return css_matrix, bio_log


def run_eeg_cpm_loso(config=None, p_threshold=0.01, output_dir='./results_eeg_cpm'):
    """Full EEG CPM LOSO pipeline across ERP, PW, and CONN modalities.

    Loads data using existing loaders, builds subject matrices, runs LOSO CPM
    for each modality, and saves CSS + stability outputs.

    Args:
        config: Config object (created if None).
        p_threshold: CPM significance threshold.
        output_dir: directory for output files.

    Returns:
        css_df: DataFrame with per-subject CSS values and labels.
    """
    if config is None:
        config = Config()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load labels ---
    logger.info('Loading EEG labels...')
    labels = load_eeg_labels(config.label_path, binary=True)
    logger.info(f'Labels loaded: {len(labels)} subjects')

    # --- Load features ---
    logger.info('Loading EEG ERP features...')
    erp_feats = load_eeg_erp_features(
        config.eeg_path_erp, config.subject_list,
        list(config.bands.keys()), config.eeg_segments
    )

    logger.info('Loading EEG power spectrum features...')
    pw_feats = load_eeg_pw_features(
        config.eeg_path_pw, config.subject_list,
        list(config.bands.keys()), config.eeg_segments
    )

    logger.info('Loading EEG connectivity features...')
    conn_feats = load_eeg_conn_features(
        config.eeg_path_conn, config.subject_list,
        config.bands, config.func_segments
    )

    # --- Build subject matrices ---
    logger.info('Building subject matrices...')
    erp_df, erp_labels, erp_subjs = build_subject_matrix(
        erp_feats, config.subject_list, labels, modality='ERP'
    )
    pw_df, pw_labels, pw_subjs = build_subject_matrix(
        pw_feats, config.subject_list, labels, modality='PW'
    )
    conn_df, conn_labels, conn_subjs = build_subject_matrix(
        conn_feats, config.subject_list, labels, modality='CONN_EEG'
    )

    # --- Intersect subjects across modalities ---
    common_subjects = sorted(
        set(erp_subjs) & set(pw_subjs) & set(conn_subjs)
    )
    if not common_subjects:
        raise ValueError('No subjects found in common across all 3 EEG modalities.')
    logger.info(f'Common subjects across ERP/PW/CONN: {len(common_subjects)}')

    # Align to common subjects
    erp_mat = erp_df.loc[common_subjects].values
    pw_mat = pw_df.loc[common_subjects].values
    conn_mat = conn_df.loc[common_subjects].values
    y = np.array([labels[s] for s in common_subjects])

    # --- Run LOSO for each modality ---
    logger.info('Running ERP CPM LOSO...')
    erp_css, erp_log = run_single_modality_cpm_loso(
        erp_mat, y, common_subjects, p_threshold,
        feature_names=list(erp_df.columns)
    )

    logger.info('Running PW CPM LOSO...')
    pw_css, pw_log = run_single_modality_cpm_loso(
        pw_mat, y, common_subjects, p_threshold,
        feature_names=list(pw_df.columns)
    )

    logger.info('Running CONN CPM LOSO...')
    conn_css, conn_log = run_single_modality_cpm_loso(
        conn_mat, y, common_subjects, p_threshold,
        feature_names=list(conn_df.columns)
    )

    # --- Assemble output DataFrame ---
    css_df = pd.DataFrame({
        'subject_id': common_subjects,
        'CSS_ERP_pos': erp_css[:, 0],
        'CSS_ERP_neg': erp_css[:, 1],
        'CSS_PW_pos': pw_css[:, 0],
        'CSS_PW_neg': pw_css[:, 1],
        'CSS_CONN_EEG_pos': conn_css[:, 0],
        'CSS_CONN_EEG_neg': conn_css[:, 1],
        'label': y,
    })

    # --- Save outputs ---
    css_path = output_dir / 'css_matrix_eeg.csv'
    css_df.to_csv(css_path, index=False)
    logger.info(f'EEG CSS matrix saved: {css_path}')

    erp_log.to_csv(output_dir / 'erp_edge_stability.csv')
    pw_log.to_csv(output_dir / 'pw_edge_stability.csv')
    conn_log.to_csv(output_dir / 'conn_eeg_edge_stability.csv')

    # Summary
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    logger.info(f'EEG CPM LOSO complete: {len(common_subjects)} subjects '
                f'({n_pos} positive, {n_neg} negative)')

    return css_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EEG CPM LOSO pipeline')
    parser.add_argument('--p-threshold', type=float, default=0.01)
    parser.add_argument('--output-dir', type=str, default='./results_eeg_cpm')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    run_eeg_cpm_loso(p_threshold=args.p_threshold, output_dir=args.output_dir)
