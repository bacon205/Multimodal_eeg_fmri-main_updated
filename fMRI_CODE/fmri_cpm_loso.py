"""fMRI CPM Leave-One-Subject-Out pipeline.

Runs CPM feature selection + CSS computation for fMRI activation and
connectivity modalities using LOSO cross-validation.

Outputs:
    css_matrix_fmri.csv — per-subject CSS values for both modalities
    *_edge_stability.csv — per-feature stability across folds

Usage:
    python -m fMRI_CODE.fmri_cpm_loso
"""

import sys
import os
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from EEG_CODE.eeg_cpm_pipeline import build_cpm_mask, compute_css, BiomarkerLog
from EEG_CODE.eeg_cpm_loso import run_single_modality_cpm_loso
from fMRI_CODE.fmri_cpm_pipeline import (
    build_activation_matrix, build_connectivity_matrix,
    fMRIBiomarkerLog, DEFAULT_ACTIVATION_TYPES, DEFAULT_CONNECTIVITY_TYPES,
)
from fMRI_CODE.fmri_utils import (
    load_activation_features, load_connectivity_features, load_fmri_labels,
)

logger = logging.getLogger(__name__)


def run_fmri_cpm_loso(fmri_data_dir=None, label_path=None,
                       subject_list=None, p_threshold=0.01,
                       output_dir='./results_fmri_cpm',
                       activation_types=None, connectivity_types=None):
    """Full fMRI CPM LOSO pipeline for activation and connectivity.

    Args:
        fmri_data_dir: path to fMRI data (sub-N folders).
        label_path: path to labels directory.
        subject_list: list of subject IDs.
        p_threshold: CPM significance threshold.
        output_dir: directory for output files.
        activation_types: list of activation types (default: all 5).
        connectivity_types: list of connectivity types (default: ['DMN']).

    Returns:
        css_df: DataFrame with per-subject CSS values and labels.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Defaults
    if fmri_data_dir is None:
        base = Path(os.getenv('EEG_DATA_PATH',
                              r'E:\Intermediate\BACON_ERIC\Head_neck'))
        fmri_data_dir = base / 'fMRI' / 'DATA'
    fmri_data_dir = Path(fmri_data_dir)

    if label_path is None:
        label_path = fmri_data_dir

    if subject_list is None:
        # Discover subjects from directory
        subject_list = []
        if fmri_data_dir.exists():
            for d in fmri_data_dir.iterdir():
                if d.is_dir() and d.name.startswith('sub-'):
                    try:
                        subject_list.append(int(d.name.replace('sub-', '')))
                    except ValueError:
                        pass
        subject_list = sorted(subject_list)
        if not subject_list:
            # Fallback: use 1-32 range
            subject_list = list(range(1, 33))

    if activation_types is None:
        activation_types = DEFAULT_ACTIVATION_TYPES
    if connectivity_types is None:
        connectivity_types = DEFAULT_CONNECTIVITY_TYPES

    # --- Load labels ---
    logger.info('Loading fMRI labels...')
    labels = load_fmri_labels(label_path, subject_list)
    logger.info(f'Labels loaded: {len(labels)} subjects')

    # --- Load features ---
    logger.info('Loading fMRI activation features...')
    act_feats = load_activation_features(
        fmri_data_dir, subject_list, activation_types
    )

    logger.info('Loading fMRI connectivity features...')
    conn_feats = load_connectivity_features(
        fmri_data_dir, subject_list, connectivity_types
    )

    # --- Build subject matrices ---
    logger.info('Building subject matrices...')
    act_df, act_labels, act_subjs = build_activation_matrix(
        act_feats, subject_list, labels
    )
    conn_df, conn_labels, conn_subjs = build_connectivity_matrix(
        conn_feats, subject_list, labels
    )

    # --- Intersect subjects ---
    common_subjects = sorted(set(act_subjs) & set(conn_subjs))
    if not common_subjects:
        raise ValueError('No subjects found in common across fMRI modalities.')
    logger.info(f'Common subjects across ACT/CONN: {len(common_subjects)}')

    # Align
    act_mat = act_df.loc[common_subjects].values
    conn_mat = conn_df.loc[common_subjects].values
    y = np.array([labels[s] for s in common_subjects])

    # --- Run LOSO ---
    logger.info('Running fMRI Activation CPM LOSO...')
    act_css, act_log = run_single_modality_cpm_loso(
        act_mat, y, common_subjects, p_threshold,
        feature_names=list(act_df.columns)
    )

    logger.info('Running fMRI Connectivity CPM LOSO...')
    conn_css, conn_log = run_single_modality_cpm_loso(
        conn_mat, y, common_subjects, p_threshold,
        feature_names=list(conn_df.columns)
    )

    # --- Assemble output ---
    css_df = pd.DataFrame({
        'subject_id': common_subjects,
        'CSS_fMRI_ACT_pos': act_css[:, 0],
        'CSS_fMRI_ACT_neg': act_css[:, 1],
        'CSS_fMRI_CONN_pos': conn_css[:, 0],
        'CSS_fMRI_CONN_neg': conn_css[:, 1],
        'label': y,
    })

    # --- Save ---
    css_path = output_dir / 'css_matrix_fmri.csv'
    css_df.to_csv(css_path, index=False)
    logger.info(f'fMRI CSS matrix saved: {css_path}')

    act_log.to_csv(output_dir / 'fmri_act_edge_stability.csv')
    conn_log.to_csv(output_dir / 'fmri_conn_edge_stability.csv')

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    logger.info(f'fMRI CPM LOSO complete: {len(common_subjects)} subjects '
                f'({n_pos} positive, {n_neg} negative)')

    return css_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='fMRI CPM LOSO pipeline')
    parser.add_argument('--fmri-data-dir', type=str, default=None)
    parser.add_argument('--label-path', type=str, default=None)
    parser.add_argument('--p-threshold', type=float, default=0.01)
    parser.add_argument('--output-dir', type=str, default='./results_fmri_cpm')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    run_fmri_cpm_loso(
        fmri_data_dir=args.fmri_data_dir,
        label_path=args.label_path,
        p_threshold=args.p_threshold,
        output_dir=args.output_dir,
    )
