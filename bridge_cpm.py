"""Bridge CPM — Cross-modal EEG+fMRI fusion with Linear SVR.

Aligns EEG and fMRI CSS matrices on shared subjects, runs a 10-feature
SVR LOSO with nested CV, and computes cross-modal convergence scores.

Usage:
    python bridge_cpm.py
"""

import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from EEG_CODE.eeg_cpm_fusion import (
    run_fusion_loso, permutation_test, bootstrap_ci,
    modality_ablation, clinical_utility_metrics, run_comparison_classifiers,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EEG-to-fMRI region mapping
# ---------------------------------------------------------------------------

EEG_TO_FMRI_MAP = {
    'Frontal': 'Default Mode',      # Fp1,Fp2,F3,F4,Fz → mPFC
    'Central': 'Sensory',           # C3,C4,Cz → sensorimotor
    'Temporal': 'Auditory',         # T3,T4,T5,T6 → auditory/language
    'Parietal': 'Cognitive Control', # P3,P4,Pz → attention
    'Occipital': 'Sensory',         # O1,O2 → visual
}

# EEG channel → region mapping (from eeg_xai_analysis.py BRAIN_REGIONS)
EEG_CHANNEL_REGIONS = {
    'Fp1': 'Frontal', 'Fp2': 'Frontal', 'Fpz': 'Frontal',
    'F7': 'Frontal', 'F3': 'Frontal', 'Fz': 'Frontal',
    'F4': 'Frontal', 'F8': 'Frontal', 'AF3': 'Frontal', 'AF4': 'Frontal',
    'C3': 'Central', 'Cz': 'Central', 'C4': 'Central',
    'FC1': 'Central', 'FC2': 'Central', 'FC5': 'Central', 'FC6': 'Central',
    'T3': 'Temporal', 'T4': 'Temporal', 'T5': 'Temporal', 'T6': 'Temporal',
    'T7': 'Temporal', 'T8': 'Temporal', 'P7': 'Temporal', 'P8': 'Temporal',
    'P3': 'Parietal', 'Pz': 'Parietal', 'P4': 'Parietal',
    'CP1': 'Parietal', 'CP2': 'Parietal', 'CP5': 'Parietal', 'CP6': 'Parietal',
    'O1': 'Occipital', 'Oz': 'Occipital', 'O2': 'Occipital',
    'PO3': 'Occipital', 'PO4': 'Occipital',
}

# fMRI network labels (matching FMRI_NETWORK_MAP values)
FMRI_NETWORKS = ['Default Mode', 'Sensory', 'Auditory', 'Language', 'Cognitive Control']


# ---------------------------------------------------------------------------
# Load and align CSS matrices
# ---------------------------------------------------------------------------

def load_and_align_css(eeg_path, fmri_path):
    """Inner-join EEG and fMRI CSS matrices on subject_id.

    Verifies label consistency between the two modalities.

    Args:
        eeg_path: path to css_matrix_eeg.csv.
        fmri_path: path to css_matrix_fmri.csv.

    Returns:
        aligned_df: DataFrame with all CSS columns and label, aligned on subject_id.
    """
    eeg_df = pd.read_csv(eeg_path)
    fmri_df = pd.read_csv(fmri_path)

    logger.info(f'EEG CSS: {eeg_df.shape[0]} subjects, '
                f'fMRI CSS: {fmri_df.shape[0]} subjects')

    # Inner join on subject_id
    merged = pd.merge(eeg_df, fmri_df, on='subject_id', suffixes=('_eeg', '_fmri'))

    # Verify label consistency
    if 'label_eeg' in merged.columns and 'label_fmri' in merged.columns:
        mismatches = merged[merged['label_eeg'] != merged['label_fmri']]
        if len(mismatches) > 0:
            logger.warning(
                f'{len(mismatches)} subjects with label mismatch between '
                f'EEG and fMRI. Using EEG labels.'
            )
        merged['label'] = merged['label_eeg']
        merged = merged.drop(columns=['label_eeg', 'label_fmri'])
    elif 'label' not in merged.columns:
        # If merge didn't create duplicates, label column is already present
        pass

    logger.info(f'Aligned CSS matrix: {merged.shape[0]} subjects, '
                f'{merged.shape[1]} columns')

    return merged


# ---------------------------------------------------------------------------
# Bridge fusion LOSO
# ---------------------------------------------------------------------------

def run_bridge_fusion_loso(aligned_df, output_dir='./results_bridge_cpm',
                            n_permutations=1000):
    """Run full bridge fusion analysis on aligned EEG+fMRI CSS features.

    Args:
        aligned_df: DataFrame from load_and_align_css.
        output_dir: directory for output files.
        n_permutations: number of permutation test iterations.

    Returns:
        summary: dict with all metrics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save aligned matrix
    aligned_df.to_csv(output_dir / 'bridge_css_aligned.csv', index=False)

    # 1. Main fusion LOSO
    logger.info('=== Bridge Fusion LOSO (Linear SVR) ===')
    results_df, metrics = run_fusion_loso(aligned_df, 'linear_svr')
    results_df.to_csv(output_dir / 'bridge_predictions.csv', index=False)

    # 2. Clinical utility
    logger.info('=== Clinical Utility Metrics ===')
    clinical = clinical_utility_metrics(
        results_df['true_label'].values,
        results_df['pred_score'].values
    )

    # 3. Bootstrap CI
    logger.info('=== Bootstrap CI ===')
    boot = bootstrap_ci(results_df)

    # 4. Permutation test
    logger.info(f'=== Permutation Test ({n_permutations} perms) ===')
    perm = permutation_test(aligned_df, n_permutations)
    perm_summary = {
        'observed_auc': perm['observed_auc'],
        'null_mean': perm['null_mean'],
        'null_std': perm['null_std'],
        'p_value': perm['p_value'],
    }
    pd.DataFrame([perm_summary]).to_csv(
        output_dir / 'bridge_permutation_test.csv', index=False
    )

    # 5. Modality ablation
    logger.info('=== Modality Ablation ===')
    ablation_df = modality_ablation(aligned_df)
    ablation_df.to_csv(output_dir / 'bridge_ablation.csv', index=False)

    # 6. Classifier comparison
    logger.info('=== Classifier Comparison ===')
    comparison_df = run_comparison_classifiers(aligned_df)
    comparison_df.to_csv(output_dir / 'bridge_classifier_comparison.csv', index=False)

    # Summary
    summary = {
        'auc': metrics['auc'],
        'accuracy': metrics['accuracy'],
        'ci_lower': boot['ci_lower'],
        'ci_upper': boot['ci_upper'],
        'p_value': perm['p_value'],
        **clinical,
    }
    pd.DataFrame([summary]).to_csv(output_dir / 'bridge_summary_metrics.csv', index=False)

    return summary


# ---------------------------------------------------------------------------
# Cross-modal convergence
# ---------------------------------------------------------------------------

def cross_modal_convergence(eeg_stability_paths, fmri_stability_paths,
                             output_path=None):
    """Compute EEG–fMRI convergence: which EEG regions and fMRI networks
    are both consistently predictive across LOSO folds.

    Uses the stability CSVs from both EEG and fMRI stages.

    Args:
        eeg_stability_paths: dict mapping modality name to stability CSV path,
            e.g. {'ERP': 'erp_edge_stability.csv', 'PW': '...', 'CONN': '...'}.
        fmri_stability_paths: dict mapping modality name to stability CSV path,
            e.g. {'ACT': 'fmri_act_edge_stability.csv', 'CONN': '...'}.
        output_path: optional path to save convergence CSV.

    Returns:
        convergence_df: DataFrame with eeg_region, fmri_network, convergence_score.
    """
    eeg_regions = sorted(EEG_TO_FMRI_MAP.keys())

    # --- Compute EEG region-level stability ---
    eeg_region_stability = {r: 0.0 for r in eeg_regions}

    for mod_name, path in eeg_stability_paths.items():
        path = Path(path)
        if not path.exists():
            logger.warning(f'EEG stability file not found: {path}')
            continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            name = row['feature_name']
            total_stab = row['stability_pos'] + row['stability_neg']
            # Try to extract channel name from feature name
            # Format: MODALITY_channel_idx or MODALITY_chA--chB
            parts = name.split('_', 1)
            if len(parts) < 2:
                continue
            feat_part = parts[1]

            if '--' in feat_part:
                # Connectivity: credit both channels
                ch_parts = feat_part.split('--')
                for ch in ch_parts:
                    ch = ch.strip()
                    region = EEG_CHANNEL_REGIONS.get(ch)
                    if region:
                        eeg_region_stability[region] += total_stab
            else:
                # ERP/PW: try to match channel name
                for ch, region in EEG_CHANNEL_REGIONS.items():
                    if ch.lower() in feat_part.lower():
                        eeg_region_stability[region] += total_stab
                        break

    # Normalize EEG region stability
    max_eeg = max(eeg_region_stability.values()) if any(eeg_region_stability.values()) else 1.0
    if max_eeg > 0:
        for r in eeg_region_stability:
            eeg_region_stability[r] /= max_eeg

    # --- Compute fMRI network-level stability ---
    fmri_network_stability = {n: 0.0 for n in FMRI_NETWORKS}

    for mod_name, path in fmri_stability_paths.items():
        path = Path(path)
        if not path.exists():
            logger.warning(f'fMRI stability file not found: {path}')
            continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            total_stab = row['stability_pos'] + row['stability_neg']
            name = row['feature_name'].lower()
            for net_key in ['dmn', 'sensory', 'an', 'ln', 'cognitive']:
                if net_key in name:
                    from fMRI_CODE.fmri_cpm_pipeline import FMRI_NETWORK_MAP
                    net_label = FMRI_NETWORK_MAP.get(net_key.upper(),
                                FMRI_NETWORK_MAP.get(net_key, 'Other'))
                    if net_label in fmri_network_stability:
                        fmri_network_stability[net_label] += total_stab
                    break

    # Normalize fMRI network stability
    max_fmri = max(fmri_network_stability.values()) if any(fmri_network_stability.values()) else 1.0
    if max_fmri > 0:
        for n in fmri_network_stability:
            fmri_network_stability[n] /= max_fmri

    # --- Compute convergence scores ---
    rows = []
    for eeg_region in eeg_regions:
        mapped_network = EEG_TO_FMRI_MAP[eeg_region]
        for fmri_network in FMRI_NETWORKS:
            eeg_s = eeg_region_stability.get(eeg_region, 0.0)
            fmri_s = fmri_network_stability.get(fmri_network, 0.0)

            # Convergence = geometric mean (high only if both are high)
            convergence = np.sqrt(eeg_s * fmri_s)

            # Bonus if this is the expected mapping
            is_expected = (mapped_network == fmri_network)

            rows.append({
                'eeg_region': eeg_region,
                'fmri_network': fmri_network,
                'eeg_stability': eeg_s,
                'fmri_stability': fmri_s,
                'convergence_score': convergence,
                'expected_mapping': is_expected,
            })

    convergence_df = pd.DataFrame(rows)

    if output_path:
        convergence_df.to_csv(output_path, index=False)
        logger.info(f'Convergence table saved: {output_path}')

    return convergence_df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Bridge CPM fusion')
    parser.add_argument('--eeg-css', type=str,
                        default='./results_eeg_cpm/css_matrix_eeg.csv')
    parser.add_argument('--fmri-css', type=str,
                        default='./results_fmri_cpm/css_matrix_fmri.csv')
    parser.add_argument('--output-dir', type=str,
                        default='./results_bridge_cpm')
    parser.add_argument('--n-permutations', type=int, default=1000)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    output_dir = Path(args.output_dir)

    # 1. Load and align
    logger.info('=== Loading and aligning CSS matrices ===')
    aligned_df = load_and_align_css(args.eeg_css, args.fmri_css)

    # 2. Bridge fusion
    logger.info('=== Running bridge fusion ===')
    summary = run_bridge_fusion_loso(
        aligned_df, args.output_dir, args.n_permutations
    )
    logger.info(f'Bridge summary: {summary}')

    # 3. Cross-modal convergence
    logger.info('=== Computing cross-modal convergence ===')
    eeg_dir = Path(args.eeg_css).parent
    fmri_dir = Path(args.fmri_css).parent

    eeg_stab = {
        'ERP': eeg_dir / 'erp_edge_stability.csv',
        'PW': eeg_dir / 'pw_edge_stability.csv',
        'CONN': eeg_dir / 'conn_eeg_edge_stability.csv',
    }
    fmri_stab = {
        'ACT': fmri_dir / 'fmri_act_edge_stability.csv',
        'CONN': fmri_dir / 'fmri_conn_edge_stability.csv',
    }

    convergence_df = cross_modal_convergence(
        eeg_stab, fmri_stab,
        output_path=output_dir / 'bridge_convergence.csv'
    )
    logger.info(f'Convergence table:\n{convergence_df.to_string(index=False)}')

    logger.info('Bridge CPM analysis complete.')


if __name__ == '__main__':
    main()
