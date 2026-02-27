"""fMRI CPM Fusion — Linear SVR classifier on fMRI CSS features.

Same structure as eeg_cpm_fusion.py but for fMRI data. Imports shared
fusion functions and adds fMRI-specific ROI/network visualizations.

Usage:
    python -m fMRI_CODE.fmri_cpm_fusion
"""

import sys
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from EEG_CODE.eeg_cpm_fusion import (
    run_fusion_loso, permutation_test, bootstrap_ci,
    modality_ablation, clinical_utility_metrics, run_comparison_classifiers,
)
from fMRI_CODE.fmri_cpm_pipeline import FMRI_NETWORK_MAP

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# fMRI-specific visualization
# ---------------------------------------------------------------------------

def plot_roi_degree(stability_path, output_path, top_k=15):
    """Plot top ROI degree from stability CSV.

    Args:
        stability_path: path to fmri_conn_edge_stability.csv.
        output_path: path for saved figure.
        top_k: number of top features to show.
    """
    df = pd.read_csv(stability_path)
    df['total_stability'] = df['stability_pos'] + df['stability_neg']
    df_sorted = df.nlargest(top_k, 'total_stability')

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = range(len(df_sorted))
    ax.barh(y_pos, df_sorted['stability_pos'], label='Positive', color='#e74c3c', alpha=0.8)
    ax.barh(y_pos, -df_sorted['stability_neg'], label='Negative', color='#3498db', alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted['feature_name'], fontsize=8)
    ax.set_xlabel('Stability (fraction of folds)')
    ax.set_title('fMRI Feature Stability (Top Edges)')
    ax.legend()
    ax.axvline(0, color='black', linewidth=0.5)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'ROI degree plot saved: {output_path}')


def plot_network_heatmap(stability_path, output_path):
    """Plot network-level stability heatmap from connectivity features.

    Args:
        stability_path: path to fmri_conn_edge_stability.csv.
        output_path: path for saved figure.
    """
    df = pd.read_csv(stability_path)

    # Map features to networks
    networks = sorted(FMRI_NETWORK_MAP.values())
    net_matrix = np.zeros((len(networks), 2))  # pos, neg

    for _, row in df.iterrows():
        name = row['feature_name']
        for net_key, net_label in FMRI_NETWORK_MAP.items():
            if net_key.lower() in name.lower():
                idx = networks.index(net_label)
                net_matrix[idx, 0] += row['stability_pos']
                net_matrix[idx, 1] += row['stability_neg']
                break

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(net_matrix, cmap='RdBu_r', aspect='auto')
    ax.set_yticks(range(len(networks)))
    ax.set_yticklabels(networks)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Positive\nStability', 'Negative\nStability'])
    ax.set_title('fMRI Network Stability Summary')
    plt.colorbar(im, ax=ax, label='Cumulative stability')
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'Network heatmap saved: {output_path}')


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_fmri_fusion_analysis(css_path='./results_fmri_cpm/css_matrix_fmri.csv',
                              output_dir='./results_fmri_cpm',
                              n_permutations=1000):
    """Run full fMRI fusion analysis pipeline.

    Args:
        css_path: path to css_matrix_fmri.csv.
        output_dir: directory for outputs.
        n_permutations: number of permutation test iterations.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    css_df = pd.read_csv(css_path)
    logger.info(f'Loaded fMRI CSS matrix: {css_df.shape}')

    # 1. Main fusion LOSO
    logger.info('=== fMRI Fusion LOSO (Linear SVR) ===')
    results_df, metrics = run_fusion_loso(css_df, 'linear_svr')
    results_df.to_csv(output_dir / 'fmri_predictions.csv', index=False)

    # 2. Clinical utility
    logger.info('=== Clinical Utility Metrics ===')
    clinical = clinical_utility_metrics(
        results_df['true_label'].values,
        results_df['pred_score'].values
    )
    logger.info(f'Clinical metrics: {clinical}')

    # 3. Bootstrap CI
    logger.info('=== Bootstrap CI ===')
    boot = bootstrap_ci(results_df)

    # 4. Permutation test
    logger.info(f'=== Permutation Test ({n_permutations} perms) ===')
    perm = permutation_test(css_df, n_permutations)
    perm_summary = {
        'observed_auc': perm['observed_auc'],
        'null_mean': perm['null_mean'],
        'null_std': perm['null_std'],
        'p_value': perm['p_value'],
    }
    pd.DataFrame([perm_summary]).to_csv(
        output_dir / 'fmri_permutation_test.csv', index=False
    )

    # 5. Modality ablation
    logger.info('=== Modality Ablation ===')
    ablation_df = modality_ablation(css_df)
    ablation_df.to_csv(output_dir / 'fmri_ablation.csv', index=False)

    # 6. Classifier comparison
    logger.info('=== Classifier Comparison ===')
    comparison_df = run_comparison_classifiers(css_df)
    comparison_df.to_csv(output_dir / 'fmri_classifier_comparison.csv', index=False)

    # 7. fMRI-specific plots
    conn_stab_path = output_dir / 'fmri_conn_edge_stability.csv'
    if conn_stab_path.exists():
        plot_roi_degree(conn_stab_path, output_dir / 'fmri_roi_degree.png')
        plot_network_heatmap(conn_stab_path, output_dir / 'fmri_network_heatmap.png')

    # Summary
    summary = {
        'auc': metrics['auc'],
        'accuracy': metrics['accuracy'],
        'ci_lower': boot['ci_lower'],
        'ci_upper': boot['ci_upper'],
        'p_value': perm['p_value'],
        **clinical,
    }
    pd.DataFrame([summary]).to_csv(output_dir / 'fmri_summary_metrics.csv', index=False)
    logger.info(f'fMRI fusion analysis complete. Results in {output_dir}')

    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='fMRI CPM Fusion analysis')
    parser.add_argument('--css-path', type=str,
                        default='./results_fmri_cpm/css_matrix_fmri.csv')
    parser.add_argument('--output-dir', type=str, default='./results_fmri_cpm')
    parser.add_argument('--n-permutations', type=int, default=1000)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    run_fmri_fusion_analysis(args.css_path, args.output_dir, args.n_permutations)
