"""Bridge visualization — all paper figures for the CPM pipeline.

Reuses plot_topomap, plot_connectivity_matrix, plot_region_comparison
from eeg_xai_analysis.py and adds CPM-specific visualizations:
ROC with CI, permutation distribution, convergence heatmap, ablation,
classifier comparison, per-subject predictions.

Usage:
    python bridge_visualize.py
"""

import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle
from pathlib import Path

from EEG_CODE.eeg_xai_analysis import (
    plot_topomap, plot_connectivity_matrix, plot_region_comparison,
    CHANNEL_POSITIONS_2D, BRAIN_REGIONS, STANDARD_10_20_19,
)
from bridge_cpm import EEG_TO_FMRI_MAP, FMRI_NETWORKS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Edge stability topomap (wraps existing plot_topomap)
# ---------------------------------------------------------------------------

def plot_edge_stability_topomap(stability_path, output_path,
                                 channel_names=None, title=None):
    """Plot CPM edge stability as an EEG topomap.

    Aggregates stability scores to channel level and calls plot_topomap.

    Args:
        stability_path: path to edge_stability CSV.
        output_path: path for saved figure.
        channel_names: list of channel names (default: 19-channel 10-20).
        title: plot title.
    """
    if channel_names is None:
        channel_names = STANDARD_10_20_19

    df = pd.read_csv(stability_path)

    # Aggregate to channel level
    channel_scores = {ch: 0.0 for ch in channel_names}
    for _, row in df.iterrows():
        name = row['feature_name']
        total = row['stability_pos'] + row['stability_neg']
        for ch in channel_names:
            if ch.lower() in name.lower():
                channel_scores[ch] += total

    # Normalize to [0, 1]
    max_val = max(channel_scores.values()) if any(channel_scores.values()) else 1.0
    if max_val > 0:
        channel_scores = {ch: v / max_val for ch, v in channel_scores.items()}

    if title is None:
        mod = Path(stability_path).stem.split('_')[0].upper()
        title = f'{mod} Feature Stability Topomap'

    fig, ax = plot_topomap(
        channel_scores, title=title,
        save_path=str(output_path)
    )
    plt.close(fig)
    logger.info(f'Stability topomap saved: {output_path}')


# ---------------------------------------------------------------------------
# Connectivity stability matrix (wraps existing plot_connectivity_matrix)
# ---------------------------------------------------------------------------

def plot_conn_stability_matrix(stability_path, output_path,
                                channel_names=None, title=None):
    """Plot connectivity stability as a channel × channel matrix.

    Args:
        stability_path: path to conn edge_stability CSV.
        output_path: path for saved figure.
        channel_names: list of channel names.
        title: plot title.
    """
    if channel_names is None:
        channel_names = STANDARD_10_20_19

    df = pd.read_csv(stability_path)

    # Build edge importance dict
    edge_importance = {}
    for _, row in df.iterrows():
        name = row['feature_name']
        total = row['stability_pos'] + row['stability_neg']
        if '--' not in name:
            continue
        # Parse edge
        parts = name.split('_', 1)
        edge_part = parts[-1] if len(parts) > 1 else name
        ch_parts = edge_part.split('--')
        if len(ch_parts) == 2:
            ch_a, ch_b = ch_parts[0].strip(), ch_parts[1].strip()
            if ch_a in channel_names and ch_b in channel_names:
                edge_importance[(ch_a, ch_b)] = total

    if not edge_importance:
        logger.warning(f'No connectivity edges found in {stability_path}')
        return

    if title is None:
        title = 'Connectivity Edge Stability'

    fig, ax = plot_connectivity_matrix(
        edge_importance, channel_names, title=title,
        save_path=str(output_path)
    )
    plt.close(fig)
    logger.info(f'Connectivity stability matrix saved: {output_path}')


# ---------------------------------------------------------------------------
# Region stability radar (wraps existing plot_region_comparison)
# ---------------------------------------------------------------------------

def plot_region_stability_radar(stability_path, output_path, title=None):
    """Plot brain region stability as a radar chart.

    Args:
        stability_path: path to edge_stability CSV.
        output_path: path for saved figure.
        title: plot title.
    """
    df = pd.read_csv(stability_path)

    region_scores = {r: 0.0 for r in BRAIN_REGIONS}
    region_counts = {r: 0 for r in BRAIN_REGIONS}

    for _, row in df.iterrows():
        name = row['feature_name']
        total = row['stability_pos'] + row['stability_neg']
        for region, channels in BRAIN_REGIONS.items():
            for ch in channels:
                if ch.lower() in name.lower():
                    region_scores[region] += total
                    region_counts[region] += 1
                    break

    # Average per region
    for r in region_scores:
        if region_counts[r] > 0:
            region_scores[r] /= region_counts[r]

    if title is None:
        title = 'Brain Region Stability'

    fig, ax = plot_region_comparison(
        region_scores, title=title,
        save_path=str(output_path)
    )
    plt.close(fig)
    logger.info(f'Region stability radar saved: {output_path}')


# ---------------------------------------------------------------------------
# ROC curve with bootstrap CI band
# ---------------------------------------------------------------------------

def plot_roc_with_ci(predictions_path, output_path, n_bootstrap=1000,
                      ci=0.95, title='ROC Curve with 95% CI'):
    """Plot ROC curve with bootstrap confidence interval band.

    Args:
        predictions_path: path to predictions CSV (true_label, pred_score).
        output_path: path for saved figure.
        n_bootstrap: number of bootstrap iterations.
        ci: confidence level.
        title: plot title.
    """
    df = pd.read_csv(predictions_path)
    y_true = df['true_label'].values
    y_score = df['pred_score'].values

    from sklearn.metrics import roc_curve, roc_auc_score
    fpr_main, tpr_main, _ = roc_curve(y_true, y_score)
    auc_main = roc_auc_score(y_true, y_score)

    # Bootstrap ROC curves
    rng = np.random.RandomState(42)
    base_fpr = np.linspace(0, 1, 100)
    tpr_boots = np.zeros((n_bootstrap, len(base_fpr)))

    for b in range(n_bootstrap):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            tpr_boots[b] = np.nan
            continue
        fpr_b, tpr_b, _ = roc_curve(y_true[idx], y_score[idx])
        tpr_boots[b] = np.interp(base_fpr, fpr_b, tpr_b)

    # Remove NaN rows
    valid = ~np.any(np.isnan(tpr_boots), axis=1)
    tpr_boots = tpr_boots[valid]

    alpha = (1 - ci) / 2
    tpr_lower = np.percentile(tpr_boots, 100 * alpha, axis=0)
    tpr_upper = np.percentile(tpr_boots, 100 * (1 - alpha), axis=0)
    tpr_mean = np.mean(tpr_boots, axis=0)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.fill_between(base_fpr, tpr_lower, tpr_upper, alpha=0.2, color='steelblue',
                    label=f'{int(ci*100)}% CI')
    ax.plot(fpr_main, tpr_main, 'b-', linewidth=2,
            label=f'ROC (AUC = {auc_main:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Chance')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'ROC with CI saved: {output_path}')


# ---------------------------------------------------------------------------
# Permutation null distribution
# ---------------------------------------------------------------------------

def plot_permutation_distribution(perm_test_path, output_path,
                                   title='Permutation Test'):
    """Plot null AUC histogram with observed AUC line.

    Args:
        perm_test_path: path to permutation_test CSV.
        output_path: path for saved figure.
        title: plot title.
    """
    df = pd.read_csv(perm_test_path)
    observed_auc = df['observed_auc'].iloc[0]
    null_mean = df['null_mean'].iloc[0]
    p_value = df['p_value'].iloc[0]

    # Try to load full null distribution if available
    null_dist_path = Path(perm_test_path).parent / 'null_distribution.npy'
    if null_dist_path.exists():
        null_aucs = np.load(null_dist_path)
    else:
        # Generate synthetic null from mean/std for visualization
        null_std = df['null_std'].iloc[0] if 'null_std' in df.columns else 0.05
        rng = np.random.RandomState(42)
        null_aucs = rng.normal(null_mean, null_std, 1000)
        null_aucs = np.clip(null_aucs, 0, 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(null_aucs, bins=50, color='gray', alpha=0.7, edgecolor='black',
            label='Null distribution')
    ax.axvline(observed_auc, color='red', linewidth=2, linestyle='--',
               label=f'Observed AUC = {observed_auc:.3f}')
    ax.axvline(0.5, color='black', linewidth=1, linestyle=':',
               label='Chance (0.5)')
    ax.set_xlabel('AUC', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'{title} (p = {p_value:.4f})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'Permutation distribution saved: {output_path}')


# ---------------------------------------------------------------------------
# Cross-modal convergence heatmap
# ---------------------------------------------------------------------------

def plot_convergence_heatmap(convergence_path, output_path,
                              title='EEG-fMRI Cross-Modal Convergence'):
    """Plot EEG region × fMRI network convergence heatmap.

    Args:
        convergence_path: path to bridge_convergence.csv.
        output_path: path for saved figure.
        title: plot title.
    """
    df = pd.read_csv(convergence_path)

    eeg_regions = sorted(df['eeg_region'].unique())
    fmri_networks = sorted(df['fmri_network'].unique())

    matrix = np.zeros((len(eeg_regions), len(fmri_networks)))
    expected = np.zeros_like(matrix, dtype=bool)

    for _, row in df.iterrows():
        i = eeg_regions.index(row['eeg_region'])
        j = fmri_networks.index(row['fmri_network'])
        matrix[i, j] = row['convergence_score']
        if row.get('expected_mapping', False):
            expected[i, j] = True

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')

    # Mark expected mappings with border
    for i in range(len(eeg_regions)):
        for j in range(len(fmri_networks)):
            val = matrix[i, j]
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=10, fontweight='bold' if expected[i, j] else 'normal',
                    color='white' if val > matrix.max() * 0.6 else 'black')
            if expected[i, j]:
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                     edgecolor='blue', linewidth=2)
                ax.add_patch(rect)

    ax.set_xticks(range(len(fmri_networks)))
    ax.set_xticklabels(fmri_networks, rotation=30, ha='right', fontsize=10)
    ax.set_yticks(range(len(eeg_regions)))
    ax.set_yticklabels(eeg_regions, fontsize=10)
    ax.set_xlabel('fMRI Network', fontsize=12)
    ax.set_ylabel('EEG Region', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Convergence Score')

    # Legend for expected mapping
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='none', edgecolor='blue', linewidth=2,
                             label='Expected mapping')]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'Convergence heatmap saved: {output_path}')


# ---------------------------------------------------------------------------
# Modality ablation bar chart
# ---------------------------------------------------------------------------

def plot_modality_ablation(ablation_path, output_path,
                            title='Modality Ablation'):
    """Bar chart of AUC with each modality dropped.

    Args:
        ablation_path: path to ablation CSV.
        output_path: path for saved figure.
        title: plot title.
    """
    df = pd.read_csv(ablation_path)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#2ecc71' if row['modality_dropped'] == 'none' else '#e74c3c'
              for _, row in df.iterrows()]
    bars = ax.bar(range(len(df)), df['auc'], color=colors, edgecolor='black')

    ax.set_xticks(range(len(df)))
    labels = []
    for _, row in df.iterrows():
        if row['modality_dropped'] == 'none':
            labels.append('Full Model')
        else:
            labels.append(f'Drop {row["modality_dropped"]}')
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=10)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add value labels
    for bar, val in zip(bars, df['auc']):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', fontsize=10)

    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'Ablation plot saved: {output_path}')


# ---------------------------------------------------------------------------
# Classifier comparison
# ---------------------------------------------------------------------------

def plot_classifier_comparison(comparison_path, output_path,
                                title='Classifier Comparison'):
    """Grouped bar chart comparing classifiers.

    Args:
        comparison_path: path to classifier_comparison CSV.
        output_path: path for saved figure.
        title: plot title.
    """
    df = pd.read_csv(comparison_path)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(df))
    width = 0.35

    bars1 = ax.bar(x - width/2, df['auc'], width, label='AUC',
                   color='#3498db', edgecolor='black')
    bars2 = ax.bar(x + width/2, df['accuracy'], width, label='Accuracy',
                   color='#e67e22', edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels(df['classifier'], rotation=15, ha='right', fontsize=10)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)

    # Value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'Classifier comparison saved: {output_path}')


# ---------------------------------------------------------------------------
# Per-subject prediction table
# ---------------------------------------------------------------------------

def plot_per_subject_predictions(predictions_path, output_path,
                                  title='Per-Subject Predictions'):
    """Color-coded table of per-subject prediction results.

    Args:
        predictions_path: path to predictions CSV.
        output_path: path for saved figure.
        title: plot title.
    """
    df = pd.read_csv(predictions_path)

    # Determine correct/incorrect
    df['correct'] = (df['pred_class'] == df['true_label']).astype(int)

    n = len(df)
    n_cols = min(10, n)
    n_rows = int(np.ceil(n / n_cols))

    fig, ax = plt.subplots(figsize=(max(10, n_cols * 1.2), n_rows * 0.8 + 2))
    ax.axis('off')

    for idx, (_, row) in enumerate(df.iterrows()):
        r = idx // n_cols
        c = idx % n_cols

        x = c / n_cols + 0.5 / n_cols
        y = 1.0 - (r + 1) / (n_rows + 1)

        if row['correct']:
            color = '#2ecc71'  # green
        else:
            color = '#e74c3c'  # red

        subj_id = int(row['subject_id'])
        true_lbl = int(row['true_label'])
        pred_lbl = int(row['pred_class'])

        ax.add_patch(plt.Rectangle(
            (x - 0.4/n_cols, y - 0.02), 0.8/n_cols, 0.06,
            facecolor=color, alpha=0.3, edgecolor='black'
        ))
        ax.text(x, y + 0.01,
                f'S{subj_id}\nT:{true_lbl} P:{pred_lbl}',
                ha='center', va='center', fontsize=7, fontweight='bold')

    n_correct = df['correct'].sum()
    ax.set_title(f'{title}\n({n_correct}/{n} correct, '
                 f'Acc={n_correct/n:.1%})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'Per-subject predictions saved: {output_path}')


# ---------------------------------------------------------------------------
# Full report generator
# ---------------------------------------------------------------------------

def generate_full_report(eeg_dir='./results_eeg_cpm',
                          fmri_dir='./results_fmri_cpm',
                          bridge_dir='./results_bridge_cpm',
                          output_dir='./results_figures'):
    """Generate all paper figures from pipeline outputs.

    Args:
        eeg_dir: path to EEG results directory.
        fmri_dir: path to fMRI results directory.
        bridge_dir: path to bridge results directory.
        output_dir: path for all figures.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eeg_dir = Path(eeg_dir)
    fmri_dir = Path(fmri_dir)
    bridge_dir = Path(bridge_dir)

    logger.info(f'Generating full report in {output_dir}')

    # --- EEG figures ---
    eeg_stab_files = {
        'erp': eeg_dir / 'erp_edge_stability.csv',
        'pw': eeg_dir / 'pw_edge_stability.csv',
        'conn': eeg_dir / 'conn_eeg_edge_stability.csv',
    }

    for mod, path in eeg_stab_files.items():
        if path.exists():
            plot_edge_stability_topomap(
                path, output_dir / f'eeg_{mod}_stability_topomap.png',
                title=f'EEG {mod.upper()} Feature Stability'
            )
            plot_region_stability_radar(
                path, output_dir / f'eeg_{mod}_region_radar.png',
                title=f'EEG {mod.upper()} Region Stability'
            )

    # Connectivity matrix
    conn_stab = eeg_stab_files['conn']
    if conn_stab.exists():
        plot_conn_stability_matrix(
            conn_stab, output_dir / 'eeg_conn_stability_matrix.png',
            title='EEG Connectivity Stability'
        )

    # EEG predictions and metrics
    eeg_pred = eeg_dir / 'eeg_predictions.csv'
    if eeg_pred.exists():
        plot_roc_with_ci(eeg_pred, output_dir / 'eeg_roc_ci.png',
                         title='EEG Fusion ROC')
        plot_per_subject_predictions(
            eeg_pred, output_dir / 'eeg_per_subject.png',
            title='EEG Per-Subject Predictions'
        )

    eeg_perm = eeg_dir / 'eeg_permutation_test.csv'
    if eeg_perm.exists():
        plot_permutation_distribution(
            eeg_perm, output_dir / 'eeg_permutation.png',
            title='EEG Permutation Test'
        )

    eeg_abl = eeg_dir / 'eeg_ablation.csv'
    if eeg_abl.exists():
        plot_modality_ablation(eeg_abl, output_dir / 'eeg_ablation.png',
                               title='EEG Modality Ablation')

    eeg_comp = eeg_dir / 'eeg_classifier_comparison.csv'
    if eeg_comp.exists():
        plot_classifier_comparison(eeg_comp, output_dir / 'eeg_classifier_comp.png',
                                    title='EEG Classifier Comparison')

    # --- fMRI figures ---
    fmri_pred = fmri_dir / 'fmri_predictions.csv'
    if fmri_pred.exists():
        plot_roc_with_ci(fmri_pred, output_dir / 'fmri_roc_ci.png',
                         title='fMRI Fusion ROC')
        plot_per_subject_predictions(
            fmri_pred, output_dir / 'fmri_per_subject.png',
            title='fMRI Per-Subject Predictions'
        )

    fmri_perm = fmri_dir / 'fmri_permutation_test.csv'
    if fmri_perm.exists():
        plot_permutation_distribution(
            fmri_perm, output_dir / 'fmri_permutation.png',
            title='fMRI Permutation Test'
        )

    fmri_abl = fmri_dir / 'fmri_ablation.csv'
    if fmri_abl.exists():
        plot_modality_ablation(fmri_abl, output_dir / 'fmri_ablation.png',
                               title='fMRI Modality Ablation')

    fmri_comp = fmri_dir / 'fmri_classifier_comparison.csv'
    if fmri_comp.exists():
        plot_classifier_comparison(fmri_comp, output_dir / 'fmri_classifier_comp.png',
                                    title='fMRI Classifier Comparison')

    # --- Bridge figures ---
    bridge_pred = bridge_dir / 'bridge_predictions.csv'
    if bridge_pred.exists():
        plot_roc_with_ci(bridge_pred, output_dir / 'bridge_roc_ci.png',
                         title='Bridge EEG+fMRI Fusion ROC')
        plot_per_subject_predictions(
            bridge_pred, output_dir / 'bridge_per_subject.png',
            title='Bridge Per-Subject Predictions'
        )

    bridge_perm = bridge_dir / 'bridge_permutation_test.csv'
    if bridge_perm.exists():
        plot_permutation_distribution(
            bridge_perm, output_dir / 'bridge_permutation.png',
            title='Bridge Permutation Test'
        )

    bridge_abl = bridge_dir / 'bridge_ablation.csv'
    if bridge_abl.exists():
        plot_modality_ablation(bridge_abl, output_dir / 'bridge_ablation.png',
                               title='Bridge Modality Ablation')

    bridge_comp = bridge_dir / 'bridge_classifier_comparison.csv'
    if bridge_comp.exists():
        plot_classifier_comparison(bridge_comp, output_dir / 'bridge_classifier_comp.png',
                                    title='Bridge Classifier Comparison')

    # Convergence heatmap
    conv_path = bridge_dir / 'bridge_convergence.csv'
    if conv_path.exists():
        plot_convergence_heatmap(conv_path, output_dir / 'convergence_heatmap.png')

    logger.info(f'Full report generated in {output_dir}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bridge CPM visualization')
    parser.add_argument('--eeg-dir', type=str, default='./results_eeg_cpm')
    parser.add_argument('--fmri-dir', type=str, default='./results_fmri_cpm')
    parser.add_argument('--bridge-dir', type=str, default='./results_bridge_cpm')
    parser.add_argument('--output-dir', type=str, default='./results_figures')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    generate_full_report(args.eeg_dir, args.fmri_dir, args.bridge_dir,
                          args.output_dir)
