"""EEG CPM Fusion — Linear SVR classifier on CSS features.

Takes the css_matrix_eeg.csv from eeg_cpm_loso and trains a Linear SVR
using nested LOSO cross-validation. Includes permutation testing,
bootstrap confidence intervals, modality ablation, and clinical utility.

Usage:
    python -m EEG_CODE.eeg_cpm_fusion
"""

import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.svm import LinearSVR, SVR
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Nested LOSO for hyperparameter selection
# ---------------------------------------------------------------------------

def nested_loso_select_C(X_train, y_train,
                          C_candidates=(0.01, 0.1, 1.0, 10.0)):
    """Inner LOSO loop to select the best SVR regularization parameter C.

    Args:
        X_train: ndarray (N_train, n_features).
        y_train: ndarray (N_train,).
        C_candidates: tuple of C values to evaluate.

    Returns:
        best_C: the C with highest inner-loop AUC.
    """
    N = X_train.shape[0]
    if N < 3:
        return C_candidates[2] if len(C_candidates) > 2 else C_candidates[0]

    best_auc = -1.0
    best_C = C_candidates[0]

    for C in C_candidates:
        scores = np.zeros(N)
        for i in range(N):
            inner_mask = np.ones(N, dtype=bool)
            inner_mask[i] = False

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_train[inner_mask])
            X_te = scaler.transform(X_train[i:i+1])

            svr = LinearSVR(C=C, max_iter=10000, dual='auto')
            svr.fit(X_tr, y_train[inner_mask])
            scores[i] = svr.predict(X_te)[0]

        # AUC requires both classes
        if len(np.unique(y_train)) < 2:
            continue
        try:
            auc = roc_auc_score(y_train, scores)
        except ValueError:
            continue

        if auc > best_auc:
            best_auc = auc
            best_C = C

    logger.debug(f'Inner CV selected C={best_C} (AUC={best_auc:.3f})')
    return best_C


# ---------------------------------------------------------------------------
# Outer LOSO fusion
# ---------------------------------------------------------------------------

def run_fusion_loso(css_df, classifier_type='linear_svr',
                     C_candidates=(0.01, 0.1, 1.0, 10.0)):
    """Outer LOSO cross-validation with per-fold scaling and nested C selection.

    Args:
        css_df: DataFrame with columns subject_id, CSS_*_pos, CSS_*_neg, label.
        classifier_type: 'linear_svr', 'rbf_svr', or 'logistic_regression'.
        C_candidates: regularization parameters to search.

    Returns:
        results_df: DataFrame with subject_id, true_label, pred_score, pred_class.
        metrics: dict with AUC, accuracy, and per-fold details.
    """
    # Extract feature columns (all CSS columns)
    feat_cols = [c for c in css_df.columns if c.startswith('CSS_')]
    X = css_df[feat_cols].values
    y = css_df['label'].values.astype(int)
    subjects = css_df['subject_id'].values

    N = len(y)
    pred_scores = np.zeros(N)

    for i in range(N):
        train_mask = np.ones(N, dtype=bool)
        train_mask[i] = False

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_mask])
        X_test = scaler.transform(X[i:i+1])
        y_train = y[train_mask]

        # Nested CV for C selection
        best_C = nested_loso_select_C(X_train, y_train, C_candidates)

        # Fit final model
        if classifier_type == 'linear_svr':
            clf = LinearSVR(C=best_C, max_iter=10000, dual='auto')
        elif classifier_type == 'rbf_svr':
            clf = SVR(C=best_C, kernel='rbf')
        elif classifier_type == 'logistic_regression':
            clf = LogisticRegression(C=best_C, max_iter=10000, solver='lbfgs')
        else:
            raise ValueError(f'Unknown classifier: {classifier_type}')

        clf.fit(X_train, y_train)

        if classifier_type == 'logistic_regression':
            pred_scores[i] = clf.predict_proba(X_test)[0, 1]
        else:
            pred_scores[i] = clf.predict(X_test)[0]

    # Compute metrics
    auc = roc_auc_score(y, pred_scores)

    # Youden's J for threshold
    fpr, tpr, thresholds = roc_curve(y, pred_scores)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]
    pred_classes = (pred_scores >= best_threshold).astype(int)
    accuracy = np.mean(pred_classes == y)

    results_df = pd.DataFrame({
        'subject_id': subjects,
        'true_label': y,
        'pred_score': pred_scores,
        'pred_class': pred_classes,
    })

    metrics = {
        'auc': auc,
        'accuracy': accuracy,
        'threshold': best_threshold,
        'classifier': classifier_type,
        'n_subjects': N,
        'n_features': len(feat_cols),
        'fpr': fpr,
        'tpr': tpr,
    }

    logger.info(f'Fusion LOSO ({classifier_type}): AUC={auc:.3f}, '
                f'Acc={accuracy:.3f}, threshold={best_threshold:.3f}')
    return results_df, metrics


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------

def permutation_test(css_df, n_permutations=1000, classifier_type='linear_svr',
                      random_state=42):
    """Permutation test for statistical significance.

    Shuffles labels and re-runs the full LOSO pipeline to build a null
    distribution of AUC values.

    Args:
        css_df: DataFrame with CSS features and labels.
        n_permutations: number of permutations.
        classifier_type: classifier to use.
        random_state: seed for reproducibility.

    Returns:
        result: dict with observed_auc, null_aucs, null_mean, null_std, p_value.
    """
    rng = np.random.RandomState(random_state)

    # Observed AUC
    _, obs_metrics = run_fusion_loso(css_df, classifier_type)
    observed_auc = obs_metrics['auc']

    # Null distribution
    null_aucs = np.zeros(n_permutations)
    feat_cols = [c for c in css_df.columns if c.startswith('CSS_')]

    for perm in range(n_permutations):
        if (perm + 1) % 100 == 0:
            logger.info(f'Permutation {perm + 1}/{n_permutations}')

        perm_df = css_df.copy()
        perm_df['label'] = rng.permutation(perm_df['label'].values)

        try:
            _, perm_metrics = run_fusion_loso(perm_df, classifier_type)
            null_aucs[perm] = perm_metrics['auc']
        except Exception:
            null_aucs[perm] = 0.5

    p_value = (np.sum(null_aucs >= observed_auc) + 1) / (n_permutations + 1)

    result = {
        'observed_auc': observed_auc,
        'null_aucs': null_aucs,
        'null_mean': null_aucs.mean(),
        'null_std': null_aucs.std(),
        'p_value': p_value,
    }
    logger.info(f'Permutation test: observed AUC={observed_auc:.3f}, '
                f'null mean={result["null_mean"]:.3f}, p={p_value:.4f}')
    return result


# ---------------------------------------------------------------------------
# Bootstrap confidence interval
# ---------------------------------------------------------------------------

def bootstrap_ci(results_df, n_bootstrap=1000, ci=0.95, random_state=42):
    """Bootstrap 95% CI on AUC using per-subject predictions.

    Args:
        results_df: DataFrame with true_label and pred_score columns.
        n_bootstrap: number of bootstrap iterations.
        ci: confidence level.
        random_state: seed.

    Returns:
        dict with auc, ci_lower, ci_upper, bootstrap_aucs.
    """
    rng = np.random.RandomState(random_state)
    y_true = results_df['true_label'].values
    y_score = results_df['pred_score'].values
    N = len(y_true)

    observed_auc = roc_auc_score(y_true, y_score)
    boot_aucs = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        idx = rng.choice(N, size=N, replace=True)
        y_b = y_true[idx]
        s_b = y_score[idx]
        if len(np.unique(y_b)) < 2:
            boot_aucs[b] = np.nan
            continue
        boot_aucs[b] = roc_auc_score(y_b, s_b)

    boot_aucs = boot_aucs[~np.isnan(boot_aucs)]
    alpha = (1 - ci) / 2
    ci_lower = np.percentile(boot_aucs, 100 * alpha)
    ci_upper = np.percentile(boot_aucs, 100 * (1 - alpha))

    logger.info(f'Bootstrap CI: AUC={observed_auc:.3f} '
                f'[{ci_lower:.3f}, {ci_upper:.3f}]')
    return {
        'auc': observed_auc,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'bootstrap_aucs': boot_aucs,
    }


# ---------------------------------------------------------------------------
# Modality ablation
# ---------------------------------------------------------------------------

def modality_ablation(css_df, classifier_type='linear_svr'):
    """Drop one modality pair (pos+neg) at a time and re-evaluate.

    Args:
        css_df: DataFrame with CSS features and labels.
        classifier_type: classifier to use.

    Returns:
        ablation_df: DataFrame with modality_dropped, auc, accuracy.
    """
    feat_cols = [c for c in css_df.columns if c.startswith('CSS_')]

    # Identify modality groups (e.g., CSS_ERP_pos and CSS_ERP_neg)
    modalities = set()
    for col in feat_cols:
        # Strip _pos or _neg suffix to get modality name
        if col.endswith('_pos'):
            modalities.add(col[4:-4])  # CSS_XXX_pos → XXX
        elif col.endswith('_neg'):
            modalities.add(col[4:-4])  # CSS_XXX_neg → XXX

    results = []

    # Full model
    _, full_metrics = run_fusion_loso(css_df, classifier_type)
    results.append({
        'modality_dropped': 'none',
        'auc': full_metrics['auc'],
        'accuracy': full_metrics['accuracy'],
    })

    # Drop each modality
    for mod in sorted(modalities):
        drop_cols = [f'CSS_{mod}_pos', f'CSS_{mod}_neg']
        keep_cols = [c for c in feat_cols if c not in drop_cols]
        if not keep_cols:
            continue

        ablated_df = css_df[['subject_id'] + keep_cols + ['label']].copy()
        _, abl_metrics = run_fusion_loso(ablated_df, classifier_type)
        results.append({
            'modality_dropped': mod,
            'auc': abl_metrics['auc'],
            'accuracy': abl_metrics['accuracy'],
        })

    ablation_df = pd.DataFrame(results)
    logger.info(f'Ablation results:\n{ablation_df.to_string(index=False)}')
    return ablation_df


# ---------------------------------------------------------------------------
# Clinical utility metrics
# ---------------------------------------------------------------------------

def clinical_utility_metrics(true_labels, pred_scores):
    """Compute sensitivity, specificity, PPV, NPV at Youden's J threshold.

    Args:
        true_labels: ndarray of binary labels.
        pred_scores: ndarray of continuous prediction scores.

    Returns:
        dict with sensitivity, specificity, ppv, npv, threshold, auc.
    """
    fpr, tpr, thresholds = roc_curve(true_labels, pred_scores)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    threshold = thresholds[best_idx]

    pred_classes = (pred_scores >= threshold).astype(int)
    y = true_labels.astype(int)

    tp = np.sum((pred_classes == 1) & (y == 1))
    tn = np.sum((pred_classes == 0) & (y == 0))
    fp = np.sum((pred_classes == 1) & (y == 0))
    fn = np.sum((pred_classes == 0) & (y == 1))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'threshold': threshold,
        'auc': roc_auc_score(true_labels, pred_scores),
    }


# ---------------------------------------------------------------------------
# Classifier comparison
# ---------------------------------------------------------------------------

def run_comparison_classifiers(css_df):
    """Compare Linear SVR, RBF SVR, and Logistic Regression.

    Args:
        css_df: DataFrame with CSS features and labels.

    Returns:
        comparison_df: DataFrame with classifier, auc, accuracy.
    """
    classifiers = ['linear_svr', 'rbf_svr', 'logistic_regression']
    results = []

    for clf_type in classifiers:
        logger.info(f'Running comparison: {clf_type}')
        _, metrics = run_fusion_loso(css_df, clf_type)
        results.append({
            'classifier': clf_type,
            'auc': metrics['auc'],
            'accuracy': metrics['accuracy'],
        })

    comparison_df = pd.DataFrame(results)
    logger.info(f'Classifier comparison:\n{comparison_df.to_string(index=False)}')
    return comparison_df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_eeg_fusion_analysis(css_path='./results_eeg_cpm/css_matrix_eeg.csv',
                             output_dir='./results_eeg_cpm',
                             n_permutations=1000):
    """Run full EEG fusion analysis pipeline.

    Args:
        css_path: path to css_matrix_eeg.csv.
        output_dir: directory for outputs.
        n_permutations: number of permutation test iterations.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    css_df = pd.read_csv(css_path)
    logger.info(f'Loaded CSS matrix: {css_df.shape}')

    # 1. Main fusion LOSO
    logger.info('=== EEG Fusion LOSO (Linear SVR) ===')
    results_df, metrics = run_fusion_loso(css_df, 'linear_svr')
    results_df.to_csv(output_dir / 'eeg_predictions.csv', index=False)

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
        output_dir / 'eeg_permutation_test.csv', index=False
    )

    # 5. Modality ablation
    logger.info('=== Modality Ablation ===')
    ablation_df = modality_ablation(css_df)
    ablation_df.to_csv(output_dir / 'eeg_ablation.csv', index=False)

    # 6. Classifier comparison
    logger.info('=== Classifier Comparison ===')
    comparison_df = run_comparison_classifiers(css_df)
    comparison_df.to_csv(output_dir / 'eeg_classifier_comparison.csv', index=False)

    # Summary
    summary = {
        'auc': metrics['auc'],
        'accuracy': metrics['accuracy'],
        'ci_lower': boot['ci_lower'],
        'ci_upper': boot['ci_upper'],
        'p_value': perm['p_value'],
        **clinical,
    }
    pd.DataFrame([summary]).to_csv(output_dir / 'eeg_summary_metrics.csv', index=False)
    logger.info(f'EEG fusion analysis complete. Results in {output_dir}')

    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EEG CPM Fusion analysis')
    parser.add_argument('--css-path', type=str,
                        default='./results_eeg_cpm/css_matrix_eeg.csv')
    parser.add_argument('--output-dir', type=str, default='./results_eeg_cpm')
    parser.add_argument('--n-permutations', type=int, default=1000)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    run_eeg_fusion_analysis(args.css_path, args.output_dir, args.n_permutations)
