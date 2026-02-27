"""
EEG Explainable AI (XAI) Analysis Module
=========================================

This module provides interpretability tools for EEG multimodal fusion models:
- Gradient-based saliency maps
- Integrated Gradients attribution
- SHAP analysis
- Channel importance extraction for 10-20 system mapping

Usage:
    from eeg_xai_analysis import EEGExplainer, plot_channel_importance, plot_topomap
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict
import warnings

# ============================================================================
# 1. STANDARD 10-20 EEG CHANNEL DEFINITIONS
# ============================================================================

# Standard 10-20 system channel names (common configurations)
STANDARD_10_20_19 = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'T3', 'C3', 'Cz', 'C4', 'T4',
    'T5', 'P3', 'Pz', 'P4', 'T6',
    'O1', 'O2'
]

STANDARD_10_20_21 = STANDARD_10_20_19 + ['A1', 'A2']  # With mastoids

# Extended 10-10 system (common 32-channel)
EXTENDED_10_10_32 = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'FC5', 'FC1', 'FC2', 'FC6',
    'T7', 'C3', 'Cz', 'C4', 'T8',
    'CP5', 'CP1', 'CP2', 'CP6',
    'P7', 'P3', 'Pz', 'P4', 'P8',
    'PO3', 'PO4', 'O1', 'Oz', 'O2',
    'AF3', 'AF4'
]

# 2D coordinates for topographic plotting (normalized 0-1)
# Based on standard 10-20 positions
CHANNEL_POSITIONS_2D = {
    # Frontal
    'Fp1': (0.35, 0.95), 'Fp2': (0.65, 0.95), 'Fpz': (0.50, 0.95),
    'AF3': (0.38, 0.88), 'AF4': (0.62, 0.88), 'AFz': (0.50, 0.88),
    'F7': (0.15, 0.75), 'F3': (0.35, 0.75), 'Fz': (0.50, 0.75),
    'F4': (0.65, 0.75), 'F8': (0.85, 0.75),
    'FC5': (0.22, 0.65), 'FC1': (0.40, 0.65), 'FC2': (0.60, 0.65), 'FC6': (0.78, 0.65),
    # Central
    'T7': (0.08, 0.50), 'T3': (0.08, 0.50),  # T7 = T3 in older nomenclature
    'C3': (0.30, 0.50), 'Cz': (0.50, 0.50), 'C4': (0.70, 0.50),
    'T8': (0.92, 0.50), 'T4': (0.92, 0.50),  # T8 = T4
    'CP5': (0.22, 0.35), 'CP1': (0.40, 0.35), 'CP2': (0.60, 0.35), 'CP6': (0.78, 0.35),
    # Parietal
    'T5': (0.15, 0.25), 'P7': (0.15, 0.25),  # P7 = T5
    'P3': (0.35, 0.25), 'Pz': (0.50, 0.25), 'P4': (0.65, 0.25),
    'T6': (0.85, 0.25), 'P8': (0.85, 0.25),  # P8 = T6
    'PO3': (0.38, 0.15), 'PO4': (0.62, 0.15), 'POz': (0.50, 0.15),
    # Occipital
    'O1': (0.35, 0.05), 'Oz': (0.50, 0.05), 'O2': (0.65, 0.05),
    # Mastoids
    'A1': (0.02, 0.50), 'A2': (0.98, 0.50),
    'M1': (0.02, 0.50), 'M2': (0.98, 0.50),
}

# Brain region groupings for summary statistics
BRAIN_REGIONS = {
    'Frontal': ['Fp1', 'Fp2', 'Fpz', 'F7', 'F3', 'Fz', 'F4', 'F8', 'AF3', 'AF4'],
    'Central': ['C3', 'Cz', 'C4', 'FC1', 'FC2', 'FC5', 'FC6'],
    'Temporal': ['T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'P7', 'P8'],
    'Parietal': ['P3', 'Pz', 'P4', 'CP1', 'CP2', 'CP5', 'CP6'],
    'Occipital': ['O1', 'Oz', 'O2', 'PO3', 'PO4']
}


# ============================================================================
# 2. GRADIENT-BASED SALIENCY METHODS
# ============================================================================

class GradientSaliency:
    """Compute gradient-based saliency maps for EEG models."""

    def __init__(self, model: nn.Module, device: torch.device = None):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    def vanilla_gradient(self, erp: torch.Tensor, pw: torch.Tensor,
                         conn: torch.Tensor = None, target_class: int = None) -> Dict[str, np.ndarray]:
        """
        Compute vanilla gradient saliency (dOutput/dInput).

        Returns:
            Dict with 'erp', 'pw', 'conn' gradient magnitudes per channel
        """
        erp = erp.clone().detach().to(self.device).requires_grad_(True)
        pw = pw.clone().detach().to(self.device).requires_grad_(True)

        if conn is not None:
            conn = conn.clone().detach().to(self.device).requires_grad_(True)
            logits = self.model(pw, erp, conn)
        else:
            logits = self.model(pw, erp)

        # Target class (default: predicted class)
        if target_class is None:
            target_class = logits.argmax(dim=1)

        # Compute gradients
        self.model.zero_grad()

        # One-hot encoding for target
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, target_class.view(-1, 1), 1)

        logits.backward(gradient=one_hot)

        results = {
            'erp': erp.grad.abs().cpu().numpy(),
            'pw': pw.grad.abs().cpu().numpy()
        }

        if conn is not None and conn.grad is not None:
            results['conn'] = conn.grad.abs().cpu().numpy()

        return results

    def gradient_x_input(self, erp: torch.Tensor, pw: torch.Tensor,
                         conn: torch.Tensor = None, target_class: int = None) -> Dict[str, np.ndarray]:
        """
        Compute Gradient × Input saliency (more discriminative than vanilla gradient).
        """
        grads = self.vanilla_gradient(erp, pw, conn, target_class)

        results = {
            'erp': grads['erp'] * np.abs(erp.detach().cpu().numpy()),
            'pw': grads['pw'] * np.abs(pw.detach().cpu().numpy())
        }

        if 'conn' in grads:
            results['conn'] = grads['conn'] * np.abs(conn.detach().cpu().numpy())

        return results


class IntegratedGradients:
    """
    Integrated Gradients attribution method.
    More accurate than vanilla gradients, satisfies axioms of attribution.
    """

    def __init__(self, model: nn.Module, device: torch.device = None, n_steps: int = 50):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_steps = n_steps
        self.model.to(self.device)
        self.model.eval()

    def compute(self, erp: torch.Tensor, pw: torch.Tensor,
                conn: torch.Tensor = None, target_class: int = None,
                baseline: str = 'zero') -> Dict[str, np.ndarray]:
        """
        Compute Integrated Gradients attribution.

        Args:
            baseline: 'zero' or 'mean' - reference point for integration
        """
        erp = erp.to(self.device)
        pw = pw.to(self.device)

        # Create baselines
        if baseline == 'zero':
            erp_baseline = torch.zeros_like(erp)
            pw_baseline = torch.zeros_like(pw)
        else:  # mean
            erp_baseline = erp.mean(dim=0, keepdim=True).expand_as(erp)
            pw_baseline = pw.mean(dim=0, keepdim=True).expand_as(pw)

        # Interpolation
        erp_diff = erp - erp_baseline
        pw_diff = pw - pw_baseline

        erp_grads = []
        pw_grads = []
        conn_grads = []

        for alpha in np.linspace(0, 1, self.n_steps):
            erp_interp = erp_baseline + alpha * erp_diff
            pw_interp = pw_baseline + alpha * pw_diff

            erp_interp.requires_grad_(True)
            pw_interp.requires_grad_(True)

            if conn is not None:
                conn_interp = conn.clone().to(self.device).requires_grad_(True)
                logits = self.model(pw_interp, erp_interp, conn_interp)
            else:
                logits = self.model(pw_interp, erp_interp)

            if target_class is None:
                target_class = logits.argmax(dim=1)

            self.model.zero_grad()
            one_hot = torch.zeros_like(logits)
            one_hot.scatter_(1, target_class.view(-1, 1), 1)
            logits.backward(gradient=one_hot)

            erp_grads.append(erp_interp.grad.detach().cpu().numpy())
            pw_grads.append(pw_interp.grad.detach().cpu().numpy())

            if conn is not None:
                conn_grads.append(conn_interp.grad.detach().cpu().numpy())

        # Integrate (approximate with mean)
        erp_ig = erp_diff.cpu().numpy() * np.mean(erp_grads, axis=0)
        pw_ig = pw_diff.cpu().numpy() * np.mean(pw_grads, axis=0)

        results = {
            'erp': np.abs(erp_ig),
            'pw': np.abs(pw_ig)
        }

        if conn_grads:
            conn_diff = conn.cpu().numpy()
            results['conn'] = np.abs(conn_diff * np.mean(conn_grads, axis=0))

        return results


# ============================================================================
# 3. SHAP ANALYSIS WRAPPER
# ============================================================================

class SHAPExplainer:
    """
    SHAP (SHapley Additive exPlanations) wrapper for EEG models.
    Uses DeepExplainer for neural networks.
    """

    def __init__(self, model: nn.Module, background_data: Dict[str, torch.Tensor],
                 device: torch.device = None):
        """
        Args:
            model: The trained model
            background_data: Dict with 'erp', 'pw', and optionally 'conn' tensors
                            Used as reference distribution (typically training subset)
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.background = background_data
        self.model.to(self.device)
        self.model.eval()

        self._shap_available = self._check_shap()

    def _check_shap(self) -> bool:
        try:
            import shap
            self.shap = shap
            return True
        except ImportError:
            warnings.warn("SHAP not installed. Run: pip install shap")
            return False

    def _model_wrapper_trimodal(self, inputs):
        """Wrapper for tri-modal model to work with SHAP."""
        # inputs is concatenated [erp, pw, conn]
        erp_size = self.erp_shape[1] * self.erp_shape[2]
        pw_size = self.pw_shape[1] * self.pw_shape[2]

        erp = inputs[:, :erp_size].reshape(-1, *self.erp_shape[1:])
        pw = inputs[:, erp_size:erp_size+pw_size].reshape(-1, *self.pw_shape[1:])
        conn = inputs[:, erp_size+pw_size:]

        erp = torch.tensor(erp, dtype=torch.float32, device=self.device)
        pw = torch.tensor(pw, dtype=torch.float32, device=self.device)
        conn = torch.tensor(conn, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            logits = self.model(pw, erp, conn)
        return logits.cpu().numpy()

    def _model_wrapper_bimodal(self, inputs):
        """Wrapper for bi-modal model to work with SHAP."""
        erp_size = self.erp_shape[1] * self.erp_shape[2]

        erp = inputs[:, :erp_size].reshape(-1, *self.erp_shape[1:])
        pw = inputs[:, erp_size:].reshape(-1, *self.pw_shape[1:])

        erp = torch.tensor(erp, dtype=torch.float32, device=self.device)
        pw = torch.tensor(pw, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            logits = self.model(pw, erp)
        return logits.cpu().numpy()

    def compute_shap_values(self, erp: torch.Tensor, pw: torch.Tensor,
                           conn: torch.Tensor = None, n_background: int = 100) -> Dict[str, np.ndarray]:
        """
        Compute SHAP values for input samples.

        Returns:
            Dict with SHAP values for each modality, shape matches input
        """
        if not self._shap_available:
            raise RuntimeError("SHAP not available. Install with: pip install shap")

        self.erp_shape = erp.shape
        self.pw_shape = pw.shape

        # Flatten and concatenate for SHAP
        erp_flat = erp.cpu().numpy().reshape(erp.shape[0], -1)
        pw_flat = pw.cpu().numpy().reshape(pw.shape[0], -1)

        if conn is not None:
            conn_flat = conn.cpu().numpy().reshape(conn.shape[0], -1)
            test_data = np.concatenate([erp_flat, pw_flat, conn_flat], axis=1)
            wrapper = self._model_wrapper_trimodal

            # Background
            bg_erp = self.background['erp'][:n_background].cpu().numpy().reshape(n_background, -1)
            bg_pw = self.background['pw'][:n_background].cpu().numpy().reshape(n_background, -1)
            bg_conn = self.background['conn'][:n_background].cpu().numpy().reshape(n_background, -1)
            background = np.concatenate([bg_erp, bg_pw, bg_conn], axis=1)
        else:
            test_data = np.concatenate([erp_flat, pw_flat], axis=1)
            wrapper = self._model_wrapper_bimodal

            bg_erp = self.background['erp'][:n_background].cpu().numpy().reshape(n_background, -1)
            bg_pw = self.background['pw'][:n_background].cpu().numpy().reshape(n_background, -1)
            background = np.concatenate([bg_erp, bg_pw], axis=1)

        # Create explainer
        explainer = self.shap.KernelExplainer(wrapper, background)
        shap_values = explainer.shap_values(test_data, nsamples=100)

        # Reshape SHAP values back to original dimensions
        erp_size = np.prod(self.erp_shape[1:])
        pw_size = np.prod(self.pw_shape[1:])

        # shap_values is list of [n_samples, n_features] for each class
        # Take class 1 (positive class) importance
        if isinstance(shap_values, list):
            sv = shap_values[1]  # Class 1
        else:
            sv = shap_values

        results = {
            'erp': np.abs(sv[:, :erp_size]).reshape(-1, *self.erp_shape[1:]),
            'pw': np.abs(sv[:, erp_size:erp_size+pw_size]).reshape(-1, *self.pw_shape[1:])
        }

        if conn is not None:
            results['conn'] = np.abs(sv[:, erp_size+pw_size:])

        return results


# ============================================================================
# 4. CHANNEL IMPORTANCE EXTRACTION
# ============================================================================

class ChannelImportanceExtractor:
    """
    Extract and aggregate channel-level importance from attribution maps.
    Maps to 10-20 system channel labels.
    """

    def __init__(self, channel_names: List[str] = None, n_channels: int = None):
        """
        Args:
            channel_names: List of channel names in order (e.g., ['Fp1', 'Fp2', ...])
            n_channels: Number of channels (used if names not provided)
        """
        if channel_names is not None:
            self.channel_names = channel_names
            self.n_channels = len(channel_names)
        elif n_channels is not None:
            self.n_channels = n_channels
            # Use standard 10-20 if matching, otherwise generic names
            if n_channels == 19:
                self.channel_names = STANDARD_10_20_19
            elif n_channels == 21:
                self.channel_names = STANDARD_10_20_21
            elif n_channels == 32:
                self.channel_names = EXTENDED_10_10_32
            else:
                self.channel_names = [f'Ch{i+1}' for i in range(n_channels)]
        else:
            raise ValueError("Must provide either channel_names or n_channels")

    def extract_channel_importance(self, attribution: np.ndarray,
                                   modality: str = 'erp') -> Dict[str, float]:
        """
        Extract per-channel importance scores from attribution map.

        Args:
            attribution: Attribution array of shape (batch, channels, time/freq) or (batch, features)
            modality: 'erp', 'pw', or 'conn'

        Returns:
            Dict mapping channel names to importance scores
        """
        if attribution.ndim == 2:
            # Already flattened - assume channels × features
            # Reshape assuming equal features per channel
            n_samples = attribution.shape[0]
            n_features = attribution.shape[1]
            features_per_channel = n_features // self.n_channels
            attribution = attribution.reshape(n_samples, self.n_channels, features_per_channel)

        # Aggregate over time/frequency dimension (axis=2) and samples (axis=0)
        channel_importance = np.mean(np.mean(attribution, axis=2), axis=0)

        # Normalize to sum to 1
        channel_importance = channel_importance / (channel_importance.sum() + 1e-8)

        return {name: float(imp) for name, imp in zip(self.channel_names, channel_importance)}

    def extract_connectivity_importance(self, attribution: np.ndarray) -> Dict[Tuple[str, str], float]:
        """
        Extract channel-pair importance from connectivity attribution.

        Args:
            attribution: Attribution array for connectivity features

        Returns:
            Dict mapping (channel1, channel2) tuples to importance scores
        """
        n_samples = attribution.shape[0]
        n_conn_features = attribution.shape[1] if attribution.ndim == 2 else np.prod(attribution.shape[1:])

        # Flatten if needed
        attr_flat = attribution.reshape(n_samples, -1)

        # Number of connectivity metrics (PLV, Coherence, wPLI = 3)
        n_pairs = self.n_channels * (self.n_channels - 1) // 2
        n_metrics = n_conn_features // n_pairs

        # Reshape to (samples, metrics, pairs)
        attr_reshaped = attr_flat.reshape(n_samples, n_metrics, n_pairs)

        # Average over metrics and samples
        pair_importance = np.mean(np.mean(attr_reshaped, axis=1), axis=0)

        # Map to channel pairs
        pair_dict = {}
        idx = 0
        for i in range(self.n_channels):
            for j in range(i+1, self.n_channels):
                pair_dict[(self.channel_names[i], self.channel_names[j])] = float(pair_importance[idx])
                idx += 1

        # Normalize
        total = sum(pair_dict.values()) + 1e-8
        return {k: v/total for k, v in pair_dict.items()}

    def get_region_importance(self, channel_importance: Dict[str, float]) -> Dict[str, float]:
        """
        Aggregate channel importance by brain region.
        """
        region_importance = {}

        for region, channels in BRAIN_REGIONS.items():
            matching = [channel_importance.get(ch, 0) for ch in channels if ch in channel_importance]
            if matching:
                region_importance[region] = float(np.mean(matching))
            else:
                region_importance[region] = 0.0

        return region_importance

    def get_top_channels(self, channel_importance: Dict[str, float], k: int = 5) -> List[Tuple[str, float]]:
        """Return top-k most important channels."""
        sorted_channels = sorted(channel_importance.items(), key=lambda x: x[1], reverse=True)
        return sorted_channels[:k]

    def get_top_connections(self, conn_importance: Dict[Tuple[str, str], float],
                           k: int = 10) -> List[Tuple[Tuple[str, str], float]]:
        """Return top-k most important connections."""
        sorted_conns = sorted(conn_importance.items(), key=lambda x: x[1], reverse=True)
        return sorted_conns[:k]


# ============================================================================
# 5. COMPREHENSIVE EEG EXPLAINER
# ============================================================================

class EEGExplainer:
    """
    Unified explainability interface for EEG multimodal models.
    Combines multiple attribution methods and provides channel-level analysis.
    """

    def __init__(self, model: nn.Module, channel_names: List[str] = None,
                 n_channels: int = None, device: torch.device = None):
        """
        Args:
            model: Trained EEG model (ImprovedTriModalFusionNet or ImprovedSmartFusionNet)
            channel_names: EEG channel names in order
            n_channels: Number of EEG channels (auto-detect if not provided)
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

        # Initialize analysis tools
        self.gradient_saliency = GradientSaliency(model, device)
        self.integrated_gradients = IntegratedGradients(model, device)

        # Channel mapper (will be set when first sample is analyzed)
        self.channel_extractor = None
        self.channel_names = channel_names
        self.n_channels = n_channels

        # Store results
        self.results_history = []

    def _init_channel_extractor(self, sample_erp: torch.Tensor):
        """Initialize channel extractor from sample data."""
        if self.channel_extractor is None:
            n_ch = sample_erp.shape[1] if sample_erp.dim() == 3 else sample_erp.shape[0]
            self.channel_extractor = ChannelImportanceExtractor(
                channel_names=self.channel_names,
                n_channels=self.n_channels or n_ch
            )

    def analyze_sample(self, erp: torch.Tensor, pw: torch.Tensor,
                      conn: torch.Tensor = None, target_class: int = None,
                      methods: List[str] = ['gradient', 'integrated_gradients']) -> Dict:
        """
        Comprehensive analysis of a single sample or batch.

        Args:
            erp: ERP tensor (batch, channels, time)
            pw: Power tensor (batch, channels, time)
            conn: Connectivity tensor (optional)
            target_class: Target class for attribution (None = predicted)
            methods: List of methods to use

        Returns:
            Dict with channel importance, region importance, and raw attributions
        """
        self._init_channel_extractor(erp)

        results = {
            'attributions': {},
            'channel_importance': {},
            'region_importance': {},
            'top_channels': {},
            'prediction': None
        }

        # Get prediction
        with torch.no_grad():
            erp_dev = erp.to(self.device)
            pw_dev = pw.to(self.device)
            if conn is not None:
                conn_dev = conn.to(self.device)
                logits = self.model(pw_dev, erp_dev, conn_dev)
            else:
                logits = self.model(pw_dev, erp_dev)

            probs = F.softmax(logits, dim=1)
            results['prediction'] = {
                'class': logits.argmax(dim=1).cpu().numpy(),
                'probabilities': probs.cpu().numpy()
            }

        # Compute attributions with each method
        for method in methods:
            if method == 'gradient':
                attr = self.gradient_saliency.gradient_x_input(erp, pw, conn, target_class)
            elif method == 'integrated_gradients':
                attr = self.integrated_gradients.compute(erp, pw, conn, target_class)
            else:
                continue

            results['attributions'][method] = attr

            # Extract channel importance for each modality
            results['channel_importance'][method] = {}
            results['region_importance'][method] = {}
            results['top_channels'][method] = {}

            for modality in ['erp', 'pw']:
                if modality in attr:
                    ch_imp = self.channel_extractor.extract_channel_importance(
                        attr[modality], modality
                    )
                    results['channel_importance'][method][modality] = ch_imp
                    results['region_importance'][method][modality] = \
                        self.channel_extractor.get_region_importance(ch_imp)
                    results['top_channels'][method][modality] = \
                        self.channel_extractor.get_top_channels(ch_imp, k=5)

            # Connectivity importance
            if 'conn' in attr:
                conn_imp = self.channel_extractor.extract_connectivity_importance(attr['conn'])
                results['channel_importance'][method]['connectivity'] = conn_imp
                results['top_channels'][method]['connectivity'] = \
                    self.channel_extractor.get_top_connections(conn_imp, k=10)

        self.results_history.append(results)
        return results

    def analyze_dataset(self, dataloader, methods: List[str] = ['gradient'],
                       max_samples: int = 100) -> Dict:
        """
        Analyze multiple samples and aggregate results.

        Returns:
            Aggregated channel importance across samples
        """
        all_channel_importance = defaultdict(lambda: defaultdict(list))
        all_region_importance = defaultdict(lambda: defaultdict(list))

        n_analyzed = 0
        for batch in dataloader:
            if n_analyzed >= max_samples:
                break

            # Handle different dataset formats
            if len(batch) == 5:  # TriModalDataset
                erp, pw, conn, _, labels = batch
            elif len(batch) == 4:  # BiModal
                erp, pw, labels = batch[0], batch[1], batch[3]
                conn = None
            else:
                continue

            results = self.analyze_sample(erp, pw, conn, methods=methods)

            for method in methods:
                for modality in ['erp', 'pw']:
                    if modality in results['channel_importance'].get(method, {}):
                        ch_imp = results['channel_importance'][method][modality]
                        for ch, imp in ch_imp.items():
                            all_channel_importance[method][f'{modality}_{ch}'].append(imp)

                        reg_imp = results['region_importance'][method][modality]
                        for reg, imp in reg_imp.items():
                            all_region_importance[method][f'{modality}_{reg}'].append(imp)

            n_analyzed += erp.shape[0]

        # Aggregate (mean across samples)
        aggregated = {
            'channel_importance': {},
            'region_importance': {},
            'n_samples': n_analyzed
        }

        for method in methods:
            aggregated['channel_importance'][method] = {
                k: float(np.mean(v)) for k, v in all_channel_importance[method].items()
            }
            aggregated['region_importance'][method] = {
                k: float(np.mean(v)) for k, v in all_region_importance[method].items()
            }

        return aggregated

    def get_channel_ranking(self, modality: str = 'erp', method: str = 'gradient') -> List[Tuple[str, float]]:
        """
        Get overall channel ranking from analysis history.
        """
        if not self.results_history:
            raise ValueError("No analysis results. Run analyze_sample first.")

        # Aggregate across all analyzed samples
        channel_scores = defaultdict(list)

        for result in self.results_history:
            if method in result['channel_importance']:
                if modality in result['channel_importance'][method]:
                    for ch, score in result['channel_importance'][method][modality].items():
                        channel_scores[ch].append(score)

        # Mean scores
        mean_scores = {ch: np.mean(scores) for ch, scores in channel_scores.items()}

        return sorted(mean_scores.items(), key=lambda x: x[1], reverse=True)


# ============================================================================
# 6. VISUALIZATION UTILITIES
# ============================================================================

def plot_channel_importance(channel_importance: Dict[str, float],
                           title: str = "Channel Importance",
                           top_k: int = None,
                           figsize: Tuple[int, int] = (12, 6),
                           save_path: str = None):
    """
    Bar plot of channel importance scores.
    """
    import matplotlib.pyplot as plt

    # Sort by importance
    sorted_items = sorted(channel_importance.items(), key=lambda x: x[1], reverse=True)

    if top_k:
        sorted_items = sorted_items[:top_k]

    channels, scores = zip(*sorted_items)

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(range(len(channels)), scores, color='steelblue')
    ax.set_yticks(range(len(channels)))
    ax.set_yticklabels(channels)
    ax.invert_yaxis()
    ax.set_xlabel('Importance Score')
    ax.set_title(title)

    # Add value labels
    for bar, score in zip(bars, scores):
        ax.text(score + 0.001, bar.get_y() + bar.get_height()/2,
                f'{score:.3f}', va='center', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, ax


def plot_topomap(channel_importance: Dict[str, float],
                title: str = "EEG Topographic Map",
                figsize: Tuple[int, int] = (8, 8),
                cmap: str = 'RdYlBu_r',
                save_path: str = None):
    """
    Plot topographic map of channel importance on a head schematic.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    import matplotlib.colors as mcolors

    fig, ax = plt.subplots(figsize=figsize)

    # Draw head outline
    head = Circle((0.5, 0.5), 0.45, fill=False, linewidth=2, color='black')
    ax.add_patch(head)

    # Draw nose
    ax.plot([0.5, 0.5], [0.95, 1.0], 'k-', linewidth=2)
    ax.plot([0.45, 0.5, 0.55], [0.95, 1.0, 0.95], 'k-', linewidth=2)

    # Draw ears
    ax.plot([0.05, 0.02, 0.05], [0.55, 0.5, 0.45], 'k-', linewidth=2)
    ax.plot([0.95, 0.98, 0.95], [0.55, 0.5, 0.45], 'k-', linewidth=2)

    # Get importance values
    values = list(channel_importance.values())
    vmin, vmax = min(values), max(values)

    # Normalize
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.cm.get_cmap(cmap)

    # Plot channels
    for channel, importance in channel_importance.items():
        if channel in CHANNEL_POSITIONS_2D:
            x, y = CHANNEL_POSITIONS_2D[channel]
            color = cmap_obj(norm(importance))

            # Channel marker
            circle = Circle((x, y), 0.04, color=color, ec='black', linewidth=1)
            ax.add_patch(circle)

            # Channel label
            ax.text(x, y, channel, ha='center', va='center', fontsize=8, fontweight='bold')

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, label='Importance')

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.15)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, ax


def plot_region_comparison(region_importance: Dict[str, float],
                          title: str = "Brain Region Importance",
                          figsize: Tuple[int, int] = (10, 6),
                          save_path: str = None):
    """
    Radar/spider plot comparing importance across brain regions.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    regions = list(region_importance.keys())
    values = list(region_importance.values())

    # Close the radar plot
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(regions), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    ax.plot(angles, values, 'o-', linewidth=2, color='steelblue')
    ax.fill(angles, values, alpha=0.25, color='steelblue')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(regions, fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, ax


def plot_connectivity_matrix(conn_importance: Dict[Tuple[str, str], float],
                            channel_names: List[str],
                            title: str = "Connectivity Importance",
                            figsize: Tuple[int, int] = (10, 8),
                            save_path: str = None):
    """
    Plot connectivity importance as a matrix heatmap.
    """
    import matplotlib.pyplot as plt

    n_channels = len(channel_names)
    matrix = np.zeros((n_channels, n_channels))

    ch_to_idx = {ch: i for i, ch in enumerate(channel_names)}

    for (ch1, ch2), importance in conn_importance.items():
        if ch1 in ch_to_idx and ch2 in ch_to_idx:
            i, j = ch_to_idx[ch1], ch_to_idx[ch2]
            matrix[i, j] = importance
            matrix[j, i] = importance

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(matrix, cmap='hot', aspect='auto')
    ax.set_xticks(range(n_channels))
    ax.set_yticks(range(n_channels))
    ax.set_xticklabels(channel_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(channel_names, fontsize=8)

    plt.colorbar(im, ax=ax, label='Importance')
    ax.set_title(title, fontsize=14, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, ax


def create_analysis_report(explainer: EEGExplainer,
                          output_dir: str,
                          modalities: List[str] = ['erp', 'pw']) -> Dict:
    """
    Generate comprehensive XAI report with all visualizations.
    """
    import os
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        'channel_rankings': {},
        'region_rankings': {},
        'figures': []
    }

    for modality in modalities:
        ranking = explainer.get_channel_ranking(modality=modality, method='gradient')
        report['channel_rankings'][modality] = ranking

        # Channel importance bar plot
        ch_imp = dict(ranking[:15])  # Top 15
        fig, _ = plot_channel_importance(
            ch_imp,
            title=f'{modality.upper()} Channel Importance (Top 15)',
            save_path=str(output_dir / f'{modality}_channel_importance.png')
        )
        report['figures'].append(str(output_dir / f'{modality}_channel_importance.png'))

        # Topomap
        fig, _ = plot_topomap(
            dict(ranking),
            title=f'{modality.upper()} Topographic Map',
            save_path=str(output_dir / f'{modality}_topomap.png')
        )
        report['figures'].append(str(output_dir / f'{modality}_topomap.png'))

    # Save text report
    with open(output_dir / 'xai_report.txt', 'w') as f:
        f.write("EEG Explainability Analysis Report\n")
        f.write("=" * 50 + "\n\n")

        for modality in modalities:
            f.write(f"\n{modality.upper()} Channel Ranking:\n")
            f.write("-" * 30 + "\n")
            for rank, (ch, score) in enumerate(report['channel_rankings'][modality][:10], 1):
                f.write(f"{rank:2d}. {ch:6s}: {score:.4f}\n")

    print(f"Report saved to {output_dir}")
    return report


# ============================================================================
# MODULE INFO
# ============================================================================

print("=" * 60)
print("EEG XAI Analysis Module Loaded")
print("=" * 60)
print("\nMain classes:")
print("  - EEGExplainer: Unified explainability interface")
print("  - GradientSaliency: Vanilla gradient and Gradient×Input")
print("  - IntegratedGradients: Axiomatic attribution method")
print("  - SHAPExplainer: SHAP values (requires: pip install shap)")
print("  - ChannelImportanceExtractor: Map attributions to 10-20 system")
print("\nVisualization functions:")
print("  - plot_channel_importance(): Bar chart of channel scores")
print("  - plot_topomap(): EEG topographic head map")
print("  - plot_region_comparison(): Radar plot by brain region")
print("  - plot_connectivity_matrix(): Connectivity heatmap")
print("  - create_analysis_report(): Full report generation")
print("\nPredefined channel configurations:")
print("  - STANDARD_10_20_19: 19-channel 10-20 system")
print("  - STANDARD_10_20_21: 21-channel with mastoids")
print("  - EXTENDED_10_10_32: 32-channel extended system")
print("=" * 60 + "\n")
