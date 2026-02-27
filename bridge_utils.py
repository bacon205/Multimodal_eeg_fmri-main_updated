"""Bridge-specific components for EEG-fMRI cross-modal fusion.

Contains the bridge fusion model, dataset, and XAI analysis tools.
Extracted from bridge notebook cells 10, 11, 14, 15, 17.
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import List

from EEG_CODE.crossmodal_v4_enhancements import LearnedFusionModule

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bridge Fusion Model 
# ---------------------------------------------------------------------------
class EEGfMRIBridgeFusionNet(nn.Module):
    """Cross-modality bridge fusion: EEG (128-d) + fMRI (64-d).

    Projects both modalities to a shared space, applies cross-modal
    attention, learned temperature-scaled fusion, and classification.
    """
    def __init__(self, eeg_dim=128, fmri_dim=64, bridge_dim=128,
                 num_classes=2, num_heads=4, dropout=0.3):
        super().__init__()
        self.bridge_dim = bridge_dim

        # Project to shared space
        self.eeg_proj = nn.Sequential(
            nn.Linear(eeg_dim, bridge_dim),
            nn.LayerNorm(bridge_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.fmri_proj = nn.Sequential(
            nn.Linear(fmri_dim, bridge_dim),
            nn.LayerNorm(bridge_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Cross-modal attention (EEG and fMRI attend to each other)
        self.cross_attn = nn.MultiheadAttention(
            bridge_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # Learned fusion with temperature scaling
        self.fusion = LearnedFusionModule(
            num_modalities=2,
            hidden_dim=bridge_dim,
            use_temperature=True
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(bridge_dim, bridge_dim // 2),
            nn.LayerNorm(bridge_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(bridge_dim // 2, num_classes)
        )

    def forward(self, eeg_feats, fmri_feats, return_features=False, return_weights=False):
      
        # Project to shared space
        eeg_proj = self.eeg_proj(eeg_feats)    # (batch, bridge_dim)
        fmri_proj = self.fmri_proj(fmri_feats)  # (batch, bridge_dim)

        # Cross-modal attention: stack as sequence of 2 tokens
        modality_seq = torch.stack([eeg_proj, fmri_proj], dim=1)  # (batch, 2, bridge_dim)

        # EEG attends to both modalities
        eeg_q = eeg_proj.unsqueeze(1)  # (batch, 1, bridge_dim)
        attn_out, attn_weights_raw = self.cross_attn(
            eeg_q, modality_seq, modality_seq
        )
        eeg_enhanced = attn_out.squeeze(1)  # (batch, bridge_dim)

        # Learned fusion
        if return_weights:
            fused, fusion_weights = self.fusion(
                [eeg_enhanced, fmri_proj], return_weights=True
            )
        else:
            fused = self.fusion([eeg_enhanced, fmri_proj])
            fusion_weights = None

        # Classify
        logits = self.classifier(fused)

        results = [logits]
        if return_features:
            results.append(fused)
        if return_weights:
            results.append(fusion_weights)
            results.append(attn_weights_raw)

        return results[0] if len(results) == 1 else tuple(results)

    def get_fusion_weights(self):
        with torch.no_grad():
            logits = self.fusion.fusion_logits
            temp = self.fusion.temperature
            weights = F.softmax(logits / temp, dim=0)
            return {
                'eeg_weight': weights[0].item(),
                'fmri_weight': weights[1].item(),
                'temperature': temp.item()
            }


# ---------------------------------------------------------------------------
# Bridge Feature Dataset + collate (Cell 11)
# ---------------------------------------------------------------------------
class BridgeFeatureDataset(Dataset):
    """Dataset of pre-extracted EEG and fMRI features, aligned by subject."""
    def __init__(self, eeg_features, fmri_features, labels, subject_list):
        self.samples = []
        
        # Ensure we are comparing the same types (Force all to int)
        # This fixes the alignment error where '001' != 1
        standardized_eeg = {int(k): v for k, v in eeg_features.items()}
        standardized_fmri = {int(k): v for k, v in fmri_features.items()}
        standardized_labels = {int(k): v for k, v in labels.items()}

        for subj in sorted(subject_list):
            s_id = int(subj)
            if s_id in standardized_eeg and s_id in standardized_fmri and s_id in standardized_labels:
                self.samples.append({
                    'eeg': standardized_eeg[s_id],
                    'fmri': standardized_fmri[s_id],
                    'label': standardized_labels[s_id],
                    'subject': s_id
                })
        
        if len(self.samples) == 0:
            logger.error("!!! NO SAMPLES ALIGNED !!! Check subject IDs in EEG and fMRI feature dicts.")
        else:
            logger.info(f'BridgeFeatureDataset: {len(self.samples)} aligned samples found.')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        # Returns exactly what the model expects: (eeg_tensor, fmri_tensor, label, subject_id)
        return s['eeg'], s['fmri'], s['label'], s['subject']


# ---------------------------------------------------------------------------
# XAI: Gradient Saliency (Cell 14)
# ---------------------------------------------------------------------------
class BridgeGradientSaliency:
    """Gradient saliency for the bridge model."""
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def compute(self, eeg_feats, fmri_feats, target_class=None):
        self.model.eval()
        eeg_feats = eeg_feats.clone().detach().to(self.device).requires_grad_(True)
        fmri_feats = fmri_feats.clone().detach().to(self.device).requires_grad_(True)

        logits = self.model(eeg_feats, fmri_feats)

        if target_class is None:
            target_class = logits.argmax(dim=1)

        self.model.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, target_class.view(-1, 1), 1)
        logits.backward(gradient=one_hot)

        return {
            'eeg': eeg_feats.grad.abs().cpu().numpy(),
            'fmri': fmri_feats.grad.abs().cpu().numpy()
        }


# ---------------------------------------------------------------------------
# XAI: Integrated Gradients (Cell 15)
# ---------------------------------------------------------------------------

class BridgeIntegratedGradients:
    """Integrated Gradients for the bridge model."""
    def __init__(self, model, device, n_steps=50):
        self.model = model
        self.device = device
        self.n_steps = n_steps

    def compute(self, eeg_feats, fmri_feats, target_class=None):
        self.model.eval()
        eeg_feats = eeg_feats.to(self.device)
        fmri_feats = fmri_feats.to(self.device)

        eeg_baseline = torch.zeros_like(eeg_feats)
        fmri_baseline = torch.zeros_like(fmri_feats)

        eeg_diff = eeg_feats - eeg_baseline
        fmri_diff = fmri_feats - fmri_baseline

        eeg_grads, fmri_grads = [], []

        for alpha in np.linspace(0, 1, self.n_steps):
            eeg_interp = (eeg_baseline + alpha * eeg_diff).requires_grad_(True)
            fmri_interp = (fmri_baseline + alpha * fmri_diff).requires_grad_(True)

            logits = self.model(eeg_interp, fmri_interp)

            if target_class is None:
                target_class = logits.argmax(dim=1)

            self.model.zero_grad()
            one_hot = torch.zeros_like(logits)
            one_hot.scatter_(1, target_class.view(-1, 1), 1)
            logits.backward(gradient=one_hot)

            eeg_grads.append(eeg_interp.grad.detach().cpu().numpy())
            fmri_grads.append(fmri_interp.grad.detach().cpu().numpy())

        eeg_ig = eeg_diff.cpu().numpy() * np.mean(eeg_grads, axis=0)
        fmri_ig = fmri_diff.cpu().numpy() * np.mean(fmri_grads, axis=0)

        return {'eeg': np.abs(eeg_ig), 'fmri': np.abs(fmri_ig)}


# ---------------------------------------------------------------------------
# XAI: Attention & Fusion Weight Extraction (Cell 17)
# ---------------------------------------------------------------------------

def extract_attention_and_fusion_weights(model, dataset, device):
    """Extract cross-modal attention weights and fusion weights per subject.

    Args:
        model: EEGfMRIBridgeFusionNet instance.
        dataset: BridgeFeatureDataset instance.
        device: torch device.

    Returns:
        List of dicts with subject, label, prediction, fusion_weights, attn_weights.
    """
    model.eval()
    subject_data = []

    with torch.no_grad():
        for idx in range(len(dataset)):
            eeg, fmri, label, subj = dataset[idx]
            eeg_t = eeg.unsqueeze(0).to(device)
            fmri_t = fmri.unsqueeze(0).to(device)

            logits, fused, fusion_weights, attn_weights = model(
                eeg_t, fmri_t, return_features=True, return_weights=True
            )

            subject_data.append({
                'subject': subj,
                'label': label,
                'prediction': logits.argmax(dim=1).item(),
                'fusion_weights': fusion_weights.cpu().numpy().squeeze(),
                'attn_weights': attn_weights.cpu().numpy().squeeze(),
            })

    return subject_data
