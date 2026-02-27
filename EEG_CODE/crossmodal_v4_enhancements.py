"""
CrossModal V4 Enhancements
===========================

This module contains all enhanced components for CrossModal V4.
Import this in your notebook and use the enhanced models.

New Features:
- Temporal transformers for sequential modeling
- Learned fusion weights with temperature scaling
- Multi-scale convolutions
- Optional GNN connectivity encoder
- Enhanced model architectures

Usage in notebook:
    from crossmodal_v4_enhancements import *
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple

# ============================================================================
# 1. TEMPORAL TRANSFORMER COMPONENTS
# ============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3 and x.size(1) != 1:
            x = x.transpose(0, 1)
            x = x + self.pe[:x.size(0)]
            x = x.transpose(0, 1)
        else:
            x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TemporalTransformerBlock(nn.Module):
    """Multi-head self-attention transformer for temporal modeling."""

    def __init__(self, d_model: int, nhead: int = 4, dim_feedforward: int = 512,
                 dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual
        x2 = self.norm1(x)
        x2, _ = self.self_attn(x2, x2, x2, attn_mask=mask)
        x = x + self.dropout1(x2)

        # Feedforward with residual
        x2 = self.norm2(x)
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x2))))
        x = x + self.dropout2(x2)

        return x


# ============================================================================
# 2. ENHANCED ENCODERS
# ============================================================================

class EnhancedERPEncoder(nn.Module):
    """Enhanced ERP encoder with 1D CNN + Temporal Transformer."""

    def __init__(self, in_channels: int, hidden_dim: int = 128,
                 num_transformer_layers: int = 2, num_heads: int = 4, dropout: float = 0.3):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),

            nn.Conv1d(128, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=dropout)

        self.transformer_layers = nn.ModuleList([
            TemporalTransformerBlock(hidden_dim, num_heads, hidden_dim*4, dropout)
            for _ in range(num_transformer_layers)
        ])

        self.output_proj = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = x.transpose(1, 2)
        x = self.pos_encoder(x)

        for transformer in self.transformer_layers:
            x = transformer(x)

        x = x.transpose(1, 2)
        x = self.output_proj(x)
        return x


class EnhancedPowerEncoder(nn.Module):
    """Enhanced Power encoder with multi-scale CNN + Temporal Transformer."""

    def __init__(self, in_channels: int, hidden_dim: int = 128,
                 num_transformer_layers: int = 2, num_heads: int = 4, dropout: float = 0.3):
        super().__init__()

        # Multi-scale convolutions (capture different temporal patterns)
        self.conv_scale1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU()
        )
        self.conv_scale2 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU()
        )
        self.conv_scale3 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU()
        )

        self.fusion = nn.Sequential(
            nn.Conv1d(192, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=dropout)

        self.transformer_layers = nn.ModuleList([
            TemporalTransformerBlock(hidden_dim, num_heads, hidden_dim*4, dropout)
            for _ in range(num_transformer_layers)
        ])

        self.output_proj = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multi-scale feature extraction
        s1 = self.conv_scale1(x)
        s2 = self.conv_scale2(x)
        s3 = self.conv_scale3(x)

        x = torch.cat([s1, s2, s3], dim=1)
        x = self.fusion(x)

        x = x.transpose(1, 2)
        x = self.pos_encoder(x)

        for transformer in self.transformer_layers:
            x = transformer(x)

        x = x.transpose(1, 2)
        x = self.output_proj(x)
        return x


# ============================================================================
# 3. LEARNED FUSION MODULE
# ============================================================================

class LearnedFusionModule(nn.Module):
    """Learned fusion weights with temperature scaling for dynamic modality weighting."""

    def __init__(self, num_modalities: int, hidden_dim: int,
                 use_temperature: bool = True, init_temperature: float = 1.0):
        super().__init__()
        self.num_modalities = num_modalities
        self.use_temperature = use_temperature

        # Learnable fusion weights (logits)
        self.fusion_logits = nn.Parameter(torch.ones(num_modalities))

        if use_temperature:
            self.temperature = nn.Parameter(torch.tensor(init_temperature))
        else:
            self.register_buffer('temperature', torch.tensor(1.0))

        # Attention-based gating (input-dependent weighting)
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * num_modalities, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_modalities)
        )

    def forward(self, modality_features: List[torch.Tensor],
                return_weights: bool = False) -> torch.Tensor:
        """
        Args:
            modality_features: List of (batch, hidden_dim) tensors
            return_weights: Whether to return fusion weights

        Returns:
            fused: (batch, hidden_dim)
            weights: (batch, num_modalities) if return_weights=True
        """
        stacked = torch.stack(modality_features, dim=1)

        # Static fusion weights (learned globally)
        static_weights = F.softmax(self.fusion_logits / self.temperature, dim=0)

        # Dynamic gating (sample-specific)
        concat_features = torch.cat(modality_features, dim=1)
        dynamic_logits = self.gate_net(concat_features)
        dynamic_weights = F.softmax(dynamic_logits / self.temperature, dim=1)

        # Combine static and dynamic (50/50 mix)
        combined_weights = 0.5 * static_weights.unsqueeze(0) + 0.5 * dynamic_weights

        # Apply weights
        weighted = stacked * combined_weights.unsqueeze(2)
        fused = weighted.sum(dim=1)

        if return_weights:
            return fused, combined_weights
        return fused


# ============================================================================
# 4. ENHANCED TRI-MODAL FUSION MODEL (V4)
# ============================================================================

class EnhancedTriModalFusionNetV4(nn.Module):
    """
    Enhanced tri-modal fusion model with:
    - Temporal transformers for ERP and Power
    - Learned fusion weights with temperature scaling
    - Multi-scale convolutions
    - Cross-modal attention

    This is the V4 replacement for ImprovedTriModalFusionNet.
    """

    def __init__(self, erp_channels: int, pw_channels: int, conn_features: int,
                 hidden_dim: int = 128, num_classes: int = 2, dropout: float = 0.3,
                 num_transformer_layers: int = 2, num_heads: int = 4):
        super().__init__()

        # Enhanced encoders with transformers
        self.erp_encoder = EnhancedERPEncoder(
            erp_channels, hidden_dim, num_transformer_layers, num_heads, dropout
        )

        self.pw_encoder = EnhancedPowerEncoder(
            pw_channels, hidden_dim, num_transformer_layers, num_heads, dropout
        )

        # Connectivity encoder (keep MLP for simplicity)
        self.conn_encoder = nn.Sequential(
            nn.Linear(conn_features, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Cross-modal attention (ERP attends to PW and CONN)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # Learned fusion module (replaces hard-coded weights)
        self.fusion = LearnedFusionModule(
            num_modalities=3,
            hidden_dim=hidden_dim,
            use_temperature=True
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, erp: torch.Tensor, pw: torch.Tensor, conn: torch.Tensor,
                return_fusion_weights: bool = False, return_fused_feats: bool = False):
        """
        Args:
            erp: (batch, channels, time)
            pw: (batch, channels, time)
            conn: (batch, conn_features) - flattened connectivity
            return_fusion_weights: Return fusion weights for analysis
            return_fused_feats: Return fused features before classifier

        Returns:
            logits: (batch, num_classes)
            weights: (batch, 3) if return_fusion_weights=True
            fused: (batch, hidden_dim) if return_fused_feats=True
        """
        # Encode each modality
        erp_feat = self.erp_encoder(erp)
        pw_feat = self.pw_encoder(pw)

        # Flatten connectivity if needed
        if conn.dim() > 2:
            batch_size = conn.size(0)
            conn = conn.view(batch_size, -1)
        conn_feat = self.conn_encoder(conn)

        # Cross-modal attention (ERP attends to others)
        modality_stack = torch.stack([erp_feat, pw_feat, conn_feat], dim=1)
        enhanced_erp, _ = self.cross_attn(
            erp_feat.unsqueeze(1),
            modality_stack,
            modality_stack
        )
        enhanced_erp = enhanced_erp.squeeze(1)

        # Learned fusion
        if return_fusion_weights:
            fused, weights = self.fusion(
                [enhanced_erp, pw_feat, conn_feat],
                return_weights=True
            )
        else:
            fused = self.fusion([enhanced_erp, pw_feat, conn_feat])
            weights = None

        # Classification
        logits = self.classifier(fused)

        # Return based on flags
        if return_fusion_weights and return_fused_feats:
            return logits, weights, fused
        elif return_fusion_weights:
            return logits, weights
        elif return_fused_feats:
            return logits, fused
        return logits


# ============================================================================
# 5. BI-DIRECTIONAL CROSS-MODAL ATTENTION
# ============================================================================

class BiDirectionalCrossAttention(nn.Module):
    """
    Bi-directional cross-modal attention module.
    Both modalities attend to each other, capturing mutual relationships.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.3):
        super().__init__()

        # ERP attends to PW
        self.erp_to_pw_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        # PW attends to ERP
        self.pw_to_erp_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # Layer norms for stability
        self.norm_erp = nn.LayerNorm(hidden_dim)
        self.norm_pw = nn.LayerNorm(hidden_dim)

        # Gating mechanism to control cross-modal influence
        self.erp_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        self.pw_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, erp_feat: torch.Tensor, pw_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            erp_feat: (batch, hidden_dim)
            pw_feat: (batch, hidden_dim)

        Returns:
            enhanced_erp: (batch, hidden_dim)
            enhanced_pw: (batch, hidden_dim)
        """
        # Add sequence dimension for attention
        erp_seq = erp_feat.unsqueeze(1)  # (batch, 1, hidden_dim)
        pw_seq = pw_feat.unsqueeze(1)    # (batch, 1, hidden_dim)

        # Stack for key/value (each modality can attend to both)
        combined = torch.cat([erp_seq, pw_seq], dim=1)  # (batch, 2, hidden_dim)

        # Cross-attention: ERP attends to PW
        erp_attended, _ = self.erp_to_pw_attn(erp_seq, combined, combined)
        erp_attended = erp_attended.squeeze(1)

        # Cross-attention: PW attends to ERP
        pw_attended, _ = self.pw_to_erp_attn(pw_seq, combined, combined)
        pw_attended = pw_attended.squeeze(1)

        # Gated residual connection
        erp_gate_input = torch.cat([erp_feat, erp_attended], dim=1)
        erp_gate_val = self.erp_gate(erp_gate_input)
        enhanced_erp = self.norm_erp(erp_feat + self.dropout(erp_gate_val * erp_attended))

        pw_gate_input = torch.cat([pw_feat, pw_attended], dim=1)
        pw_gate_val = self.pw_gate(pw_gate_input)
        enhanced_pw = self.norm_pw(pw_feat + self.dropout(pw_gate_val * pw_attended))

        return enhanced_erp, enhanced_pw


# ============================================================================
# 6. ENHANCED BI-MODAL FUSION MODEL (V4) - WITH CROSS-MODAL ATTENTION
# ============================================================================

class EnhancedSmartFusionNetV4(nn.Module):
    """
    Enhanced bi-modal fusion model (ERP + PW) with:
    - Temporal transformers for sequential modeling
    - Bi-directional cross-modal attention (NEW)
    - Learned gated fusion with temperature scaling
    - Multi-scale convolutions
    - Stronger regularization for small datasets

    This is the V4 replacement for ImprovedSmartFusionNet.
    """

    def __init__(self, erp_channels: int, pw_channels: int,
                 hidden_dim: int = 128, num_classes: int = 2, dropout: float = 0.4,
                 num_transformer_layers: int = 2, num_heads: int = 4,
                 use_cross_attention: bool = True):
        super().__init__()

        self.use_cross_attention = use_cross_attention

        # Enhanced encoders with transformers
        self.erp_encoder = EnhancedERPEncoder(
            erp_channels, hidden_dim, num_transformer_layers, num_heads, dropout
        )

        self.pw_encoder = EnhancedPowerEncoder(
            pw_channels, hidden_dim, num_transformer_layers, num_heads, dropout
        )

        # Bi-directional cross-modal attention (NEW)
        if use_cross_attention:
            self.cross_attention = BiDirectionalCrossAttention(
                hidden_dim, num_heads=num_heads, dropout=dropout
            )

        # Learned fusion with temperature scaling
        self.fusion = LearnedFusionModule(
            num_modalities=2,
            hidden_dim=hidden_dim,
            use_temperature=True
        )

        # Deeper classifier head (matching tri-modal architecture)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, erp: torch.Tensor, pw: torch.Tensor,
                return_fusion_weights: bool = False, return_fused_feats: bool = False):
        """
        Args:
            erp: (batch, channels, time)
            pw: (batch, channels, time)
            return_fusion_weights: Return fusion weights for analysis
            return_fused_feats: Return fused features before classifier

        Returns:
            logits: (batch, num_classes)
            weights: (batch, 2) if return_fusion_weights=True
            fused: (batch, hidden_dim) if return_fused_feats=True
        """
        # Encode each modality
        erp_feat = self.erp_encoder(erp)
        pw_feat = self.pw_encoder(pw)

        # Apply bi-directional cross-modal attention
        if self.use_cross_attention:
            erp_feat, pw_feat = self.cross_attention(erp_feat, pw_feat)

        # Learned fusion
        if return_fusion_weights:
            fused, weights = self.fusion([erp_feat, pw_feat], return_weights=True)
        else:
            fused = self.fusion([erp_feat, pw_feat])
            weights = None

        # Classification
        logits = self.classifier(fused)

        # Return based on flags
        if return_fusion_weights and return_fused_feats:
            return logits, weights, fused
        elif return_fusion_weights:
            return logits, weights
        elif return_fused_feats:
            return logits, fused
        return logits


# ============================================================================
# 7. UTILITY FUNCTIONS
# ============================================================================

def get_fusion_weights_from_model(model):
    """Extract current fusion weights from enhanced model."""
    if not hasattr(model, 'fusion') or not hasattr(model.fusion, 'fusion_logits'):
        return None

    with torch.no_grad():
        logits = model.fusion.fusion_logits
        temp = model.fusion.temperature
        weights = F.softmax(logits / temp, dim=0)

        result = {
            'temperature': temp.item()
        }

        if len(weights) == 3:
            result.update({
                'erp_weight': weights[0].item(),
                'pw_weight': weights[1].item(),
                'conn_weight': weights[2].item()
            })
        elif len(weights) == 2:
            result.update({
                'erp_weight': weights[0].item(),
                'pw_weight': weights[1].item()
            })

        return result


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# MODULE INFO
# ============================================================================

print("="*60)
print("CrossModal V4 Enhancements Module Loaded")
print("="*60)
print("\nAvailable classes:")
print("  - EnhancedTriModalFusionNetV4 (tri-modal: ERP + PW + Connectivity)")
print("  - EnhancedSmartFusionNetV4 (bi-modal: ERP + PW, with cross-attention)")
print("  - BiDirectionalCrossAttention (NEW: mutual cross-modal attention)")
print("  - EnhancedERPEncoder (1D CNN + temporal transformers)")
print("  - EnhancedPowerEncoder (multi-scale CNN + transformers)")
print("  - LearnedFusionModule (learned fusion weights with temperature)")
print("\nUtility functions:")
print("  - get_fusion_weights_from_model(model)")
print("  - count_parameters(model)")
print("\nV4.1 Updates:")
print("  - Bi-modal now has bi-directional cross-modal attention")
print("  - Increased dropout (0.3 -> 0.4) for small datasets")
print("  - Deeper classifier head matching tri-modal")
print("="*60 + "\n")


# ============================================================================
# 8. STOCHASTIC DEPTH (DROP PATH) FOR REGULARIZATION
# ============================================================================

def drop_path(x: torch.Tensor, drop_prob: float = 0., training: bool = False) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


# ============================================================================
# 9. LABEL SMOOTHING CROSS ENTROPY
# ============================================================================

class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy with label smoothing for better generalization."""
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logprobs = F.log_softmax(pred, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


# ============================================================================
# 10. ENHANCED CONNECTIVITY ENCODER WITH ATTENTION
# ============================================================================

class EnhancedConnEncoder(nn.Module):
    """
    Enhanced connectivity encoder with:
    - Residual connections
    - Feature attention mechanism
    - Better normalization
    """
    def __init__(self, conn_features: int, hidden_dim: int = 96, dropout: float = 0.4):
        super().__init__()

        # First projection with residual
        self.proj1 = nn.Sequential(
            nn.Linear(conn_features, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Second projection
        self.proj2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Feature attention (learn which connections matter)
        self.attention = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Sigmoid()
        )

        # Output projection
        self.output = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        x = self.proj1(x)
        x = self.proj2(x)

        # Apply feature attention
        attn_weights = self.attention(x)
        x = x * attn_weights

        x = self.output(x)
        return x


# ============================================================================
# 11. HYBRID FUSION MODULE (EARLY FOR ERP-PW, LATE FOR CONN)
# ============================================================================

class HybridFusionModule(nn.Module):
    """
    Hybrid fusion strategy:
    - Early fusion for ERP + PW (similar signal types)
    - Late fusion for Connectivity (different modality type)

    This prevents connectivity from being overwhelmed by the other modalities.
    """
    def __init__(self, hidden_dim: int, dropout: float = 0.3, conn_boost: float = 1.2):
        super().__init__()
        self.conn_boost = conn_boost  # Boost connectivity contribution

        # ERP-PW early fusion (learned gating)
        self.erp_pw_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1)
        )

        # Late fusion with connectivity
        self.late_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Final gating between early-fused and connectivity
        self.final_gate = nn.Parameter(torch.tensor([0.6, 0.4]))  # Initial bias toward ERP-PW

    def forward(self, erp_feat: torch.Tensor, pw_feat: torch.Tensor,
                conn_feat: torch.Tensor, return_weights: bool = False):
        """
        Args:
            erp_feat: (batch, hidden_dim)
            pw_feat: (batch, hidden_dim)
            conn_feat: (batch, hidden_dim)
        """
        # Early fusion: ERP + PW
        erp_pw_concat = torch.cat([erp_feat, pw_feat], dim=1)
        gate_weights = self.erp_pw_gate(erp_pw_concat)

        erp_pw_fused = gate_weights[:, 0:1] * erp_feat + gate_weights[:, 1:2] * pw_feat

        # Late fusion: (ERP-PW) + CONN
        # Boost connectivity to prevent it from being ignored
        conn_boosted = conn_feat * self.conn_boost

        final_weights = F.softmax(self.final_gate, dim=0)

        # Concatenate and project
        combined = torch.cat([erp_pw_fused, conn_boosted], dim=1)
        fused = self.late_fusion(combined)

        if return_weights:
            weights = {
                'erp_weight': gate_weights[:, 0].mean().item() * final_weights[0].item(),
                'pw_weight': gate_weights[:, 1].mean().item() * final_weights[0].item(),
                'conn_weight': final_weights[1].item() * self.conn_boost
            }
            return fused, weights

        return fused


# ============================================================================
# 12. LIGHTWEIGHT TRIMODAL MODEL (V4-LITE)
# ============================================================================

class LiteERPEncoder(nn.Module):
    """Lightweight ERP encoder - reduced complexity."""
    def __init__(self, in_channels: int, hidden_dim: int = 96, dropout: float = 0.4):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels, 48, kernel_size=7, padding=3),
            nn.BatchNorm1d(48),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(2),

            nn.Conv1d(48, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1)
        )

        self.output = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        return self.output(x)


class LitePowerEncoder(nn.Module):
    """Lightweight Power encoder - reduced complexity."""
    def __init__(self, in_channels: int, hidden_dim: int = 96, dropout: float = 0.4):
        super().__init__()

        # Single scale instead of multi-scale
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(2),

            nn.Conv1d(64, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1)
        )

        self.output = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        return self.output(x)


class EnhancedTriModalFusionNetV4Lite(nn.Module):
    """
    Lightweight tri-modal fusion model designed for small datasets.

    Key differences from V4:
    - ~400K params instead of 1.26M
    - No transformer layers (just CNN)
    - Hybrid fusion (early ERP-PW, late CONN)
    - Stronger regularization
    - Connectivity boost to prevent weight collapse

    Use this when you have < 500 subjects.
    """
    def __init__(self, erp_channels: int, pw_channels: int, conn_features: int,
                 hidden_dim: int = 96, num_classes: int = 2, dropout: float = 0.4,
                 conn_boost: float = 1.3):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Lightweight encoders (no transformers)
        self.erp_encoder = LiteERPEncoder(erp_channels, hidden_dim, dropout)
        self.pw_encoder = LitePowerEncoder(pw_channels, hidden_dim, dropout)
        self.conn_encoder = EnhancedConnEncoder(conn_features, hidden_dim, dropout)

        # Hybrid fusion module
        self.fusion = HybridFusionModule(hidden_dim, dropout, conn_boost)

        # Simpler classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # For tracking
        self._fusion_weights = None

    def forward(self, erp: torch.Tensor, pw: torch.Tensor, conn: torch.Tensor,
                return_fusion_weights: bool = False, return_fused_feats: bool = False):
        # Encode
        erp_feat = self.erp_encoder(erp)
        pw_feat = self.pw_encoder(pw)
        conn_feat = self.conn_encoder(conn)

        # Hybrid fusion
        if return_fusion_weights:
            fused, weights = self.fusion(erp_feat, pw_feat, conn_feat, return_weights=True)
            self._fusion_weights = weights
        else:
            fused = self.fusion(erp_feat, pw_feat, conn_feat)
            weights = None

        # Classify
        logits = self.classifier(fused)

        if return_fusion_weights and return_fused_feats:
            return logits, weights, fused
        elif return_fusion_weights:
            return logits, weights
        elif return_fused_feats:
            return logits, fused
        return logits

    def get_fusion_weights(self):
        """Return last computed fusion weights."""
        return self._fusion_weights


# ============================================================================
# 13. BALANCED TRIMODAL DATASET
# ============================================================================

class BalancedTriModalDataset(torch.utils.data.Dataset):
    """
    Balanced tri-modal dataset that handles modality sample count mismatch.

    Strategy: Aggregate ERP and PW per-subject to match CONN granularity.
    This ensures each modality contributes equally during training.
    """
    def __init__(self, erp_features: dict, pw_features: dict, conn_features: dict,
                 label_dict: dict, transform=None, agg_method: str = 'mean'):
        """
        Args:
            erp_features: Dict[subject_id] -> (feature_tensor, metadata)
            pw_features: Dict[subject_id] -> (feature_tensor, metadata)
            conn_features: Dict[subject_id] -> (feature_tensor, metadata)
            label_dict: Dict[subject_id] -> label
            transform: Optional transform to apply
            agg_method: 'mean' or 'max' for aggregating multiple samples per subject
        """
        self.transform = transform
        self.agg_method = agg_method
        self.samples = []

        # Get common subjects across all modalities
        erp_subjects = set(self._extract_subjects(erp_features))
        pw_subjects = set(self._extract_subjects(pw_features))
        conn_subjects = set(self._extract_subjects(conn_features))

        common_subjects = erp_subjects & pw_subjects & conn_subjects
        print(f"BalancedTriModalDataset: Found {len(common_subjects)} common subjects")
        print(f"  ERP subjects: {len(erp_subjects)}, PW subjects: {len(pw_subjects)}, CONN subjects: {len(conn_subjects)}")

        # Aggregate features per subject
        erp_by_subj = self._aggregate_by_subject(erp_features, agg_method)
        pw_by_subj = self._aggregate_by_subject(pw_features, agg_method)
        conn_by_subj = self._aggregate_by_subject(conn_features, agg_method)

        # Build samples for common subjects only
        for subj in sorted(common_subjects):
            if subj in label_dict:
                erp_feat = erp_by_subj.get(subj)
                pw_feat = pw_by_subj.get(subj)
                conn_feat = conn_by_subj.get(subj)
                label = label_dict[subj]

                if erp_feat is not None and pw_feat is not None and conn_feat is not None:
                    self.samples.append({
                        'erp': erp_feat,
                        'pw': pw_feat,
                        'conn': conn_feat,
                        'label': label,
                        'subject': subj
                    })

        print(f"BalancedTriModalDataset: Created {len(self.samples)} balanced samples")

    def _extract_subjects(self, features_dict):
        """Extract unique subject IDs from feature dictionary."""
        subjects = set()
        for key in features_dict.keys():
            if isinstance(key, tuple):
                subjects.add(key[0])  # (subject, band, freq) format
            else:
                subjects.add(key)
        return subjects

    def _aggregate_by_subject(self, features_dict, method='mean'):
        """Aggregate multiple samples per subject."""
        from collections import defaultdict
        import numpy as np

        subj_features = defaultdict(list)

        for key, value in features_dict.items():
            if isinstance(key, tuple):
                subj = key[0]
            else:
                subj = key

            # Handle different value formats
            if isinstance(value, tuple):
                feat = value[0]  # (feature, metadata) format
            else:
                feat = value

            if isinstance(feat, torch.Tensor):
                feat = feat.numpy()
            #subj_features[subj].append(feat)
            # Flatten to 1D for consistent aggregation across different shapes
            feat_flat = feat.flatten()
            subj_features[subj].append(feat_flat)

        # Aggregate
        aggregated = {}
        for subj, feats in subj_features.items():
            stacked = np.stack(feats, axis=0)
            if method == 'mean':
                agg_feat = np.mean(stacked, axis=0)
            elif method == 'max':
                agg_feat = np.max(stacked, axis=0)
            else:
                agg_feat = stacked[0]  # Just take first
            aggregated[subj] = torch.tensor(agg_feat, dtype=torch.float32)

        return aggregated

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        erp = sample['erp']
        pw = sample['pw']
        conn = sample['conn']
        label = sample['label']
        subj = sample['subject']

        # Apply transforms if any
        if self.transform:
            erp = self.transform(erp)
            pw = self.transform(pw)

        return erp, pw, conn, label, subj


# ============================================================================
# 14. IMPROVED TRAINING UTILITIES
# ============================================================================

class CosineAnnealingWarmup:
    """Learning rate scheduler with warmup + cosine annealing."""
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int,
                 min_lr: float = 1e-6, base_lr: float = None):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = base_lr or optimizer.param_groups[0]['lr']
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1

        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (self.current_epoch / self.warmup_epochs)
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


def get_lite_fusion_weights(model):
    """Extract fusion weights from V4-Lite model."""
    if hasattr(model, 'get_fusion_weights'):
        return model.get_fusion_weights()
    elif hasattr(model, '_fusion_weights'):
        return model._fusion_weights
    return None


# ============================================================================
# MODULE INFO UPDATE
# ============================================================================

print("\n" + "="*60)
print("V4-LITE Components for Small Datasets Added")
print("="*60)
print("\nNew classes for improved trimodal performance:")
print("  - EnhancedTriModalFusionNetV4Lite (lightweight, ~400K params)")
print("  - BalancedTriModalDataset (handles sample count mismatch)")
print("  - HybridFusionModule (early ERP-PW + late CONN fusion)")
print("  - EnhancedConnEncoder (with feature attention)")
print("  - LiteERPEncoder, LitePowerEncoder (no transformers)")
print("\nTraining utilities:")
print("  - CosineAnnealingWarmup (LR scheduler)")
print("  - EarlyStopping (prevent overfitting)")
print("  - LabelSmoothingCrossEntropy")
print("  - DropPath (stochastic depth)")
print("\nRecommended for datasets with < 500 subjects")
print("="*60 + "\n")
