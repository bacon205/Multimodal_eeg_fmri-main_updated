"""
Enhanced Models for CrossModal EEG Analysis (V4)
=================================================

This module provides enhanced versions of the models from CrossModal_V3_new.ipynb with:
1. Learned fusion weights with temperature scaling
2. Temporal transformer blocks for sequential modeling
3. GNN-based connectivity encoder (preserves spatial topology)
4. Optuna-based hyperparameter optimization

Author: Enhanced version for improved multi-modal fusion
Date: 2026-01-07
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data, Batch
import numpy as np
import optuna
from typing import Dict, List, Tuple, Optional, Any
import math


# ============================================================================
# 1. TEMPORAL TRANSFORMER BLOCKS
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
        """
        Args:
            x: Tensor of shape (seq_len, batch, d_model) or (batch, seq_len, d_model)
        """
        if x.dim() == 3 and x.size(1) != 1:  # (batch, seq_len, d_model)
            x = x.transpose(0, 1)  # -> (seq_len, batch, d_model)
            x = x + self.pe[:x.size(0)]
            x = x.transpose(0, 1)  # -> (batch, seq_len, d_model)
        else:  # (seq_len, batch, d_model)
            x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TemporalTransformerBlock(nn.Module):
    """Multi-head self-attention transformer block for temporal modeling."""

    def __init__(
        self,
        d_model: int,
        nhead: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: Optional attention mask
        Returns:
            (batch, seq_len, d_model)
        """
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
# 2. ENHANCED ENCODERS WITH TEMPORAL TRANSFORMERS
# ============================================================================

class EnhancedERPEncoder(nn.Module):
    """Enhanced ERP encoder with 1D CNN + Temporal Transformer."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 128,
        num_transformer_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3
    ):
        super().__init__()

        # 1D CNN for feature extraction
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

        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=dropout)

        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TemporalTransformerBlock(
                hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim*4,
                dropout=dropout
            )
            for _ in range(num_transformer_layers)
        ])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, time)
        Returns:
            (batch, hidden_dim)
        """
        # CNN feature extraction
        x = self.conv_layers(x)  # (batch, hidden_dim, time')

        # Prepare for transformer: (batch, time', hidden_dim)
        x = x.transpose(1, 2)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer layers
        for transformer in self.transformer_layers:
            x = transformer(x)

        # Global pooling and projection
        x = x.transpose(1, 2)  # (batch, hidden_dim, time')
        x = self.output_proj(x)

        return x


class EnhancedPowerEncoder(nn.Module):
    """Enhanced Power Spectrum encoder with multi-scale CNN + Temporal Transformer."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 128,
        num_transformer_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3
    ):
        super().__init__()

        # Multi-scale convolutions
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

        # Fusion of multi-scale features
        self.fusion = nn.Sequential(
            nn.Conv1d(192, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Positional encoding and transformer
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=dropout)

        self.transformer_layers = nn.ModuleList([
            TemporalTransformerBlock(
                hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim*4,
                dropout=dropout
            )
            for _ in range(num_transformer_layers)
        ])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, time)
        Returns:
            (batch, hidden_dim)
        """
        # Multi-scale feature extraction
        s1 = self.conv_scale1(x)
        s2 = self.conv_scale2(x)
        s3 = self.conv_scale3(x)

        # Concatenate and fuse
        x = torch.cat([s1, s2, s3], dim=1)
        x = self.fusion(x)

        # Transformer processing
        x = x.transpose(1, 2)
        x = self.pos_encoder(x)

        for transformer in self.transformer_layers:
            x = transformer(x)

        # Output projection
        x = x.transpose(1, 2)
        x = self.output_proj(x)

        return x


# ============================================================================
# 3. GNN-BASED CONNECTIVITY ENCODER
# ============================================================================

class GNNConnectivityEncoder(nn.Module):
    """
    Graph Neural Network encoder for connectivity data.
    Uses actual EEG electrode positions as graph structure.
    """

    def __init__(
        self,
        num_nodes: int,
        num_conn_types: int = 3,  # PLV, COH, WPLI
        hidden_dim: int = 128,
        num_gat_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_conn_types = num_conn_types

        # Node feature projection (each node gets connectivity features)
        self.node_proj = nn.Sequential(
            nn.Linear(num_nodes * num_conn_types, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # GAT layers for graph processing
        self.gat_layers = nn.ModuleList()
        for i in range(num_gat_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim
            self.gat_layers.append(
                GATv2Conv(
                    in_dim,
                    hidden_dim // num_heads,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True
                )
            )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def create_graph_from_connectivity(
        self,
        conn_matrix: torch.Tensor,
        threshold: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create graph edge list from connectivity matrix.

        Args:
            conn_matrix: (batch, num_nodes, num_nodes) connectivity strength
            threshold: Keep edges above this threshold

        Returns:
            edge_index: (2, num_edges)
            edge_attr: (num_edges, 1) edge weights
        """
        # Average across batch for graph structure
        avg_conn = conn_matrix.mean(dim=0)

        # Threshold to get edges
        edges = (avg_conn > threshold).nonzero(as_tuple=False)
        edge_index = edges.t().contiguous()
        edge_attr = avg_conn[edges[:, 0], edges[:, 1]].unsqueeze(1)

        return edge_index, edge_attr

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_nodes, num_nodes, num_conn_types) or flattened
            edge_index: (2, num_edges) graph structure

        Returns:
            (batch, hidden_dim)
        """
        batch_size = x.size(0)

        # Flatten connectivity to node features
        # Each node gets all its connections as features
        if x.dim() == 4:
            # (batch, num_nodes, num_nodes, num_conn_types)
            x = x.view(batch_size, self.num_nodes, -1)
        elif x.dim() == 3:
            # (batch, num_nodes, num_nodes) - single conn type
            x = x.view(batch_size, self.num_nodes, -1)
        else:
            # Already flattened
            x = x.view(batch_size, self.num_nodes, -1)

        # Project node features
        node_features = []
        for i in range(batch_size):
            node_feat = self.node_proj(x[i])  # (num_nodes, hidden_dim)
            node_features.append(node_feat)

        # Process with GAT
        batch_outputs = []
        for i in range(batch_size):
            h = node_features[i]

            for gat_layer in self.gat_layers:
                h = gat_layer(h, edge_index)
                h = F.gelu(h)

            # Global pooling
            h = h.mean(dim=0)  # (hidden_dim,)
            batch_outputs.append(h)

        # Stack batch
        output = torch.stack(batch_outputs, dim=0)  # (batch, hidden_dim)
        output = self.output_proj(output)

        return output


# ============================================================================
# 4. LEARNED FUSION WITH TEMPERATURE SCALING
# ============================================================================

class LearnedFusionModule(nn.Module):
    """
    Learned fusion weights with temperature scaling for dynamic modality weighting.
    """

    def __init__(
        self,
        num_modalities: int,
        hidden_dim: int,
        use_temperature: bool = True,
        init_temperature: float = 1.0
    ):
        super().__init__()
        self.num_modalities = num_modalities
        self.use_temperature = use_temperature

        # Learnable fusion weights (logits)
        self.fusion_logits = nn.Parameter(torch.ones(num_modalities))

        # Temperature parameter
        if use_temperature:
            self.temperature = nn.Parameter(torch.tensor(init_temperature))
        else:
            self.register_buffer('temperature', torch.tensor(1.0))

        # Attention-based gating
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * num_modalities, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_modalities)
        )

    def forward(
        self,
        modality_features: List[torch.Tensor],
        return_weights: bool = False
    ) -> torch.Tensor:
        """
        Args:
            modality_features: List of (batch, hidden_dim) tensors
            return_weights: Whether to return fusion weights

        Returns:
            fused: (batch, hidden_dim)
            weights: (batch, num_modalities) if return_weights=True
        """
        # Stack modalities
        stacked = torch.stack(modality_features, dim=1)  # (batch, num_mod, hidden_dim)
        batch_size = stacked.size(0)

        # Compute static fusion weights (softmax of learned logits)
        static_weights = F.softmax(self.fusion_logits / self.temperature, dim=0)

        # Compute dynamic gating (input-dependent)
        concat_features = torch.cat(modality_features, dim=1)  # (batch, num_mod*hidden_dim)
        dynamic_logits = self.gate_net(concat_features)  # (batch, num_mod)
        dynamic_weights = F.softmax(dynamic_logits / self.temperature, dim=1)

        # Combine static and dynamic weights
        combined_weights = 0.5 * static_weights.unsqueeze(0) + 0.5 * dynamic_weights

        # Apply weights
        weighted = stacked * combined_weights.unsqueeze(2)  # (batch, num_mod, hidden_dim)
        fused = weighted.sum(dim=1)  # (batch, hidden_dim)

        if return_weights:
            return fused, combined_weights
        return fused


# ============================================================================
# 5. ENHANCED TRI-MODAL FUSION MODEL
# ============================================================================

class EnhancedTriModalFusionNet(nn.Module):
    """
    Enhanced tri-modal fusion model with:
    - Temporal transformers for ERP and Power
    - GNN for connectivity
    - Learned fusion weights
    """

    def __init__(
        self,
        erp_channels: int,
        pw_channels: int,
        num_conn_nodes: int,
        num_conn_types: int = 3,
        hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.3,
        num_transformer_layers: int = 2,
        num_heads: int = 4,
        use_gnn: bool = True
    ):
        super().__init__()
        self.use_gnn = use_gnn

        # Encoders
        self.erp_encoder = EnhancedERPEncoder(
            erp_channels,
            hidden_dim,
            num_transformer_layers,
            num_heads,
            dropout
        )

        self.pw_encoder = EnhancedPowerEncoder(
            pw_channels,
            hidden_dim,
            num_transformer_layers,
            num_heads,
            dropout
        )

        if use_gnn:
            self.conn_encoder = GNNConnectivityEncoder(
                num_conn_nodes,
                num_conn_types,
                hidden_dim,
                num_gat_layers=2,
                num_heads=num_heads,
                dropout=dropout
            )
        else:
            # Fallback to MLP encoder
            conn_features = num_conn_nodes * num_conn_nodes * num_conn_types
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

        # Learned fusion
        self.fusion = LearnedFusionModule(
            num_modalities=3,
            hidden_dim=hidden_dim,
            use_temperature=True
        )

        # Cross-modal attention (optional enhancement)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Classifier
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

        # Store edge index for GNN (created once)
        self.edge_index = None

    def forward(
        self,
        erp: torch.Tensor,
        pw: torch.Tensor,
        conn: torch.Tensor,
        return_fusion_weights: bool = False
    ) -> torch.Tensor:
        """
        Args:
            erp: (batch, channels, time)
            pw: (batch, channels, time)
            conn: (batch, nodes, nodes, conn_types) or flattened
            return_fusion_weights: Return fusion weights for analysis

        Returns:
            logits: (batch, num_classes)
            weights: (batch, 3) if return_fusion_weights=True
        """
        # Encode each modality
        erp_feat = self.erp_encoder(erp)  # (batch, hidden_dim)
        pw_feat = self.pw_encoder(pw)    # (batch, hidden_dim)

        if self.use_gnn:
            # Create edge index from connectivity matrix (first batch)
            if self.edge_index is None:
                # Use first sample to create graph structure
                if conn.dim() == 4:
                    conn_matrix = conn[0, :, :, 0]  # Use first connectivity type
                else:
                    conn_matrix = conn[0].view(conn.size(1), conn.size(1))

                self.edge_index, _ = self.conn_encoder.create_graph_from_connectivity(
                    conn_matrix.unsqueeze(0)
                )
                self.edge_index = self.edge_index.to(conn.device)

            conn_feat = self.conn_encoder(conn, self.edge_index)
        else:
            # Flatten and encode
            batch_size = conn.size(0)
            conn_flat = conn.view(batch_size, -1)
            conn_feat = self.conn_encoder(conn_flat)

        # Cross-modal attention (ERP attends to PW and CONN)
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

        if return_fusion_weights:
            return logits, weights
        return logits


# ============================================================================
# 6. HYPERPARAMETER OPTIMIZATION WITH OPTUNA
# ============================================================================

class OptunaHPOTrainer:
    """
    Optuna-based hyperparameter optimization for the enhanced models.
    """

    def __init__(
        self,
        train_loader,
        val_loader,
        device: str = 'cuda',
        n_trials: int = 50,
        timeout: int = 3600
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.n_trials = n_trials
        self.timeout = timeout
        self.best_params = None

    def objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function for hyperparameter search.

        Searches over:
        - Learning rate
        - Hidden dimension
        - Dropout rate
        - Number of transformer layers
        - Number of attention heads
        - Batch size (via data loader)
        """
        # Suggest hyperparameters
        params = {
            'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
            'hidden_dim': trial.suggest_categorical('hidden_dim', [64, 128, 256]),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'num_transformer_layers': trial.suggest_int('num_transformer_layers', 1, 3),
            'num_heads': trial.suggest_categorical('num_heads', [2, 4, 8]),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
            'use_gnn': trial.suggest_categorical('use_gnn', [True, False])
        }

        # Get sample batch to infer dimensions
        sample_batch = next(iter(self.train_loader))
        erp_channels = sample_batch['erp'].size(1)
        pw_channels = sample_batch['pw'].size(1)
        num_conn_nodes = sample_batch['conn'].size(1)

        # Create model
        model = EnhancedTriModalFusionNet(
            erp_channels=erp_channels,
            pw_channels=pw_channels,
            num_conn_nodes=num_conn_nodes,
            num_conn_types=3,
            hidden_dim=params['hidden_dim'],
            num_classes=2,
            dropout=params['dropout'],
            num_transformer_layers=params['num_transformer_layers'],
            num_heads=params['num_heads'],
            use_gnn=params['use_gnn']
        ).to(self.device)

        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params['lr'],
            weight_decay=params['weight_decay']
        )

        # Loss function
        criterion = nn.CrossEntropyLoss()

        # Training loop (fixed number of epochs for fair comparison)
        num_epochs = 10
        best_val_acc = 0.0

        for epoch in range(num_epochs):
            # Training
            model.train()
            for batch in self.train_loader:
                erp = batch['erp'].to(self.device)
                pw = batch['pw'].to(self.device)
                conn = batch['conn'].to(self.device)
                labels = batch['label'].to(self.device)

                optimizer.zero_grad()
                logits = model(erp, pw, conn)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for batch in self.val_loader:
                    erp = batch['erp'].to(self.device)
                    pw = batch['pw'].to(self.device)
                    conn = batch['conn'].to(self.device)
                    labels = batch['label'].to(self.device)

                    logits = model(erp, pw, conn)
                    preds = logits.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            val_acc = correct / total
            best_val_acc = max(best_val_acc, val_acc)

            # Report intermediate value for pruning
            trial.report(val_acc, epoch)

            # Handle pruning
            if trial.should_prune():
                raise optuna.TrialPruned()

        return best_val_acc

    def optimize(self) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.

        Returns:
            best_params: Dictionary of best hyperparameters
        """
        study = optuna.create_study(
            direction='maximize',
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=5
            )
        )

        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )

        self.best_params = study.best_params

        print(f"\n{'='*60}")
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best validation accuracy: {study.best_value:.4f}")
        print(f"\nBest hyperparameters:")
        for key, value in self.best_params.items():
            print(f"  {key}: {value}")
        print(f"{'='*60}\n")

        return self.best_params


# ============================================================================
# 7. UTILITY FUNCTIONS
# ============================================================================

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_fusion_weights(model: EnhancedTriModalFusionNet) -> Dict[str, float]:
    """Extract current fusion weights from model."""
    with torch.no_grad():
        logits = model.fusion.fusion_logits
        temp = model.fusion.temperature
        weights = F.softmax(logits / temp, dim=0)

        return {
            'erp_weight': weights[0].item(),
            'pw_weight': weights[1].item(),
            'conn_weight': weights[2].item(),
            'temperature': temp.item()
        }


if __name__ == "__main__":
    # Example usage
    print("Enhanced Models V4 - Testing Components")
    print("=" * 60)

    # Test dimensions
    batch_size = 4
    erp_channels = 32
    pw_channels = 32
    time_steps = 250
    num_nodes = 32

    # Create dummy data
    erp = torch.randn(batch_size, erp_channels, time_steps)
    pw = torch.randn(batch_size, pw_channels, time_steps)
    conn = torch.randn(batch_size, num_nodes, num_nodes, 3)

    # Test EnhancedTriModalFusionNet
    print("\n1. Testing EnhancedTriModalFusionNet...")
    model = EnhancedTriModalFusionNet(
        erp_channels=erp_channels,
        pw_channels=pw_channels,
        num_conn_nodes=num_nodes,
        hidden_dim=128,
        num_transformer_layers=2,
        use_gnn=False  # Use MLP for testing (GNN requires torch_geometric)
    )

    logits, weights = model(erp, pw, conn, return_fusion_weights=True)
    print(f"   Output shape: {logits.shape}")
    print(f"   Fusion weights shape: {weights.shape}")
    print(f"   Fusion weights (sample): {weights[0]}")
    print(f"   Parameters: {count_parameters(model):,}")

    # Test individual encoders
    print("\n2. Testing EnhancedERPEncoder...")
    erp_encoder = EnhancedERPEncoder(erp_channels, 128, 2, 4)
    erp_out = erp_encoder(erp)
    print(f"   Output shape: {erp_out.shape}")

    print("\n3. Testing EnhancedPowerEncoder...")
    pw_encoder = EnhancedPowerEncoder(pw_channels, 128, 2, 4)
    pw_out = pw_encoder(pw)
    print(f"   Output shape: {pw_out.shape}")

    print("\n" + "="*60)
    print("All tests passed!")
