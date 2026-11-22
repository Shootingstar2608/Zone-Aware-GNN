"""Temporal encoders for graph models that previously consumed flattened X."""
from __future__ import annotations

import torch
import torch.nn as nn


class TemporalGraphWrapper(nn.Module):
    """Encode each node's T_in-step sequence, then apply a graph forecaster.

    The wrapped graph model receives a node embedding of shape ``(B,N,H)``;
    consequently all Zone-Aware variants and AH-GNN operate on an actual
    temporal representation rather than on a flattened history vector.
    """
    def __init__(self, graph_model: nn.Module, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.graph_model = graph_model
        self.temporal_gru = nn.GRU(feature_dim, hidden_dim, batch_first=True)
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

    @property
    def zone_emb(self):
        return getattr(self.graph_model, "zone_emb", None)

    def forward(self, X: torch.Tensor, Z=None, time_idx=None, A_static=None) -> torch.Tensor:
        if X.ndim != 4:
            raise ValueError(f"TemporalGraphWrapper expects (B,N,T,F), got {tuple(X.shape)}")
        B, N, _, F = X.shape
        if F != self.feature_dim:
            raise ValueError(f"Feature dimension {F} != model feature_dim {self.feature_dim}")
        sequence = X.reshape(B * N, X.shape[2], F)
        _, h = self.temporal_gru(sequence)
        node_embedding = h[-1].reshape(B, N, self.hidden_dim)
        return self.graph_model(node_embedding, Z, time_idx, A_static)
