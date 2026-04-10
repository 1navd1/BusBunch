from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class STGNN(nn.Module):
    """Lightweight STGNN: Conv1d temporal encoding + GRU + adjacency mixing."""

    def __init__(self, in_features: int, hidden_dim: int, num_nodes: int):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes

        self.temporal_conv = nn.Conv1d(in_channels=in_features, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.temporal_gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)

        self.self_proj = nn.Linear(hidden_dim, hidden_dim)
        self.nei_proj = nn.Linear(hidden_dim, hidden_dim)

        self.travel_head = nn.Linear(hidden_dim, 1)
        self.demand_head = nn.Linear(hidden_dim, 1)
        self.bunching_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: [B, T, N, F]
        bsz, t_hist, n_nodes, in_feat = x.shape
        if n_nodes != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} nodes, got {n_nodes}")
        if in_feat != self.in_features:
            raise ValueError(f"Expected {self.in_features} features, got {in_feat}")

        xt = x.permute(0, 2, 3, 1).reshape(bsz * n_nodes, in_feat, t_hist)
        xt = F.relu(self.temporal_conv(xt))
        xt = xt.transpose(1, 2)

        _, h_last = self.temporal_gru(xt)
        h = h_last.squeeze(0).reshape(bsz, n_nodes, self.hidden_dim)

        neigh = torch.matmul(adj_norm, h)
        h_mix = F.relu(self.self_proj(h) + self.nei_proj(neigh))

        return {
            "travel_time": self.travel_head(h_mix).squeeze(-1),
            "demand": self.demand_head(h_mix).squeeze(-1),
            "bunching": self.bunching_head(h_mix).squeeze(-1),
        }


@dataclass
class LossWeights:
    travel_time: float = 1.0
    demand: float = 0.7
    bunching: float = 0.9


def stgnn_loss(
    pred: Dict[str, torch.Tensor],
    target_travel_time: torch.Tensor,
    target_demand: torch.Tensor,
    target_bunching: torch.Tensor,
    weights: LossWeights | None = None,
) -> Dict[str, torch.Tensor]:
    w = weights or LossWeights()

    loss_tt = F.l1_loss(pred["travel_time"], target_travel_time)
    loss_dem = F.l1_loss(pred["demand"], target_demand)
    loss_bun = F.mse_loss(torch.sigmoid(pred["bunching"]), target_bunching)

    total = w.travel_time * loss_tt + w.demand * loss_dem + w.bunching * loss_bun
    return {
        "total": total,
        "travel_time": loss_tt,
        "demand": loss_dem,
        "bunching": loss_bun,
    }
