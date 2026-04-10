from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List

import numpy as np
import torch

from src.data.graph_builder import build_corridor_graph
from src.models.contracts import PredictionBundle, SystemState
from src.models.stgnn import STGNN


class STGNNPredictor:
    def __init__(
        self,
        checkpoint_path: str = "artifacts/models/stgnn_best.pt",
        norm_path: str = "artifacts/models/stgnn_norm.json",
        device: str = "cpu",
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.norm_path = Path(norm_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Missing STGNN checkpoint at {self.checkpoint_path}. Run `python3 -m src.train.train_stgnn` first."
            )
        if not self.norm_path.exists():
            raise FileNotFoundError(
                f"Missing STGNN normalization stats at {self.norm_path}. Run `python3 -m src.train.train_stgnn` first."
            )

        norm = json.loads(self.norm_path.read_text(encoding="utf-8"))
        self.stop_ids = list(norm["stop_ids"])
        self.history = int(norm["history"])
        self.x_mean = np.asarray(norm["x_mean"], dtype=np.float32).reshape(1, 1, 1, -1)
        self.x_std = np.asarray(norm["x_std"], dtype=np.float32).reshape(1, 1, 1, -1)

        graph = build_corridor_graph()
        self.stop_index = graph.stop_index
        self.adj_norm = torch.tensor(graph.normalized_adjacency, dtype=torch.float32, device=device)

        payload = torch.load(self.checkpoint_path, map_location=device)
        self.model = STGNN(
            in_features=int(payload["in_features"]),
            hidden_dim=int(payload["hidden_dim"]),
            num_nodes=int(payload["num_nodes"]),
        )
        self.model.load_state_dict(payload["state_dict"])
        self.model.to(device)
        self.model.eval()
        self.device = device
        self._hist: Deque[np.ndarray] = deque(maxlen=self.history)

    def _features_from_state(self, sim_state: SystemState) -> np.ndarray:
        n = len(self.stop_ids)
        queue = np.zeros((n,), dtype=np.float32)
        demand_level = float(sim_state.global_traffic_context.get("demand_level", 0.5))
        time_sin = float(sim_state.global_traffic_context.get("time_sin", 0.0))
        time_cos = float(np.sqrt(max(0.0, 1.0 - min(1.0, time_sin**2))))

        traffic = np.ones((n,), dtype=np.float32)
        travel_time = np.zeros((n,), dtype=np.float32)
        near_occ = np.zeros((n,), dtype=np.float32)
        occ_count = np.zeros((n,), dtype=np.float32)

        for st in sim_state.stops:
            idx = self.stop_index.get(st.stop_id)
            if idx is not None:
                queue[idx] = float(st.queue_len)

        for seg in sim_state.segments:
            from_idx = self.stop_index.get(seg.from_stop_id)
            if from_idx is not None:
                traffic[from_idx] = float(seg.traffic_multiplier)
                travel_time[from_idx] = float(seg.base_travel_time_sec) * float(seg.traffic_multiplier)

        for bus in sim_state.buses:
            idx = int(bus.stop_index) % n
            near_occ[idx] += float(bus.occupancy)
            occ_count[idx] += 1.0
        near_occ = np.where(occ_count > 0.0, near_occ / np.clip(occ_count, 1.0, None), 0.0).astype(np.float32)

        demand = np.full((n,), demand_level, dtype=np.float32)
        time_s = np.full((n,), time_sin, dtype=np.float32)
        time_c = np.full((n,), time_cos, dtype=np.float32)

        return np.stack([queue, demand, near_occ, traffic, travel_time, time_s, time_c], axis=-1).astype(np.float32)

    def predict(self, sim_state: SystemState) -> PredictionBundle:
        feats = self._features_from_state(sim_state)
        self._hist.append(feats)

        while len(self._hist) < self.history:
            self._hist.appendleft(feats)

        x = np.stack(list(self._hist), axis=0)[None, ...]
        x = (x - self.x_mean) / self.x_std

        xt = torch.tensor(x, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            pred = self.model(xt, self.adj_norm)
            travel = pred["travel_time"].squeeze(0).cpu().numpy()
            demand = pred["demand"].squeeze(0).cpu().numpy()
            bunch = torch.sigmoid(pred["bunching"]).squeeze(0).cpu().numpy()

        travel = np.clip(travel, 20.0, 420.0)
        demand = np.clip(demand, 0.0, 80.0)
        bunch = np.clip(bunch, 0.0, 1.0)

        congestion = float(np.clip(np.mean(travel) / 180.0, 0.0, 1.0))
        bunching_risk = float(np.clip(np.mean(bunch), 0.0, 1.0))

        return PredictionBundle(
            downstream_travel_time=[float(round(x, 2)) for x in travel.tolist()],
            stop_demand_forecast=[float(round(x, 2)) for x in demand.tolist()],
            bunching_risk_score=float(round(bunching_risk, 4)),
            congestion_score=float(round(congestion, 4)),
        )
