from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np


@dataclass
class GraphArtifacts:
    stop_ids: List[str]
    stop_index: Dict[str, int]
    adjacency: np.ndarray
    normalized_adjacency: np.ndarray


def _load_stop_ids(stops_csv: str = "src/data/stops.csv") -> List[str]:
    path = Path(stops_csv)
    rows = path.read_text(encoding="utf-8").strip().splitlines()
    header = rows[0].split(",")
    stop_id_idx = header.index("stop_id")
    order_idx = header.index("order")

    parsed = []
    for row in rows[1:]:
        cols = row.split(",")
        parsed.append((int(cols[order_idx]), cols[stop_id_idx]))
    parsed.sort(key=lambda x: x[0])
    return [sid for _, sid in parsed]


def build_corridor_graph(
    corridor_cfg_path: str = "src/data/corridor_config.json",
    stops_csv: str = "src/data/stops.csv",
) -> GraphArtifacts:
    _ = json.loads(Path(corridor_cfg_path).read_text(encoding="utf-8"))
    stop_ids = _load_stop_ids(stops_csv)
    n = len(stop_ids)

    adjacency = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        j = (i + 1) % n
        adjacency[i, j] = 1.0
        adjacency[j, i] = 1.0

    # Self loops make message passing numerically stable for small graphs.
    adjacency += np.eye(n, dtype=np.float32)
    deg = adjacency.sum(axis=1)
    deg_inv_sqrt = np.diag(np.power(np.clip(deg, 1e-6, None), -0.5)).astype(np.float32)
    normalized = deg_inv_sqrt @ adjacency @ deg_inv_sqrt

    return GraphArtifacts(
        stop_ids=stop_ids,
        stop_index={sid: i for i, sid in enumerate(stop_ids)},
        adjacency=adjacency,
        normalized_adjacency=normalized.astype(np.float32),
    )
