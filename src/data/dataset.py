from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

from src.data.graph_builder import GraphArtifacts, build_corridor_graph


@dataclass
class RolloutFrame:
    timestamp: int
    queue_len: np.ndarray
    demand_level: float
    nearby_occupancy: np.ndarray
    traffic_multiplier: np.ndarray
    recent_travel_time: np.ndarray
    time_sin: float
    time_cos: float
    target_travel_time: np.ndarray
    target_demand: np.ndarray
    target_bunching: np.ndarray


def _mean_occupancy_by_stop(system_state: Dict, n_stops: int) -> np.ndarray:
    occ = np.zeros(n_stops, dtype=np.float32)
    cnt = np.zeros(n_stops, dtype=np.float32)
    for bus in system_state["buses"]:
        idx = int(bus["stop_index"]) % n_stops
        occ[idx] += float(bus["occupancy"])
        cnt[idx] += 1.0
    return np.where(cnt > 0.0, occ / np.clip(cnt, 1.0, None), 0.0).astype(np.float32)


def frame_from_snapshot(snapshot: Dict, max_steps: int) -> RolloutFrame:
    sys = snapshot["system_state"]
    stops = sorted(sys["stops"], key=lambda s: s["order"])
    segs = sorted(sys["segments"], key=lambda s: int(str(s["segment_id"]).split("_")[-1]))

    n = len(stops)
    queue_len = np.array([float(s["queue_len"]) for s in stops], dtype=np.float32)
    demand_level = float(sys["global_traffic_context"]["demand_level"])

    traffic_multiplier = np.array([float(s["traffic_multiplier"]) for s in segs], dtype=np.float32)
    recent_travel_time = np.array(
        [float(s["base_travel_time_sec"]) * float(s["traffic_multiplier"]) for s in segs],
        dtype=np.float32,
    )

    nearby_occupancy = _mean_occupancy_by_stop(sys, n)
    bunching = float(snapshot.get("bunching", 0.0))
    bunching_proxy = np.full((n,), min(1.0, bunching / max(1.0, len(sys["buses"]))), dtype=np.float32)

    t = int(snapshot["step"])
    phase = 2.0 * np.pi * (t / max(1, max_steps))
    time_sin = float(np.sin(phase))
    time_cos = float(np.cos(phase))

    return RolloutFrame(
        timestamp=t,
        queue_len=queue_len,
        demand_level=demand_level,
        nearby_occupancy=nearby_occupancy,
        traffic_multiplier=traffic_multiplier,
        recent_travel_time=recent_travel_time,
        time_sin=time_sin,
        time_cos=time_cos,
        target_travel_time=recent_travel_time.copy(),
        target_demand=queue_len.copy(),
        target_bunching=bunching_proxy,
    )


def build_feature_matrix(frame: RolloutFrame) -> np.ndarray:
    n = frame.queue_len.shape[0]
    demand = np.full((n,), frame.demand_level, dtype=np.float32)
    time_sin = np.full((n,), frame.time_sin, dtype=np.float32)
    time_cos = np.full((n,), frame.time_cos, dtype=np.float32)

    feats = np.stack(
        [
            frame.queue_len,
            demand,
            frame.nearby_occupancy,
            frame.traffic_multiplier,
            frame.recent_travel_time,
            time_sin,
            time_cos,
        ],
        axis=-1,
    )
    return feats.astype(np.float32)


def build_windows(
    rollouts: Sequence[Sequence[Dict]],
    history: int,
    horizon: int,
    max_steps: int,
) -> Dict[str, np.ndarray]:
    xs: List[np.ndarray] = []
    y_tt: List[np.ndarray] = []
    y_dem: List[np.ndarray] = []
    y_bunch: List[np.ndarray] = []

    for rollout in rollouts:
        frames = [frame_from_snapshot(s, max_steps=max_steps) for s in rollout]
        if len(frames) < history + horizon:
            continue

        feat_seq = [build_feature_matrix(f) for f in frames]

        for t in range(history - 1, len(frames) - horizon):
            x = np.stack(feat_seq[t - history + 1 : t + 1], axis=0)
            target_idx = t + horizon
            target = frames[target_idx]

            xs.append(x)
            y_tt.append(target.target_travel_time)
            y_dem.append(target.target_demand)
            y_bunch.append(target.target_bunching)

    return {
        "x": np.asarray(xs, dtype=np.float32),
        "y_travel_time": np.asarray(y_tt, dtype=np.float32),
        "y_demand": np.asarray(y_dem, dtype=np.float32),
        "y_bunching": np.asarray(y_bunch, dtype=np.float32),
    }


def compute_norm(x: np.ndarray) -> Dict[str, np.ndarray]:
    mean = x.mean(axis=(0, 1, 2), keepdims=True)
    std = x.std(axis=(0, 1, 2), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def normalize_x(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


def split_indices(n: int, seed: int = 7, train_ratio: float = 0.7, val_ratio: float = 0.15) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_idx = np.sort(idx[:n_train])
    val_idx = np.sort(idx[n_train : n_train + n_val])
    test_idx = np.sort(idx[n_train + n_val :])

    return {"train": train_idx, "val": val_idx, "test": test_idx}


def save_dataset(
    out_path: str,
    arrays: Dict[str, np.ndarray],
    split: Dict[str, np.ndarray],
    graph: GraphArtifacts,
    norm: Dict[str, np.ndarray],
    metadata: Dict,
) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x=arrays["x"],
        y_travel_time=arrays["y_travel_time"],
        y_demand=arrays["y_demand"],
        y_bunching=arrays["y_bunching"],
        idx_train=split["train"],
        idx_val=split["val"],
        idx_test=split["test"],
        adj_norm=graph.normalized_adjacency,
        stop_ids=np.array(graph.stop_ids),
        x_mean=norm["mean"],
        x_std=norm["std"],
    )

    meta_out = path.with_suffix(".json")
    meta_out.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def save_norm_json(out_path: str, norm: Dict[str, np.ndarray], graph: GraphArtifacts, history: int) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "x_mean": norm["mean"].reshape(-1).tolist(),
        "x_std": norm["std"].reshape(-1).tolist(),
        "stop_ids": graph.stop_ids,
        "history": int(history),
        "feature_order": [
            "queue_len",
            "demand_level",
            "nearby_occupancy",
            "traffic_multiplier",
            "recent_travel_time",
            "time_sin",
            "time_cos",
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_dataset(npz_path: str) -> Dict[str, np.ndarray]:
    data = np.load(npz_path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def default_graph() -> GraphArtifacts:
    return build_corridor_graph()
