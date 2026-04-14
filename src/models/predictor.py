from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Dict, Iterable, List

from .contracts import PredictionBundle, SystemState

if TYPE_CHECKING:
    from .stgnn_infer import STGNNPredictor


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class Predictor:
    """Interface-compatible predictor base."""

    def predict(self, sim_state: SystemState) -> PredictionBundle:
        raise NotImplementedError


def load_default_predictor() -> Predictor:
    """Default predictor backend: trained STGNN checkpoint."""
    from .stgnn_infer import STGNNPredictor

    return STGNNPredictor()


class GraphAwarePredictor(Predictor):
    """Graph-temporal forecaster (lightweight fallback, GNN-ready interface)."""

    def __init__(self, num_buses: int, route_edges: Iterable[tuple[int, int]] | None = None):
        self.num_buses = num_buses
        self.route_edges = list(route_edges or self._default_ring_edges(num_buses))

    @staticmethod
    def _default_ring_edges(n: int) -> List[tuple[int, int]]:
        return [(i, (i + 1) % n) for i in range(n)]

    def _neighbor_aggregate(self, values: List[float]) -> List[float]:
        agg = [0.0 for _ in values]
        deg = [0 for _ in values]
        for u, v in self.route_edges:
            agg[v] += values[u]
            agg[u] += values[v]
            deg[u] += 1
            deg[v] += 1
        return [agg[i] / max(1, deg[i]) for i in range(len(values))]

    def predict_from_features(
        self,
        headways_sec: List[float],
        occupancies: List[float],
        demand_level: float,
        traffic_level: float,
        time_sin: float,
    ) -> PredictionBundle:
        neighbor_headway = self._neighbor_aggregate(headways_sec)
        downstream_tt = [max(30.0, 55.0 + 0.08 * h + 35.0 * traffic_level) for h in neighbor_headway]
        demand_forecast = [_clamp(2.5 + 4.0 * demand_level + 2.0 * o, 0.0, 12.0) for o in occupancies]

        mean_hw = sum(headways_sec) / len(headways_sec)
        headway_std = (sum((x - mean_hw) ** 2 for x in headways_sec) / len(headways_sec)) ** 0.5
        bunching_risk = _clamp((180.0 - min(headways_sec)) / 180.0 + headway_std / 200.0, 0.0, 1.0)
        congestion = _clamp(0.5 * traffic_level + 0.3 * demand_level + 0.2 * (time_sin + 1) / 2, 0.0, 1.0)

        return PredictionBundle(
            downstream_travel_time=[round(x, 2) for x in downstream_tt],
            stop_demand_forecast=[round(x, 2) for x in demand_forecast],
            bunching_risk_score=round(bunching_risk, 4),
            congestion_score=round(congestion, 4),
        )

    def predict(self, sim_state: SystemState) -> PredictionBundle:
        headways = [b.headway_forward_sec for b in sim_state.buses]
        occupancies = [b.occupancy for b in sim_state.buses]
        traffic = float(sim_state.global_traffic_context.get("traffic_level", 0.5))
        demand = float(sim_state.global_traffic_context.get("demand_level", 0.5))
        time_sin = float(sim_state.global_traffic_context.get("time_sin", 0.0))
        return self.predict_from_features(headways, occupancies, demand, traffic, time_sin)

    @staticmethod
    def to_dict(bundle: PredictionBundle) -> Dict[str, float | list[float]]:
        return asdict(bundle)
