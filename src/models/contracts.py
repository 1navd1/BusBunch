from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class BusState:
    bus_id: str
    stop_index: int
    position_progress: float
    cycle_position_sec: float
    lat: float
    lon: float
    current_stop_id: str
    current_stop_name: str
    next_stop_id: str
    next_stop_name: str
    occupancy: float
    delay_sec: float
    headway_forward_sec: float
    headway_backward_sec: float
    status: str = "in_service"


@dataclass
class StopState:
    stop_id: str
    stop_name: str
    lat: float
    lon: float
    order: int
    queue_len: int
    is_terminal: bool = False


@dataclass
class SegmentState:
    segment_id: str
    from_stop_id: str
    to_stop_id: str
    from_lat: float
    from_lon: float
    to_lat: float
    to_lon: float
    base_travel_time_sec: float
    traffic_multiplier: float


@dataclass
class SystemState:
    timestamp: int
    buses: List[BusState]
    stops: List[StopState] = field(default_factory=list)
    segments: List[SegmentState] = field(default_factory=list)
    terminal_queue: Dict[str, int] = field(default_factory=dict)
    global_traffic_context: Dict[str, float] = field(default_factory=dict)


@dataclass
class ControlState:
    focus_bus_id: str
    focus_stop_id: str
    focus_stop_name: str
    current_bus_delay_sec: float
    forward_headway_sec: float
    backward_headway_sec: float
    occupancy_ratio: float
    stop_demand_estimate: float
    predicted_corridor_congestion_score: float
    predicted_bunching_risk: float
    is_terminal: bool


@dataclass
class PredictionBundle:
    downstream_travel_time: List[float]
    stop_demand_forecast: List[float]
    bunching_risk_score: float
    congestion_score: float


@dataclass
class ControlAction:
    hold_sec: float = 0.0
    speed_delta_pct: float = 0.0
    dispatch_offset_sec: float = 0.0


@dataclass
class EpisodeMetrics:
    bunching_count: int
    headway_std: float
    avg_wait_time: float
    occupancy_std: float
    fuel_proxy: float
    total_delay: float


@dataclass
class EpisodeReport:
    policy_name: str
    seed: int
    total_reward: float
    metrics: EpisodeMetrics
    trace: List[Dict[str, Any]]
