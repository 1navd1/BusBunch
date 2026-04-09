from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class Stop:
    stop_id: str
    stop_name: str
    lat: float
    lon: float
    order: int


@dataclass
class RouteSegment:
    segment_id: str
    from_stop_id: str
    to_stop_id: str
    base_travel_time_sec: float


@dataclass
class PassengerQueue:
    stop_id: str
    waiting: int = 0


@dataclass
class Bus:
    bus_id: str
    stop_index: int
    position_progress: float
    occupancy_passengers: int
    delay_sec: float
    headway_forward_sec: float
    headway_backward_sec: float
    status: str = "in_service"
    dwell_remaining_sec: float = 0.0
    hold_remaining_sec: float = 0.0
    dispatch_delay_remaining_sec: float = 0.0


@dataclass
class EpisodeScenario:
    day_type: str
    peak_profile: str
    max_steps: int
    seed: int


@dataclass
class RouteGraph:
    stops: List[Stop] = field(default_factory=list)
    segments: List[RouteSegment] = field(default_factory=list)

    @staticmethod
    def load(stops: List[Stop], segments: List[RouteSegment]) -> "RouteGraph":
        return RouteGraph(stops=stops, segments=segments)
