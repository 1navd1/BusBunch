from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.models.contracts import EpisodeMetrics


def _std(values: List[float]) -> float:
    if not values:
        return 0.0
    m = sum(values) / len(values)
    return (sum((x - m) ** 2 for x in values) / len(values)) ** 0.5


@dataclass
class MetricsTracker:
    bunching_count: int = 0
    total_delay: float = 0.0
    total_hold_time: float = 0.0
    fuel_proxy: float = 0.0
    headway_stds: List[float] | None = None
    wait_times: List[float] | None = None
    occupancy_stds: List[float] | None = None

    def __post_init__(self) -> None:
        if self.headway_stds is None:
            self.headway_stds = []
        if self.wait_times is None:
            self.wait_times = []
        if self.occupancy_stds is None:
            self.occupancy_stds = []

    def update(
        self,
        headways: List[float],
        occupancies: List[float],
        bunching_step: int,
        hold_sec: float,
        delay_sec: float,
        fuel_step: float,
    ) -> None:
        self.bunching_count += int(bunching_step)
        self.total_hold_time += float(hold_sec)
        self.total_delay += float(delay_sec)
        self.fuel_proxy += float(fuel_step)

        hw_std = _std(headways)
        occ_std = _std(occupancies)
        wait = sum(headways) / len(headways) / 2.0 + 0.35 * hw_std

        self.headway_stds.append(hw_std)
        self.wait_times.append(wait)
        self.occupancy_stds.append(occ_std)

    def to_episode_metrics(self) -> EpisodeMetrics:
        return EpisodeMetrics(
            bunching_count=int(self.bunching_count),
            headway_std=float(sum(self.headway_stds) / max(1, len(self.headway_stds))),
            avg_wait_time=float(sum(self.wait_times) / max(1, len(self.wait_times))),
            occupancy_std=float(sum(self.occupancy_stds) / max(1, len(self.occupancy_stds))),
            fuel_proxy=float(self.fuel_proxy),
            total_delay=float(self.total_delay),
        )
