from __future__ import annotations

import math
import random
from typing import Dict, List


def demand_context(step_idx: int, profile: Dict[str, float]) -> tuple[float, float]:
    phase = 2 * math.pi * (step_idx % 60) / 60.0
    time_sin = math.sin(phase)
    demand_level = profile["base_per_min"] + profile["amplifier"] * max(0.0, time_sin)
    return time_sin, demand_level


def traffic_multiplier(step_idx: int, profile: Dict[str, float]) -> float:
    phase = 2 * math.pi * (step_idx % 60) / 60.0
    return max(0.7, profile["base"] + profile["amplitude"] * max(0.0, math.sin(phase)))


def update_stop_queues(
    queues: List[int],
    tick_sec: float,
    demand_profile: Dict[str, float],
    step_idx: int,
    rng: random.Random,
) -> tuple[List[int], float, float]:
    time_sin, demand_level = demand_context(step_idx, demand_profile)
    weights = demand_profile["stop_weights"]

    arrivals = []
    for w in weights:
        lam = (demand_level * w) * (tick_sec / 60.0)
        noise = rng.uniform(0.8, 1.2)
        arrivals.append(int(max(0.0, round(lam * noise))))

    new_q = [queues[i] + arrivals[i] for i in range(len(queues))]
    return new_q, time_sin, demand_level


def board_alight(
    queue_len: int,
    occupancy: int,
    capacity: int,
    rng: random.Random,
) -> tuple[int, int, int]:
    alight = int(round(occupancy * rng.uniform(0.05, 0.2)))
    alight = max(0, min(alight, occupancy))
    occ_after_alight = occupancy - alight

    can_board = max(0, capacity - occ_after_alight)
    board = min(queue_len, can_board)
    new_queue = queue_len - board
    new_occ = occ_after_alight + board
    return board, alight, new_occ
