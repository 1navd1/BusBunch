from __future__ import annotations


def compute_dispatch_delay(
    is_terminal: bool,
    forward_headway_sec: float,
    target_headway_sec: float,
    dispatch_offset_sec: float,
) -> float:
    if not is_terminal:
        return 0.0
    gap_deficit = max(0.0, target_headway_sec - forward_headway_sec)
    base_delay = 0.35 * gap_deficit
    return max(0.0, min(base_delay + dispatch_offset_sec, 180.0))
