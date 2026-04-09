from __future__ import annotations

from src.models.contracts import ControlAction, ControlState, PredictionBundle


class StaticPolicy:
    """Static timetable baseline: no intervention."""

    name = "static"

    def act(self, control_state: ControlState, prediction_bundle: PredictionBundle) -> ControlAction:
        return ControlAction(hold_sec=0.0, speed_delta_pct=0.0, dispatch_offset_sec=0.0)

    def action_vector(self, observation: list[float]) -> list[float]:
        return [0.0, 0.0, 0.0]
