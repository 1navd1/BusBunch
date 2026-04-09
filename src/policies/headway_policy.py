from __future__ import annotations

from src.models.contracts import ControlAction, ControlState, PredictionBundle


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class HeadwayPolicy:
    """Simple headway-rule baseline with hold/speed/dispatch controls."""

    name = "headway"

    def __init__(self, target_headway_sec: float = 300.0):
        self.target_headway_sec = target_headway_sec

    def act(self, control_state: ControlState, prediction_bundle: PredictionBundle) -> ControlAction:
        fwd = control_state.forward_headway_sec
        back = control_state.backward_headway_sec
        risk = prediction_bundle.bunching_risk_score

        gap_deficit = max(0.0, (self.target_headway_sec * 0.7 - fwd) / self.target_headway_sec)
        hold_norm = _clamp(0.8 * gap_deficit + 0.35 * risk, 0.0, 1.0)

        imbalance = max(0.0, (back - fwd) / self.target_headway_sec)
        speed_reduction_norm = _clamp(0.45 * risk + 0.3 * imbalance, 0.0, 1.0)

        dispatch_norm = 0.0
        if control_state.is_terminal:
            dispatch_norm = _clamp(max(0.0, (self.target_headway_sec - fwd) / self.target_headway_sec), 0.0, 1.0)

        return ControlAction(
            hold_sec=hold_norm * 60.0,
            speed_delta_pct=-speed_reduction_norm * 0.2,
            dispatch_offset_sec=dispatch_norm * 90.0,
        )

    def action_vector(self, observation: list[float]) -> list[float]:
        control_state = ControlState(
            current_bus_delay_sec=observation[0] * 600.0,
            forward_headway_sec=observation[1] * 600.0,
            backward_headway_sec=observation[2] * 600.0,
            occupancy_ratio=observation[3],
            stop_demand_estimate=observation[4],
            predicted_corridor_congestion_score=observation[5],
            predicted_bunching_risk=observation[6],
            is_terminal=observation[7] > 0.5,
        )
        bundle = PredictionBundle(
            downstream_travel_time=[120.0],
            stop_demand_forecast=[control_state.stop_demand_estimate],
            bunching_risk_score=control_state.predicted_bunching_risk,
            congestion_score=control_state.predicted_corridor_congestion_score,
        )
        action = self.act(control_state, bundle)
        return [
            _clamp(action.hold_sec / 60.0, -1.0, 1.0),
            _clamp(action.speed_delta_pct / 0.2, -1.0, 1.0),
            _clamp(action.dispatch_offset_sec / 90.0, -1.0, 1.0),
        ]
