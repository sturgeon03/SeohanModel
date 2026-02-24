"""
Steering inner-loop controller (computed-torque + SMC).
"""

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Union

import numpy as np


@dataclass
class SteeringInnerLoopGains:
    """Per-wheel inner-loop gains and limits."""
    lambda_smc: float = 12.0
    k_smc: float = 4.0
    boundary: float = 0.02
    torque_limit: float = 10000.0
    max_accel: float = 200.0


class SteeringInnerLoopController:
    """
    Computes T_steer for each wheel to track delta_cmd.

    Model: J*delta_ddot + B*delta_dot = T_steer - T_align
    """

    def __init__(
        self,
        dt: float,
        gains: Optional[SteeringInnerLoopGains] = None,
        gains_by_corner: Optional[Dict[str, SteeringInnerLoopGains]] = None,
    ) -> None:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self.dt = float(dt)
        self.default_gains = gains or SteeringInnerLoopGains()
        self.gains_by_corner = gains_by_corner or {}
        self.prev_delta_cmd: Dict[str, float] = {}
        self.prev_delta_dot_cmd: Dict[str, float] = {}

    def reset(self) -> None:
        """Reset internal command history."""
        self.prev_delta_cmd.clear()
        self.prev_delta_dot_cmd.clear()

    def update(
        self,
        vehicle_body,
        delta_cmd: Union[float, Dict[str, float]],
        delta_dot_cmd: Optional[Union[float, Dict[str, float]]] = None,
        delta_ddot_cmd: Optional[Union[float, Dict[str, float]]] = None,
    ) -> Dict[str, float]:
        """
        Compute steering torques per wheel.

        Args:
            vehicle_body: VehicleBody instance.
            delta_cmd: desired steering angle(s) [rad].
            delta_dot_cmd: optional desired steering rate(s) [rad/s].
            delta_ddot_cmd: optional desired steering accel(s) [rad/s^2].
        """
        labels = list(vehicle_body.wheel_labels)
        delta_cmd_map = self._normalize_command(delta_cmd, labels)
        delta_dot_map = self._normalize_command(delta_dot_cmd, labels) if delta_dot_cmd is not None else {}
        delta_ddot_map = self._normalize_command(delta_ddot_cmd, labels) if delta_ddot_cmd is not None else {}

        torques: Dict[str, float] = {}
        for label in labels:
            corner = vehicle_body.corners[label]
            params = corner.steering.params
            gains = self.gains_by_corner.get(label, self.default_gains)

            delta_cmd_i = float(delta_cmd_map[label])
            delta_cmd_i = self._clip_angle(delta_cmd_i, params.max_angle_neg, params.max_angle_pos)

            prev_delta = self.prev_delta_cmd.get(label, delta_cmd_i)
            has_prev_delta_dot = label in self.prev_delta_dot_cmd
            prev_delta_dot = self.prev_delta_dot_cmd.get(label, 0.0)

            if delta_dot_cmd is None:
                delta_dot_cmd_i = (delta_cmd_i - prev_delta) / self.dt
            else:
                delta_dot_cmd_i = float(delta_dot_map[label])
            delta_dot_cmd_i = float(np.clip(delta_dot_cmd_i, -params.max_rate, params.max_rate))

            if delta_ddot_cmd is None:
                if not has_prev_delta_dot:
                    # Avoid startup spike from numerical differentiation.
                    delta_ddot_cmd_i = 0.0
                else:
                    delta_ddot_cmd_i = (delta_dot_cmd_i - prev_delta_dot) / self.dt
            else:
                delta_ddot_cmd_i = float(delta_ddot_map[label])
            if gains.max_accel is not None:
                delta_ddot_cmd_i = float(np.clip(delta_ddot_cmd_i, -gains.max_accel, gains.max_accel))

            delta = float(corner.steering.state.steering_angle)
            delta_dot = float(corner.steering.state.steering_rate)
            t_align = float(corner.steering.state.self_aligning_torque)

            e = delta - delta_cmd_i
            e_dot = delta_dot - delta_dot_cmd_i
            s = e_dot + gains.lambda_smc * e

            torque = (
                t_align
                + params.B_cq * delta_dot
                + params.J_cq * (delta_ddot_cmd_i - gains.lambda_smc * e_dot)
                - gains.k_smc * self._sat(s / max(gains.boundary, 1e-6))
            )

            if gains.torque_limit is not None:
                torque = float(np.clip(torque, -gains.torque_limit, gains.torque_limit))

            torques[label] = torque

            self.prev_delta_cmd[label] = delta_cmd_i
            self.prev_delta_dot_cmd[label] = delta_dot_cmd_i

        return torques

    @staticmethod
    def _normalize_command(
        value: Optional[Union[float, Dict[str, float]]],
        labels: Iterable[str],
    ) -> Dict[str, float]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return {label: float(value.get(label, 0.0)) for label in labels}
        return {label: float(value) for label in labels}

    @staticmethod
    def _clip_angle(angle: float, neg: float, pos: float) -> float:
        lower = min(neg, pos)
        upper = max(neg, pos)
        return float(np.clip(angle, lower, upper))

    @staticmethod
    def _sat(x: float) -> float:
        return float(np.clip(x, -1.0, 1.0))
