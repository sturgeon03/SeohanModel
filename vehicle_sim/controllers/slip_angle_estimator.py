"""
Slip angle estimator using IMU and wheel speed inputs.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class SlipAngleEstimatorOptions:
    """Options for slip angle estimation."""
    ay_bias: float = 0.0
    lowpass_tau: Optional[float] = None
    vy_init: float = 0.0
    vy_limit: Optional[float] = None
    leak_tau: Optional[float] = None
    min_vx: float = 0.1


class SlipAngleEstimator:
    """
    Estimate wheel slip angles from measured yaw rate, lateral accel, wheel speeds, and steering.
    """

    def __init__(
        self,
        dt: float,
        wheel_xy: Dict[str, Tuple[float, float]],
        options: Optional[SlipAngleEstimatorOptions] = None,
    ) -> None:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self.dt = float(dt)
        self.wheel_xy = dict(wheel_xy)
        self.options = options or SlipAngleEstimatorOptions()
        self._vy = float(self.options.vy_init)
        self._v_dot_y_filtered: Optional[float] = None

    def reset(self, vy_init: Optional[float] = None) -> None:
        """Reset estimator state."""
        if vy_init is not None:
            self._vy = float(vy_init)
        else:
            self._vy = float(self.options.vy_init)
        self._v_dot_y_filtered = None

    def update(
        self,
        vx: float,
        yaw_rate: float,
        ay_meas: float,
        delta_map: Dict[str, float],
    ) -> Tuple[Dict[str, float], float]:
        """
        Args:
            vx: measured longitudinal speed [m/s].
            yaw_rate: measured yaw rate [rad/s].
            ay_meas: measured lateral acceleration [m/s^2].
            delta_map: per-wheel steering angles [rad].

        Returns:
            (alpha_est_map, vy_est)
        """
        vx = float(vx)
        yaw_rate = float(yaw_rate)
        ay = float(ay_meas) - float(self.options.ay_bias)

        v_dot_y = ay - yaw_rate * vx
        if self.options.lowpass_tau is not None:
            tau = float(self.options.lowpass_tau)
            if tau <= 0.0:
                raise ValueError("lowpass_tau must be positive")
            alpha = self.dt / (tau + self.dt)
            if self._v_dot_y_filtered is None:
                self._v_dot_y_filtered = v_dot_y
            else:
                self._v_dot_y_filtered = float(
                    self._v_dot_y_filtered + alpha * (v_dot_y - self._v_dot_y_filtered)
                )
            v_dot_y_used = float(self._v_dot_y_filtered)
        else:
            self._v_dot_y_filtered = v_dot_y
            v_dot_y_used = float(v_dot_y)

        if self.options.leak_tau is not None:
            leak_tau = float(self.options.leak_tau)
            if leak_tau <= 0.0:
                raise ValueError("leak_tau must be positive")
            self._vy -= self._vy * (self.dt / leak_tau)

        self._vy += v_dot_y_used * self.dt
        if self.options.vy_limit is not None:
            limit = abs(float(self.options.vy_limit))
            self._vy = float(np.clip(self._vy, -limit, limit))

        min_vx = max(1e-6, float(self.options.min_vx))
        alpha_est: Dict[str, float] = {}
        for label, (x_i, y_i) in self.wheel_xy.items():
            v_wx = vx - yaw_rate * float(y_i)
            if abs(v_wx) < min_vx:
                v_wx = min_vx if v_wx >= 0.0 else -min_vx
            v_wy = self._vy + yaw_rate * float(x_i)
            delta = float(delta_map.get(label, 0.0))
            alpha_est[label] = float(np.arctan2(v_wy, v_wx) - delta)

        return alpha_est, float(self._vy)
