"""
Longitudinal speed PID controller (v2) using measured acceleration.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SpeedControllerV2Gains:
    """Gains and limits for speed control."""
    kp: float = 50.0
    ki: float = 50.0
    kd: float = 0.1
    accel_limit: Optional[float] = 5.0
    torque_limit: Optional[float] = 400.0
    torque_rate_limit: Optional[float] = None
    integrator_limit: Optional[float] = None


class SpeedControllerV2:
    """
    Speed controller: v_cmd -> T_drv (per wheel input) using measured acceleration.

    e_dot = v_cmd_dot - v_dot (measured)
    a_cmd = kp * e + kd * e_dot + ki * integral
    T_total = m * a_cmd * R_eff
    T_per_wheel = T_total / n_wheels
    """

    def __init__(
        self,
        dt: float,
        mass: float,
        wheel_radius: float,
        gains: Optional[SpeedControllerV2Gains] = None,
        num_wheels: int = 4,
    ) -> None:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if mass <= 0.0:
            raise ValueError("mass must be positive")
        if wheel_radius <= 0.0:
            raise ValueError("wheel_radius must be positive")
        if num_wheels <= 0:
            raise ValueError("num_wheels must be positive")

        self.dt = float(dt)
        self.mass = float(mass)
        self.wheel_radius = float(wheel_radius)
        self.num_wheels = int(num_wheels)
        self.gains = gains or SpeedControllerV2Gains()

        self._prev_speed_cmd: Optional[float] = None
        self._prev_speed: Optional[float] = None
        self._integral_error: float = 0.0
        self._prev_torque: float = 0.0

    def reset(self) -> None:
        """Reset controller history."""
        self._prev_speed_cmd = None
        self._prev_speed = None
        self._integral_error = 0.0
        self._prev_torque = 0.0

    def update(
        self,
        speed_cmd: float,
        speed: float,
        speed_dot: Optional[float] = None,
    ) -> float:
        """
        Compute per-wheel drive torque to track speed input.

        Args:
            speed_cmd: desired speed [m/s].
            speed: measured speed [m/s].
            speed_dot: measured acceleration [m/s^2].
        """
        v_cmd = float(speed_cmd)
        v = float(speed)

        if self._prev_speed_cmd is None:
            v_cmd_dot = 0.0
        else:
            v_cmd_dot = (v_cmd - self._prev_speed_cmd) / self.dt

        if speed_dot is None:
            if self._prev_speed is None:
                v_dot = 0.0
            else:
                v_dot = (v - self._prev_speed) / self.dt
        else:
            v_dot = float(speed_dot)

        error = v_cmd - v
        error_dot = v_cmd_dot - v_dot

        if self.gains.ki != 0.0:
            self._integral_error += error * self.dt
            if self.gains.integrator_limit is not None:
                limit = abs(float(self.gains.integrator_limit))
                self._integral_error = float(np.clip(self._integral_error, -limit, limit))

        accel_cmd = (
            self.gains.kp * error
            + self.gains.kd * error_dot
            + self.gains.ki * self._integral_error
        )

        if self.gains.accel_limit is not None:
            limit = abs(float(self.gains.accel_limit))
            accel_cmd = float(np.clip(accel_cmd, -limit, limit))

        total_torque = self.mass * accel_cmd * self.wheel_radius
        torque_per_wheel = total_torque / float(self.num_wheels)

        if self.gains.torque_rate_limit is not None:
            rate = abs(float(self.gains.torque_rate_limit))
            delta = torque_per_wheel - self._prev_torque
            delta = float(np.clip(delta, -rate * self.dt, rate * self.dt))
            torque_per_wheel = self._prev_torque + delta

        if self.gains.torque_limit is not None:
            limit = abs(float(self.gains.torque_limit))
            torque_per_wheel = float(np.clip(torque_per_wheel, -limit, limit))

        self._prev_speed_cmd = v_cmd
        self._prev_speed = v
        self._prev_torque = torque_per_wheel

        return float(torque_per_wheel)
