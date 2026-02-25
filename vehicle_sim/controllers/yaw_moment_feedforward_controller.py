"""
Yaw moment feedforward controller using r_dot = Mz / Izz (roll/pitch ignored).
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class YawMomentFeedforwardOptions:
    max_yaw_accel: Optional[float] = None
    torque_limit: Optional[float] = None


class YawMomentFeedforwardController:
    """
    Compute yaw moment feedforward from desired yaw rate.

    Model (roll/pitch ignored): Mz = Izz * r_dot
    """

    def __init__(
        self,
        dt: float,
        options: Optional[YawMomentFeedforwardOptions] = None,
    ) -> None:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self.dt = float(dt)
        self.options = options or YawMomentFeedforwardOptions()
        self.prev_yaw_rate_cmd: Optional[float] = None

    def reset(self) -> None:
        """Reset internal command history."""
        self.prev_yaw_rate_cmd = None

    def compute_moment(
        self,
        vehicle_body,
        yaw_rate_cmd: float,
        yaw_accel_cmd: Optional[float] = None,
    ) -> float:
        """
        Args:
            vehicle_body: VehicleBody instance (for Izz).
            yaw_rate_cmd: desired yaw rate [rad/s].
            yaw_accel_cmd: optional desired yaw accel [rad/s^2].
        """
        r_cmd = float(yaw_rate_cmd)
        if yaw_accel_cmd is None:
            if self.prev_yaw_rate_cmd is None:
                r_dot_cmd = 0.0
            else:
                r_dot_cmd = (r_cmd - self.prev_yaw_rate_cmd) / self.dt
        else:
            r_dot_cmd = float(yaw_accel_cmd)

        if self.options.max_yaw_accel is not None:
            r_dot_cmd = float(
                np.clip(r_dot_cmd, -self.options.max_yaw_accel, self.options.max_yaw_accel)
            )

        Izz = float(vehicle_body.params.Izz)
        if Izz <= 0.0:
            raise ValueError("Izz must be positive for yaw moment feedforward")

        Mz = Izz * r_dot_cmd
        if self.options.torque_limit is not None:
            Mz = float(np.clip(Mz, -self.options.torque_limit, self.options.torque_limit))

        self.prev_yaw_rate_cmd = r_cmd
        return float(Mz)
