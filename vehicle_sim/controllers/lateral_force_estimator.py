"""
Lateral force estimator using measured lateral acceleration (ay).
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class LateralForceEstimatorOptions:
    """Configuration for lateral force estimation from ay."""
    ay_bias: float = 0.0
    lowpass_tau: Optional[float] = None
    max_abs_ay: Optional[float] = None


class LateralForceEstimator:
    """Estimate total lateral force from lateral acceleration measurements."""

    def __init__(
        self,
        dt: float,
        mass: float,
        options: Optional[LateralForceEstimatorOptions] = None,
    ) -> None:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if mass <= 0.0:
            raise ValueError("mass must be positive")

        self.dt = float(dt)
        self.mass = float(mass)
        self.options = options or LateralForceEstimatorOptions()
        self._ay_filtered: Optional[float] = None

    def reset(self) -> None:
        """Reset internal filter state."""
        self._ay_filtered = None

    def update(self, ay_meas: float) -> float:
        """
        Estimate total lateral force from ay measurement.

        Args:
            ay_meas: measured lateral acceleration [m/s^2].
        """
        ay = float(ay_meas) - float(self.options.ay_bias)

        if self.options.max_abs_ay is not None:
            limit = abs(float(self.options.max_abs_ay))
            ay = float(np.clip(ay, -limit, limit))

        if self.options.lowpass_tau is not None:
            tau = float(self.options.lowpass_tau)
            if tau <= 0.0:
                raise ValueError("lowpass_tau must be positive")
            alpha = self.dt / (tau + self.dt)
            if self._ay_filtered is None:
                self._ay_filtered = ay
            else:
                self._ay_filtered = float(self._ay_filtered + alpha * (ay - self._ay_filtered))
            ay_used = float(self._ay_filtered)
        else:
            self._ay_filtered = ay
            ay_used = float(ay)

        return float(self.mass * ay_used)

    @property
    def last_ay(self) -> Optional[float]:
        """Latest filtered ay used for the force estimate."""
        return self._ay_filtered
