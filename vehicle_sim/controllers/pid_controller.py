"""
Minimal PID controller (discrete-time, Euler integration).

u = kp * e + ki * ∫e dt + kd * de/dt
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PIDGains:
    kp: float = 0.0
    ki: float = 0.0
    kd: float = 0.0


class PIDController:
    """Simple PID controller without extra features."""

    def __init__(self, dt: float, gains: Optional[PIDGains] = None) -> None:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self.dt = float(dt)
        self.gains = gains or PIDGains()
        self._integral = 0.0
        self._prev_error: Optional[float] = None

    def reset(self) -> None:
        """Reset internal state."""
        self._integral = 0.0
        self._prev_error = None

    def update(self, error: float) -> float:
        """
        Compute PID output from the given error.

        Args:
            error: setpoint - measurement
        """
        e = float(error)
        if self._prev_error is None:
            de = 0.0
        else:
            de = (e - self._prev_error) / self.dt

        self._integral += e * self.dt
        self._prev_error = e

        return (
            self.gains.kp * e
            + self.gains.ki * self._integral
            + self.gains.kd * de
        )
