#!/usr/bin/env python3
"""
Lateral tire model.

Inputs:
    - V_wheel_x, V_wheel_y: wheel-frame longitudinal/lateral velocities [m/s]
    - F_tire: normal load on tire [N]

Outputs:
    - F_y_tire: lateral force [N]
    - aligning torque (stored in state)
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass

from vehicle_sim.utils.config_loader import load_param


@dataclass
class LateralTireParameters:
    """Parameters for a single lateral tire model."""

    C_alpha: float = 80000.0   # Cornering stiffness [N/rad]
    mu: float = 0.9            # Tire-road friction coefficient [-]
    trail: float = 0.05        # Wheel trail [m]


@dataclass
class LateralTireState:
    """Runtime state for lateral tire model."""

    slip_angle: float = 0.0      # Slip angle [rad]
    lateral_force: float = 0.0   # F_y [N]
    aligning_torque: float = 0.0 # M_z [N*m]


class LateralTireModel:
    """Lateral tire model.

    Simple linear relation in slip angle with friction limit.
    """

    def __init__(self, parameters: Optional[LateralTireParameters] = None,
                 config_path: Optional[str] = None):
        if parameters is not None:
            self.params = parameters
        else:
            tire_param = load_param('tire', config_path)
            lateral_param = tire_param.get('lateral', {})
            self.params = LateralTireParameters(
                C_alpha=float(lateral_param.get('C_alpha', LateralTireParameters.C_alpha)),
                mu=float(lateral_param.get('mu', tire_param.get('mu', LateralTireParameters.mu))),
                trail=float(lateral_param.get('trail', LateralTireParameters.trail)),
            )
        self.state = LateralTireState()

    def update(self, V_wheel_x: float, V_wheel_y: float, F_tire: float) -> float:
        """Update lateral force and aligning torque.

        Args:
            V_wheel_x: wheel-frame longitudinal velocity [m/s]
            V_wheel_y: wheel-frame lateral velocity [m/s]
            F_tire: normal load [N]

        Returns:
            lateral force F_y [N]
        """
        alpha = self.calculate_slip_angle(V_wheel_x, V_wheel_y)
        Fy = self.calculate_force(alpha, F_tire)
        M_z = self.calculate_aligning_torque(alpha, F_tire, Fy_override=Fy)

        self.state.slip_angle = alpha
        self.state.lateral_force = Fy
        self.state.aligning_torque = M_z

        return Fy

    def calculate_slip_angle(self, V_wheel_x: float, V_wheel_y: float) -> float:
        """Calculate slip angle in wheel frame."""
        return float(np.arctan2(V_wheel_y, V_wheel_x))

    def calculate_force(self, alpha: float, F_tire: float) -> float:
        """Calculate lateral force with linear + friction limit."""
        Fy = -self.params.C_alpha * alpha
        Fy_max = abs(self.params.mu * F_tire)
        return float(np.clip(Fy, -Fy_max, Fy_max))

    def calculate_aligning_torque(self, alpha: float, F_tire: float,
                                  Fy_override: Optional[float] = None) -> float:
        """Calculate steering self-aligning torque from lateral force.

        The simplified model uses `M_z = trail * F_y`.
        """
        Fy = Fy_override if Fy_override is not None else self.calculate_force(alpha, F_tire)
        return float(self.params.trail * Fy)

    def get_state(self) -> Dict:
        """Return current tire state."""
        return {
            "slip_angle": self.state.slip_angle,
            "lateral_force": self.state.lateral_force,
            "aligning_torque": self.state.aligning_torque,
        }

    def reset(self) -> None:
        """Reset lateral tire state."""
        self.state = LateralTireState()
