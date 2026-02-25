"""
Estimate per-wheel and total lateral forces from slip angles.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class TireLateralForceEstimatorOptions:
    """Options for tire lateral force estimation."""
    use_command_limit: bool = True
    min_cmd_abs: float = 1e-3


class TireLateralForceEstimator:
    """Estimate lateral forces with a linear tire model and optional command limits."""

    def __init__(self, options: Optional[TireLateralForceEstimatorOptions] = None) -> None:
        self.options = options or TireLateralForceEstimatorOptions()

    def estimate(
        self,
        alpha_map: Dict[str, float],
        fy_cmd_map: Dict[str, float],
        c_alpha_map: Dict[str, float],
    ) -> Tuple[Dict[str, float], float]:
        fy_est: Dict[str, float] = {}
        total = 0.0
        labels = set(alpha_map) | set(c_alpha_map) | set(fy_cmd_map)

        for label in labels:
            alpha = float(alpha_map.get(label, 0.0))
            c_alpha = float(c_alpha_map.get(label, 0.0))
            fy_model = -c_alpha * alpha
            if self.options.use_command_limit:
                limit = max(float(self.options.min_cmd_abs), abs(float(fy_cmd_map.get(label, 0.0))))
                fy_est_i = float(np.clip(fy_model, -limit, limit))
            else:
                fy_est_i = float(fy_model)
            fy_est[label] = fy_est_i
            total += fy_est_i

        return fy_est, float(total)
