"""
Yaw moment allocator using equal per-wheel moment split.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class YawMomentAllocator:
    """Allocate per-wheel lateral forces for a desired yaw moment.

    Note:
        This allocator returns per-wheel lateral force commands in the wheel frame (Fy_wheel_cmd).
        It uses the body-frame yaw moment relationship (Mz = Σ(x_i*Fy - y_i*Fx)) with a small-angle
        approximation that treats Fy_wheel ≈ Fy_body. This avoids an extra δ-dependent body→wheel
        conversion stage downstream. If Fy_total_cmd is provided, it allocates Fy to satisfy both
        total lateral force and yaw moment.
    """

    min_abs_x: float = 1e-6

    def allocate(
        self,
        vehicle_body,
        Mz_d: float,
        Fx_body: Optional[Dict[str, float]] = None,
        Fy_total_cmd: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Args:
            vehicle_body: VehicleBody instance (geometry + wheel labels).
            Mz_d: desired yaw moment about CG [N*m], body frame.
            Fx_body: optional per-wheel longitudinal forces [N] in body frame.
            Fy_total_cmd: optional total lateral force command [N] in body frame.

        Returns:
            per-wheel lateral force commands Fy_wheel_cmd [N] in wheel frame.
        """
        labels = list(vehicle_body.wheel_labels)
        if not labels:
            return {}

        fx_map = Fx_body or {}
        fy_wheel_cmd: Dict[str, float] = {}

        if Fy_total_cmd is None:
            Mz_i = float(Mz_d) / len(labels)
            for label in labels:
                signs = vehicle_body.corner_signs[label]
                x_i = (vehicle_body.params.L_wheelbase / 2.0) * signs["pitch"]
                y_i = (vehicle_body.params.L_track / 2.0) * signs["roll"]
                if abs(x_i) < self.min_abs_x:
                    raise ValueError("Wheelbase too small for yaw moment allocation")
                Fx_i = float(fx_map.get(label, 0.0))
                # Small-angle approximation: treat Fy_wheel ≈ Fy_body in the yaw moment balance.
                fy_wheel_cmd[label] = (Mz_i + y_i * Fx_i) / x_i
            return fy_wheel_cmd

        x_list = []
        y_list = []
        fx_list = []
        for label in labels:
            signs = vehicle_body.corner_signs[label]
            x_i = (vehicle_body.params.L_wheelbase / 2.0) * signs["pitch"]
            y_i = (vehicle_body.params.L_track / 2.0) * signs["roll"]
            if abs(x_i) < self.min_abs_x:
                raise ValueError("Wheelbase too small for yaw moment allocation")
            x_list.append(float(x_i))
            y_list.append(float(y_i))
            fx_list.append(float(fx_map.get(label, 0.0)))

        x = np.asarray(x_list, dtype=float)
        y = np.asarray(y_list, dtype=float)
        fx = np.asarray(fx_list, dtype=float)

        b_total = float(Fy_total_cmd)
        b_yaw = float(Mz_d) + float(np.dot(y, fx))

        # Least-norm allocation for A * Fy = b with A=[1; x].
        A = np.vstack([np.ones_like(x), x])
        ata = A @ A.T
        b = np.array([b_total, b_yaw], dtype=float)
        try:
            weights = np.linalg.solve(ata, b)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Yaw moment allocation matrix is singular") from exc
        fy_vec = A.T @ weights

        for label, fy in zip(labels, fy_vec):
            fy_wheel_cmd[label] = float(fy)

        return fy_wheel_cmd
