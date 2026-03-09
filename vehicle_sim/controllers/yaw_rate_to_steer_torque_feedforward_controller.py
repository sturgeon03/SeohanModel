"""
Yaw-rate to steering-torque feedforward wrapper.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

from .steer_angle_ff import (
    SteeringFeedforwardController,
    SteeringFeedforwardOptions,
)
from .steer_motor_ff import (
    SteeringMotorTorqueFeedforwardController,
    SteeringMotorTorqueFeedforwardOptions,
)
from .yaw_moment_allocator import YawMomentAllocator
from .yaw_moment_feedforward_controller import (
    YawMomentFeedforwardController,
    YawMomentFeedforwardOptions,
)
from .pid_controller import PIDController, PIDGains
from vehicle_sim.utils.config_loader import load_param


class YawRateToSteerTorqueFeedforwardController:
    """
    Pipeline: yaw_rate_cmd -> Mz -> Fy (per wheel) -> delta_cmd -> T_steer.
    """

    def __init__(
        self,
        dt: float,
        yaw_moment_options: Optional[YawMomentFeedforwardOptions] = None,
        steering_ff_options: Optional[SteeringFeedforwardOptions] = None,
        steering_torque_options: Optional[SteeringMotorTorqueFeedforwardOptions] = None,
        allocator: Optional[YawMomentAllocator] = None,
        gains_path: Optional[str] = None,
    ) -> None:
        self.dt = float(dt)
        self.yaw_moment_ff = YawMomentFeedforwardController(self.dt, yaw_moment_options)
        self.allocator = allocator or YawMomentAllocator()
        self.steering_ff = SteeringFeedforwardController(steering_ff_options)
        self.steering_torque_ff = SteeringMotorTorqueFeedforwardController(
            self.dt,
            steering_torque_options,
        )

        gains_path = gains_path or self._default_gains_path()
        self._yaw_pid = PIDController(self.dt, self._load_pid_gains("yaw_rate_pid", gains_path))
        self._fy_pid_gains = self._load_pid_gains("fy_pid", gains_path)
        self._steer_pid_gains = self._load_pid_gains("steering_pid", gains_path)
        self._fy_pids: Optional[Dict[str, PIDController]] = None
        self._steer_pids: Optional[Dict[str, PIDController]] = None

    def reset(self) -> None:
        """Reset internal command history."""
        self.yaw_moment_ff.reset()
        self.steering_torque_ff.reset()
        self._yaw_pid.reset()
        if self._fy_pids:
            for pid in self._fy_pids.values():
                pid.reset()
        if self._steer_pids:
            for pid in self._steer_pids.values():
                pid.reset()

    @staticmethod
    def _default_gains_path() -> str:
        base = Path(__file__).resolve().parents[1]
        return str(base / "models" / "params" / "controller_gains.yaml")

    @staticmethod
    def _load_pid_gains(section: str, config_path: str) -> PIDGains:
        cfg = load_param(section, config_path)
        return PIDGains(
            kp=float(cfg.get("kp", 0.0)),
            ki=float(cfg.get("ki", 0.0)),
            kd=float(cfg.get("kd", 0.0)),
        )

    def compute_torque(
        self,
        vehicle_body,
        yaw_rate_cmd: float,
        vx_cmd: float,
        yaw_accel_cmd: Optional[float] = None,
        fx_body: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Compute per-wheel steering torque feedforward (motor torque)."""
        t_steer, _ = self.compute_torque_with_debug(
            vehicle_body,
            yaw_rate_cmd=yaw_rate_cmd,
            vx_cmd=vx_cmd,
            yaw_accel_cmd=yaw_accel_cmd,
            fx_body=fx_body,
        )
        return t_steer

    def compute_torque_with_debug(
        self,
        vehicle_body,
        yaw_rate_cmd: float,
        vx_cmd: float,
        yaw_accel_cmd: Optional[float] = None,
        fx_body: Optional[Dict[str, float]] = None,
    ) -> Tuple[Dict[str, float], Dict[str, Union[Dict[str, float], float]]]:
        """
        Compute steering torque feedforward (motor torque) and return intermediate signals.

        Returns:
            (t_steer, debug) where debug has keys: Mz_cmd, fy_cmd, delta_cmd.
        """
        labels = list(vehicle_body.wheel_labels)
        if self._fy_pids is None or set(self._fy_pids.keys()) != set(labels):
            self._fy_pids = {label: PIDController(self.dt, self._fy_pid_gains) for label in labels}
        if self._steer_pids is None or set(self._steer_pids.keys()) != set(labels):
            self._steer_pids = {label: PIDController(self.dt, self._steer_pid_gains) for label in labels}

        Mz_ff = self.yaw_moment_ff.compute_moment(
            vehicle_body,
            yaw_rate_cmd=yaw_rate_cmd,
            yaw_accel_cmd=yaw_accel_cmd,
        )
        yaw_error = float(yaw_rate_cmd) - float(vehicle_body.state.yaw_rate)
        Mz_fb = self._yaw_pid.update(yaw_error)
        Mz_cmd = Mz_ff + Mz_fb

        fy_total_cmd = float(vehicle_body.params.m) * float(vx_cmd) * float(yaw_rate_cmd)
        fy_wheel_cmd = self.allocator.allocate(
            vehicle_body,
            Mz_cmd,
            Fx_body=fx_body,
            Fy_total_cmd=fy_total_cmd,
        )
        fy_wheel_actual: Dict[str, float] = {}
        fy_body_actual: Dict[str, float] = {}
        for label in labels:
            corner = vehicle_body.corners[label]
            delta_actual = float(corner.state.steering_angle)
            c, s = np.cos(delta_actual), np.sin(delta_actual)
            f_x = float(corner.state.F_x_tire)
            f_y = float(corner.state.F_y_tire)
            fy_wheel_actual[label] = f_y
            fy_body_actual[label] = s * f_x + c * f_y

        delta_cmd = self.steering_ff.compute_delta_cmd(
            vehicle_body,
            fy_wheel_cmd,
            vx_cmd=vx_cmd,
            yaw_rate_cmd=yaw_rate_cmd,
        )
        delta_fb = {}
        for label in labels:
            fy_error = float(fy_wheel_cmd.get(label, 0.0)) - float(fy_wheel_actual.get(label, 0.0))
            delta_correction = self._fy_pids[label].update(fy_error)
            delta_cmd[label] = float(delta_cmd.get(label, 0.0)) + float(delta_correction)
            delta_fb[label] = float(delta_correction)

        t_align_cmd: Dict[str, float] = {}
        for label in labels:
            corner = vehicle_body.corners[label]
            fy_wheel = float(fy_wheel_cmd.get(label, 0.0))
            fz = float(corner.state.F_z)
            if fz > 0.0:
                mu = float(corner.lateral_tire.params.mu)
                fy_wheel = float(np.clip(fy_wheel, -mu * abs(fz), mu * abs(fz)))
            trail = float(corner.lateral_tire.params.trail)
            t_align_cmd[label] = float(trail * fy_wheel)

        t_steer_ff = self.steering_torque_ff.compute_torque(
            vehicle_body,
            delta_cmd,
            aligning_torque_cmd=t_align_cmd,
        )
        t_steer = {}
        t_steer_fb = {}
        for label in labels:
            delta_error = float(delta_cmd.get(label, 0.0)) - float(vehicle_body.corners[label].state.steering_angle)
            torque_correction = self._steer_pids[label].update(delta_error)
            params = vehicle_body.corners[label].steering.params
            scale = float(params.gear_ratio)
            if abs(scale) < 1e-6:
                torque_correction_motor = float(torque_correction)
            else:
                torque_correction_motor = float(torque_correction) / scale
            t_steer[label] = float(t_steer_ff.get(label, 0.0)) + torque_correction_motor
            t_steer_fb[label] = torque_correction_motor
        debug = {
            "Mz_cmd": Mz_cmd,
            "Mz_ff": Mz_ff,
            "Mz_fb": Mz_fb,
            "fy_total_cmd": fy_total_cmd,
            "fy_cmd": fy_wheel_cmd,
            "fy_wheel_cmd": fy_wheel_cmd,
            "fy_wheel_actual": fy_wheel_actual,
            "fy_body_actual": fy_body_actual,
            "delta_cmd": delta_cmd,
            "delta_fb": delta_fb,
            "t_align_cmd": t_align_cmd,
            "t_steer_ff": t_steer_ff,
            "t_steer_fb": t_steer_fb,
        }
        return t_steer, debug
