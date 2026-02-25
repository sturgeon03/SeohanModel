"""
Run target speed + per-wheel steering scenario and export a GIF.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import yaml

# Path setup
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = (CURRENT_DIR / "../../..").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from vehicle_sim.models.vehicle_body.vehicle_body import VehicleBody
from vehicle_sim.models.visualization.vehicle_visualizer import VehicleVisualizer
from vehicle_sim.controllers.speed_controller_v2 import SpeedControllerV2, SpeedControllerV2Gains
from vehicle_sim.controllers.steer_motor_ff import (
    SteeringMotorTorqueFeedforwardController,
    SteeringMotorTorqueFeedforwardOptions,
)
from vehicle_sim.controllers.pid_controller import PIDController, PIDGains
from vehicle_sim.controllers.slip_angle_estimator import SlipAngleEstimator, SlipAngleEstimatorOptions
from vehicle_sim.controllers.tire_lateral_force_estimator import (
    TireLateralForceEstimator,
    TireLateralForceEstimatorOptions,
)

WHEEL_LABELS = ["FL", "FR", "RL", "RR"]


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def normalize_wheel_map(value, labels, default=0.0) -> Dict[str, float]:
    if value is None:
        return {label: float(default) for label in labels}
    if isinstance(value, dict):
        return {label: float(value.get(label, default)) for label in labels}
    return {label: float(value) for label in labels}


def steering_targets_from_cfg(targets_cfg: Dict, labels) -> Dict[str, float]:
    unit = str(targets_cfg.get("steering_unit", "deg")).lower()
    if "steering_rad" in targets_cfg:
        steering = normalize_wheel_map(targets_cfg.get("steering_rad"), labels)
        return steering
    if "steering_deg" in targets_cfg:
        steering_deg = normalize_wheel_map(targets_cfg.get("steering_deg"), labels)
        return {label: float(np.deg2rad(val)) for label, val in steering_deg.items()}
    if "steering" in targets_cfg:
        steering = normalize_wheel_map(targets_cfg.get("steering"), labels)
        if unit == "deg":
            return {label: float(np.deg2rad(val)) for label, val in steering.items()}
        return steering
    return {label: 0.0 for label in labels}


def build_steering_provider(
    targets_cfg: Dict,
    labels: List[str],
) -> Tuple[Callable[[float], Dict[str, float]], Dict[str, Dict[str, float]], bool]:
    profile_cfg = targets_cfg.get("steering_profile", {}) or {}
    if bool(profile_cfg.get("enabled", False)):
        profile_type = str(profile_cfg.get("type", "sine")).lower()
        if profile_type != "sine":
            raise ValueError(f"Unsupported steering_profile.type: {profile_type}")

        unit = str(profile_cfg.get("unit", targets_cfg.get("steering_unit", "deg"))).lower()
        if unit not in {"deg", "rad"}:
            raise ValueError(f"steering_profile.unit must be 'deg' or 'rad', got {unit}")

        default_cfg = profile_cfg.get("default", {}) or {}
        wheel_cfgs = profile_cfg.get("wheels", {}) or {}
        params_rad: Dict[str, Dict[str, float]] = {}

        for label in labels:
            wheel_cfg = wheel_cfgs.get(label, {}) or {}
            amp = float(wheel_cfg.get("amp", default_cfg.get("amp", 0.0)))
            bias = float(wheel_cfg.get("bias", default_cfg.get("bias", 0.0)))
            phase = float(wheel_cfg.get("phase", default_cfg.get("phase", 0.0)))
            freq = float(wheel_cfg.get("freq_hz", default_cfg.get("freq_hz", 0.0)))
            if unit == "deg":
                amp = float(np.deg2rad(amp))
                bias = float(np.deg2rad(bias))
                phase = float(np.deg2rad(phase))
            params_rad[label] = {
                "amp": amp,
                "bias": bias,
                "phase": phase,
                "freq_hz": freq,
            }

        def provider(time_s: float) -> Dict[str, float]:
            return {
                label: float(p["bias"] + p["amp"] * np.sin(2 * np.pi * p["freq_hz"] * time_s + p["phase"]))
                for label, p in params_rad.items()
            }

        return provider, params_rad, True

    static_targets = steering_targets_from_cfg(targets_cfg, labels)
    return lambda _time_s: static_targets, static_targets, False


def build_wheel_xy(vehicle) -> Dict[str, Tuple[float, float]]:
    wheelbase = float(vehicle.params.L_wheelbase)
    track = float(vehicle.params.L_track)
    a = 0.5 * wheelbase
    b = 0.5 * wheelbase
    half_track = 0.5 * track
    return {
        "FL": (a, half_track),
        "FR": (a, -half_track),
        "RL": (-b, half_track),
        "RR": (-b, -half_track),
    }


def plot_steering_subplots(
    time_s: np.ndarray,
    delta_map: Dict[str, np.ndarray],
    labels: List[str],
    output_cfg: Dict,
    save_dir: Path,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    for label in labels:
        delta_deg = np.rad2deg(delta_map[label])
        ax.plot(time_s, delta_deg, linewidth=1.5, label=label)
    ax.set_title("Steering Angles (Per Wheel)", fontweight="bold")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Steering [deg]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()

    if bool(output_cfg.get("save_steering_plot", True)):
        filename = output_cfg.get("steering_plot_filename", "steering_angles.png")
        fig.savefig(save_dir / filename, dpi=150, bbox_inches="tight")


def estimate_align_torque(
    vehicle,
    slip_estimator: SlipAngleEstimator,
    tire_estimator: TireLateralForceEstimator,
    vx: float,
    yaw_rate: float,
    ay_meas: float,
) -> Dict[str, float]:
    delta_map = {
        label: float(vehicle.corners[label].state.steering_angle)
        for label in WHEEL_LABELS
    }
    alpha_est, _ = slip_estimator.update(vx, yaw_rate, ay_meas, delta_map)
    c_alpha_map = {
        label: float(vehicle.corners[label].lateral_tire.params.C_alpha)
        for label in WHEEL_LABELS
    }
    fy_cmd_map = {label: 0.0 for label in WHEEL_LABELS}
    fy_est_map, _ = tire_estimator.estimate(alpha_est, fy_cmd_map, c_alpha_map)

    align_cmd = {}
    for label in WHEEL_LABELS:
        corner = vehicle.corners[label]
        fy = float(fy_est_map.get(label, 0.0))
        fz = float(corner.state.F_z)
        if fz > 0.0:
            mu = float(corner.lateral_tire.params.mu)
            fy = float(np.clip(fy, -mu * abs(fz), mu * abs(fz)))
        trail = float(corner.lateral_tire.params.trail)
        align_cmd[label] = float(trail * fy)
    return align_cmd


def run(config_path: Path) -> None:
    cfg = load_yaml(config_path)
    sim_cfg = cfg.get("sim", {})
    targets_cfg = cfg.get("targets", {})
    speed_cfg = cfg.get("speed_controller", {})
    steering_cfg = cfg.get("steering_controller", {})
    estimator_cfg = cfg.get("estimator", {})
    output_cfg = cfg.get("output", {})

    dt = float(sim_cfg.get("dt", 0.001))
    total_time = float(sim_cfg.get("time", 5.0))
    direction = int(sim_cfg.get("direction", 1))
    log_stride = max(1, int(sim_cfg.get("log_stride", 1)))
    vehicle_config_path = sim_cfg.get("vehicle_config_path", None)
    if not vehicle_config_path:
        vehicle_config_path = None

    vehicle = VehicleBody(config_path=vehicle_config_path)
    vehicle.reset()

    speed_cmd = float(targets_cfg.get("speed_mps", 0.0))
    steering_provider, steering_meta, steering_dynamic = build_steering_provider(
        targets_cfg, WHEEL_LABELS
    )
    print("Loaded targets:")
    print(f"  speed_mps: {speed_cmd:.3f}")
    if steering_dynamic:
        print("  steering_profile: sine")
        for label in WHEEL_LABELS:
            params = steering_meta[label]
            print(
                "    "
                f"{label}: amp={np.rad2deg(params['amp']):.1f} deg, "
                f"freq={params['freq_hz']:.3f} Hz, "
                f"phase={np.rad2deg(params['phase']):.1f} deg, "
                f"bias={np.rad2deg(params['bias']):.1f} deg"
            )
    else:
        print(
            "  steering_deg: "
            + ", ".join(
                f"{label}={np.rad2deg(steering_meta[label]):.1f}"
                for label in WHEEL_LABELS
            )
        )

    gains_cfg = speed_cfg.get("gains", {})
    speed_gains = SpeedControllerV2Gains(
        kp=float(gains_cfg.get("kp", 50.0)),
        ki=float(gains_cfg.get("ki", 50.0)),
        kd=float(gains_cfg.get("kd", 0.1)),
        accel_limit=gains_cfg.get("accel_limit", 5.0),
        torque_limit=gains_cfg.get("torque_limit", 400.0),
        torque_rate_limit=gains_cfg.get("torque_rate_limit", None),
        integrator_limit=gains_cfg.get("integrator_limit", None),
    )
    wheel_radius = float(vehicle.corners["FL"].drive.params.R_wheel)
    speed_controller = SpeedControllerV2(
        dt=dt,
        mass=vehicle.params.m,
        wheel_radius=wheel_radius,
        gains=speed_gains,
        num_wheels=len(WHEEL_LABELS),
    )

    ff_cfg = steering_cfg.get("feedforward", {})
    steer_ff_opts = SteeringMotorTorqueFeedforwardOptions(
        max_accel=ff_cfg.get("max_accel", None),
        torque_limit=ff_cfg.get("torque_limit", None),
    )
    steering_ff = SteeringMotorTorqueFeedforwardController(dt=dt, options=steer_ff_opts)

    align_source = str(steering_cfg.get("align_torque_source", "estimate")).lower()
    feedback_cfg = steering_cfg.get("feedback", {})
    feedback_enabled = bool(feedback_cfg.get("enabled", True))
    steer_pids = None
    if feedback_enabled:
        gains = PIDGains(
            kp=float(feedback_cfg.get("kp", 100.0)),
            ki=float(feedback_cfg.get("ki", 50.0)),
            kd=float(feedback_cfg.get("kd", 10.0)),
        )
        steer_pids = {label: PIDController(dt, gains) for label in WHEEL_LABELS}
    slip_estimator = None
    tire_estimator = None
    if align_source == "estimate":
        slip_cfg = estimator_cfg.get("slip_angle", {})
        slip_opts = SlipAngleEstimatorOptions(
            ay_bias=float(slip_cfg.get("ay_bias", 0.0)),
            lowpass_tau=slip_cfg.get("lowpass_tau", None),
            vy_init=float(slip_cfg.get("vy_init", 0.0)),
            vy_limit=slip_cfg.get("vy_limit", None),
            leak_tau=slip_cfg.get("leak_tau", None),
            min_vx=float(slip_cfg.get("min_vx", 0.1)),
        )
        wheel_xy = build_wheel_xy(vehicle)
        slip_estimator = SlipAngleEstimator(dt=dt, wheel_xy=wheel_xy, options=slip_opts)

        lat_cfg = estimator_cfg.get("lateral_force", {})
        tire_opts = TireLateralForceEstimatorOptions(
            use_command_limit=bool(lat_cfg.get("use_command_limit", False)),
            min_cmd_abs=float(lat_cfg.get("min_cmd_abs", 1e-3)),
        )
        tire_estimator = TireLateralForceEstimator(options=tire_opts)

    if align_source not in {"estimate", "true", "ignore"}:
        raise ValueError(f"Invalid align_torque_source: {align_source}")

    time_log: List[float] = []
    x_log: List[float] = []
    y_log: List[float] = []
    yaw_log: List[float] = []
    vx_log: List[float] = []
    vy_log: List[float] = []
    yaw_rate_log: List[float] = []
    delta_log = {label: [] for label in WHEEL_LABELS}

    t = 0.0
    prev_vy = None
    steps = int(np.ceil(total_time / dt))

    for step in range(steps):
        state = vehicle.state
        vx = float(state.velocity_x)
        vy = float(state.velocity_y)
        yaw_rate = float(state.yaw_rate)

        if prev_vy is None:
            vy_dot = 0.0
        else:
            vy_dot = (vy - prev_vy) / dt
        ay_meas = vy_dot + yaw_rate * vx

        if align_source == "estimate":
            align_cmd = estimate_align_torque(
                vehicle,
                slip_estimator,
                tire_estimator,
                vx=vx,
                yaw_rate=yaw_rate,
                ay_meas=ay_meas,
            )
        elif align_source == "true":
            align_cmd = {
                label: float(vehicle.corners[label].lateral_tire.state.aligning_torque)
                for label in WHEEL_LABELS
            }
        else:
            align_cmd = {label: 0.0 for label in WHEEL_LABELS}

        steering_cmd = steering_provider(t)
        t_steer = steering_ff.compute_torque(
            vehicle,
            delta_cmd=steering_cmd,
            aligning_torque_cmd=align_cmd,
        )
        if steer_pids is not None:
            for label in WHEEL_LABELS:
                delta_error = float(steering_cmd[label]) - float(
                    vehicle.corners[label].state.steering_angle
                )
                torque_correction = steer_pids[label].update(delta_error)
                params = vehicle.corners[label].steering.params
                scale = float(params.gear_ratio)
                if abs(scale) < 1e-6:
                    torque_correction_motor = float(torque_correction)
                else:
                    torque_correction_motor = float(torque_correction) / scale
                t_steer[label] = float(t_steer.get(label, 0.0)) + torque_correction_motor
        t_drv = speed_controller.update(speed_cmd, vx)

        corner_inputs = {
            label: {
                "T_steer": float(t_steer.get(label, 0.0)),
                "T_brk": 0.0,
                "T_Drv": float(t_drv),
                "T_susp": 0.0,
                "z_road": 0.0,
                "z_road_dot": 0.0,
            }
            for label in WHEEL_LABELS
        }

        prev_vy = vy
        vehicle.update(dt, corner_inputs, direction=direction)
        t += dt

        if step % log_stride == 0:
            state = vehicle.state
            time_log.append(t)
            x_log.append(float(state.x))
            y_log.append(float(state.y))
            yaw_log.append(float(state.yaw))
            vx_log.append(float(state.velocity_x))
            vy_log.append(float(state.velocity_y))
            yaw_rate_log.append(float(state.yaw_rate))
            for label in WHEEL_LABELS:
                delta_log[label].append(float(vehicle.corners[label].state.steering_angle))

    if len(time_log) < 2:
        print("Not enough data points to animate.")
        return

    vehicle_params = {
        "L_wheelbase": float(vehicle.params.L_wheelbase),
        "L_track": float(vehicle.params.L_track),
        "h_CG": float(vehicle.params.h_CG),
        "R_wheel": float(vehicle.corners["FL"].drive.params.R_wheel),
    }

    save_dir = output_cfg.get("save_dir", "output")
    save_dir_path = Path(save_dir)
    if not save_dir_path.is_absolute():
        save_dir_path = CURRENT_DIR / save_dir_path
    save_dir_path.mkdir(parents=True, exist_ok=True)

    delta_log_np = {label: np.array(vals) for label, vals in delta_log.items()}
    plot_steering_subplots(
        time_s=np.array(time_log),
        delta_map=delta_log_np,
        labels=WHEEL_LABELS,
        output_cfg=output_cfg,
        save_dir=save_dir_path,
    )

    gif_filename = output_cfg.get("gif_filename", "target_speed_steering.gif")
    gif_path = save_dir_path / gif_filename

    visualizer = VehicleVisualizer(mode="animation", vehicle_params=vehicle_params)
    visualizer.animate_trajectory(
        time=np.array(time_log),
        x_log=np.array(x_log),
        y_log=np.array(y_log),
        yaw_log=np.array(yaw_log),
        delta_log=delta_log_np,
        velocity_x_log=np.array(vx_log),
        velocity_y_log=np.array(vy_log),
        yaw_rate_log=np.array(yaw_rate_log),
        labels=WHEEL_LABELS,
        stride=int(output_cfg.get("stride", 20)),
        save_gif=bool(output_cfg.get("save_gif", True)),
        gif_filename=str(gif_path),
        max_gif_frames=int(output_cfg.get("max_gif_frames", 500)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Target speed + steering GIF demo")
    parser.add_argument(
        "--config",
        type=str,
        default=str(CURRENT_DIR / "config.yaml"),
        help="Path to scenario YAML config",
    )
    args = parser.parse_args()
    run(Path(args.config))


if __name__ == "__main__":
    main()
