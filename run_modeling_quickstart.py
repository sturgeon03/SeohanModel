#!/usr/bin/env python3
"""
Modeling quick start runner.

This script provides one stable entry point for first-time users:
1) Run a simple vehicle simulation scenario.
2) Save logs and figures to a fixed output location.
3) Print a short summary so users can confirm it worked.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as mpl_animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from vehicle_sim.models.vehicle_body.vehicle_body import VehicleBody


CORNER_LABELS = ("FL", "FR", "RL", "RR")
DEFAULT_OUTPUT_ROOT = Path("outputs") / "modeling_quickstart"
DEFAULT_DURATION_SCENARIO_S = 12.0
DEFAULT_DT_SCENARIO_S = 0.01
DEFAULT_DATA_ROOT = Path("vehicle_sim") / "Data"


@dataclass
class ScenarioCommand:
    """Per-step command values for the whole vehicle."""

    drive_torque: float
    brake_torque: float
    steer_torque_left: float
    steer_torque_right: float
    z_road_front: float
    z_road_dot_front: float


@dataclass
class CBNUReplayData:
    """Parsed CBNU log data used as replay references."""

    source_file: Path
    time_s: np.ndarray
    dt_s: np.ndarray
    speed_kph: np.ndarray
    steer_deg: Dict[str, np.ndarray]
    steer_rate_deg_s: Dict[str, np.ndarray]
    steer_bias_deg: Dict[str, float]


@dataclass
class ReplayControllerConfig:
    """Simple tracking controller gains for replay mode."""

    speed_kp: float = 90.0
    speed_ki: float = 12.0
    speed_integral_limit: float = 10.0
    max_drive_torque: float = 220.0
    max_brake_torque: float = 140.0
    brake_gain: float = 1.0
    steer_kp: float = 8.0
    steer_kd: float = 35.0
    max_steer_torque: float = 8.0


def road_bump(t: float, t_start: float = 4.0, duration: float = 0.5, height: float = 0.02) -> Tuple[float, float]:
    """Smooth half-sine bump profile and derivative."""
    if t < t_start or t > t_start + duration:
        return 0.0, 0.0

    phase = (t - t_start) / duration
    omega = np.pi / duration
    z = height * np.sin(np.pi * phase)
    z_dot = height * omega * np.cos(np.pi * phase)
    return float(z), float(z_dot)


def command_profile(t: float) -> ScenarioCommand:
    """
    Deterministic beginner-friendly scenario.
    - 0-3s: acceleration
    - 3-7s: acceleration + steering sweep
    - 7-9s: straight hold
    - 9-11s: braking
    - 11s+: coast
    """
    if t < 3.0:
        drive_torque = 180.0 * (t / 3.0)
        brake_torque = 0.0
        steer_command = 0.0
    elif t < 7.0:
        drive_torque = 180.0
        brake_torque = 0.0
        steer_command = 15.0 * np.sin(1.0 * (t - 3.0))
    elif t < 9.0:
        drive_torque = 120.0
        brake_torque = 0.0
        steer_command = 0.0
    elif t < 11.0:
        drive_torque = 0.0
        brake_torque = 80.0
        steer_command = 0.0
    else:
        drive_torque = 0.0
        brake_torque = 0.0
        steer_command = 0.0

    # Small left/right steering split for simple inside/outside behavior.
    if steer_command >= 0.0:
        steer_left = 1.1 * steer_command
        steer_right = 0.9 * steer_command
    else:
        steer_left = 0.9 * steer_command
        steer_right = 1.1 * steer_command

    z_road_front, z_road_dot_front = road_bump(t)
    return ScenarioCommand(
        drive_torque=float(drive_torque),
        brake_torque=float(brake_torque),
        steer_torque_left=float(steer_left),
        steer_torque_right=float(steer_right),
        z_road_front=float(z_road_front),
        z_road_dot_front=float(z_road_dot_front),
    )


def build_corner_inputs(cmd: ScenarioCommand) -> Dict[str, Dict[str, float]]:
    """Convert scenario command to per-corner model inputs."""
    return {
        "FL": {
            "T_steer": cmd.steer_torque_left,
            "T_brk": cmd.brake_torque,
            "T_Drv": cmd.drive_torque,
            "T_susp": 0.0,
            "z_road": cmd.z_road_front,
            "z_road_dot": cmd.z_road_dot_front,
        },
        "FR": {
            "T_steer": cmd.steer_torque_right,
            "T_brk": cmd.brake_torque,
            "T_Drv": cmd.drive_torque,
            "T_susp": 0.0,
            "z_road": cmd.z_road_front,
            "z_road_dot": cmd.z_road_dot_front,
        },
        "RL": {
            "T_steer": 0.0,
            "T_brk": cmd.brake_torque,
            "T_Drv": cmd.drive_torque,
            "T_susp": 0.0,
            "z_road": 0.0,
            "z_road_dot": 0.0,
        },
        "RR": {
            "T_steer": 0.0,
            "T_brk": cmd.brake_torque,
            "T_Drv": cmd.drive_torque,
            "T_susp": 0.0,
            "z_road": 0.0,
            "z_road_dot": 0.0,
        },
    }


def _as_float(raw: str, default: float = 0.0) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _discover_cbnu_dir(data_root: Path, explicit_dir: Optional[str]) -> Path:
    if explicit_dir:
        path = Path(explicit_dir)
        if not path.exists() or not path.is_dir():
            raise ValueError(f"Invalid --cbnu-dir: {path}")
        return path

    if not data_root.exists():
        raise ValueError(f"Data root does not exist: {data_root}")

    candidates = sorted(p for p in data_root.iterdir() if p.is_dir() and any(p.glob("*.txt")))
    if not candidates:
        raise ValueError(f"No TXT dataset directory found under: {data_root}")
    if len(candidates) == 1:
        return candidates[0]

    for candidate in candidates:
        lowered = candidate.name.lower()
        if "cbnu" in lowered or "chung" in lowered:
            return candidate
    return candidates[0]


def list_cbnu_files(cbnu_dir: Path) -> List[Path]:
    return sorted(cbnu_dir.glob("*.txt"))


def _resolve_cbnu_file(cbnu_dir: Path, selector: Optional[str]) -> Path:
    files = list_cbnu_files(cbnu_dir)
    if not files:
        raise ValueError(f"No TXT files found in {cbnu_dir}")

    if selector is None or selector.strip() == "":
        raise ValueError("Missing --cbnu-file. Use --list-cbnu-files first.")

    selector = selector.strip()

    if selector.isdigit():
        idx = int(selector)
        if idx < 1 or idx > len(files):
            raise ValueError(f"--cbnu-file index out of range: {idx} (1..{len(files)})")
        return files[idx - 1]

    selector_path = Path(selector)
    if selector_path.exists() and selector_path.is_file():
        return selector_path

    for p in files:
        if p.name.lower() == selector.lower():
            return p

    matches = [p for p in files if selector.lower() in p.name.lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        preview = ", ".join(m.name for m in matches[:5])
        raise ValueError(f"Ambiguous --cbnu-file '{selector}'. Matches: {preview}")

    raise ValueError(f"Dataset not found: {selector}")


def _read_text_lines_with_fallback(path: Path) -> List[str]:
    for encoding in ("cp949", "utf-8-sig", "utf-8"):
        try:
            return path.read_text(encoding=encoding).splitlines()
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="cp949", errors="replace").splitlines()


def _find_header_line(lines: Sequence[str]) -> int:
    for idx, line in enumerate(lines):
        if "[Time]" in line and "\t" in line:
            return idx
    raise ValueError("Could not find tabular header line containing [Time].")


def _parse_log_timestamp(raw: str) -> Optional[datetime]:
    raw = raw.strip()
    for fmt in ("%y%m%d_%H%M%S.%f", "%y%m%d_%H%M%S"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def _monotonic_time_or_fallback(time_tokens: Sequence[str], fallback_dt: float = 0.01) -> np.ndarray:
    parsed = [_parse_log_timestamp(token) for token in time_tokens]
    if not parsed or parsed[0] is None:
        return np.arange(len(time_tokens), dtype=float) * fallback_dt

    t0 = parsed[0]
    out = np.zeros(len(time_tokens), dtype=float)
    for i, dt_obj in enumerate(parsed):
        if dt_obj is None:
            out[i] = out[i - 1] + fallback_dt if i > 0 else 0.0
        else:
            out[i] = (dt_obj - t0).total_seconds()

    for i in range(1, len(out)):
        if out[i] <= out[i - 1]:
            out[i] = out[i - 1] + fallback_dt
    return out


def load_cbnu_replay_data(file_path: Path, steer_bias_window_s: float = 1.0) -> CBNUReplayData:
    lines = _read_text_lines_with_fallback(file_path)
    header_idx = _find_header_line(lines)

    headers = [h.strip() for h in lines[header_idx].strip().split("\t")]
    rows: List[List[str]] = []
    for line in lines[header_idx + 1:]:
        if not line.strip():
            continue
        cols = [c.strip() for c in line.split("\t")]
        if len(cols) >= len(headers):
            rows.append(cols[:len(headers)])
    if len(rows) < 2:
        raise ValueError(f"Not enough data rows in {file_path}")

    col_map = {name: idx for idx, name in enumerate(headers)}

    def require_col(substring: str) -> int:
        for name, idx in col_map.items():
            if substring in name:
                return idx
        raise ValueError(f"Required column not found ({substring}) in {file_path}")

    idx_time = require_col("[Time]")
    idx_speed = require_col("Km/h")
    idx_steer = {
        "FL": require_col("Steer_Enc_FL"),
        "FR": require_col("Steer_Enc_FR"),
        "RL": require_col("Steer_Enc_RL"),
        "RR": require_col("Steer_Enc_RR"),
    }

    time_tokens = [r[idx_time] for r in rows]
    time_s = _monotonic_time_or_fallback(time_tokens, fallback_dt=0.01)

    dt_s = np.diff(time_s, prepend=time_s[0])
    valid_dt = dt_s[1:][dt_s[1:] > 0.0]
    default_dt = float(np.median(valid_dt)) if valid_dt.size > 0 else 0.01
    dt_s[0] = default_dt
    dt_s[dt_s <= 0.0] = default_dt
    dt_s = np.clip(dt_s, 1e-4, 0.2)

    speed_kph = np.array([_as_float(r[idx_speed], 0.0) for r in rows], dtype=float)
    speed_kph = np.maximum(speed_kph, 0.0)

    steer_deg_raw = {
        label: np.array([_as_float(r[idx], 0.0) for r in rows], dtype=float)
        for label, idx in idx_steer.items()
    }

    window_n = max(1, int(steer_bias_window_s / default_dt))
    steer_bias_deg = {
        label: float(np.median(values[:window_n]))
        for label, values in steer_deg_raw.items()
    }
    steer_deg = {
        label: values - steer_bias_deg[label]
        for label, values in steer_deg_raw.items()
    }
    steer_rate_deg_s = {
        label: np.gradient(values, time_s)
        for label, values in steer_deg.items()
    }

    return CBNUReplayData(
        source_file=file_path,
        time_s=time_s,
        dt_s=dt_s,
        speed_kph=speed_kph,
        steer_deg=steer_deg,
        steer_rate_deg_s=steer_rate_deg_s,
        steer_bias_deg=steer_bias_deg,
    )


def _infer_driven_wheels(source_name: str, drivetrain_mode: str) -> Tuple[str, ...]:
    if drivetrain_mode == "2wd-front":
        return ("FL", "FR")
    if drivetrain_mode == "4wd":
        return CORNER_LABELS

    lower = source_name.lower()
    if "2wd" in lower:
        return ("FL", "FR")
    return CORNER_LABELS


def _collect_log_row(
    t: float,
    dt: float,
    vehicle: VehicleBody,
    extra: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    speed = float(np.hypot(vehicle.state.velocity_x, vehicle.state.velocity_y))
    corner_states = {label: vehicle.corners[label].state for label in CORNER_LABELS}
    row = {
        "time_s": t,
        "dt_s": dt,
        "x_m": float(vehicle.state.x),
        "y_m": float(vehicle.state.y),
        "speed_mps": speed,
        "speed_kph": speed * 3.6,
        "yaw_deg": float(np.rad2deg(vehicle.state.yaw)),
        "yaw_rate_deg_s": float(np.rad2deg(vehicle.state.yaw_rate)),
        "roll_deg": float(np.rad2deg(vehicle.state.roll)),
        "pitch_deg": float(np.rad2deg(vehicle.state.pitch)),
        "fl_steer_deg": float(np.rad2deg(corner_states["FL"].steering_angle)),
        "fr_steer_deg": float(np.rad2deg(corner_states["FR"].steering_angle)),
        "fl_fx_n": float(corner_states["FL"].F_x_tire),
        "fr_fx_n": float(corner_states["FR"].F_x_tire),
        "fl_fy_n": float(corner_states["FL"].F_y_tire),
        "fr_fy_n": float(corner_states["FR"].F_y_tire),
    }
    if extra:
        row.update(extra)
    return row


def run_simulation_scenario(duration: float, dt: float) -> List[Dict[str, float]]:
    """Run vehicle model and return step logs for built-in scenario."""
    vehicle = VehicleBody()
    vehicle.reset()

    n_steps = int(np.floor(duration / dt))
    logs: List[Dict[str, float]] = []

    for step in range(n_steps):
        t = step * dt
        cmd = command_profile(t)
        corner_inputs = build_corner_inputs(cmd)
        vehicle.update(dt, corner_inputs, direction=1)

        logs.append(
            _collect_log_row(
                t=t,
                dt=dt,
                vehicle=vehicle,
                extra={
                    "ref_speed_kph": np.nan,
                    "ref_steer_fl_deg": np.nan,
                    "ref_steer_fr_deg": np.nan,
                    "cmd_drive_torque_nm": float(cmd.drive_torque),
                    "cmd_brake_torque_nm": float(cmd.brake_torque),
                },
            )
        )

    return logs


def run_simulation_cbnu(
    replay_data: CBNUReplayData,
    duration_limit_s: Optional[float],
    drivetrain_mode: str,
    ctrl_cfg: Optional[ReplayControllerConfig] = None,
) -> List[Dict[str, float]]:
    """Run vehicle model by tracking CBNU references (speed + steering)."""
    vehicle = VehicleBody()
    vehicle.reset()

    ctrl = ctrl_cfg or ReplayControllerConfig()
    driven_wheels = _infer_driven_wheels(replay_data.source_file.name, drivetrain_mode)
    speed_error_integral = 0.0

    logs: List[Dict[str, float]] = []
    n = len(replay_data.time_s)
    for i in range(n):
        t = float(replay_data.time_s[i])
        if duration_limit_s is not None and t > duration_limit_s:
            break

        dt = float(replay_data.dt_s[i])
        ref_speed_kph = float(replay_data.speed_kph[i])
        ref_speed_mps = max(0.0, ref_speed_kph / 3.6)

        current_speed_mps = float(np.hypot(vehicle.state.velocity_x, vehicle.state.velocity_y))
        speed_error = ref_speed_mps - current_speed_mps
        speed_error_integral += speed_error * dt
        speed_error_integral = float(
            np.clip(speed_error_integral, -ctrl.speed_integral_limit, ctrl.speed_integral_limit)
        )

        speed_cmd = ctrl.speed_kp * speed_error + ctrl.speed_ki * speed_error_integral
        if speed_cmd >= 0.0:
            drive_torque = float(np.clip(speed_cmd, 0.0, ctrl.max_drive_torque))
            brake_torque = 0.0
        else:
            drive_torque = 0.0
            brake_torque = float(np.clip(-speed_cmd * ctrl.brake_gain, 0.0, ctrl.max_brake_torque))

        corner_inputs: Dict[str, Dict[str, float]] = {}
        steer_cmd_log: Dict[str, float] = {}
        for label in CORNER_LABELS:
            target_angle_rad = float(np.deg2rad(replay_data.steer_deg[label][i]))
            target_rate_rad = float(np.deg2rad(replay_data.steer_rate_deg_s[label][i]))

            corner = vehicle.corners[label]
            angle_error = target_angle_rad - float(corner.state.steering_angle)
            current_rate = float(corner.steering.state.steering_rate)
            desired_rate = target_rate_rad + ctrl.steer_kp * angle_error
            desired_accel = ctrl.steer_kd * (desired_rate - current_rate)

            # Inverse steering dynamics:
            # J*ddelta + B*delta_dot = T_str*gear - T_align
            steering_model = corner.steering
            J = float(steering_model.params.J_cq)
            B = float(steering_model.params.B_cq)
            gear = float(steering_model.params.gear_ratio)
            T_align = float(steering_model.state.self_aligning_torque)
            steer_torque = (J * desired_accel + B * current_rate + T_align) / gear
            steer_torque = float(np.clip(steer_torque, -ctrl.max_steer_torque, ctrl.max_steer_torque))

            corner_inputs[label] = {
                "T_steer": steer_torque,
                "T_brk": brake_torque,
                "T_Drv": drive_torque if label in driven_wheels else 0.0,
                "T_susp": 0.0,
                "z_road": 0.0,
                "z_road_dot": 0.0,
            }
            steer_cmd_log[label] = steer_torque

        vehicle.update(dt, corner_inputs, direction=1)

        logs.append(
            _collect_log_row(
                t=t,
                dt=dt,
                vehicle=vehicle,
                extra={
                    "ref_speed_kph": ref_speed_kph,
                    "ref_steer_fl_deg": float(replay_data.steer_deg["FL"][i]),
                    "ref_steer_fr_deg": float(replay_data.steer_deg["FR"][i]),
                    "cmd_drive_torque_nm": drive_torque,
                    "cmd_brake_torque_nm": brake_torque,
                    "cmd_steer_fl_nm": steer_cmd_log["FL"],
                    "cmd_steer_fr_nm": steer_cmd_log["FR"],
                    "cmd_steer_rl_nm": steer_cmd_log["RL"],
                    "cmd_steer_rr_nm": steer_cmd_log["RR"],
                },
            )
        )

    return logs


def save_csv(logs: List[Dict[str, float]], output_file: Path) -> None:
    if not logs:
        return
    with output_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(logs[0].keys()))
        writer.writeheader()
        writer.writerows(logs)


def save_trajectory_plot(logs: List[Dict[str, float]], output_file: Path) -> None:
    if not logs:
        return
    x = np.array([row["x_m"] for row in logs])
    y = np.array([row["y_m"] for row in logs])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y, linewidth=2.0, color="tab:blue")
    ax.set_title("Quick Start Trajectory")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)


def save_state_plot(logs: List[Dict[str, float]], output_file: Path) -> None:
    if not logs:
        return

    t = np.array([row["time_s"] for row in logs], dtype=float)
    speed = np.array([row["speed_kph"] for row in logs], dtype=float)
    yaw_rate = np.array([row["yaw_rate_deg_s"] for row in logs], dtype=float)
    roll = np.array([row["roll_deg"] for row in logs], dtype=float)
    pitch = np.array([row["pitch_deg"] for row in logs], dtype=float)
    has_ref_speed = "ref_speed_kph" in logs[0] and np.isfinite(logs[-1].get("ref_speed_kph", np.nan))

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    axes = axes.ravel()

    axes[0].plot(t, speed, color="tab:green", label="Sim")
    if has_ref_speed:
        ref_speed = np.array([row["ref_speed_kph"] for row in logs], dtype=float)
        axes[0].plot(t, ref_speed, color="tab:gray", linestyle="--", linewidth=1.2, label="Ref")
        axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_ylabel("Speed [km/h]")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, yaw_rate, color="tab:orange")
    axes[1].set_ylabel("Yaw Rate [deg/s]")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, roll, color="tab:red")
    axes[2].set_ylabel("Roll [deg]")
    axes[2].set_xlabel("Time [s]")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(t, pitch, color="tab:purple")
    axes[3].set_ylabel("Pitch [deg]")
    axes[3].set_xlabel("Time [s]")
    axes[3].grid(True, alpha=0.3)

    fig.suptitle("Quick Start State Signals")
    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)


def save_input_tracking_plot(logs: List[Dict[str, float]], output_file: Path) -> None:
    if not logs:
        return

    required_keys = ("ref_speed_kph", "ref_steer_fl_deg", "ref_steer_fr_deg")
    if not all(key in logs[0] for key in required_keys):
        return

    t = np.array([row["time_s"] for row in logs], dtype=float)
    ref_speed = np.array([row["ref_speed_kph"] for row in logs], dtype=float)
    sim_speed = np.array([row["speed_kph"] for row in logs], dtype=float)
    ref_fl = np.array([row["ref_steer_fl_deg"] for row in logs], dtype=float)
    ref_fr = np.array([row["ref_steer_fr_deg"] for row in logs], dtype=float)
    sim_fl = np.array([row["fl_steer_deg"] for row in logs], dtype=float)
    sim_fr = np.array([row["fr_steer_deg"] for row in logs], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(t, ref_speed, "k--", linewidth=1.2, label="Ref Speed")
    axes[0].plot(t, sim_speed, "tab:green", linewidth=1.4, label="Sim Speed")
    axes[0].set_ylabel("Speed [km/h]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].plot(t, ref_fl, "tab:blue", linestyle="--", linewidth=1.2, label="Ref FL")
    axes[1].plot(t, sim_fl, "tab:blue", linewidth=1.4, label="Sim FL")
    axes[1].plot(t, ref_fr, "tab:orange", linestyle="--", linewidth=1.2, label="Ref FR")
    axes[1].plot(t, sim_fr, "tab:orange", linewidth=1.4, label="Sim FR")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Steer [deg]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right", fontsize=8, ncol=2)

    fig.suptitle("Replay Tracking (Reference vs Simulation)")
    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)


def save_dashboard_view(logs: List[Dict[str, float]], output_file: Path) -> None:
    if not logs:
        return

    time = np.array([row["time_s"] for row in logs])
    x_log = np.array([row["x_m"] for row in logs])
    y_log = np.array([row["y_m"] for row in logs])
    yaw_log = np.deg2rad(np.array([row["yaw_deg"] for row in logs]))
    steer_fl = np.array([row["fl_steer_deg"] for row in logs])
    steer_fr = np.array([row["fr_steer_deg"] for row in logs])
    speed = np.array([row["speed_mps"] for row in logs])
    yaw_rate = np.deg2rad(np.array([row["yaw_rate_deg_s"] for row in logs]))

    # Same style as the dashboard-like animation layout in vehicle_visualizer.
    fig = plt.figure(figsize=(15, 10))
    view_ax = plt.subplot2grid((4, 3), (0, 0), rowspan=4, colspan=2)
    zoom_ax = plt.subplot2grid((4, 3), (0, 2))
    steer_ax = plt.subplot2grid((4, 3), (1, 2), rowspan=2)
    speed_ax = plt.subplot2grid((4, 3), (3, 2))

    wheelbase = 2.8
    track = 1.6
    vehicle_length = wheelbase * 1.2
    vehicle_width = track * 0.9
    wheel_width = 0.3
    wheel_length = 0.5

    min_x, max_x = float(np.min(x_log)), float(np.max(x_log))
    min_y, max_y = float(np.min(y_log)), float(np.max(y_log))
    span = max(max_x - min_x, max_y - min_y, 10.0)
    pad = 0.15 * span
    view_ax.set_aspect("equal")
    view_ax.set_xlim(min_x - pad, max_x + pad)
    view_ax.set_ylim(min_y - pad, max_y + pad)
    view_ax.set_xlabel("X [m]", fontsize=12)
    view_ax.set_ylabel("Y [m]", fontsize=12)
    view_ax.set_title("Vehicle Animation (Global View)", fontsize=14, fontweight="bold")
    view_ax.grid(True, alpha=0.3)

    zoom_span = max(vehicle_length * 1.2, vehicle_width * 2.0, 3.0)
    zoom_ax.set_aspect("equal")
    zoom_ax.set_xlim(-zoom_span, zoom_span)
    zoom_ax.set_ylim(-zoom_span, zoom_span)
    zoom_ax.set_xlabel("X [m]", fontsize=9)
    zoom_ax.set_ylabel("Y [m]", fontsize=9)
    zoom_ax.set_title("Zoom View", fontsize=10, fontweight="bold")
    zoom_ax.grid(True, alpha=0.3)

    max_steer = max(1.0, float(np.max(np.abs(np.concatenate([steer_fl, steer_fr])))))
    steer_ax.set_title("Steering Angles", fontsize=10, fontweight="bold")
    steer_ax.set_xlim(float(time[0]), float(time[-1]))
    steer_ax.set_ylim(-1.1 * max_steer, 1.1 * max_steer)
    steer_ax.set_xlabel("Time [s]", fontsize=9)
    steer_ax.set_ylabel("Steer [deg]", fontsize=9)
    steer_ax.grid(True, alpha=0.3)

    speed_ax.set_title("Vehicle Speed", fontsize=10)
    speed_ax.set_xlim(-1.2, 1.2)
    speed_ax.set_ylim(-1.2, 1.2)
    speed_ax.set_aspect("equal")
    speed_ax.axis("off")

    def affine_transform(x_list, y_list, angle, translation=(0.0, 0.0)):
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        x_new = [x * cos_a - y * sin_a + translation[0] for x, y in zip(x_list, y_list)]
        y_new = [x * sin_a + y * cos_a + translation[1] for x, y in zip(x_list, y_list)]
        return x_new, y_new

    def draw_vehicle(ax, center_x, center_y, yaw, steer_map, draw_marker=True):
        vl, vw = vehicle_length, vehicle_width
        body_x = [-0.4 * vl, -0.4 * vl, 0.4 * vl, 0.4 * vl, -0.4 * vl]
        body_y = [-0.5 * vw, 0.5 * vw, 0.5 * vw, -0.5 * vw, -0.5 * vw]
        windshield_x = [0.2 * vl, 0.2 * vl, 0.35 * vl, 0.35 * vl, 0.2 * vl]
        windshield_y = [-0.35 * vw, 0.35 * vw, 0.35 * vw, -0.35 * vw, -0.35 * vw]

        body_x_g, body_y_g = affine_transform(body_x, body_y, yaw, (center_x, center_y))
        ax.plot(body_x_g, body_y_g, "black", linewidth=1.5, zorder=3)
        ax.fill(body_x_g, body_y_g, color="lightblue", alpha=0.7, zorder=2)

        ws_x_g, ws_y_g = affine_transform(windshield_x, windshield_y, yaw, (center_x, center_y))
        ax.fill(ws_x_g, ws_y_g, color="lightcyan", alpha=0.6, zorder=3)

        wl, ww = wheel_length, wheel_width
        wheel_shape_x = [-0.5 * wl, -0.5 * wl, 0.5 * wl, 0.5 * wl, -0.5 * wl]
        wheel_shape_y = [0.0, 0.5 * ww, 0.5 * ww, -0.5 * ww, -0.5 * ww, 0.0]
        wheel_centers = {
            "FL": (0.35 * vl, 0.35 * vw, steer_map["FL"]),
            "FR": (0.35 * vl, -0.35 * vw, steer_map["FR"]),
            "RL": (-0.35 * vl, 0.35 * vw, steer_map["RL"]),
            "RR": (-0.35 * vl, -0.35 * vw, steer_map["RR"]),
        }
        for _, (center_local_x, center_local_y, delta) in wheel_centers.items():
            wx_local, wy_local = affine_transform(wheel_shape_x, wheel_shape_y, delta, (center_local_x, center_local_y))
            wx_global, wy_global = affine_transform(wx_local, wy_local, yaw, (center_x, center_y))
            ax.fill(wx_global, wy_global, color="black", zorder=4)

        if draw_marker:
            ax.scatter([center_x], [center_y], c="red", s=150, zorder=5)

    idx = len(logs) - 1
    current_x, current_y, current_yaw = x_log[idx], y_log[idx], yaw_log[idx]
    steer_map = {
        "FL": np.deg2rad(steer_fl[idx]),
        "FR": np.deg2rad(steer_fr[idx]),
        "RL": 0.0,
        "RR": 0.0,
    }

    view_ax.plot(x_log, y_log, color="purple", linewidth=1.5, alpha=0.7)
    draw_vehicle(view_ax, current_x, current_y, current_yaw, steer_map, draw_marker=True)
    view_ax.arrow(
        current_x,
        current_y,
        0.6 * vehicle_length * np.cos(current_yaw),
        0.6 * vehicle_length * np.sin(current_yaw),
        head_width=0.3,
        head_length=0.4,
        fc="tab:orange",
        ec="tab:orange",
        zorder=6,
    )
    view_ax.text(
        0.5,
        0.02,
        f"t={time[idx]:.2f}s  |  v={speed[idx]:.2f}m/s  |  yaw_rate={yaw_rate[idx]:.3f}rad/s",
        ha="center",
        transform=view_ax.transAxes,
        fontsize=11,
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    draw_vehicle(zoom_ax, 0.0, 0.0, current_yaw, steer_map, draw_marker=False)
    zoom_ax.arrow(
        0.0,
        0.0,
        0.45 * vehicle_length * np.cos(current_yaw),
        0.45 * vehicle_length * np.sin(current_yaw),
        head_width=0.2,
        head_length=0.3,
        fc="tab:orange",
        ec="tab:orange",
        zorder=6,
    )

    steer_ax.plot(time, steer_fl, linewidth=1.2, color="tab:blue", label="FL")
    steer_ax.plot(time, steer_fr, linewidth=1.2, color="tab:orange", label="FR")
    steer_ax.plot([time[idx]], [steer_fl[idx]], "o", color="tab:blue", markersize=3)
    steer_ax.plot([time[idx]], [steer_fr[idx]], "o", color="tab:orange", markersize=3)
    steer_ax.legend(loc="upper right", fontsize=8)

    max_speed = 25.0
    pie_rate = 0.75
    pie_start = 225
    normalized_speed = min(float(speed[idx]), max_speed)
    speed_ax.pie(
        [
            normalized_speed * pie_rate,
            (max_speed - normalized_speed) * pie_rate,
            max_speed * (1.0 - pie_rate),
        ],
        startangle=pie_start,
        counterclock=False,
        colors=["black", "lightgray", "white"],
        wedgeprops={"linewidth": 0, "edgecolor": "white", "width": 0.4},
    )
    speed_ax.text(0, -1, f"{speed[idx]:.1f} m/s", size=12, ha="center", va="center", fontfamily="monospace")

    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)


def save_trajectory_gif(logs: List[Dict[str, float]], output_file: Path, max_frames: int = 400) -> None:
    """Save dashboard-style animation GIF (global view + zoom + steer + speed)."""
    if len(logs) < 2:
        return

    time = np.array([row["time_s"] for row in logs], dtype=float)
    x_log = np.array([row["x_m"] for row in logs], dtype=float)
    y_log = np.array([row["y_m"] for row in logs], dtype=float)
    yaw_log = np.deg2rad(np.array([row["yaw_deg"] for row in logs], dtype=float))
    steer_fl_deg = np.array([row["fl_steer_deg"] for row in logs], dtype=float)
    steer_fr_deg = np.array([row["fr_steer_deg"] for row in logs], dtype=float)
    speed_log = np.array([row["speed_mps"] for row in logs], dtype=float)
    yaw_rate_log = np.deg2rad(np.array([row["yaw_rate_deg_s"] for row in logs], dtype=float))

    n = len(logs)
    stride = max(1, n // max_frames)
    frame_indices = np.arange(0, n, stride, dtype=int)
    if frame_indices[-1] != n - 1:
        frame_indices = np.append(frame_indices, n - 1)

    wheelbase = 2.8
    track = 1.6
    vehicle_length = wheelbase * 1.2
    vehicle_width = track * 0.9
    wheel_length = 0.5
    wheel_width = 0.3

    fig = plt.figure(figsize=(15, 10))
    view_ax = plt.subplot2grid((4, 3), (0, 0), rowspan=4, colspan=2)
    zoom_ax = plt.subplot2grid((4, 3), (0, 2))
    steer_ax = plt.subplot2grid((4, 3), (1, 2), rowspan=2)
    speed_ax = plt.subplot2grid((4, 3), (3, 2))

    min_x, max_x = float(np.min(x_log)), float(np.max(x_log))
    min_y, max_y = float(np.min(y_log)), float(np.max(y_log))
    span = max(max_x - min_x, max_y - min_y, 10.0)
    pad = 0.15 * span
    view_ax.set_aspect("equal")
    view_ax.set_xlim(min_x - pad, max_x + pad)
    view_ax.set_ylim(min_y - pad, max_y + pad)
    view_ax.set_xlabel("X [m]")
    view_ax.set_ylabel("Y [m]")
    view_ax.set_title("Vehicle Animation (Global View)", fontsize=14, fontweight="bold")
    view_ax.grid(True, alpha=0.3)

    zoom_span = max(vehicle_length * 1.2, vehicle_width * 2.0, 3.0)
    zoom_ax.set_aspect("equal")
    zoom_ax.set_xlim(-zoom_span, zoom_span)
    zoom_ax.set_ylim(-zoom_span, zoom_span)
    zoom_ax.set_xlabel("X [m]", fontsize=9)
    zoom_ax.set_ylabel("Y [m]", fontsize=9)
    zoom_ax.set_title("Zoom View", fontsize=10, fontweight="bold")
    zoom_ax.grid(True, alpha=0.3)

    max_steer = max(1.0, float(np.max(np.abs(np.concatenate([steer_fl_deg, steer_fr_deg])))))
    steer_ax.set_title("Steering Angles", fontsize=10, fontweight="bold")
    steer_ax.set_xlim(float(time[0]), float(time[-1]))
    steer_ax.set_ylim(-1.1 * max_steer, 1.1 * max_steer)
    steer_ax.set_xlabel("Time [s]", fontsize=9)
    steer_ax.set_ylabel("Steer [deg]", fontsize=9)
    steer_ax.grid(True, alpha=0.3)

    speed_ax.set_title("Vehicle Speed", fontsize=10)
    speed_ax.set_xlim(-1.2, 1.2)
    speed_ax.set_ylim(-1.2, 1.2)
    speed_ax.set_aspect("equal")
    speed_ax.axis("off")

    def affine_transform(x_list, y_list, angle, translation=(0.0, 0.0)):
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        out_x = [x * cos_a - y * sin_a + translation[0] for x, y in zip(x_list, y_list)]
        out_y = [x * sin_a + y * cos_a + translation[1] for x, y in zip(x_list, y_list)]
        return out_x, out_y

    def vehicle_polygons(center_x: float, center_y: float, yaw_value: float, steer_map: Dict[str, float]):
        body_x = [-0.4 * vehicle_length, -0.4 * vehicle_length, 0.4 * vehicle_length, 0.4 * vehicle_length]
        body_y = [-0.5 * vehicle_width, 0.5 * vehicle_width, 0.5 * vehicle_width, -0.5 * vehicle_width]
        body_xy = np.array(affine_transform(body_x, body_y, yaw_value, (center_x, center_y))).T

        ws_x = [0.2 * vehicle_length, 0.2 * vehicle_length, 0.35 * vehicle_length, 0.35 * vehicle_length]
        ws_y = [-0.35 * vehicle_width, 0.35 * vehicle_width, 0.35 * vehicle_width, -0.35 * vehicle_width]
        ws_xy = np.array(affine_transform(ws_x, ws_y, yaw_value, (center_x, center_y))).T

        wheel_shape_x = [-0.5 * wheel_length, -0.5 * wheel_length, 0.5 * wheel_length, 0.5 * wheel_length]
        wheel_shape_y = [-0.5 * wheel_width, 0.5 * wheel_width, 0.5 * wheel_width, -0.5 * wheel_width]
        wheel_centers = {
            "FL": (0.35 * vehicle_length, 0.35 * vehicle_width, steer_map["FL"]),
            "FR": (0.35 * vehicle_length, -0.35 * vehicle_width, steer_map["FR"]),
            "RL": (-0.35 * vehicle_length, 0.35 * vehicle_width, steer_map["RL"]),
            "RR": (-0.35 * vehicle_length, -0.35 * vehicle_width, steer_map["RR"]),
        }

        wheels_xy: Dict[str, np.ndarray] = {}
        for label, (wx, wy, delta) in wheel_centers.items():
            local_x, local_y = affine_transform(wheel_shape_x, wheel_shape_y, delta, (wx, wy))
            global_x, global_y = affine_transform(local_x, local_y, yaw_value, (center_x, center_y))
            wheels_xy[label] = np.array([global_x, global_y]).T

        return body_xy, ws_xy, wheels_xy

    traj_line, = view_ax.plot([], [], color="purple", linewidth=1.5, alpha=0.7)
    view_body = Polygon(np.zeros((4, 2)), closed=True, facecolor="lightblue", edgecolor="black", linewidth=1.5)
    view_ws = Polygon(np.zeros((4, 2)), closed=True, facecolor="lightcyan", edgecolor="black", linewidth=1.0)
    view_ax.add_patch(view_body)
    view_ax.add_patch(view_ws)
    view_wheels = {
        label: Polygon(np.zeros((4, 2)), closed=True, facecolor="black", edgecolor="black", linewidth=1.0)
        for label in CORNER_LABELS
    }
    for patch in view_wheels.values():
        view_ax.add_patch(patch)
    view_cg, = view_ax.plot([], [], "o", color="red", markersize=8)
    view_arrow = None

    zoom_body = Polygon(np.zeros((4, 2)), closed=True, facecolor="lightblue", edgecolor="black", linewidth=1.5)
    zoom_ws = Polygon(np.zeros((4, 2)), closed=True, facecolor="lightcyan", edgecolor="black", linewidth=1.0)
    zoom_ax.add_patch(zoom_body)
    zoom_ax.add_patch(zoom_ws)
    zoom_wheels = {
        label: Polygon(np.zeros((4, 2)), closed=True, facecolor="black", edgecolor="black", linewidth=1.0)
        for label in CORNER_LABELS
    }
    for patch in zoom_wheels.values():
        zoom_ax.add_patch(patch)
    zoom_arrow = None

    fl_line, = steer_ax.plot([], [], linewidth=1.2, color="tab:blue", label="FL")
    fr_line, = steer_ax.plot([], [], linewidth=1.2, color="tab:orange", label="FR")
    fl_dot, = steer_ax.plot([], [], "o", color="tab:blue", markersize=3)
    fr_dot, = steer_ax.plot([], [], "o", color="tab:orange", markersize=3)
    steer_ax.legend(loc="upper right", fontsize=8)

    status_text = view_ax.text(
        0.5,
        0.02,
        "",
        ha="center",
        transform=view_ax.transAxes,
        fontsize=11,
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    def update(frame_idx: int):
        nonlocal view_arrow, zoom_arrow

        idx = int(frame_indices[frame_idx])
        cx, cy, yaw_value = float(x_log[idx]), float(y_log[idx]), float(yaw_log[idx])
        steer_map = {
            "FL": float(np.deg2rad(steer_fl_deg[idx])),
            "FR": float(np.deg2rad(steer_fr_deg[idx])),
            "RL": 0.0,
            "RR": 0.0,
        }

        traj_line.set_data(x_log[: idx + 1], y_log[: idx + 1])
        body_xy, ws_xy, wheels_xy = vehicle_polygons(cx, cy, yaw_value, steer_map)
        view_body.set_xy(body_xy)
        view_ws.set_xy(ws_xy)
        for label in CORNER_LABELS:
            view_wheels[label].set_xy(wheels_xy[label])
        view_cg.set_data([cx], [cy])

        if view_arrow is not None:
            view_arrow.remove()
        view_arrow = view_ax.arrow(
            cx,
            cy,
            0.6 * vehicle_length * np.cos(yaw_value),
            0.6 * vehicle_length * np.sin(yaw_value),
            head_width=0.3,
            head_length=0.4,
            fc="tab:orange",
            ec="tab:orange",
            zorder=6,
        )

        status_text.set_text(
            f"t={time[idx]:.2f}s  |  v={speed_log[idx]:.2f}m/s  |  yaw_rate={yaw_rate_log[idx]:.3f}rad/s"
        )

        body_xy, ws_xy, wheels_xy = vehicle_polygons(0.0, 0.0, yaw_value, steer_map)
        zoom_body.set_xy(body_xy)
        zoom_ws.set_xy(ws_xy)
        for label in CORNER_LABELS:
            zoom_wheels[label].set_xy(wheels_xy[label])

        if zoom_arrow is not None:
            zoom_arrow.remove()
        zoom_arrow = zoom_ax.arrow(
            0.0,
            0.0,
            0.45 * vehicle_length * np.cos(yaw_value),
            0.45 * vehicle_length * np.sin(yaw_value),
            head_width=0.2,
            head_length=0.3,
            fc="tab:orange",
            ec="tab:orange",
            zorder=6,
        )

        fl_line.set_data(time[: idx + 1], steer_fl_deg[: idx + 1])
        fr_line.set_data(time[: idx + 1], steer_fr_deg[: idx + 1])
        fl_dot.set_data([time[idx]], [steer_fl_deg[idx]])
        fr_dot.set_data([time[idx]], [steer_fr_deg[idx]])

        speed_ax.clear()
        speed_ax.set_title("Vehicle Speed", fontsize=10)
        speed_ax.set_xlim(-1.2, 1.2)
        speed_ax.set_ylim(-1.2, 1.2)
        speed_ax.set_aspect("equal")
        speed_ax.axis("off")
        max_speed = 25.0
        pie_rate = 0.75
        current_speed = min(float(speed_log[idx]), max_speed)
        speed_ax.pie(
            [current_speed * pie_rate, (max_speed - current_speed) * pie_rate, max_speed * (1.0 - pie_rate)],
            startangle=225,
            counterclock=False,
            colors=["black", "lightgray", "white"],
            wedgeprops={"linewidth": 0, "edgecolor": "white", "width": 0.4},
        )
        speed_ax.text(0, -1, f"{speed_log[idx]:.1f} m/s", size=12, ha="center", va="center", fontfamily="monospace")

        return ()

    median_dt = _median_dt_from_logs(logs, fallback_dt=DEFAULT_DT_SCENARIO_S)
    effective_dt = max(1e-3, median_dt * stride)
    fps = int(np.clip(round(1.0 / effective_dt), 8, 20))
    interval_ms = int(round(1000.0 / fps))

    anim = mpl_animation.FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=interval_ms,
        blit=False,
        repeat=False,
    )
    writer = mpl_animation.PillowWriter(fps=fps)
    anim.save(str(output_file), writer=writer, dpi=80)
    plt.close(fig)


def _median_dt_from_logs(logs: List[Dict[str, float]], fallback_dt: float) -> float:
    if len(logs) < 2:
        return fallback_dt
    t = np.array([row["time_s"] for row in logs], dtype=float)
    dt = np.diff(t)
    dt = dt[dt > 0.0]
    return float(np.median(dt)) if dt.size > 0 else fallback_dt


def save_summary(
    logs: List[Dict[str, float]],
    dt: float,
    output_file: Path,
    source_mode: str,
    source_detail: Optional[str] = None,
) -> None:
    if not logs:
        return

    speed = np.array([row["speed_kph"] for row in logs], dtype=float)
    roll = np.array([row["roll_deg"] for row in logs], dtype=float)
    pitch = np.array([row["pitch_deg"] for row in logs], dtype=float)

    lines = [
        "Modeling Quick Start Summary",
        "============================",
        f"input_mode: {source_mode}",
    ]
    if source_detail:
        lines.append(f"input_source: {source_detail}")
    lines.extend(
        [
            f"steps: {len(logs)}",
            f"dt_s(median): {dt}",
            f"total_time_s: {logs[-1]['time_s']:.3f}",
            f"final_x_m: {logs[-1]['x_m']:.3f}",
            f"final_y_m: {logs[-1]['y_m']:.3f}",
            f"final_speed_kph: {logs[-1]['speed_kph']:.3f}",
            f"max_speed_kph: {speed.max():.3f}",
            f"max_abs_roll_deg: {np.abs(roll).max():.3f}",
            f"max_abs_pitch_deg: {np.abs(pitch).max():.3f}",
        ]
    )

    if "ref_speed_kph" in logs[0]:
        ref_speed = np.array([row.get("ref_speed_kph", np.nan) for row in logs], dtype=float)
        mask = np.isfinite(ref_speed)
        if np.any(mask):
            mae_speed = np.mean(np.abs(speed[mask] - ref_speed[mask]))
            lines.append(f"tracking_mae_speed_kph: {mae_speed:.3f}")

    if "ref_steer_fl_deg" in logs[0]:
        ref_fl = np.array([row.get("ref_steer_fl_deg", np.nan) for row in logs], dtype=float)
        sim_fl = np.array([row.get("fl_steer_deg", np.nan) for row in logs], dtype=float)
        mask = np.isfinite(ref_fl) & np.isfinite(sim_fl)
        if np.any(mask):
            mae_fl = np.mean(np.abs(sim_fl[mask] - ref_fl[mask]))
            lines.append(f"tracking_mae_steer_fl_deg: {mae_fl:.3f}")

    if "ref_steer_fr_deg" in logs[0]:
        ref_fr = np.array([row.get("ref_steer_fr_deg", np.nan) for row in logs], dtype=float)
        sim_fr = np.array([row.get("fr_steer_deg", np.nan) for row in logs], dtype=float)
        mask = np.isfinite(ref_fr) & np.isfinite(sim_fr)
        if np.any(mask):
            mae_fr = np.mean(np.abs(sim_fr[mask] - ref_fr[mask]))
            lines.append(f"tracking_mae_steer_fr_deg: {mae_fr:.3f}")

    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_run_directory(output_root: Path) -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run modeling quick start scenario.")
    parser.add_argument(
        "--input-source",
        type=str,
        default="scenario",
        choices=("scenario", "cbnu"),
        help="Input source mode: built-in scenario or CBNU replay file.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Simulation duration limit in seconds. Default: 12s for scenario, full length for cbnu.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=DEFAULT_DT_SCENARIO_S,
        help="Simulation time step in seconds (scenario mode only).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Output root directory for logs and figures.",
    )
    parser.add_argument(
        "--cbnu-dir",
        type=str,
        default=None,
        help="Directory containing CBNU TXT files. If omitted, auto-detected under vehicle_sim/Data.",
    )
    parser.add_argument(
        "--list-cbnu-files",
        action="store_true",
        help="List detected CBNU TXT files and exit.",
    )
    parser.add_argument(
        "--cbnu-file",
        type=str,
        default=None,
        help="CBNU TXT file selector: index(1-based), filename, partial name, or full path.",
    )
    parser.add_argument(
        "--drivetrain",
        type=str,
        choices=("auto", "2wd-front", "4wd"),
        default="auto",
        help="Driven-wheel mapping for CBNU replay mode.",
    )
    return parser.parse_args()


def _print_cbnu_files(cbnu_dir: Path) -> None:
    files = list_cbnu_files(cbnu_dir)
    if not files:
        print(f"[QuickStart] no TXT files in: {cbnu_dir}")
        return
    print(f"[QuickStart] cbnu_dir: {cbnu_dir}")
    for idx, path in enumerate(files, start=1):
        print(f"{idx:2d}. {path.name}")


def main() -> None:
    args = parse_args()

    if args.list_cbnu_files:
        try:
            cbnu_dir = _discover_cbnu_dir(DEFAULT_DATA_ROOT, args.cbnu_dir)
            _print_cbnu_files(cbnu_dir)
        except ValueError as err:
            print(f"[QuickStart] error: {err}")
        return

    output_root = Path(args.output_root)
    run_dir = create_run_directory(output_root)

    if args.input_source == "scenario":
        duration_s = args.duration if args.duration is not None else DEFAULT_DURATION_SCENARIO_S
        logs = run_simulation_scenario(duration=duration_s, dt=args.dt)
        source_mode = "scenario"
        source_detail = "built-in profile"
        extra_files: List[str] = []
    else:
        try:
            cbnu_dir = _discover_cbnu_dir(DEFAULT_DATA_ROOT, args.cbnu_dir)
            selected_file = _resolve_cbnu_file(cbnu_dir, args.cbnu_file)
            replay_data = load_cbnu_replay_data(selected_file)
            logs = run_simulation_cbnu(
                replay_data=replay_data,
                duration_limit_s=args.duration,
                drivetrain_mode=args.drivetrain,
            )
        except ValueError as err:
            print(f"[QuickStart] error: {err}")
            print("[QuickStart] hint: use --list-cbnu-files and then pass --cbnu-file <index>")
            return

        source_mode = "cbnu"
        source_detail = str(selected_file)
        extra_files = ["input_tracking.png"]

    save_csv(logs, run_dir / "signals.csv")
    save_summary(
        logs,
        dt=_median_dt_from_logs(logs, fallback_dt=args.dt),
        output_file=run_dir / "summary.txt",
        source_mode=source_mode,
        source_detail=source_detail,
    )
    save_trajectory_plot(logs, run_dir / "trajectory.png")
    save_state_plot(logs, run_dir / "states.png")
    save_dashboard_view(logs, run_dir / "dashboard_view.png")
    try:
        save_trajectory_gif(logs, run_dir / "trajectory.gif")
        extra_files.append("trajectory.gif")
    except Exception as err:
        print(f"[QuickStart] warning: GIF export failed ({err})")
    if source_mode == "cbnu":
        save_input_tracking_plot(logs, run_dir / "input_tracking.png")

    print(f"[QuickStart] finished: {run_dir}")
    files = ["signals.csv", "summary.txt", "trajectory.png", "states.png", "dashboard_view.png", *extra_files]
    print(f"[QuickStart] files: {', '.join(files)}")
    print(f"[QuickStart] input_mode: {source_mode}")
    if source_mode == "cbnu":
        print(f"[QuickStart] source: {source_detail}")


if __name__ == "__main__":
    main()
