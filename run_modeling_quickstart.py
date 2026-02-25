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
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vehicle_sim.models.vehicle_body.vehicle_body import VehicleBody


CORNER_LABELS = ("FL", "FR", "RL", "RR")
DEFAULT_OUTPUT_ROOT = Path("outputs") / "modeling_quickstart"


@dataclass
class ScenarioCommand:
    """Per-step command values for the whole vehicle."""

    drive_torque: float
    brake_torque: float
    steer_torque_left: float
    steer_torque_right: float
    z_road_front: float
    z_road_dot_front: float


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


def run_simulation(duration: float, dt: float) -> List[Dict[str, float]]:
    """Run vehicle model and return step logs."""
    vehicle = VehicleBody()
    vehicle.reset()

    n_steps = int(np.floor(duration / dt))
    logs: List[Dict[str, float]] = []

    for step in range(n_steps):
        t = step * dt
        cmd = command_profile(t)
        corner_inputs = build_corner_inputs(cmd)
        vehicle.update(dt, corner_inputs, direction=1)

        speed = float(np.hypot(vehicle.state.velocity_x, vehicle.state.velocity_y))
        corner_states = {label: vehicle.corners[label].state for label in CORNER_LABELS}

        logs.append(
            {
                "time_s": t,
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
    t = np.array([row["time_s"] for row in logs])
    speed = np.array([row["speed_kph"] for row in logs])
    yaw_rate = np.array([row["yaw_rate_deg_s"] for row in logs])
    roll = np.array([row["roll_deg"] for row in logs])
    pitch = np.array([row["pitch_deg"] for row in logs])

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    axes = axes.ravel()

    axes[0].plot(t, speed, color="tab:green")
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


def save_summary(logs: List[Dict[str, float]], dt: float, output_file: Path) -> None:
    if not logs:
        return

    speed = np.array([row["speed_kph"] for row in logs])
    roll = np.array([row["roll_deg"] for row in logs])
    pitch = np.array([row["pitch_deg"] for row in logs])

    lines = [
        "Modeling Quick Start Summary",
        "============================",
        f"steps: {len(logs)}",
        f"dt_s: {dt}",
        f"total_time_s: {logs[-1]['time_s']:.3f}",
        f"final_x_m: {logs[-1]['x_m']:.3f}",
        f"final_y_m: {logs[-1]['y_m']:.3f}",
        f"final_speed_kph: {logs[-1]['speed_kph']:.3f}",
        f"max_speed_kph: {speed.max():.3f}",
        f"max_abs_roll_deg: {np.abs(roll).max():.3f}",
        f"max_abs_pitch_deg: {np.abs(pitch).max():.3f}",
    ]
    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_run_directory(output_root: Path) -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run modeling quick start scenario.")
    parser.add_argument("--duration", type=float, default=12.0, help="Simulation duration in seconds.")
    parser.add_argument("--dt", type=float, default=0.01, help="Simulation time step in seconds.")
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Output root directory for logs and figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    run_dir = create_run_directory(output_root)

    logs = run_simulation(duration=args.duration, dt=args.dt)

    save_csv(logs, run_dir / "signals.csv")
    save_summary(logs, dt=args.dt, output_file=run_dir / "summary.txt")
    save_trajectory_plot(logs, run_dir / "trajectory.png")
    save_state_plot(logs, run_dir / "states.png")
    save_dashboard_view(logs, run_dir / "dashboard_view.png")

    print(f"[QuickStart] finished: {run_dir}")
    print("[QuickStart] files: signals.csv, summary.txt, trajectory.png, states.png, dashboard_view.png")


if __name__ == "__main__":
    main()
