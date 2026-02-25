#!/usr/bin/env python3
"""
Yaw-rate tracking study with configurable controllers and parameter estimation.

Run:
  python SeohanModel/vehicle_sim/scenarios/yaw_rate_control_suite/run_yaw_rate_control_suite.py \
    --config SeohanModel/vehicle_sim/scenarios/yaw_rate_control_suite/config.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import yaml

# Ensure the project package root is on sys.path when running directly.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vehicle_sim.controllers.pid_controller import PIDController, PIDGains
from vehicle_sim.controllers.speed_controller_v2 import SpeedControllerV2
from vehicle_sim.controllers.steer_angle_ff import SteeringFeedforwardController
from vehicle_sim.controllers.yaw_moment_allocator import YawMomentAllocator
from vehicle_sim.controllers.yaw_moment_feedforward_controller import (
    YawMomentFeedforwardController,
)
from vehicle_sim.controllers.lateral_force_estimator import (
    LateralForceEstimator,
    LateralForceEstimatorOptions,
)
from vehicle_sim.controllers.slip_angle_estimator import (
    SlipAngleEstimator,
    SlipAngleEstimatorOptions,
)
from vehicle_sim.models.vehicle_body.vehicle_body import VehicleBody
from vehicle_sim.utils.config_loader import load_param

from yaw_rate_profiles import build_profile
from steer_torque_ff import SteeringFFParams, SteeringMotorTorqueFF, SteeringTorqueFFOptions
from steering_param_estimator import ScalarClamp, ScalarRLS


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def build_speed_controller(
    dt: float,
    vehicle: VehicleBody,
    config_path: Optional[str],
) -> SpeedControllerV2:
    vehicle_spec = load_param("vehicle_spec", config_path)
    wheel_radius = float(vehicle_spec.get("wheel", {}).get("R_eff", 0.3))
    return SpeedControllerV2(dt, mass=vehicle.params.m, wheel_radius=wheel_radius)


def load_pid_gains(section: str, gains_path: Path) -> PIDGains:
    cfg = load_param(section, gains_path)
    return PIDGains(
        kp=float(cfg.get("kp", 0.0)),
        ki=float(cfg.get("ki", 0.0)),
        kd=float(cfg.get("kd", 0.0)),
    )


def lock_speed(vehicle: VehicleBody, speed: float) -> None:
    vehicle.state.velocity_x = float(speed)
    for label in vehicle.wheel_labels:
        corner = vehicle.corners[label]
        r_wheel = float(corner.drive.params.R_wheel)
        if r_wheel > 0.0:
            omega = float(speed / r_wheel)
            corner.drive.state.wheel_speed = omega
            corner.state.omega_wheel = omega


def init_random_factors(
    rng: np.random.Generator,
    percent: float,
    per_wheel: bool,
    labels: list[str],
    randomize_b: bool,
    randomize_c: bool,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    if per_wheel:
        factor_b = {
            label: float(rng.uniform(1.0 - percent, 1.0 + percent)) if randomize_b else 1.0
            for label in labels
        }
        factor_c = {
            label: float(rng.uniform(1.0 - percent, 1.0 + percent)) if randomize_c else 1.0
            for label in labels
        }
    else:
        shared_b = float(rng.uniform(1.0 - percent, 1.0 + percent)) if randomize_b else 1.0
        shared_c = float(rng.uniform(1.0 - percent, 1.0 + percent)) if randomize_c else 1.0
        factor_b = {label: shared_b for label in labels}
        factor_c = {label: shared_c for label in labels}
    return factor_b, factor_c


def apply_random_factors(
    vehicle: VehicleBody,
    base_b: float,
    base_c_alpha: float,
    factor_b: Dict[str, float],
    factor_c: Dict[str, float],
) -> Dict[str, Tuple[float, float]]:
    actual_params: Dict[str, Tuple[float, float]] = {}
    for label in vehicle.wheel_labels:
        b_val = max(1e-6, base_b * float(factor_b[label]))
        c_val = max(1e-6, base_c_alpha * float(factor_c[label]))
        corner = vehicle.corners[label]
        corner.steering.params.B_cq = float(b_val)
        corner.lateral_tire.params.C_alpha = float(c_val)
        actual_params[label] = (float(b_val), float(c_val))
    return actual_params


def clamp_random_factors(
    factor_b: Dict[str, float],
    factor_c: Dict[str, float],
    min_factor: float,
    max_factor: float,
) -> None:
    for label in factor_b:
        factor_b[label] = float(np.clip(factor_b[label], min_factor, max_factor))
        factor_c[label] = float(np.clip(factor_c[label], min_factor, max_factor))


def update_gaussian_factors(
    rng: np.random.Generator,
    factor_b: Dict[str, float],
    factor_c: Dict[str, float],
    sigma_step: float,
    per_wheel: bool,
    randomize_b: bool,
    randomize_c: bool,
) -> None:
    if per_wheel:
        for label in factor_b:
            if randomize_b:
                factor_b[label] += float(rng.normal(0.0, sigma_step))
            if randomize_c:
                factor_c[label] += float(rng.normal(0.0, sigma_step))
    else:
        delta_b = float(rng.normal(0.0, sigma_step)) if randomize_b else 0.0
        delta_c = float(rng.normal(0.0, sigma_step)) if randomize_c else 0.0
        for label in factor_b:
            if randomize_b:
                factor_b[label] += delta_b
            if randomize_c:
                factor_c[label] += delta_c


def build_ff_params(
    vehicle: VehicleBody,
    j_map: Dict[str, float],
    b_map: Dict[str, float],
) -> Dict[str, SteeringFFParams]:
    params = {}
    for label in vehicle.wheel_labels:
        corner = vehicle.corners[label]
        sparams = corner.steering.params
        params[label] = SteeringFFParams(
            J_cq=float(j_map[label]),
            B_cq=float(b_map[label]),
            gear_ratio=float(sparams.gear_ratio),
            max_rate=float(sparams.max_rate),
            max_angle_pos=float(sparams.max_angle_pos),
            max_angle_neg=float(sparams.max_angle_neg),
        )
    return params


def compute_align_cmd(vehicle: VehicleBody, fy_wheel_cmd: Dict[str, float]) -> Dict[str, float]:
    t_align_cmd: Dict[str, float] = {}
    for label in vehicle.wheel_labels:
        corner = vehicle.corners[label]
        fy_wheel = float(fy_wheel_cmd.get(label, 0.0))
        fz = float(corner.state.F_z)
        if fz > 0.0:
            mu = float(corner.lateral_tire.params.mu)
            fy_wheel = float(np.clip(fy_wheel, -mu * abs(fz), mu * abs(fz)))
        trail = float(corner.lateral_tire.params.trail)
        t_align_cmd[label] = float(trail * fy_wheel)
    return t_align_cmd


def should_update_b_estimator(
    t: float,
    delta: float,
    delta_dot: float,
    delta_ddot: float,
    params,
    cfg: dict,
) -> bool:
    if t < float(cfg.get("start_time", 0.0)):
        return False
    dot_min_abs = float(cfg.get("dot_min_abs", cfg.get("ddot_min_abs", 0.0)))
    if abs(delta_dot) < dot_min_abs:
        return False
    angle_margin = float(cfg.get("angle_margin_rad", 0.0))
    rate_margin = float(cfg.get("rate_margin_rad_s", 0.0))
    lower = min(params.max_angle_neg, params.max_angle_pos)
    upper = max(params.max_angle_neg, params.max_angle_pos)
    if delta <= lower + angle_margin or delta >= upper - angle_margin:
        return False
    if abs(delta_dot) >= float(params.max_rate) - rate_margin:
        return False
    if not np.isfinite(delta) or not np.isfinite(delta_dot) or not np.isfinite(delta_ddot):
        return False
    return True


def should_update_calpha_estimator(
    t: float,
    alpha: float,
    fy: float,
    fz: float,
    mu: float,
    cfg: dict,
) -> bool:
    if t < float(cfg.get("start_time", 0.0)):
        return False
    alpha_min = float(cfg.get("alpha_min_abs", 0.0))
    if abs(alpha) < alpha_min:
        return False
    fz_min = float(cfg.get("fz_min", 0.0))
    if abs(fz) < fz_min:
        return False
    fy_max = abs(mu * fz)
    if fy_max <= 0.0:
        return False
    sat_margin = float(cfg.get("fy_saturation_margin", 0.0))
    if abs(fy) >= fy_max * (1.0 - sat_margin):
        return False
    if not np.isfinite(alpha) or not np.isfinite(fy) or not np.isfinite(fz):
        return False
    return True


def _resolve_plot_mode(output_cfg: dict) -> str:
    plot_mode = str(output_cfg.get("plot_mode", "debug")).strip().lower()
    if plot_mode not in {"debug", "paper"}:
        raise ValueError("output.plot_mode must be one of: ['debug', 'paper']")
    return plot_mode


def _resolve_paper_wheel(labels: list[str], output_cfg: dict) -> str:
    if not labels:
        raise ValueError("No wheel labels available for plotting")
    requested = output_cfg.get("paper_wheel", None)
    if requested is None:
        return labels[0]
    requested = str(requested)
    if requested not in labels:
        return labels[0]
    return requested


def _apply_plot_style(plt, output_cfg: dict) -> dict:
    style_cfg = output_cfg.get("style", {})
    if not isinstance(style_cfg, dict):
        style_cfg = {}

    font_size = float(style_cfg.get("font_size", 12.0))
    if font_size <= 0.0:
        raise ValueError("output.style.font_size must be positive")

    grid_alpha = float(style_cfg.get("grid_alpha", 0.3))
    if not (0.0 <= grid_alpha <= 1.0):
        raise ValueError("output.style.grid_alpha must be in [0, 1]")

    line_width = float(style_cfg.get("line_width", 2.0))
    if line_width <= 0.0:
        raise ValueError("output.style.line_width must be positive")

    input_color = style_cfg.get("input_color", "0.35")
    input_alpha = float(style_cfg.get("input_alpha", 0.6))
    if not (0.0 <= input_alpha <= 1.0):
        raise ValueError("output.style.input_alpha must be in [0, 1]")

    output_alpha = float(style_cfg.get("output_alpha", 0.9))
    if not (0.0 <= output_alpha <= 1.0):
        raise ValueError("output.style.output_alpha must be in [0, 1]")

    mode_tint = float(style_cfg.get("mode_tint", 0.35))
    if not (0.0 <= mode_tint <= 1.0):
        raise ValueError("output.style.mode_tint must be in [0, 1]")

    default_mode_colors = {
        "ff": "#1f77b4",
        "ff_fb": "#2ca02c",
        "ff_ls": "#ff7f0e",
        "ff_fb_ls": "#d62728",
    }
    mode_colors_cfg = style_cfg.get("mode_colors", {})
    mode_colors = dict(default_mode_colors)
    if isinstance(mode_colors_cfg, dict):
        for key, value in mode_colors_cfg.items():
            mode_colors[str(key)] = value

    plt.rcParams.update(
        {
            "font.size": font_size,
            "axes.titlesize": float(style_cfg.get("title_size", font_size * 1.15)),
            "axes.labelsize": float(style_cfg.get("label_size", font_size)),
            "xtick.labelsize": float(style_cfg.get("tick_size", font_size * 0.9)),
            "ytick.labelsize": float(style_cfg.get("tick_size", font_size * 0.9)),
            "legend.fontsize": float(style_cfg.get("legend_size", font_size * 0.9)),
            "lines.linewidth": line_width,
            "axes.grid": True,
            "grid.alpha": grid_alpha,
            "grid.linestyle": str(style_cfg.get("grid_linestyle", "-")),
            "figure.titlesize": float(style_cfg.get("suptitle_size", font_size * 1.25)),
        }
    )

    return {
        "grid_alpha": grid_alpha,
        "input_color": input_color,
        "input_alpha": input_alpha,
        "output_alpha": output_alpha,
        "paper_output_color": style_cfg.get("paper_output_color", None),
        "true_color": style_cfg.get("true_color", "0.2"),
        "estimate_color": style_cfg.get("estimate_color", None),
        "beta_color": style_cfg.get("beta_color", None),
        "error_color": style_cfg.get("error_color", "tab:red"),
        "mode_tint": mode_tint,
        "mode_colors": mode_colors,
    }


def _plot_paper(
    plt,
    history: dict,
    t_axis: np.ndarray,
    labels: list[str],
    control_mode: str,
    estimator_enabled: bool,
    estimator_start_time: float,
    output_cfg: dict,
    style: dict,
) -> None:
    grid_alpha = float(style.get("grid_alpha", 0.3))
    nm_unit = r"$\mathrm{N\cdot m}$"
    nms_unit = r"$\mathrm{N\cdot m\cdot s/rad}$"
    b_label = r"$B_\delta$"
    c_label = r"$C_\alpha$"
    input_color = style.get("input_color", "0.35")
    input_alpha = float(style.get("input_alpha", 0.6))
    output_alpha = float(style.get("output_alpha", 0.9))
    mode_tint = float(style.get("mode_tint", 0.35))
    mode_colors = style.get("mode_colors", {})
    output_color = mode_colors.get(control_mode, "tab:blue")
    error_color = str(style.get("error_color", "tab:red"))
    true_color = str(style.get("true_color", "0.2"))
    estimate_color = style.get("estimate_color", None)
    estimate_color = output_color if estimate_color is None else str(estimate_color)
    beta_color = style.get("beta_color", None)
    beta_color = output_color if beta_color is None else str(beta_color)
    from matplotlib import colors as mcolors

    wheel_base_colors = {
        "FL": "tab:blue",
        "FR": "tab:orange",
        "RR": "tab:green",
        "RL": "tab:red",
    }

    def _blend_color(base_color: str) -> tuple[float, float, float]:
        base_rgb = np.array(mcolors.to_rgb(base_color))
        tint_rgb = np.array(mcolors.to_rgb(output_color))
        mixed = (1.0 - mode_tint) * base_rgb + mode_tint * tint_rgb
        return tuple(mixed.tolist())
    input_color = style.get("input_color", "0.35")
    input_alpha = float(style.get("input_alpha", 0.6))
    output_alpha = float(style.get("output_alpha", 0.9))
    mode_colors = style.get("mode_colors", {})
    output_color = mode_colors.get(control_mode, "tab:blue")
    error_color = str(style.get("error_color", "tab:red"))
    paper_wheel = _resolve_paper_wheel(labels, output_cfg)
    wheel_color = str(wheel_base_colors.get(paper_wheel, output_color))
    paper_output_color = style.get("paper_output_color", None)
    paper_output_color = wheel_color if paper_output_color is None else str(paper_output_color)
    estimate_color = style.get("estimate_color", None)
    estimate_color = paper_output_color if estimate_color is None else str(estimate_color)
    beta_color = style.get("beta_color", None)
    beta_color = paper_output_color if beta_color is None else str(beta_color)
    true_color = str(style.get("true_color", "0.2"))

    fig, axes = plt.subplots(4, 2, figsize=(12, 13), sharex=True)

    ax = axes[0, 0]
    ax.plot(
        t_axis,
        history["yaw_rate_cmd"],
        label=r"$r_\mathrm{ref}$",
        linestyle="--",
        color=input_color,
        alpha=input_alpha,
    )
    ax.plot(
        t_axis,
        history["yaw_rate"],
        label=r"$r$",
        color=paper_output_color,
        alpha=output_alpha,
    )
    ax.set_title("Yaw rate tracking")
    ax.set_ylabel("[rad/s]")
    ax.grid(True, alpha=grid_alpha)
    ax.legend(loc="upper right")

    ax = axes[0, 1]
    ax.plot(
        t_axis,
        history["yaw_error"],
        label=r"$r_\mathrm{ref}-r$",
        color=error_color,
    )
    ax.set_title("Yaw rate error")
    ax.set_ylabel("[rad/s]")
    ax.grid(True, alpha=grid_alpha)
    ax.legend(loc="upper right")

    ax = axes[1, 0]
    ax.plot(
        t_axis,
        history["Mz_cmd"],
        label=r"$M_{z,\mathrm{ref}}$",
        linestyle="--",
        color=input_color,
        alpha=input_alpha,
    )
    ax.plot(
        t_axis,
        history["Mz_actual"],
        label=r"$M_z$",
        color=paper_output_color,
        alpha=output_alpha,
    )
    ax.set_title("Yaw moment tracking")
    ax.set_ylabel(f"[{nm_unit}]")
    ax.grid(True, alpha=grid_alpha)
    ax.legend(loc="upper right")

    ax = axes[1, 1]
    ax.plot(
        t_axis,
        history["Mz_error"],
        label=r"$M_{z,\mathrm{ref}}-M_z$",
        color=error_color,
    )
    ax.set_title("Yaw moment error")
    ax.set_ylabel(f"[{nm_unit}]")
    ax.grid(True, alpha=grid_alpha)
    ax.legend(loc="upper right")

    ax = axes[2, 0]
    ax.plot(
        t_axis,
        history["fy_cmd"][paper_wheel],
        label=r"$F_{y,\mathrm{ref}}$",
        linestyle="--",
        color=input_color,
        alpha=input_alpha,
    )
    ax.plot(
        t_axis,
        history["fy_actual"][paper_wheel],
        label=r"$F_y$",
        color=paper_output_color,
        alpha=output_alpha,
    )
    ax.set_title(f"Lateral force tracking ({paper_wheel})")
    ax.set_ylabel("[N]")
    ax.grid(True, alpha=grid_alpha)
    ax.legend(loc="upper right")

    ax = axes[2, 1]
    ax.plot(
        t_axis,
        history["fy_error"][paper_wheel],
        label=r"$F_{y,\mathrm{ref}}-F_y$",
        color=error_color,
    )
    ax.set_title(f"Lateral force error ({paper_wheel})")
    ax.set_ylabel("[N]")
    ax.grid(True, alpha=grid_alpha)
    ax.legend(loc="upper right")

    ax = axes[3, 0]
    ax.plot(
        t_axis,
        history["delta_cmd"][paper_wheel],
        label=r"$\delta_\mathrm{ref}$",
        linestyle="--",
        color=input_color,
        alpha=input_alpha,
    )
    ax.plot(
        t_axis,
        history["delta_actual"][paper_wheel],
        label=r"$\delta$",
        color=paper_output_color,
        alpha=output_alpha,
    )
    ax.set_title(f"Steering angle tracking ({paper_wheel})")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("[rad]")
    ax.grid(True, alpha=grid_alpha)
    ax.legend(loc="upper right")

    ax = axes[3, 1]
    ax.plot(
        t_axis,
        history["delta_error"][paper_wheel],
        label=r"$\delta_\mathrm{ref}-\delta$",
        color=error_color,
    )
    ax.set_title(f"Steering angle error ({paper_wheel})")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("[rad]")
    ax.grid(True, alpha=grid_alpha)
    ax.legend(loc="upper right")

    fig.tight_layout()

    # Parameter estimation results (no error subplots).
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes = np.atleast_1d(axes)
    b_hat_name = "estimate" if estimator_enabled else "used"
    c_hat_name = "estimate" if estimator_enabled else "used"

    b_true = np.array(history["B_true"][paper_wheel], dtype=float)
    b_hat = np.array(history["B_hat"][paper_wheel], dtype=float)
    ax = axes[0]
    ax.plot(
        t_axis,
        b_hat,
        label=f"{b_label} ({b_hat_name})",
        color=estimate_color,
        alpha=output_alpha,
    )
    ax.plot(
        t_axis,
        b_true,
        label=f"{b_label} (true)",
        linestyle="--",
        color=true_color,
        alpha=output_alpha,
    )
    ax.set_title(f"{b_label} estimate ({paper_wheel})")
    ax.set_ylabel(f"{b_label} [{nms_unit}]")
    ax.grid(True, alpha=grid_alpha)
    ax.legend(loc="upper right")

    c_true = np.array(history["C_alpha_true"][paper_wheel], dtype=float)
    c_hat = np.array(history["C_alpha_hat"][paper_wheel], dtype=float)
    ax = axes[1]
    ax.plot(
        t_axis,
        c_hat,
        label=f"{c_label} ({c_hat_name})",
        color=estimate_color,
        alpha=output_alpha,
    )
    ax.plot(
        t_axis,
        c_true,
        label=f"{c_label} (true)",
        linestyle="--",
        color=true_color,
        alpha=output_alpha,
    )
    ax.set_title(f"{c_label} estimate ({paper_wheel})")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(f"{c_label} [N/rad]")
    ax.grid(True, alpha=grid_alpha)
    ax.legend(loc="upper right")

    # Show C_alpha in scientific notation around 1e5.
    from matplotlib.ticker import ScalarFormatter

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((5, 5))
    ax.yaxis.set_major_formatter(formatter)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(5, 5))

    fig.tight_layout()

    # One-figure B estimator diagnostics (main + small panels).
    if estimator_enabled and "delta_ddot" in history and "align_true" in history:
        fig = plt.figure(figsize=(12, 6))
        gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1.0], hspace=0.35, wspace=0.3)
        ax_main = fig.add_subplot(gs[0, :])
        ax_ddot = fig.add_subplot(gs[1, 0], sharex=ax_main)
        ax_align = fig.add_subplot(gs[1, 1], sharex=ax_main)

        b_hat = np.array(history["B_hat"][paper_wheel], dtype=float)
        b_true = np.array(history["B_true"][paper_wheel], dtype=float)
        ax_main.plot(
            t_axis,
            b_hat,
            label=f"{b_label} ({b_hat_name})",
            color=estimate_color,
            alpha=output_alpha,
        )
        ax_main.plot(
            t_axis,
            b_true,
            label=f"{b_label} (true)",
            linestyle="--",
            color=true_color,
            alpha=output_alpha,
        )
        if b_hat.size and b_true.size:
            ax_main.fill_between(
                t_axis,
                b_true,
                b_hat,
                color=error_color,
                alpha=0.15,
                label="error",
            )
        ax_main.set_title(
            f"Scenario 3: Dynamic Mismatch with Ramp-up ({b_label} estimate + diagnostics, {paper_wheel})"
        )
        ax_main.set_ylabel(f"{b_label} [{nms_unit}]")
        ax_main.grid(True, alpha=grid_alpha)
        ax_main.legend(loc="upper right")

        delta_ddot = np.array(history["delta_ddot"][paper_wheel], dtype=float)
        ax_ddot.plot(
            t_axis,
            delta_ddot,
            color=error_color,
            alpha=0.8,
            label=r"$\ddot{\delta}$",
        )
        ax_ddot.set_title(r"Steer accel $\ddot{\delta}$ (raw)")
        ax_ddot.set_xlabel("Time [s]")
        ax_ddot.set_ylabel(r"[rad/s$^2$]")
        ax_ddot.grid(True, alpha=grid_alpha)
        ax_ddot.legend(loc="upper right")

        align_true = np.array(history["align_true"][paper_wheel], dtype=float)
        align_est = np.array(history["align_est"][paper_wheel], dtype=float)
        ax_align.plot(
            t_axis,
            align_true,
            label=r"$T_\mathrm{align}$ (true)",
            color=true_color,
            alpha=output_alpha,
        )
        if np.any(np.isfinite(align_est)):
            ax_align.plot(
                t_axis,
                align_est,
                label=r"$T_\mathrm{align}$ (est)",
                color=estimate_color,
                alpha=output_alpha,
            )
        ax_align.set_title("Aligning torque")
        ax_align.set_xlabel("Time [s]")
        ax_align.set_ylabel(f"[{nm_unit}]")
        ax_align.grid(True, alpha=grid_alpha)
        ax_align.legend(loc="upper right")

        fig.tight_layout()

    if "vy" in history:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4), sharex=True)
        ax.plot(
            t_axis,
            history["vy"],
            label=r"$v_y$",
            color=paper_output_color,
            alpha=output_alpha,
        )
        ax.set_title("Lateral velocity")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("[m/s]")
        ax.grid(True, alpha=grid_alpha)
        ax.legend(loc="upper right")
        fig.tight_layout()

    # (Disabled) sideslip plot: keep paper output focused on tracking + estimator results.

    if "fy_total_est" in history:
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        ax = axes[0]
        ax.plot(
            t_axis,
            history["fy_total_actual"],
            label=r"$F_{y,\mathrm{total}}$",
            color="0.2",
        )
        ax.plot(
            t_axis,
            history["fy_total_est"],
            label=r"$\hat{F}_{y,\mathrm{total}}$",
            linestyle="--",
            color=estimate_color,
            alpha=output_alpha,
        )
        ax.set_title("Total lateral force estimate")
        ax.set_ylabel("[N]")
        ax.grid(True, alpha=grid_alpha)
        ax.legend(loc="upper right")

        ax = axes[1]
        ax.plot(
            t_axis,
            history["fy_total_error"],
            label=r"$\hat{F}_{y,\mathrm{total}}-F_{y,\mathrm{total}}$",
            color=error_color,
        )
        ax.set_title("Total lateral force error")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("[N]")
        ax.grid(True, alpha=grid_alpha)
        ax.legend(loc="upper right")
        fig.tight_layout()


def _plot_debug(
    plt,
    history: dict,
    t_axis: np.ndarray,
    labels: list[str],
    control_mode: str,
    estimator_enabled: bool,
    style: dict,
) -> None:
    grid_alpha = float(style.get("grid_alpha", 0.3))
    nm_unit = r"$\mathrm{N\cdot m}$"
    nms_unit = r"$\mathrm{N\cdot m\cdot s/rad}$"
    b_label = r"$B_\delta$"
    c_label = r"$C_\alpha$"
    input_color = style.get("input_color", "0.35")
    input_alpha = float(style.get("input_alpha", 0.6))
    output_alpha = float(style.get("output_alpha", 0.9))
    mode_tint = float(style.get("mode_tint", 0.35))
    mode_colors = style.get("mode_colors", {})
    output_color = mode_colors.get(control_mode, "tab:blue")
    error_color = str(style.get("error_color", "tab:red"))
    true_color = str(style.get("true_color", "0.2"))
    estimate_color = style.get("estimate_color", None)
    estimate_color = output_color if estimate_color is None else str(estimate_color)
    beta_color = style.get("beta_color", None)
    beta_color = output_color if beta_color is None else str(beta_color)
    from matplotlib import colors as mcolors

    wheel_base_colors = {
        "FL": "tab:blue",
        "FR": "tab:orange",
        "RR": "tab:green",
        "RL": "tab:red",
    }

    def _blend_color(base_color: str) -> tuple[float, float, float]:
        base_rgb = np.array(mcolors.to_rgb(base_color))
        tint_rgb = np.array(mcolors.to_rgb(output_color))
        mixed = (1.0 - mode_tint) * base_rgb + mode_tint * tint_rgb
        return tuple(mixed.tolist())

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    ax = axes[0, 0]
    ax.plot(
        t_axis,
        history["yaw_rate_cmd"],
        label=r"$r_\mathrm{ref}$",
        color=input_color,
        alpha=input_alpha,
    )
    ax.plot(
        t_axis,
        history["yaw_rate"],
        label=r"$r$",
        color=output_color,
        alpha=output_alpha,
    )
    ax.set_title("Yaw rate tracking")
    ax.set_ylabel("[rad/s]")
    ax.grid(True, alpha=grid_alpha)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(
        t_axis,
        history["yaw_error"],
        label=r"$r_\mathrm{ref}-r$",
        color=error_color,
    )
    ax.set_title("Yaw rate error")
    ax.set_ylabel("[rad/s]")
    ax.grid(True, alpha=grid_alpha)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(
        t_axis,
        history["Mz_cmd"],
        label=r"$M_{z,\mathrm{ref}}$",
        color=input_color,
        alpha=input_alpha,
    )
    ax.plot(
        t_axis,
        history["Mz_actual"],
        label=r"$M_z$",
        color=output_color,
        alpha=output_alpha,
    )
    ax.set_title("Yaw moment")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(f"[{nm_unit}]")
    ax.grid(True, alpha=grid_alpha)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(
        t_axis,
        history["Mz_error"],
        label=r"$M_{z,\mathrm{ref}}-M_z$",
        color=error_color,
    )
    ax.set_title("Yaw moment error")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(f"[{nm_unit}]")
    ax.grid(True, alpha=grid_alpha)
    ax.legend()

    fig.suptitle("Tracking Summary")
    fig.tight_layout()

    if "vy" in history:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4), sharex=True)
        ax.plot(
            t_axis,
            history["vy"],
            label=r"$v_y$",
            color=output_color,
            alpha=output_alpha,
        )
        ax.set_title("Lateral velocity")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("[m/s]")
        ax.grid(True, alpha=grid_alpha)
        ax.legend()
        fig.tight_layout()

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax = axes[0]
    ax.plot(
        t_axis,
        history["speed_cmd"],
        label=r"$V_{x,\mathrm{ref}}$",
        color=input_color,
        alpha=input_alpha,
    )
    ax.plot(
        t_axis,
        history["speed"],
        label=r"$V_x$",
        color=output_color,
        alpha=output_alpha,
    )
    ax.set_title("Speed tracking")
    ax.set_ylabel("[m/s]")
    ax.grid(True, alpha=grid_alpha)
    ax.legend()

    ax = axes[1]
    speed_error = np.array(history["speed_cmd"], dtype=float) - np.array(history["speed"], dtype=float)
    ax.plot(
        t_axis,
        speed_error,
        label=r"$V_{x,\mathrm{ref}}-V_x$",
        color=error_color,
    )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("[m/s]")
    ax.grid(True, alpha=grid_alpha)
    ax.legend()
    fig.tight_layout()

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    mz_sum = np.array(history["mz_from_fy"], dtype=float) + np.array(history["mz_from_fx"], dtype=float)
    ax = axes[0]
    ax.plot(
        t_axis,
        history["Mz_actual"],
        label=r"$M_z$",
        color=output_color,
        alpha=output_alpha,
    )
    ax.plot(
        t_axis,
        mz_sum,
        label=r"$M_z^{(F_y+F_x)}$",
        linestyle="--",
        color="0.2",
    )
    ax.set_title("Yaw moment reconstruction")
    ax.set_ylabel(f"[{nm_unit}]")
    ax.grid(True, alpha=grid_alpha)
    ax.legend()

    ax = axes[1]
    ax.plot(t_axis, history["mz_from_fy"], label=r"$M_z^{(F_y)}$", color="0.2")
    ax.plot(t_axis, history["mz_from_fx"], label=r"$M_z^{(F_x)}$", color="0.5")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(f"[{nm_unit}]")
    ax.grid(True, alpha=grid_alpha)
    ax.legend()
    fig.tight_layout()

    if "fy_total_actual" in history and "fy_total_from_r" in history:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4), sharex=True)
        ax.plot(
            t_axis,
            history["fy_total_actual"],
            label=r"$\sum F_y$",
            color=output_color,
            alpha=output_alpha,
        )
        ax.plot(
            t_axis,
            history["fy_total_from_r"],
            label=r"$m r v_x$",
            linestyle="--",
            color=input_color,
            alpha=input_alpha,
        )
        ax.set_title("Total lateral force balance")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("[N]")
        ax.grid(True, alpha=grid_alpha)
        ax.legend(loc="upper right")
        fig.tight_layout()

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    ax = axes[0]
    ax.plot(
        t_axis,
        history["fy_total_cmd"],
        label=r"$F_{y,\mathrm{ref}}$",
        color=input_color,
        alpha=input_alpha,
    )
    ax.plot(
        t_axis,
        history["fy_bias"],
        label=r"$F_{y,\mathrm{bias}}$",
        linestyle="--",
        color="0.2",
    )
    ax.set_title("Total lateral force command and bias")
    ax.set_ylabel("[N]")
    ax.grid(True, alpha=grid_alpha)
    ax.legend()

    ax = axes[1]
    for label in labels:
        ax.plot(
            t_axis,
            history["fy_cmd"][label],
            label=rf"$F_{{y,\mathrm{{ref}},\mathrm{{{label}}}}}$",
            linewidth=1.0,
            color=input_color,
            alpha=input_alpha,
        )
    ax.plot(
        t_axis,
        history["fy_bias"],
        label=r"$F_{y,\mathrm{bias}}$",
        color="black",
        linestyle="--",
        linewidth=1.2,
    )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("[N]")
    ax.grid(True, alpha=grid_alpha)
    ax.legend(ncol=2)
    fig.tight_layout()

    fig, axes = plt.subplots(len(labels), 2, figsize=(12, 2.6 * len(labels)), sharex=True)
    axes = np.atleast_2d(axes)
    for idx, label in enumerate(labels):
        ax_cmd = axes[idx, 0]
        ax_err = axes[idx, 1]
        output_wheel_color = _blend_color(wheel_base_colors.get(label, output_color))
        ax_cmd.plot(
            t_axis,
            history["delta_cmd"][label],
            linestyle="--",
            label=r"$\delta_\mathrm{ref}$",
            color=input_color,
            alpha=input_alpha,
        )
        ax_cmd.plot(
            t_axis,
            history["delta_actual"][label],
            label=r"$\delta$",
            color=output_wheel_color,
            alpha=output_alpha,
        )
        ax_cmd.set_ylabel(f"{label} [rad]")
        ax_cmd.grid(True, alpha=grid_alpha)
        if idx == 0:
            ax_cmd.set_title("Steering angle tracking")
        if idx == len(labels) - 1:
            ax_cmd.set_xlabel("Time [s]")
        ax_cmd.legend(loc="upper right")

        ax_err.plot(
            t_axis,
            history["delta_error"][label],
            label=r"$\delta_\mathrm{ref}-\delta$",
            color=error_color,
        )
        ax_err.grid(True, alpha=grid_alpha)
        if idx == 0:
            ax_err.set_title("Steering angle error")
        if idx == len(labels) - 1:
            ax_err.set_xlabel("Time [s]")
        ax_err.legend(loc="upper right")

    fig.tight_layout()

    fig, axes = plt.subplots(len(labels), 2, figsize=(12, 2.6 * len(labels)), sharex=True)
    axes = np.atleast_2d(axes)
    for idx, label in enumerate(labels):
        ax_fy = axes[idx, 0]
        ax_fx = axes[idx, 1]
        output_wheel_color = _blend_color(wheel_base_colors.get(label, output_color))
        ax_fy.plot(
            t_axis,
            history["fy_actual"][label],
            label=rf"$F_{{y,\mathrm{{tire}},\mathrm{{{label}}}}}$",
            color=output_wheel_color,
            alpha=output_alpha,
        )
        ax_fy.plot(
            t_axis,
            history["fy_body"][label],
            label=rf"$F_{{y,\mathrm{{body}},\mathrm{{{label}}}}}$",
            linestyle="--",
            color=output_wheel_color,
            alpha=output_alpha,
        )
        ax_fy.set_ylabel(f"{label} [N]")
        ax_fy.grid(True, alpha=grid_alpha)
        if idx == 0:
            ax_fy.set_title("Lateral force (tire vs body)")
        if idx == len(labels) - 1:
            ax_fy.set_xlabel("Time [s]")
        ax_fy.legend(loc="upper right")

        ax_fx.plot(
            t_axis,
            history["fx_tire"][label],
            label=rf"$F_{{x,\mathrm{{tire}},\mathrm{{{label}}}}}$",
            color=output_wheel_color,
            alpha=output_alpha,
        )
        ax_fx.plot(
            t_axis,
            history["fx_body"][label],
            label=rf"$F_{{x,\mathrm{{body}},\mathrm{{{label}}}}}$",
            linestyle="--",
            color=output_wheel_color,
            alpha=output_alpha,
        )
        ax_fx.grid(True, alpha=grid_alpha)
        if idx == 0:
            ax_fx.set_title("Longitudinal force (tire vs body)")
        if idx == len(labels) - 1:
            ax_fx.set_xlabel("Time [s]")
        ax_fx.legend(loc="upper right")
    fig.tight_layout()

    fig, axes = plt.subplots(len(labels) + 1, 1, figsize=(10, 2.4 * (len(labels) + 1)), sharex=True)
    axes = np.atleast_1d(axes)
    ax = axes[0]
    for label in labels:
        ax.plot(
            t_axis,
            history["beta_ref"][label],
            label=rf"$\beta_\mathrm{{ref}},\mathrm{{{label}}}$",
            linestyle="--",
            color=input_color,
            alpha=input_alpha,
        )
    if "beta_ref_body" in history:
        ax.plot(
            t_axis,
            history["beta_ref_body"],
            label=r"$\beta_\mathrm{ref}$",
            linestyle=":",
            color=input_color,
            alpha=min(1.0, input_alpha + 0.15),
        )
    ax.plot(
        t_axis,
        history["beta"],
        label=r"$\beta$",
        color=beta_color,
        alpha=output_alpha,
    )
    ax.set_title("Sideslip angles")
    ax.set_ylabel("[rad]")
    ax.grid(True, alpha=grid_alpha)
    ax.legend(loc="upper right")
    for idx, label in enumerate(labels, start=1):
        ax = axes[idx]
        output_wheel_color = _blend_color(wheel_base_colors.get(label, output_color))
        ax.plot(
            t_axis,
            history["alpha_cmd"][label],
            label=rf"$\alpha_\mathrm{{ref}},\mathrm{{{label}}}$",
            linestyle="--",
            color=input_color,
            alpha=input_alpha,
        )
        ax.plot(
            t_axis,
            history["alpha"][label],
            label=rf"$\alpha,\mathrm{{{label}}}$",
            color=output_wheel_color,
            alpha=output_alpha,
        )
        ax.set_ylabel(f"{label} [rad]")
        ax.grid(True, alpha=grid_alpha)
        if idx == len(labels):
            ax.set_xlabel("Time [s]")
        ax.legend(loc="upper right")
    fig.tight_layout()

    fig, axes = plt.subplots(len(labels), 2, figsize=(12, 2.6 * len(labels)), sharex=True)
    axes = np.atleast_2d(axes)
    for idx, label in enumerate(labels):
        ax_cmd = axes[idx, 0]
        ax_err = axes[idx, 1]
        output_wheel_color = _blend_color(wheel_base_colors.get(label, output_color))
        ax_cmd.plot(
            t_axis,
            history["fy_cmd"][label],
            linestyle="--",
            label=r"$F_{y,\mathrm{ref}}$",
            color=input_color,
            alpha=input_alpha,
        )
        ax_cmd.plot(
            t_axis,
            history["fy_actual"][label],
            label=r"$F_y$",
            color=output_wheel_color,
            alpha=output_alpha,
        )
        ax_cmd.set_ylabel(f"{label} [N]")
        ax_cmd.grid(True, alpha=grid_alpha)
        if idx == 0:
            ax_cmd.set_title("Lateral force tracking (wheel)")
        if idx == len(labels) - 1:
            ax_cmd.set_xlabel("Time [s]")
        ax_cmd.legend(loc="upper right")

        ax_err.plot(
            t_axis,
            history["fy_error"][label],
            label=r"$F_{y,\mathrm{ref}}-F_y$",
            color=error_color,
        )
        ax_err.grid(True, alpha=grid_alpha)
        if idx == 0:
            ax_err.set_title("Lateral force error (wheel)")
        if idx == len(labels) - 1:
            ax_err.set_xlabel("Time [s]")
        ax_err.legend(loc="upper right")

    fig.tight_layout()

    fig, axes = plt.subplots(len(labels), 2, figsize=(12, 2.6 * len(labels)), sharex=True)
    axes = np.atleast_2d(axes)
    for idx, label in enumerate(labels):
        ax_cmd = axes[idx, 0]
        ax_err = axes[idx, 1]
        output_wheel_color = _blend_color(wheel_base_colors.get(label, output_color))
        ax_cmd.plot(
            t_axis,
            history["t_steer_cmd"][label],
            linestyle="--",
            label=r"$T_{\mathrm{steer,ref}}$",
            color=input_color,
            alpha=input_alpha,
        )
        ax_cmd.plot(
            t_axis,
            history["t_steer_actual"][label],
            label=r"$T_{\mathrm{steer}}$",
            color=output_wheel_color,
            alpha=output_alpha,
        )
        ax_cmd.set_ylabel(f"{label} [{nm_unit}]")
        ax_cmd.grid(True, alpha=grid_alpha)
        if idx == 0:
            ax_cmd.set_title("Steering torque tracking (motor)")
        if idx == len(labels) - 1:
            ax_cmd.set_xlabel("Time [s]")
        ax_cmd.legend(loc="upper right")

        ax_err.plot(
            t_axis,
            history["t_steer_error"][label],
            label=r"$T_{\mathrm{steer,ref}}-T_{\mathrm{steer}}$",
            color=error_color,
        )
        ax_err.grid(True, alpha=grid_alpha)
        if idx == 0:
            ax_err.set_title("Steering torque error (motor)")
        if idx == len(labels) - 1:
            ax_err.set_xlabel("Time [s]")
        ax_err.legend(loc="upper right")

    fig.tight_layout()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(history["x"], history["y"], label="trajectory")
    ax.set_title("XY trajectory")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_aspect("equal", "box")
    ax.grid(True, alpha=grid_alpha)
    ax.legend()

    label = labels[0]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    b_hat_name = "estimate" if estimator_enabled else "used"
    c_hat_name = "estimate" if estimator_enabled else "used"

    ax = axes[0]
    ax.plot(
        t_axis,
        history["B_hat"][label],
        label=f"{label} {b_label} ({b_hat_name})",
        color=estimate_color,
        alpha=output_alpha,
    )
    ax.plot(
        t_axis,
        history["B_true"][label],
        label=f"{label} {b_label} (true)",
        color=true_color,
    )
    ax.set_title(f"{b_label} ({label})")
    ax.set_ylabel(f"{b_label} [{nms_unit}]")
    ax.grid(True, alpha=grid_alpha)
    ax.legend()

    ax = axes[1]
    ax.plot(
        t_axis,
        history["C_alpha_hat"][label],
        label=f"{label} {c_label} ({c_hat_name})",
        color=estimate_color,
        alpha=output_alpha,
    )
    ax.plot(
        t_axis,
        history["C_alpha_true"][label],
        label=f"{label} {c_label} (true)",
        color=true_color,
    )
    ax.set_title(f"{c_label} ({label})")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(f"{c_label} [N/rad]")
    ax.grid(True, alpha=grid_alpha)
    ax.legend()

    fig.tight_layout()

    if "fy_total_actual" in history and "fy_total_from_r" in history:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4), sharex=True)
        ax.plot(
            t_axis,
            history["fy_total_actual"],
            label=r"$\sum F_y$",
            color=output_color,
            alpha=output_alpha,
        )
        ax.plot(
            t_axis,
            history["fy_total_from_r"],
            label=r"$m r v_x$",
            linestyle="--",
            color=input_color,
            alpha=input_alpha,
        )
        ax.set_title("Total lateral force balance")
        ax.set_ylabel("[N]")
        ax.grid(True, alpha=grid_alpha)
        ax.legend(loc="upper right")
        fig.tight_layout()

    if "fy_total_est" in history:
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        ax = axes[0]
        ax.plot(
            t_axis,
            history["fy_total_actual"],
            label=r"$F_{y,\mathrm{total}}$",
            color="0.2",
        )
        ax.plot(
            t_axis,
            history["fy_total_est"],
            label=r"$\hat{F}_{y,\mathrm{total}}$",
            linestyle="--",
            color=output_color,
            alpha=output_alpha,
        )
        ax.set_title("Total lateral force estimate")
        ax.set_ylabel("[N]")
        ax.grid(True, alpha=grid_alpha)
        ax.legend(loc="upper right")

        ax = axes[1]
        ax.plot(
            t_axis,
            history["fy_total_error"],
            label=r"$\hat{F}_{y,\mathrm{total}}-F_{y,\mathrm{total}}$",
            color=error_color,
        )
        ax.set_title("Total lateral force error")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("[N]")
        ax.grid(True, alpha=grid_alpha)
        ax.legend(loc="upper right")
        fig.tight_layout()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).with_name("config.yaml")),
        help="Path to YAML config.",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))

    sim_cfg = cfg.get("sim", {})
    dt = float(sim_cfg.get("dt", 0.001))
    sim_time = float(sim_cfg.get("time", 10.0))
    speed_cmd = float(sim_cfg.get("speed_mps", 10.0))
    seed = int(sim_cfg.get("seed", 1))
    speed_mode = str(sim_cfg.get("speed_mode", "lock")).lower()
    if speed_mode not in {"lock", "pid"}:
        raise ValueError(f"Unknown sim.speed_mode: {speed_mode}. Use 'lock' or 'pid'.")
    vehicle_config_path = sim_cfg.get("vehicle_config_path", None)

    control_cfg = cfg.get("control", {})
    control_mode = str(control_cfg.get("mode", "ff")).lower()
    control_mode = control_mode.replace("+", "_").replace("-", "_")
    aliases = {
        "ff_fb": "ff_fb",
        "ff_fb_ls": "ff_fb_ls",
        "ff_ls": "ff_ls",
        "ff": "ff",
    }
    if control_mode not in aliases:
        valid_modes = sorted(aliases.keys())
        raise ValueError(f"Unknown control.mode: {control_mode}. Valid: {valid_modes}")
    control_mode = aliases[control_mode]
    feedback_cfg = control_cfg.get("feedback", {})

    ff_cfg = control_cfg.get("feedforward", {})
    ff_options = SteeringTorqueFFOptions(
        max_accel=ff_cfg.get("max_accel", None),
        torque_limit=ff_cfg.get("torque_limit", None),
    )

    rng = np.random.default_rng(seed)

    vehicle = VehicleBody(config_path=vehicle_config_path)
    lock_speed(vehicle, speed_cmd)

    wheel_xy = {}
    for label in vehicle.wheel_labels:
        signs = vehicle.corner_signs[label]
        x_i = (vehicle.params.L_wheelbase / 2.0) * signs["pitch"]
        y_i = (vehicle.params.L_track / 2.0) * signs["roll"]
        wheel_xy[label] = (float(x_i), float(y_i))

    steering_param = load_param("steering", vehicle_config_path)
    base_j = float(steering_param.get("J_cq", 0.0))
    base_b = float(steering_param.get("B_cq", 0.0))
    tire_param = load_param("tire", vehicle_config_path)
    lateral_param = tire_param.get("lateral", {})
    base_c_alpha = float(lateral_param.get("C_alpha", 0.0))

    ff_param_bias_enabled = bool(ff_cfg.get("param_bias_enabled", False))
    ff_param_bias_percent = float(
        ff_cfg.get("param_bias_percent", 0.10 if ff_param_bias_enabled else 0.0)
    )
    if ff_param_bias_enabled and ff_param_bias_percent != 0.0:
        ff_base_b = float(base_b) * (1.0 + float(ff_param_bias_percent))
        ff_base_c_alpha = float(base_c_alpha) * (1.0 + float(ff_param_bias_percent))
    else:
        ff_base_b = float(base_b)
        ff_base_c_alpha = float(base_c_alpha)

    rand_cfg = cfg.get("randomization", {})
    random_enabled = bool(rand_cfg.get("enabled", False))
    random_mode = str(rand_cfg.get("mode", "static")).lower()
    if random_mode not in {"static", "gaussian"}:
        raise ValueError(f"Unknown randomization.mode: {random_mode}. Use 'static' or 'gaussian'.")
    rand_percent = float(rand_cfg.get("percent", 0.0))
    rand_per_wheel = bool(rand_cfg.get("per_wheel", False))
    randomize_b = bool(rand_cfg.get("randomize_b", True))
    randomize_c = bool(rand_cfg.get("randomize_c", True))
    gaussian_cfg = rand_cfg.get("gaussian", {})
    gauss_sigma_percent = float(gaussian_cfg.get("sigma_percent", 0.0))
    gauss_update_every = max(1, int(gaussian_cfg.get("update_every", 1)))
    gauss_randomize_initial = gaussian_cfg.get("randomize_initial", None)
    if gauss_randomize_initial is None:
        # Backward compatibility with init_from_nominal (inverse semantics).
        gauss_init_from_nominal = bool(gaussian_cfg.get("init_from_nominal", False))
        gauss_randomize_initial = not gauss_init_from_nominal
    else:
        gauss_randomize_initial = bool(gauss_randomize_initial)
    min_factor = 1.0 - rand_percent
    max_factor = 1.0 + rand_percent
    factor_b: Dict[str, float] = {}
    factor_c: Dict[str, float] = {}

    if random_enabled and rand_percent > 0.0:
        if random_mode == "gaussian" and not gauss_randomize_initial:
            factor_b = {label: 1.0 for label in vehicle.wheel_labels}
            factor_c = {label: 1.0 for label in vehicle.wheel_labels}
        else:
            factor_b, factor_c = init_random_factors(
                rng,
                rand_percent,
                rand_per_wheel,
                list(vehicle.wheel_labels),
                randomize_b,
                randomize_c,
            )
        actual_params = apply_random_factors(vehicle, base_b, base_c_alpha, factor_b, factor_c)
    else:
        actual_params = {
            label: (float(base_b), float(base_c_alpha)) for label in vehicle.wheel_labels
        }
    if random_enabled and rand_percent > 0.0:
        print(f"Applied steering parameter randomization ({random_mode}):")
        for label, (b_val, c_val) in actual_params.items():
            print(f"  {label}: B_cq={b_val:.6f}, C_alpha={c_val:.6f}")

    use_feedback = control_mode in {"ff_fb", "ff_fb_ls"}
    use_estimator = control_mode in {"ff_ls", "ff_fb_ls"}

    enable_yaw_fb = use_feedback and bool(feedback_cfg.get("yaw_rate", True))
    enable_fy_fb = use_feedback and bool(feedback_cfg.get("fy", False))
    enable_steer_fb = use_feedback and bool(feedback_cfg.get("steering", True))
    fy_feedback_source = str(feedback_cfg.get("fy_source", "true")).strip().lower()
    if fy_feedback_source not in {"true", "estimate"}:
        raise ValueError("control.feedback.fy_source must be one of: ['true', 'estimate']")

    yaw_moment_ff = YawMomentFeedforwardController(dt)
    allocator = YawMomentAllocator()
    steering_ff = SteeringFeedforwardController()
    steer_torque_ff = SteeringMotorTorqueFF(dt, ff_options)

    gains_path = control_cfg.get("gains_path", None)
    if gains_path is None:
        gains_path = PROJECT_ROOT / "vehicle_sim" / "models" / "params" / "controller_gains.yaml"
    else:
        gains_path = Path(gains_path)

    yaw_pid = PIDController(dt, load_pid_gains("yaw_rate_pid", gains_path))
    fy_gains = load_pid_gains("fy_pid", gains_path)
    steer_gains = load_pid_gains("steering_pid", gains_path)
    fy_pids = {label: PIDController(dt, fy_gains) for label in vehicle.wheel_labels}
    steer_pids = {label: PIDController(dt, steer_gains) for label in vehicle.wheel_labels}

    estimator_cfg = cfg.get("estimator", {})
    estimator_enabled = use_estimator and bool(estimator_cfg.get("enabled", False))
    if use_estimator and not estimator_enabled:
        print("Warning: control.mode requests LS, but estimator.enabled is false; using nominal params.")
    default_sample_decimation = max(1, int(estimator_cfg.get("sample_decimation", 1)))
    c_alpha_source = str(estimator_cfg.get("c_alpha_source", "true")).strip().lower()
    if c_alpha_source not in {"true", "estimate"}:
        raise ValueError("estimator.c_alpha_source must be one of: ['true', 'estimate']")
    b_align_source = str(estimator_cfg.get("b_align_source", "true")).strip().lower()
    if b_align_source not in {"true", "estimate", "ignore"}:
        raise ValueError("estimator.b_align_source must be one of: ['true', 'estimate', 'ignore']")

    def _cfg_section(value) -> dict:
        return value if isinstance(value, dict) else {}

    default_lambda = float(estimator_cfg.get("lambda", 0.99))
    default_p0 = float(estimator_cfg.get("p0", 1000.0))
    default_min_samples = int(estimator_cfg.get("min_samples", 0))
    default_start_time = float(estimator_cfg.get("start_time", 0.0))

    b_cfg = _cfg_section(estimator_cfg.get("b", estimator_cfg.get("b_cq", {})))
    c_cfg = _cfg_section(estimator_cfg.get("c_alpha", estimator_cfg.get("c", {})))

    b_lambda = float(b_cfg.get("lambda", default_lambda))
    b_p0 = float(b_cfg.get("p0", default_p0))
    b_min_samples = int(b_cfg.get("min_samples", default_min_samples))
    b_sample_decimation = max(1, int(b_cfg.get("sample_decimation", default_sample_decimation)))
    b_start_time = float(b_cfg.get("start_time", default_start_time))

    c_lambda = float(c_cfg.get("lambda", default_lambda))
    c_p0 = float(c_cfg.get("p0", default_p0))
    c_min_samples = int(c_cfg.get("min_samples", default_min_samples))
    c_sample_decimation = max(1, int(c_cfg.get("sample_decimation", default_sample_decimation)))
    c_start_time = float(c_cfg.get("start_time", default_start_time))

    b_gate_cfg = dict(estimator_cfg)
    b_gate_cfg["start_time"] = b_start_time
    c_gate_cfg = dict(estimator_cfg)
    c_gate_cfg["start_time"] = c_start_time

    b_estimators: Dict[str, ScalarRLS] = {}
    c_alpha_estimators: Dict[str, ScalarRLS] = {}
    if estimator_enabled:
        clamp_cfg = estimator_cfg.get("clamp", {})
        b_clamp = ScalarClamp(
            min_value=clamp_cfg.get("b_min", None),
            max_value=clamp_cfg.get("b_max", None),
        )
        c_alpha_min = clamp_cfg.get("c_alpha_min", clamp_cfg.get("j_min", None))
        c_alpha_max = clamp_cfg.get("c_alpha_max", clamp_cfg.get("j_max", None))
        c_alpha_clamp = ScalarClamp(
            min_value=c_alpha_min,
            max_value=c_alpha_max,
        )
        for label in vehicle.wheel_labels:
            b_estimators[label] = ScalarRLS(
                init_value=ff_base_b,
                forgetting_factor=b_lambda,
                p0=b_p0,
                clamp=b_clamp,
            )
            c_alpha_estimators[label] = ScalarRLS(
                init_value=ff_base_c_alpha,
                forgetting_factor=c_lambda,
                p0=c_p0,
                clamp=c_alpha_clamp,
            )

    speed_ctrl = None
    if speed_mode == "pid":
        speed_ctrl = build_speed_controller(dt, vehicle, vehicle_config_path)

    lateral_force_cfg = cfg.get("lateral_force_estimator", {})
    lateral_force_enabled = bool(lateral_force_cfg.get("enabled", False))
    lateral_force_estimator = None
    lateral_force_source = "kinematic"
    if lateral_force_enabled:
        lateral_force_source = str(lateral_force_cfg.get("ay_source", "kinematic")).strip().lower()
        if lateral_force_source not in {"kinematic", "imu"}:
            raise ValueError(
                "lateral_force_estimator.ay_source must be one of: ['kinematic', 'imu']"
            )
        ay_bias = float(lateral_force_cfg.get("ay_bias", 0.0))
        lowpass_tau = lateral_force_cfg.get("lowpass_tau", None)
        if lowpass_tau is not None:
            lowpass_tau = float(lowpass_tau)
        max_abs_ay = lateral_force_cfg.get("max_abs_ay", None)
        if max_abs_ay is not None:
            max_abs_ay = float(max_abs_ay)
        lat_force_options = LateralForceEstimatorOptions(
            ay_bias=ay_bias,
            lowpass_tau=lowpass_tau,
            max_abs_ay=max_abs_ay,
        )
        lateral_force_estimator = LateralForceEstimator(
            dt=dt,
            mass=float(vehicle.params.m),
            options=lat_force_options,
        )
    if enable_fy_fb and fy_feedback_source == "estimate" and not lateral_force_enabled:
        raise ValueError(
            "control.feedback.fy_source='estimate' requires lateral_force_estimator.enabled=true"
        )
    slip_estimator = None
    if estimator_enabled and (c_alpha_source == "estimate" or b_align_source == "estimate"):
        if not lateral_force_enabled:
            raise ValueError(
                "estimator.c_alpha_source/b_align_source='estimate' "
                "requires lateral_force_estimator.enabled=true"
            )
        slip_estimator = SlipAngleEstimator(dt, wheel_xy, SlipAngleEstimatorOptions())

    profile = build_profile(cfg.get("yaw_profile", {}))

    n_steps = int(round(sim_time / dt))

    history = {
        "t": [],
        "yaw_rate_cmd": [],
        "yaw_rate": [],
        "yaw_accel_cmd": [],
        "yaw_accel_actual": [],
        "yaw_error": [],
        "Mz_cmd": [],
        "Mz_ff": [],
        "Mz_fb": [],
        "Mz_actual": [],
        "Mz_error": [],
        "speed_cmd": [],
        "speed": [],
        "vy": [],
        "x": [],
        "y": [],
        "beta": [],
        "beta_ref_body": [],
        "beta_ref": {label: [] for label in vehicle.wheel_labels},
        "mz_from_fy": [],
        "mz_from_fx": [],
        "fy_total_cmd": [],
        "fy_total_actual": [],
        "fy_total_from_r": [],
        "fy_bias": [],
        "fy_cmd": {label: [] for label in vehicle.wheel_labels},
        "fy_actual": {label: [] for label in vehicle.wheel_labels},
        "fy_error": {label: [] for label in vehicle.wheel_labels},
        "fy_body": {label: [] for label in vehicle.wheel_labels},
        "fx_tire": {label: [] for label in vehicle.wheel_labels},
        "fx_body": {label: [] for label in vehicle.wheel_labels},
        "alpha": {label: [] for label in vehicle.wheel_labels},
        "alpha_cmd": {label: [] for label in vehicle.wheel_labels},
        "delta_cmd": {label: [] for label in vehicle.wheel_labels},
        "delta_actual": {label: [] for label in vehicle.wheel_labels},
        "delta_error": {label: [] for label in vehicle.wheel_labels},
        "delta_dot_actual": {label: [] for label in vehicle.wheel_labels},
        "delta_ddot": {label: [] for label in vehicle.wheel_labels},
        "t_steer_cmd": {label: [] for label in vehicle.wheel_labels},
        "t_steer_ff": {label: [] for label in vehicle.wheel_labels},
        "t_steer_fb": {label: [] for label in vehicle.wheel_labels},
        "t_steer_actual": {label: [] for label in vehicle.wheel_labels},
        "t_steer_error": {label: [] for label in vehicle.wheel_labels},
        "align_true": {label: [] for label in vehicle.wheel_labels},
        "align_est": {label: [] for label in vehicle.wheel_labels},
        "y_b": {label: [] for label in vehicle.wheel_labels},
        "b_update": {label: [] for label in vehicle.wheel_labels},
        "B_hat": {label: [] for label in vehicle.wheel_labels},
        "B_true": {label: [] for label in vehicle.wheel_labels},
        "B_samples": {label: [] for label in vehicle.wheel_labels},
        "C_alpha_hat": {label: [] for label in vehicle.wheel_labels},
        "C_alpha_true": {label: [] for label in vehicle.wheel_labels},
        "C_alpha_samples": {label: [] for label in vehicle.wheel_labels},
    }
    if lateral_force_enabled:
        history.update(
            {
                "ay_meas": [],
                "ay_filtered": [],
                "fy_total_est": [],
                "fy_total_error": [],
            }
        )

    prev_speed = vehicle.state.velocity_x
    prev_yaw_rate = vehicle.state.yaw_rate
    prev_vy = vehicle.state.velocity_y
    prev_delta_dot = {label: 0.0 for label in vehicle.wheel_labels}
    fy_total_est_last: Optional[float] = None
    wheel_labels = list(vehicle.wheel_labels)

    def _distribute_fy_total(fy_total: float, fy_cmd_map: Dict[str, float]) -> Dict[str, float]:
        denom = float(sum(fy_cmd_map.values()))
        if abs(denom) > 1e-6:
            return {
                label: float(fy_total) * float(fy_cmd_map.get(label, 0.0)) / denom
                for label in wheel_labels
            }
        share = float(fy_total) / max(1, len(wheel_labels))
        return {label: share for label in wheel_labels}

    for k in range(n_steps):
        t = k * dt
        if (
            random_enabled
            and random_mode == "gaussian"
            and rand_percent > 0.0
            and gauss_sigma_percent > 0.0
            and (k % gauss_update_every == 0)
        ):
            sigma_step = gauss_sigma_percent * np.sqrt(dt * gauss_update_every)
            update_gaussian_factors(
                rng,
                factor_b,
                factor_c,
                sigma_step,
                rand_per_wheel,
                randomize_b,
                randomize_c,
            )
            clamp_random_factors(factor_b, factor_c, min_factor, max_factor)
            actual_params = apply_random_factors(vehicle, base_b, base_c_alpha, factor_b, factor_c)
        yaw_rate_cmd, yaw_accel_cmd = profile.evaluate(t)

        if enable_yaw_fb:
            yaw_error = float(yaw_rate_cmd) - float(vehicle.state.yaw_rate)
            Mz_fb = yaw_pid.update(yaw_error)
        else:
            yaw_error = float(yaw_rate_cmd) - float(vehicle.state.yaw_rate)
            Mz_fb = 0.0

        Mz_ff = yaw_moment_ff.compute_moment(
            vehicle,
            yaw_rate_cmd=yaw_rate_cmd,
            yaw_accel_cmd=yaw_accel_cmd,
        )
        Mz_cmd = float(Mz_ff + Mz_fb)

        fy_total_cmd = float(vehicle.params.m) * float(speed_cmd) * float(yaw_rate_cmd)
        fy_wheel_cmd = allocator.allocate(
            vehicle,
            Mz_cmd,
            Fx_body=None,
            Fy_total_cmd=fy_total_cmd,
        )

        fy_wheel_actual_true: Dict[str, float] = {}
        for label in vehicle.wheel_labels:
            corner = vehicle.corners[label]
            fy_wheel_actual_true[label] = float(corner.state.F_y_tire)

        fy_wheel_actual = dict(fy_wheel_actual_true)
        if enable_fy_fb and fy_feedback_source == "estimate" and fy_total_est_last is not None:
            fy_wheel_actual = _distribute_fy_total(fy_total_est_last, fy_wheel_cmd)

        j_map = {}
        b_map = {}
        c_alpha_map = {}
        for label in vehicle.wheel_labels:
            if estimator_enabled:
                b_est = b_estimators[label]
                c_est = c_alpha_estimators[label]
                if b_est.sample_count >= b_min_samples:
                    b_map[label] = float(b_est.get_value())
                else:
                    b_map[label] = float(ff_base_b)
                if c_est.sample_count >= c_min_samples:
                    c_alpha_val = c_est.get_value()
                else:
                    c_alpha_val = float(ff_base_c_alpha)
                c_alpha_map[label] = float(c_alpha_val)
            else:
                b_map[label] = float(ff_base_b)
                c_alpha_map[label] = float(ff_base_c_alpha)
            j_map[label] = float(base_j)

        vy_cmd = 0.0
        delta_cmd, steering_debug = steering_ff.compute_delta_cmd_with_debug(
            vehicle,
            fy_wheel_cmd,
            vx_cmd=speed_cmd,
            yaw_rate_cmd=yaw_rate_cmd,
            vy_cmd=vy_cmd,
            c_alpha_override=c_alpha_map,
        )
        alpha_cmd_map = steering_debug.get("alpha_cmd", {})

        if enable_fy_fb:
            for label in vehicle.wheel_labels:
                fy_error = float(fy_wheel_cmd.get(label, 0.0)) - float(fy_wheel_actual.get(label, 0.0))
                delta_cmd[label] = float(delta_cmd.get(label, 0.0)) + float(fy_pids[label].update(fy_error))

        align_cmd = compute_align_cmd(vehicle, fy_wheel_cmd)

        ff_params = build_ff_params(vehicle, j_map, b_map)
        t_steer_ff_motor, _ = steer_torque_ff.compute_torque(
            vehicle.wheel_labels,
            delta_cmd=delta_cmd,
            align_cmd=align_cmd,
            params_map=ff_params,
        )

        t_steer_cmd = dict(t_steer_ff_motor)
        if enable_steer_fb:
            for label in vehicle.wheel_labels:
                corner = vehicle.corners[label]
                delta_error = float(delta_cmd.get(label, 0.0)) - float(corner.state.steering_angle)
                torque_correction_axis = float(steer_pids[label].update(delta_error))
                scale = float(corner.steering.params.gear_ratio)
                if abs(scale) < 1e-6:
                    torque_correction_motor = torque_correction_axis
                else:
                    torque_correction_motor = torque_correction_axis / scale
                t_steer_cmd[label] = float(t_steer_cmd.get(label, 0.0)) + float(torque_correction_motor)

        if speed_mode == "pid" and speed_ctrl is not None:
            speed = vehicle.state.velocity_x
            speed_dot = (speed - prev_speed) / dt if k > 0 else 0.0
            t_drv = speed_ctrl.update(speed_cmd, speed, speed_dot)
            prev_speed = speed
        else:
            t_drv = 0.0

        align_used = {
            label: float(vehicle.corners[label].steering.state.self_aligning_torque)
            for label in vehicle.wheel_labels
        }

        corner_inputs = {}
        for label in vehicle.wheel_labels:
            corner_inputs[label] = {
                "T_steer": t_steer_cmd.get(label, 0.0),
                "T_brk": 0.0,
                "T_Drv": t_drv,
                "T_susp": 0.0,
                "z_road": 0.0,
                "z_road_dot": 0.0,
            }

        vehicle.update(dt, corner_inputs)
        if speed_mode == "lock":
            lock_speed(vehicle, speed_cmd)

        yaw_rate_actual = float(vehicle.state.yaw_rate)
        yaw_accel_actual = (yaw_rate_actual - prev_yaw_rate) / dt if k > 0 else 0.0
        prev_yaw_rate = yaw_rate_actual

        corner_outputs = {
            label: (
                vehicle.corners[label].state.F_s,
                vehicle.corners[label].state.F_x_tire,
                vehicle.corners[label].state.F_y_tire,
            )
            for label in vehicle.wheel_labels
        }
        forces, moments = vehicle.assemble_forces_moments(corner_outputs)
        mz_actual = float(moments[2])
        velocity_x = float(vehicle.state.velocity_x)
        velocity_y = float(vehicle.state.velocity_y)
        fy_total_actual = float(forces[1])
        fy_total_from_r = float(vehicle.params.m) * float(velocity_x) * float(yaw_rate_actual)
        v_dot_y = (velocity_y - prev_vy) / dt if k > 0 else 0.0
        prev_vy = velocity_y
        ay_meas = None
        if lateral_force_enabled and lateral_force_estimator is not None:
            if lateral_force_source == "imu":
                ay_meas = float(forces[1]) / float(vehicle.params.m)
            else:
                omega = np.array(
                    [
                        float(vehicle.state.roll_rate),
                        float(vehicle.state.pitch_rate),
                        float(vehicle.state.yaw_rate),
                    ]
                )
                v_body = np.array([velocity_x, velocity_y, float(vehicle.state.heave_dot)])
                omega_cross_v = np.cross(omega, v_body)
                ay_meas = v_dot_y + float(omega_cross_v[1])

            fy_total_est = float(lateral_force_estimator.update(ay_meas))
            fy_total_est_last = fy_total_est
            ay_filtered = lateral_force_estimator.last_ay
            if ay_filtered is None:
                ay_filtered = ay_meas
            history["ay_meas"].append(float(ay_meas))
            history["ay_filtered"].append(float(ay_filtered))
            history["fy_total_est"].append(float(fy_total_est))
            history["fy_total_error"].append(float(fy_total_est - fy_total_actual))

        alpha_est_map: Dict[str, float] = {}
        fy_wheel_est_map: Dict[str, float] = {}
        align_est_map: Dict[str, float] = {}
        if slip_estimator is not None:
            if ay_meas is None:
                omega = np.array(
                    [
                        float(vehicle.state.roll_rate),
                        float(vehicle.state.pitch_rate),
                        float(vehicle.state.yaw_rate),
                    ]
                )
                v_body = np.array([velocity_x, velocity_y, float(vehicle.state.heave_dot)])
                omega_cross_v = np.cross(omega, v_body)
                ay_meas = v_dot_y + float(omega_cross_v[1])
            delta_meas = {
                label: float(vehicle.corners[label].state.steering_angle)
                for label in vehicle.wheel_labels
            }
            alpha_est_map, _ = slip_estimator.update(
                velocity_x,
                yaw_rate_actual,
                float(ay_meas),
                delta_meas,
            )
        if fy_total_est_last is not None:
            fy_wheel_est_map = _distribute_fy_total(fy_total_est_last, fy_wheel_cmd)
        if fy_wheel_est_map:
            for label in vehicle.wheel_labels:
                trail = float(vehicle.corners[label].lateral_tire.params.trail)
                align_est_map[label] = float(trail * float(fy_wheel_est_map.get(label, 0.0)))

        delta_ddot_map: Dict[str, float] = {label: 0.0 for label in vehicle.wheel_labels}
        y_b_map: Dict[str, float] = {label: float("nan") for label in vehicle.wheel_labels}
        b_update_map: Dict[str, bool] = {label: False for label in vehicle.wheel_labels}

        if estimator_enabled:
            for label in vehicle.wheel_labels:
                corner = vehicle.corners[label]
                delta_dot = float(corner.steering.state.steering_rate)
                delta_ddot = (delta_dot - prev_delta_dot[label]) / dt
                prev_delta_dot[label] = delta_dot
                delta_ddot_map[label] = float(delta_ddot)
                if b_align_source == "true":
                    align_term = float(align_used.get(label, 0.0))
                elif b_align_source == "estimate":
                    align_term = float(align_est_map.get(label, 0.0))
                else:
                    align_term = 0.0
                y_b = (
                    float(corner.steering.state.steering_torque)
                    - align_term
                    - float(base_j) * float(delta_ddot)
                )
                y_b_map[label] = float(y_b)

                if (
                    (t >= b_start_time)
                    and (k % b_sample_decimation == 0)
                    and should_update_b_estimator(
                    t,
                    delta=float(corner.state.steering_angle),
                    delta_dot=delta_dot,
                    delta_ddot=delta_ddot,
                    params=corner.steering.params,
                    cfg=b_gate_cfg,
                )
                ):
                    b_estimators[label].update(y_b, delta_dot)
                    b_update_map[label] = True

                if c_alpha_source == "estimate":
                    if label not in alpha_est_map or label not in fy_wheel_est_map:
                        continue
                    alpha = float(alpha_est_map[label])
                    fy = float(fy_wheel_est_map[label])
                else:
                    alpha = float(corner.lateral_tire.state.slip_angle)
                    fy = float(corner.state.F_y_tire)
                fz = float(corner.state.F_z)
                mu = float(corner.lateral_tire.params.mu)
                if (
                    (t >= c_start_time)
                    and (k % c_sample_decimation == 0)
                    and should_update_calpha_estimator(
                    t,
                    alpha=alpha,
                    fy=fy,
                    fz=fz,
                    mu=mu,
                    cfg=c_gate_cfg,
                )
                ):
                    c_alpha_estimators[label].update(-fy, alpha)
        else:
            for label in vehicle.wheel_labels:
                delta_dot = float(vehicle.corners[label].steering.state.steering_rate)
                delta_ddot = (delta_dot - prev_delta_dot[label]) / dt
                prev_delta_dot[label] = delta_dot
                delta_ddot_map[label] = float(delta_ddot)

        history["t"].append(t)
        history["yaw_rate_cmd"].append(float(yaw_rate_cmd))
        history["yaw_rate"].append(float(yaw_rate_actual))
        history["yaw_accel_cmd"].append(float(yaw_accel_cmd))
        history["yaw_accel_actual"].append(float(yaw_accel_actual))
        history["yaw_error"].append(float(yaw_error))
        history["Mz_cmd"].append(float(Mz_cmd))
        history["Mz_ff"].append(float(Mz_ff))
        history["Mz_fb"].append(float(Mz_fb))
        history["Mz_actual"].append(float(mz_actual))
        history["Mz_error"].append(float(Mz_cmd) - float(mz_actual))
        history["fy_total_cmd"].append(float(fy_total_cmd))
        history["fy_total_actual"].append(float(fy_total_actual))
        history["fy_total_from_r"].append(float(fy_total_from_r))
        fy_bias = float(fy_total_cmd) / max(1, len(vehicle.wheel_labels))
        history["fy_bias"].append(float(fy_bias))
        history["speed_cmd"].append(float(speed_cmd))
        history["speed"].append(float(vehicle.state.velocity_x))
        history["vy"].append(float(velocity_y))
        history["x"].append(float(vehicle.state.x))
        history["y"].append(float(vehicle.state.y))
        beta_actual = float(np.arctan2(vehicle.state.velocity_y, vehicle.state.velocity_x))
        history["beta"].append(beta_actual)
        beta_ref_body = float(np.arctan2(float(vy_cmd), float(speed_cmd)))
        history["beta_ref_body"].append(beta_ref_body)
        beta_ref_map = steering_debug.get("beta_ref", {})
        for label in vehicle.wheel_labels:
            history["beta_ref"][label].append(float(beta_ref_map.get(label, 0.0)))

        mz_from_fy = 0.0
        mz_from_fx = 0.0
        for label in vehicle.wheel_labels:
            corner = vehicle.corners[label]
            f_x_tire = float(corner.state.F_x_tire)
            f_y_tire = float(corner.state.F_y_tire)
            delta_actual = float(corner.state.steering_angle)
            c, s = np.cos(delta_actual), np.sin(delta_actual)
            f_x_body = c * f_x_tire - s * f_y_tire
            f_y_body = s * f_x_tire + c * f_y_tire
            signs = vehicle.corner_signs[label]
            x_i = (vehicle.params.L_wheelbase / 2.0) * signs["pitch"]
            y_i = (vehicle.params.L_track / 2.0) * signs["roll"]
            mz_from_fy += float(x_i) * float(f_y_body)
            mz_from_fx += -float(y_i) * float(f_x_body)
            history["fy_cmd"][label].append(float(fy_wheel_cmd.get(label, 0.0)))
            history["fy_actual"][label].append(float(f_y_tire))
            history["fy_error"][label].append(
                float(fy_wheel_cmd.get(label, 0.0)) - float(f_y_tire)
            )
            history["fy_body"][label].append(float(f_y_body))
            history["fx_tire"][label].append(float(f_x_tire))
            history["fx_body"][label].append(float(f_x_body))
            history["alpha"][label].append(float(corner.lateral_tire.state.slip_angle))
            history["alpha_cmd"][label].append(float(alpha_cmd_map.get(label, 0.0)))
            history["delta_cmd"][label].append(float(delta_cmd.get(label, 0.0)))
            history["delta_actual"][label].append(float(corner.state.steering_angle))
            history["delta_error"][label].append(
                float(delta_cmd.get(label, 0.0)) - float(corner.state.steering_angle)
            )
            history["delta_dot_actual"][label].append(float(corner.steering.state.steering_rate))
            history["delta_ddot"][label].append(float(delta_ddot_map.get(label, 0.0)))
            history["t_steer_cmd"][label].append(float(t_steer_cmd.get(label, 0.0)))
            history["t_steer_ff"][label].append(float(t_steer_ff_motor.get(label, 0.0)))
            history["t_steer_fb"][label].append(
                float(t_steer_cmd.get(label, 0.0)) - float(t_steer_ff_motor.get(label, 0.0))
            )
            params = corner.steering.params
            scale = float(params.gear_ratio)
            t_axis_actual = float(corner.steering.state.steering_torque)
            t_motor_actual = t_axis_actual / scale if abs(scale) > 1e-6 else t_axis_actual
            history["t_steer_actual"][label].append(float(t_motor_actual))
            history["t_steer_error"][label].append(
                float(t_steer_cmd.get(label, 0.0)) - float(t_motor_actual)
            )
            history["align_true"][label].append(float(align_used.get(label, float("nan"))))
            history["align_est"][label].append(float(align_est_map.get(label, float("nan"))))
            history["y_b"][label].append(float(y_b_map.get(label, float("nan"))))
            history["b_update"][label].append(1 if b_update_map.get(label, False) else 0)

            if estimator_enabled:
                b_hat = b_estimators[label].get_value()
                c_alpha_hat = c_alpha_estimators[label].get_value()
                b_samples = int(b_estimators[label].sample_count)
                c_samples = int(c_alpha_estimators[label].sample_count)
            else:
                b_hat, c_alpha_hat = ff_base_b, ff_base_c_alpha
                b_samples, c_samples = 0, 0
            history["B_hat"][label].append(float(b_hat))
            history["C_alpha_hat"][label].append(float(c_alpha_hat))
            b_true, c_true = actual_params[label]
            history["B_true"][label].append(float(b_true))
            history["C_alpha_true"][label].append(float(c_true))
            history["B_samples"][label].append(b_samples)
            history["C_alpha_samples"][label].append(c_samples)

        history["mz_from_fy"].append(float(mz_from_fy))
        history["mz_from_fx"].append(float(mz_from_fx))

    t_axis = np.array(history["t"], dtype=float)

    def trapz_abs(values: list[float] | np.ndarray) -> float:
        arr = np.array(values, dtype=float)
        if arr.size < 2:
            return 0.0
        integrator = getattr(np, "trapezoid", None)
        if integrator is None:
            integrator = np.trapz
        return float(integrator(np.abs(arr), t_axis))

    yaw_err = np.array(history["yaw_error"], dtype=float)
    mz_err = np.array(history["Mz_error"], dtype=float)
    mz_cmd = np.array(history["Mz_cmd"], dtype=float)

    metrics = {
        "yaw_rate_rmse": float(np.sqrt(np.mean(yaw_err ** 2))) if yaw_err.size else 0.0,
        "yaw_rate_mae": float(np.mean(np.abs(yaw_err))) if yaw_err.size else 0.0,
        "yaw_rate_peak_abs_error": float(np.max(np.abs(yaw_err))) if yaw_err.size else 0.0,
        "yaw_moment_error_rmse": float(np.sqrt(np.mean(mz_err ** 2))) if mz_err.size else 0.0,
        "yaw_moment_error_mae": float(np.mean(np.abs(mz_err))) if mz_err.size else 0.0,
        "yaw_moment_error_peak_abs": float(np.max(np.abs(mz_err))) if mz_err.size else 0.0,
        "yaw_moment_cmd_effort_l1": trapz_abs(mz_cmd),
        "yaw_moment_cmd_peak_abs": float(np.max(np.abs(mz_cmd))) if mz_cmd.size else 0.0,
    }

    fy_err_list = []
    for label in vehicle.wheel_labels:
        fy_err = np.array(history["fy_error"][label], dtype=float)
        fy_err_list.append(fy_err)
        metrics[f"fy_error_rmse_{label}"] = float(np.sqrt(np.mean(fy_err ** 2))) if fy_err.size else 0.0
        metrics[f"fy_error_peak_abs_{label}"] = (
            float(np.max(np.abs(fy_err))) if fy_err.size else 0.0
        )
    fy_err_all = np.concatenate(fy_err_list) if fy_err_list else np.array([], dtype=float)
    metrics["fy_wheel_error_rmse"] = float(np.sqrt(np.mean(fy_err_all ** 2))) if fy_err_all.size else 0.0
    metrics["fy_wheel_error_mae"] = float(np.mean(np.abs(fy_err_all))) if fy_err_all.size else 0.0
    metrics["fy_wheel_error_peak_abs"] = float(np.max(np.abs(fy_err_all))) if fy_err_all.size else 0.0

    delta_err_list = []
    for label in vehicle.wheel_labels:
        delta_err = np.array(history["delta_error"][label], dtype=float)
        delta_err_list.append(delta_err)
        delta_rmse = float(np.sqrt(np.mean(delta_err ** 2))) if delta_err.size else 0.0
        delta_peak = float(np.max(np.abs(delta_err))) if delta_err.size else 0.0
        metrics[f"delta_error_rmse_{label}"] = delta_rmse
        metrics[f"delta_error_peak_abs_{label}"] = delta_peak
        metrics[f"steering_angle_error_rmse_{label}"] = delta_rmse
        metrics[f"steering_angle_error_peak_abs_{label}"] = delta_peak
    delta_err_all = np.concatenate(delta_err_list) if delta_err_list else np.array([], dtype=float)
    delta_rmse_all = float(np.sqrt(np.mean(delta_err_all ** 2))) if delta_err_all.size else 0.0
    delta_peak_all = float(np.max(np.abs(delta_err_all))) if delta_err_all.size else 0.0
    metrics["delta_error_rmse"] = delta_rmse_all
    metrics["delta_error_peak_abs"] = delta_peak_all
    metrics["steering_angle_error_rmse"] = delta_rmse_all
    metrics["steering_angle_error_peak_abs"] = delta_peak_all
    if lateral_force_enabled and "fy_total_error" in history:
        fy_total_err = np.array(history["fy_total_error"], dtype=float)
        metrics["fy_total_est_rmse"] = (
            float(np.sqrt(np.mean(fy_total_err ** 2))) if fy_total_err.size else 0.0
        )
        metrics["fy_total_est_mae"] = (
            float(np.mean(np.abs(fy_total_err))) if fy_total_err.size else 0.0
        )
        metrics["fy_total_est_peak_abs"] = (
            float(np.max(np.abs(fy_total_err))) if fy_total_err.size else 0.0
        )

    if estimator_enabled:
        def first_valid_index(samples: np.ndarray, min_samples: int, start_time: float) -> int:
            if samples.size == 0:
                return 0
            mask = (t_axis >= float(start_time)) & (samples >= int(min_samples))
            idx = np.argmax(mask) if np.any(mask) else samples.size - 1
            return int(idx)

        def add_estimator_metrics(
            name: str,
            true_series: np.ndarray,
            hat_series: np.ndarray,
            samples: np.ndarray,
            min_samples: int,
            start_time: float,
        ) -> None:
            idx0 = first_valid_index(samples, min_samples, start_time)
            err = hat_series - true_series
            denom = np.maximum(1e-12, np.abs(true_series))
            rel = err / denom
            err_w = err[idx0:]
            rel_w = rel[idx0:]

            metrics[f"{name}_window_start_s"] = float(t_axis[idx0]) if t_axis.size else 0.0
            metrics[f"{name}_updates"] = int(samples[-1]) if samples.size else 0
            metrics[f"{name}_abs_rmse"] = float(np.sqrt(np.mean(err_w ** 2))) if err_w.size else 0.0
            metrics[f"{name}_abs_mae"] = float(np.mean(np.abs(err_w))) if err_w.size else 0.0
            metrics[f"{name}_abs_peak"] = float(np.max(np.abs(err_w))) if err_w.size else 0.0
            metrics[f"{name}_rel_rmse_percent"] = (
                float(np.sqrt(np.mean(rel_w ** 2)) * 100.0) if rel_w.size else 0.0
            )
            metrics[f"{name}_rel_mae_percent"] = (
                float(np.mean(np.abs(rel_w)) * 100.0) if rel_w.size else 0.0
            )
            metrics[f"{name}_rel_peak_percent"] = (
                float(np.max(np.abs(rel_w)) * 100.0) if rel_w.size else 0.0
            )
            metrics[f"{name}_final_rel_percent"] = (
                float(rel[-1] * 100.0) if rel.size else 0.0
            )

        output_cfg = cfg.get("output", {})
        representative = _resolve_paper_wheel(list(vehicle.wheel_labels), output_cfg)

        b_true_rep = np.array(history["B_true"][representative], dtype=float)
        b_hat_rep = np.array(history["B_hat"][representative], dtype=float)
        b_samples_rep = np.array(history["B_samples"][representative], dtype=int)
        add_estimator_metrics(
            "est_b_cq", b_true_rep, b_hat_rep, b_samples_rep, b_min_samples, b_start_time
        )

        c_true_rep = np.array(history["C_alpha_true"][representative], dtype=float)
        c_hat_rep = np.array(history["C_alpha_hat"][representative], dtype=float)
        c_samples_rep = np.array(history["C_alpha_samples"][representative], dtype=int)
        add_estimator_metrics(
            "est_c_alpha", c_true_rep, c_hat_rep, c_samples_rep, c_min_samples, c_start_time
        )

        b_rel_rmse_list = []
        c_rel_rmse_list = []
        b_updates_list = []
        c_updates_list = []
        for label in vehicle.wheel_labels:
            b_true = np.array(history["B_true"][label], dtype=float)
            b_hat = np.array(history["B_hat"][label], dtype=float)
            b_samples = np.array(history["B_samples"][label], dtype=int)
            idx0 = first_valid_index(b_samples, b_min_samples, b_start_time)
            denom = np.maximum(1e-12, np.abs(b_true[idx0:]))
            rel = (b_hat[idx0:] - b_true[idx0:]) / denom
            if rel.size:
                b_rel_rmse_list.append(float(np.sqrt(np.mean(rel ** 2)) * 100.0))
            b_updates_list.append(int(b_samples[-1]) if b_samples.size else 0)

            c_true = np.array(history["C_alpha_true"][label], dtype=float)
            c_hat = np.array(history["C_alpha_hat"][label], dtype=float)
            c_samples = np.array(history["C_alpha_samples"][label], dtype=int)
            idx0 = first_valid_index(c_samples, c_min_samples, c_start_time)
            denom = np.maximum(1e-12, np.abs(c_true[idx0:]))
            rel = (c_hat[idx0:] - c_true[idx0:]) / denom
            if rel.size:
                c_rel_rmse_list.append(float(np.sqrt(np.mean(rel ** 2)) * 100.0))
            c_updates_list.append(int(c_samples[-1]) if c_samples.size else 0)

        metrics["est_b_cq_rel_rmse_percent_mean"] = (
            float(np.mean(b_rel_rmse_list)) if b_rel_rmse_list else 0.0
        )
        metrics["est_c_alpha_rel_rmse_percent_mean"] = (
            float(np.mean(c_rel_rmse_list)) if c_rel_rmse_list else 0.0
        )
        metrics["est_b_cq_updates_mean"] = float(np.mean(b_updates_list)) if b_updates_list else 0.0
        metrics["est_c_alpha_updates_mean"] = float(np.mean(c_updates_list)) if c_updates_list else 0.0

    steer_effort_cmd_total = 0.0
    steer_effort_ff_total = 0.0
    steer_effort_fb_total = 0.0
    steer_peak_cmd_total = 0.0
    for label in vehicle.wheel_labels:
        t_cmd = np.array(history["t_steer_cmd"][label], dtype=float)
        steer_effort_cmd_total += trapz_abs(t_cmd)
        if t_cmd.size:
            steer_peak_cmd_total = max(steer_peak_cmd_total, float(np.max(np.abs(t_cmd))))

        steer_effort_ff_total += trapz_abs(history["t_steer_ff"][label])
        steer_effort_fb_total += trapz_abs(history["t_steer_fb"][label])

    metrics["steer_torque_cmd_effort_l1_total"] = float(steer_effort_cmd_total)
    metrics["steer_torque_ff_effort_l1_total"] = float(steer_effort_ff_total)
    metrics["steer_torque_fb_effort_l1_total"] = float(steer_effort_fb_total)
    metrics["steer_torque_cmd_peak_abs_total"] = float(steer_peak_cmd_total)

    output_cfg = cfg.get("output", {})
    if output_cfg.get("save_json", False):
        out_dir = Path(output_cfg.get("save_dir", "output"))
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = str(output_cfg.get("tag", "run"))
        out_path = out_dir / f"metrics_{tag}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump({"metrics": metrics, "config": cfg}, f, indent=2)

    if output_cfg.get("save_npz", False):
        out_dir = Path(output_cfg.get("save_dir", "output"))
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = str(output_cfg.get("tag", "run"))
        out_path = out_dir / f"history_{tag}.npz"
        np.savez_compressed(out_path, history=history, config=cfg, metrics=metrics)

    if output_cfg.get("plot", False):
        import matplotlib.pyplot as plt

        t_axis = np.array(history["t"], dtype=float)
        labels = list(vehicle.wheel_labels)

        plot_mode = _resolve_plot_mode(output_cfg)
        style = _apply_plot_style(plt, output_cfg)
        if plot_mode == "paper":
            estimator_start_time = min(float(b_start_time), float(c_start_time))
            _plot_paper(
                plt,
                history,
                t_axis,
                labels,
                control_mode,
                estimator_enabled,
                estimator_start_time,
                output_cfg,
                style,
            )
        else:
            _plot_debug(plt, history, t_axis, labels, control_mode, estimator_enabled, style)

        if output_cfg.get("show_plots", True):
            plt.show()

    print("Metrics:")
    for key in sorted(metrics):
        value = metrics[key]
        print(f"  {key}: {value:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


