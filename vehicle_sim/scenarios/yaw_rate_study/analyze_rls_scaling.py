#!/usr/bin/env python3
"""
Analyze RLS regressor (phi) scale and initial gain sensitivity.

This script does NOT modify run_yaw_rate_study.py nor config.yaml in-place.
It can either:
  1) Analyze an existing `history_*.npz`, or
  2) Run a simulation using a temporary YAML (copy of your config) with save_npz enabled,
     then analyze the generated history.

Outputs:
  - phi statistics (delta_dot, delta_ddot, alpha)
  - first-update index/time for B_cq and C_alpha estimators
  - initial gain K0 and first-step parameter update magnitude (scalar RLS)
"""

from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import yaml

# Ensure the project package root is on sys.path when running directly.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vehicle_sim.controllers.slip_angle_estimator import SlipAngleEstimator, SlipAngleEstimatorOptions
from vehicle_sim.utils.config_loader import load_param


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _dump_yaml(cfg: dict[str, Any], path: Path) -> None:
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def _load_history(npz_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    data = np.load(npz_path, allow_pickle=True)
    history = data["history"].item()
    cfg = data["config"].item()
    metrics = data["metrics"].item()
    return history, cfg, metrics


def _resolve_vehicle_config_path(cfg: dict[str, Any]) -> str | None:
    sim_cfg = cfg.get("sim", {})
    raw = sim_cfg.get("vehicle_config_path", None)
    if raw in (None, "null"):
        return None
    path = Path(str(raw))
    if not path.is_absolute():
        path = _repo_root() / path
    return str(path)


def _resolve_wheel(history: dict[str, Any], cfg: dict[str, Any], requested: str | None) -> str:
    labels = []
    fy_cmd = history.get("fy_cmd", {})
    if isinstance(fy_cmd, dict) and fy_cmd:
        labels = list(fy_cmd.keys())
    if requested and requested in labels:
        return str(requested)
    output_cfg = cfg.get("output", {})
    paper = output_cfg.get("paper_wheel", None)
    if paper in labels:
        return str(paper)
    return str(labels[0] if labels else (requested or "FL"))


def _cfg_section(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _rls_params(cfg: dict[str, Any]) -> dict[str, Any]:
    est_cfg = _cfg_section(cfg.get("estimator", {}))
    default_lambda = float(est_cfg.get("lambda", 0.99))
    default_p0 = float(est_cfg.get("p0", 1000.0))
    default_min_samples = int(est_cfg.get("min_samples", 0))
    default_start_time = float(est_cfg.get("start_time", 0.0))
    default_sample_decimation = max(1, int(est_cfg.get("sample_decimation", 1)))
    c_alpha_source = str(est_cfg.get("c_alpha_source", "true")).strip().lower()
    b_align_source = str(est_cfg.get("b_align_source", "true")).strip().lower()

    b_cfg = _cfg_section(est_cfg.get("b", est_cfg.get("b_cq", {})))
    c_cfg = _cfg_section(est_cfg.get("c_alpha", est_cfg.get("c", {})))

    return {
        "start_time": default_start_time,
        "sample_decimation": default_sample_decimation,
        "c_alpha_source": c_alpha_source,
        "b_align_source": b_align_source,
        "b": {
            "start_time": float(b_cfg.get("start_time", default_start_time)),
            "lambda": float(b_cfg.get("lambda", default_lambda)),
            "p0": float(b_cfg.get("p0", default_p0)),
            "min_samples": int(b_cfg.get("min_samples", default_min_samples)),
            "sample_decimation": max(
                1, int(b_cfg.get("sample_decimation", default_sample_decimation))
            ),
        },
        "c_alpha": {
            "start_time": float(c_cfg.get("start_time", default_start_time)),
            "lambda": float(c_cfg.get("lambda", default_lambda)),
            "p0": float(c_cfg.get("p0", default_p0)),
            "min_samples": int(c_cfg.get("min_samples", default_min_samples)),
            "sample_decimation": max(
                1, int(c_cfg.get("sample_decimation", default_sample_decimation))
            ),
        },
    }


def _ff_bias(cfg: dict[str, Any]) -> tuple[bool, float]:
    control_cfg = _cfg_section(cfg.get("control", {}))
    ff_cfg = _cfg_section(control_cfg.get("feedforward", {}))
    enabled = bool(ff_cfg.get("param_bias_enabled", False))
    percent = float(ff_cfg.get("param_bias_percent", 0.0))
    return enabled, percent


def _scalar_rls_first_step(
    *,
    theta0: float,
    p0: float,
    lam: float,
    y: float,
    phi: float,
) -> dict[str, float]:
    phi = float(phi)
    denom = float(lam + p0 * phi * phi)
    if denom <= 0.0 or not np.isfinite(denom):
        return {"K0": 0.0, "err0": 0.0, "dtheta0": 0.0, "p0_phi2": float(p0 * phi * phi)}
    K0 = float((p0 * phi) / denom)
    err0 = float(y - phi * theta0)
    dtheta0 = float(K0 * err0)
    return {"K0": K0, "err0": err0, "dtheta0": dtheta0, "p0_phi2": float(p0 * phi * phi)}


def _first_update_index(samples: np.ndarray) -> int | None:
    samples = np.asarray(samples, dtype=int)
    if samples.size < 2:
        return None
    jumps = np.flatnonzero(np.diff(samples) > 0) + 1
    if jumps.size == 0:
        return None
    return int(jumps[0])


def _min_samples_reached_index(
    t_axis: np.ndarray,
    *,
    start_time: float,
    samples: np.ndarray,
    min_samples: int,
) -> int | None:
    if min_samples <= 0:
        return None
    t_axis = np.asarray(t_axis, dtype=float)
    samples = np.asarray(samples, dtype=int)
    if t_axis.size == 0 or samples.size == 0 or t_axis.size != samples.size:
        return None
    mask = (t_axis >= float(start_time)) & (samples >= int(min_samples))
    if not np.any(mask):
        return None
    return int(np.argmax(mask))


def _median_dt(t_axis: np.ndarray) -> float | None:
    t_axis = np.asarray(t_axis, dtype=float)
    if t_axis.size < 2:
        return None
    dt = np.diff(t_axis)
    dt = dt[np.isfinite(dt)]
    if dt.size == 0:
        return None
    dt_med = float(np.median(dt))
    if dt_med <= 0.0 or not np.isfinite(dt_med):
        return None
    return dt_med


def _describe_stats(name: str, values: np.ndarray) -> str:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return f"{name}: (empty)"
    abs_values = np.abs(values)
    p50, p90, p99 = np.percentile(abs_values, [50, 90, 99])
    return (
        f"{name}: rms={np.sqrt(np.mean(values**2)):.3g}, "
        f"max_abs={abs_values.max():.3g}, "
        f"|x| p50/p90/p99={p50:.3g}/{p90:.3g}/{p99:.3g}"
    )


def _compute_fy_wheel_est(
    history: dict[str, Any],
    labels: list[str],
) -> dict[str, np.ndarray] | None:
    if "fy_total_est" not in history:
        return None
    fy_total_est = np.asarray(history.get("fy_total_est", []), dtype=float)
    if fy_total_est.size == 0:
        return None
    fy_cmd = history.get("fy_cmd", {})
    if not isinstance(fy_cmd, dict):
        return None

    n = fy_total_est.size
    cmd_mat = np.zeros((n, len(labels)), dtype=float)
    for j, label in enumerate(labels):
        cmd_mat[:, j] = np.asarray(fy_cmd.get(label, []), dtype=float)
    denom = np.sum(cmd_mat, axis=1)
    out_mat = np.zeros_like(cmd_mat)
    near_zero = np.abs(denom) <= 1e-6
    if np.any(~near_zero):
        out_mat[~near_zero, :] = fy_total_est[~near_zero, None] * cmd_mat[~near_zero, :] / denom[
            ~near_zero, None
        ]
    if np.any(near_zero):
        out_mat[near_zero, :] = fy_total_est[near_zero, None] / max(1, len(labels))
    return {label: out_mat[:, j] for j, label in enumerate(labels)}


def _compute_alpha_est(
    *,
    cfg: dict[str, Any],
    history: dict[str, Any],
    labels: list[str],
    dt: float,
) -> dict[str, np.ndarray] | None:
    if "ay_meas" not in history:
        return None
    ay_meas = np.asarray(history.get("ay_meas", []), dtype=float)
    if ay_meas.size == 0:
        return None

    vx = np.asarray(history.get("speed", []), dtype=float)
    yaw_rate = np.asarray(history.get("yaw_rate", []), dtype=float)
    if vx.size != ay_meas.size or yaw_rate.size != ay_meas.size:
        return None

    vehicle_config_path = _resolve_vehicle_config_path(cfg)
    vehicle_spec = load_param("vehicle_spec", vehicle_config_path)
    geometry = _cfg_section(vehicle_spec.get("geometry", {}))
    corner_offsets = _cfg_section(geometry.get("corner_offsets", {}))
    wheel_xy: dict[str, Tuple[float, float]] = {}
    for label in labels:
        off = _cfg_section(corner_offsets.get(label, {}))
        if "x" in off and "y" in off:
            wheel_xy[label] = (float(off["x"]), float(off["y"]))
    if set(wheel_xy.keys()) != set(labels):
        l_wb = float(geometry.get("L_wheelbase", 0.0))
        l_tr = float(geometry.get("L_track", 0.0))
        half_x = 0.5 * l_wb
        half_y = 0.5 * l_tr
        sign_map = {"FL": (1.0, 1.0), "FR": (1.0, -1.0), "RL": (-1.0, 1.0), "RR": (-1.0, -1.0)}
        for label in labels:
            sx, sy = sign_map.get(label, (0.0, 0.0))
            wheel_xy[label] = (half_x * sx, half_y * sy)

    estimator = SlipAngleEstimator(dt, wheel_xy, SlipAngleEstimatorOptions())
    alpha_out = {label: np.zeros_like(vx) for label in labels}

    delta_actual = _cfg_section(history.get("delta_actual", {}))
    if not delta_actual:
        return None

    delta_series: dict[str, np.ndarray] = {}
    for label in labels:
        series = np.asarray(delta_actual.get(label, []), dtype=float)
        if series.size != vx.size:
            return None
        delta_series[label] = series

    for k in range(vx.size):
        delta_map = {label: float(delta_series[label][k]) for label in labels}
        alpha_map, _vy_est = estimator.update(float(vx[k]), float(yaw_rate[k]), float(ay_meas[k]), delta_map)
        for label in labels:
            alpha_out[label][k] = float(alpha_map.get(label, 0.0))

    return alpha_out


def _run_and_get_history(config_path: Path, out_dir: Path, tag_prefix: str) -> Path:
    base_cfg = _load_yaml(config_path)
    cfg = copy.deepcopy(base_cfg)
    out_cfg = _cfg_section(cfg.setdefault("output", {}))
    out_cfg["plot"] = False
    out_cfg["show_plots"] = False
    out_cfg["save_npz"] = True
    out_cfg["save_json"] = False
    out_cfg["save_dir"] = str(out_dir)
    tag = str(out_cfg.get("tag", "run")).strip() or "run"
    out_cfg["tag"] = f"{tag_prefix}_{tag}"

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_cfg = out_dir / "_tmp_config_analyze_rls_scaling.yaml"
    _dump_yaml(cfg, tmp_cfg)

    script_path = Path(__file__).with_name("run_yaw_rate_study.py").resolve()
    cmd = [sys.executable, str(script_path), "--config", str(tmp_cfg)]
    subprocess.check_call(cmd, cwd=str(_repo_root()))

    return out_dir / f"history_{out_cfg['tag']}.npz"


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze phi scale and initial RLS gain.")
    parser.add_argument(
        "--history",
        type=str,
        default=None,
        help="Path to history_*.npz (if omitted, runs a sim with --config).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).with_name("config.yaml")),
        help="Base YAML config path (used when --history is omitted).",
    )
    parser.add_argument("--wheel", type=str, default=None, help="Wheel label (FL/FR/RL/RR).")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Where to place a temporary run (default: under config.output.save_dir).",
    )
    args = parser.parse_args()

    if args.history is None:
        base_cfg = _load_yaml(Path(args.config))
        base_out = Path(_cfg_section(base_cfg.get("output", {})).get("save_dir", "output"))
        if not base_out.is_absolute():
            base_out = _repo_root() / base_out
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(args.out_dir) if args.out_dir else (base_out / f"analysis_rls_scaling_{stamp}")
        if not out_dir.is_absolute():
            out_dir = _repo_root() / out_dir
        history_path = _run_and_get_history(Path(args.config), out_dir, "analysis")
    else:
        history_path = Path(args.history)
        if not history_path.is_absolute():
            history_path = _repo_root() / history_path

    if not history_path.exists():
        raise FileNotFoundError(f"history file not found: {history_path}")

    history, cfg, _metrics = _load_history(history_path)
    dt_cfg = float(_cfg_section(cfg.get("sim", {})).get("dt", 0.001))
    labels = list(_cfg_section(history.get("fy_cmd", {})).keys())
    if not labels:
        raise ValueError("history.fy_cmd missing wheel labels")
    wheel = _resolve_wheel(history, cfg, args.wheel)
    t = np.asarray(history.get("t", []), dtype=float)
    dt_hist = _median_dt(t)
    dt = dt_hist if dt_hist is not None else dt_cfg

    est_params = _rls_params(cfg)
    bias_enabled, bias_percent = _ff_bias(cfg)
    vehicle_config_path = _resolve_vehicle_config_path(cfg)
    steering_param = load_param("steering", vehicle_config_path)
    tire_param = load_param("tire", vehicle_config_path)
    lateral_param = _cfg_section(tire_param.get("lateral", {}))

    base_j = float(steering_param.get("J_cq", 0.0))
    base_b = float(steering_param.get("B_cq", 0.0))
    base_c_alpha = float(lateral_param.get("C_alpha", 0.0))
    gear_ratio = float(steering_param.get("gear_ratio", 1.0))
    trail = float(lateral_param.get("trail", 0.0))

    ff_base_b = base_b * (1.0 + bias_percent) if bias_enabled else base_b
    ff_base_c_alpha = base_c_alpha * (1.0 + bias_percent) if bias_enabled else base_c_alpha

    print(f"History: {history_path}")
    if dt_hist is not None and abs(dt_hist - dt_cfg) > max(1e-12, 1e-6 * abs(dt_cfg)):
        print(
            f"dt(cfg)={dt_cfg:g}  dt(from t-axis)={dt_hist:g}  -> using dt={dt:g} (warning: mismatch)"
        )
    else:
        print(f"dt={dt:g}  n={t.size}  wheel={wheel}  labels={labels}")
    print(
        "Estimator cfg:",
        f"b_start={est_params['b']['start_time']}, c_start={est_params['c_alpha']['start_time']},",
        f"b_decim={est_params['b']['sample_decimation']}, c_decim={est_params['c_alpha']['sample_decimation']},",
        f"b_align_source={est_params['b_align_source']}, c_alpha_source={est_params['c_alpha_source']}",
    )
    print(
        "RLS(B):",
        f"lambda={est_params['b']['lambda']}, p0={est_params['b']['p0']},",
        f"min_samples={est_params['b']['min_samples']}, decim={est_params['b']['sample_decimation']}",
    )
    print(
        "RLS(C):",
        f"lambda={est_params['c_alpha']['lambda']}, p0={est_params['c_alpha']['p0']},",
        f"min_samples={est_params['c_alpha']['min_samples']}, decim={est_params['c_alpha']['sample_decimation']}",
    )
    if int(est_params["b"]["min_samples"]) > 0:
        lb = float(est_params["b"]["min_samples"]) * float(est_params["b"]["sample_decimation"]) * float(dt)
        t0 = float(est_params["b"]["start_time"])
        print(
            f"min_samples(B) lower-bound additional time ≈ {lb:.3f}s; earliest reach ≈ {t0 + lb:.3f}s"
        )
    if int(est_params["c_alpha"]["min_samples"]) > 0:
        lb = (
            float(est_params["c_alpha"]["min_samples"])
            * float(est_params["c_alpha"]["sample_decimation"])
            * float(dt)
        )
        t0 = float(est_params["c_alpha"]["start_time"])
        print(
            f"min_samples(C) lower-bound additional time ≈ {lb:.3f}s; earliest reach ≈ {t0 + lb:.3f}s"
        )
    print(
        "FF model params:",
        f"J={base_j:g}, gear_ratio={gear_ratio:g}, trail={trail:g},",
        f"B_used0={ff_base_b:g} ({'+10%' if bias_enabled else 'nominal'}),",
        f"C_used0={ff_base_c_alpha:g} ({'+10%' if bias_enabled else 'nominal'})",
    )

    # ---- B estimator (phi = delta_dot) ----
    delta_dot = np.asarray(_cfg_section(history.get("delta_dot_actual", {})).get(wheel, []), dtype=float)
    if delta_dot.size == 0:
        delta_dot = np.asarray(history.get("delta_dot_actual", {}).get(wheel, []), dtype=float)
    delta_ddot = np.zeros_like(delta_dot)
    if delta_dot.size:
        delta_ddot[0] = delta_dot[0] / dt
        if delta_dot.size > 1:
            delta_ddot[1:] = np.diff(delta_dot) / dt

    print(_describe_stats("delta_dot [rad/s]", delta_dot))
    print(_describe_stats("delta_ddot [rad/s^2]", delta_ddot))

    b_samples = np.asarray(_cfg_section(history.get("B_samples", {})).get(wheel, []), dtype=int)
    if b_samples.size == 0:
        b_samples = np.asarray(history.get("B_samples", {}).get(wheel, []), dtype=int)
    b_first = _first_update_index(b_samples)
    b_min_idx = _min_samples_reached_index(
        t,
        start_time=float(est_params["b"]["start_time"]),
        samples=b_samples,
        min_samples=int(est_params["b"]["min_samples"]),
    )
    if b_min_idx is not None:
        t_min = float(t[b_min_idx]) if b_min_idx < t.size else float(b_min_idx) * dt
        print(
            f"B_cq min_samples reached: idx={b_min_idx} t={t_min:.3f}s"
            f" (min_samples={int(est_params['b']['min_samples'])})"
        )
    if b_first is None:
        print("B_cq: no RLS updates (B_samples never increases).")
    else:
        idx = b_first
        phi = float(delta_dot[idx]) if idx < delta_dot.size else 0.0

        fy_est = _compute_fy_wheel_est(history, labels)
        if est_params["b_align_source"] == "estimate" and fy_est is not None:
            align_term = float(trail * float(fy_est[wheel][idx]))
        elif est_params["b_align_source"] == "true":
            fy_true = np.asarray(history.get("fy_actual", {}).get(wheel, []), dtype=float)
            align_term = float(trail * float(fy_true[idx])) if idx < fy_true.size else 0.0
        else:
            align_term = 0.0

        t_motor_actual_series = np.asarray(
            _cfg_section(history.get("t_steer_actual", {})).get(wheel, []), dtype=float
        )
        if t_motor_actual_series.size == 0:
            t_motor_actual_series = np.asarray(history.get("t_steer_actual", {}).get(wheel, []), dtype=float)

        t_motor_cmd_series = np.asarray(_cfg_section(history.get("t_steer_cmd", {})).get(wheel, []), dtype=float)
        if t_motor_cmd_series.size == 0:
            t_motor_cmd_series = np.asarray(history.get("t_steer_cmd", {}).get(wheel, []), dtype=float)

        # run_yaw_rate_study.py updates B using *actual* steering axis torque; we store motor torque,
        # so reconstruct axis torque via gear ratio. If actual torque is missing, fall back to command.
        if idx < t_motor_actual_series.size:
            t_motor = float(t_motor_actual_series[idx])
            torque_source = "actual"
        elif idx < t_motor_cmd_series.size:
            t_motor = float(t_motor_cmd_series[idx])
            torque_source = "cmd"
        else:
            t_motor = 0.0
            torque_source = "missing"

        t_axis = float(t_motor) * gear_ratio
        y_b = float(t_axis - align_term - base_j * float(delta_ddot[idx]))

        step = _scalar_rls_first_step(
            theta0=float(ff_base_b),
            p0=float(est_params["b"]["p0"]),
            lam=float(est_params["b"]["lambda"]),
            y=y_b,
            phi=phi,
        )
        t_s = float(t[idx]) if idx < t.size else float(idx) * dt
        print(
            "B_cq first update:",
            f"idx={idx} t={t_s:.3f}s phi=delta_dot={phi:.3g},",
            f"torque={t_motor:.3g}N*m(motor,{torque_source}) -> {t_axis:.3g}N*m(axis),",
            f"p0*phi^2={step['p0_phi2']:.3g}, K0={step['K0']:.3g},",
            f"err0={step['err0']:.3g}, dtheta0={step['dtheta0']:.3g}",
        )

    # ---- C_alpha estimator (phi = alpha) ----
    c_samples = np.asarray(_cfg_section(history.get("C_alpha_samples", {})).get(wheel, []), dtype=int)
    if c_samples.size == 0:
        c_samples = np.asarray(history.get("C_alpha_samples", {}).get(wheel, []), dtype=int)
    c_first = _first_update_index(c_samples)
    c_min_idx = _min_samples_reached_index(
        t,
        start_time=float(est_params["c_alpha"]["start_time"]),
        samples=c_samples,
        min_samples=int(est_params["c_alpha"]["min_samples"]),
    )
    if c_min_idx is not None:
        t_min = float(t[c_min_idx]) if c_min_idx < t.size else float(c_min_idx) * dt
        print(
            f"C_alpha min_samples reached: idx={c_min_idx} t={t_min:.3f}s"
            f" (min_samples={int(est_params['c_alpha']['min_samples'])})"
        )

    fy_est = _compute_fy_wheel_est(history, labels)
    alpha_series = None
    fy_series = None
    if est_params["c_alpha_source"] == "estimate":
        alpha_est = _compute_alpha_est(cfg=cfg, history=history, labels=labels, dt=dt)
        if alpha_est is not None and fy_est is not None:
            alpha_series = alpha_est.get(wheel, None)
            fy_series = fy_est.get(wheel, None)
    else:
        alpha_series = np.asarray(history.get("alpha", {}).get(wheel, []), dtype=float)
        fy_series = np.asarray(history.get("fy_actual", {}).get(wheel, []), dtype=float)

    if alpha_series is None or fy_series is None or alpha_series.size == 0:
        print("C_alpha: missing signals to analyze (need alpha and Fy series).")
        return 0

    print(_describe_stats("alpha [rad]", alpha_series))
    if c_first is None:
        print("C_alpha: no RLS updates (C_alpha_samples never increases).")
        return 0

    idx = c_first
    phi = float(alpha_series[idx]) if idx < alpha_series.size else 0.0
    y = float(-fy_series[idx]) if idx < fy_series.size else 0.0
    step = _scalar_rls_first_step(
        theta0=float(ff_base_c_alpha),
        p0=float(est_params["c_alpha"]["p0"]),
        lam=float(est_params["c_alpha"]["lambda"]),
        y=y,
        phi=phi,
    )
    t_s = float(t[idx]) if idx < t.size else float(idx) * dt
    print(
        "C_alpha first update:",
        f"idx={idx} t={t_s:.3f}s phi=alpha={phi:.3g},",
        f"p0*phi^2={step['p0_phi2']:.3g}, K0={step['K0']:.3g},",
        f"err0={step['err0']:.3g}, dtheta0={step['dtheta0']:.3g}",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
