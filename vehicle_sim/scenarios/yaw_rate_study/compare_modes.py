#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

DEFAULT_MODES = ["ff", "ff_fb", "ff_ls", "ff_fb_ls"]


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "vehicle_sim").is_dir():
            return parent
    return here.parents[3]


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _dump_yaml(cfg: dict[str, Any], path: Path) -> None:
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def _resolve_runner() -> Path:
    candidates = [
        Path(__file__).with_name("run_yaw_rate_control_suite.py"),
        Path(__file__).with_name("run_yaw_rate_study.py"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    names = ", ".join(c.name for c in candidates)
    raise FileNotFoundError(f"No run script found next to {Path(__file__).name}: {names}")


def _run_mode(
    base_cfg: dict[str, Any],
    mode: str,
    script_path: Path,
    output_dir: Path,
    tag_prefix: str,
    quiet: bool,
) -> Path:
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("control", {})["mode"] = mode

    out_cfg = cfg.setdefault("output", {})
    out_cfg["save_dir"] = str(output_dir)
    out_cfg["tag"] = f"{tag_prefix}_{mode}"
    out_cfg["plot"] = False
    out_cfg["show_plots"] = False
    out_cfg["save_npz"] = True
    out_cfg["save_json"] = True

    cfg_path = output_dir / f"_tmp_config_{mode}.yaml"
    _dump_yaml(cfg, cfg_path)

    cmd = [sys.executable, str(script_path), "--config", str(cfg_path)]
    kwargs: dict[str, Any] = {"cwd": str(_repo_root())}
    if quiet:
        kwargs["stdout"] = subprocess.DEVNULL
        kwargs["stderr"] = subprocess.DEVNULL
    subprocess.check_call(cmd, **kwargs)

    return output_dir / f"history_{out_cfg['tag']}.npz"


def _load_history(npz_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    data = np.load(npz_path, allow_pickle=True)
    history = data["history"].item()
    cfg = data["config"].item()
    metrics = data["metrics"].item()
    return history, cfg, metrics


def _rmse(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(values ** 2)))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run yaw-rate modes and compare results.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).with_name("config.yaml")),
        help="Base YAML config path.",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default=",".join(DEFAULT_MODES),
        help="Comma-separated control modes.",
    )
    parser.add_argument("--tag", type=str, default=None, help="Output tag prefix.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory to store histories and plots.",
    )
    parser.add_argument("--show", action="store_true", help="Show comparison plots.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-mode run output.")
    args = parser.parse_args()

    base_cfg_path = Path(args.config)
    base_cfg = _load_yaml(base_cfg_path)

    modes = [m.strip().lower() for m in str(args.modes).split(",") if m.strip()]
    if not modes:
        modes = list(DEFAULT_MODES)

    repo_root = _repo_root()
    base_out_dir = Path(base_cfg.get("output", {}).get("save_dir", "output"))
    if not base_out_dir.is_absolute():
        base_out_dir = repo_root / base_out_dir

    tag_prefix = args.tag
    if tag_prefix is None:
        tag_prefix = str(base_cfg.get("output", {}).get("tag", "run")).strip() or "run"

    if args.out_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = base_out_dir / f"compare_{tag_prefix}_{stamp}"
    else:
        output_dir = Path(args.out_dir)
        if not output_dir.is_absolute():
            output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    script_path = _resolve_runner()

    histories: dict[str, dict[str, Any]] = {}
    metrics_by_mode: dict[str, dict[str, Any]] = {}

    for mode in modes:
        npz_path = _run_mode(
            base_cfg,
            mode,
            script_path,
            output_dir,
            tag_prefix,
            quiet=bool(args.quiet),
        )
        history, _cfg, metrics = _load_history(npz_path)
        histories[mode] = history
        metrics_by_mode[mode] = metrics

    summary: dict[str, dict[str, float]] = {}
    for mode, history in histories.items():
        yaw_err = np.asarray(history.get("yaw_error", []), dtype=float)
        mz_err = np.asarray(history.get("Mz_error", []), dtype=float)
        summary[mode] = {
            "yaw_error_rmse": _rmse(yaw_err),
            "yaw_error_peak_abs": float(np.max(np.abs(yaw_err))) if yaw_err.size else 0.0,
            "mz_error_rmse": _rmse(mz_err),
            "mz_error_peak_abs": float(np.max(np.abs(mz_err))) if mz_err.size else 0.0,
        }

    (output_dir / "compare_summary.json").write_text(
        json.dumps({"summary": summary, "metrics": metrics_by_mode}, indent=2),
        encoding="utf-8",
    )

    if not args.show:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    for mode, history in histories.items():
        t = np.asarray(history.get("t", []), dtype=float)
        yaw = np.asarray(history.get("yaw_rate", []), dtype=float)
        yaw_cmd = np.asarray(history.get("yaw_rate_cmd", []), dtype=float)
        yaw_err = np.asarray(history.get("yaw_error", []), dtype=float)

        if t.size and yaw_cmd.size == t.size:
            axes[0].plot(t, yaw_cmd, linestyle="--", alpha=0.35, color="gray")
        if t.size and yaw.size == t.size:
            axes[0].plot(t, yaw, label=mode)
        if t.size and yaw_err.size == t.size:
            axes[1].plot(t, yaw_err, label=mode)

    axes[0].set_ylabel("yaw rate [rad/s]")
    axes[0].set_title("Yaw-rate comparison")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("yaw error [rad/s]")
    axes[1].set_title("Yaw-rate error")
    axes[1].grid(True, alpha=0.3)

    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        axes[1].legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_dir / "compare_modes.png", dpi=180)
    if args.show:
        plt.show()

    print(f"Saved comparison to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
