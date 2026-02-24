"""
Active stabilizer test (torque control) using CM_Body_17.csv.

- Uses SuspensionModel per corner.
- Uses roll/roll_rate PD to generate left-right force difference.
- Converts desired force to actuator torque and feeds the suspension model.
- Shows plots after simulation (no CSV output).

NOTE: All body inputs are converted to deviation coordinates (initial value removed).
"""

import os
import sys

# Ensure vehicle_sim is on sys.path before imports.
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vehicle_sim.models.e_corner.suspension.suspension_model import SuspensionModel


# ==================== CONFIG ====================
DATA_FILE = "CM_Body_17.csv"

# PD gains (separate front/rear)
K_ROLL_F = 1.68e5  # [N/rad]
C_ROLL_F = 8.2e3  # [N*s/rad]
K_ROLL_R = 1.16e5 # [N/rad]
C_ROLL_R = 3.15e3  # [N*s/rad]

# Roll sign: roll(+) means left goes up. Use negative to reduce roll.
SIGN_ROLL = 1.0


def _median_dt(time: np.ndarray, fallback: float) -> float:
    if len(time) < 2:
        return fallback
    dt_vals = np.diff(time)
    dt_pos = dt_vals[dt_vals > 0]
    if len(dt_pos) == 0:
        return fallback
    return float(np.median(dt_pos))


def _force_to_torque(model: SuspensionModel, F_act: float) -> float:
    max_force = model.params.F_active_max
    if max_force is not None:
        F_act = float(np.clip(F_act, -max_force, max_force))
    return float(F_act * model.params.lead / (2.0 * np.pi * model.params.efficiency))


def _init_model_state(model: SuspensionModel, X_body0: np.ndarray, z_road0: float, z_u_abs0: float, z_u_dot0: float) -> None:
    model.state.z_u_abs = float(z_u_abs0)
    model.state.z_u_dot = float(z_u_dot0)
    # Use dt=0.0 to refresh internal forces without integrating.
    model.update(0.0, 0.0, X_body0, z_road0, 0.0)


def main() -> None:
    data_path = os.path.join(project_root, "vehicle_sim", "Data", DATA_FILE)
    config_path = os.path.join(project_root, "vehicle_sim", "models", "params", "vehicle_standard.yaml")

    df = pd.read_csv(data_path)

    # Required columns
    required_cols = [
        "Time",
        "cm_Heave", "cm_Roll", "cm_Pitch",
        "cm_Heave_dot", "cm_Roll_Dot", "cm_Pitch_Dot",
        "cm_RoadZFL", "cm_RoadZFR", "cm_RoadZRL", "cm_RoadZRR",
        "cm_UnsprungZFL", "cm_UnsprungZFR", "cm_UnsprungZRL", "cm_UnsprungZRR",
        "cm_FzFL", "cm_FzFR", "cm_FzRL", "cm_FzRR",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in {data_path}")

    time = df["Time"].to_numpy(dtype=float)
    dt_default = _median_dt(time, 0.001)

    # Body signals (absolute)
    heave = df["cm_Heave"].to_numpy(dtype=float)
    roll = df["cm_Roll"].to_numpy(dtype=float)
    pitch = df["cm_Pitch"].to_numpy(dtype=float)
    heave_dot = df["cm_Heave_dot"].to_numpy(dtype=float)
    roll_dot = df["cm_Roll_Dot"].to_numpy(dtype=float)
    pitch_dot = df["cm_Pitch_Dot"].to_numpy(dtype=float)

    # Convert to deviation coordinates (IMPORTANT)
    heave_rel = heave - heave[0]
    roll_rel = roll - roll[0]
    pitch_rel = pitch - pitch[0]
    heave_dot_rel = heave_dot - heave_dot[0]
    roll_dot_rel = roll_dot - roll_dot[0]
    pitch_dot_rel = pitch_dot - pitch_dot[0]

    # Road inputs
    road_z = {
        "FL": df["cm_RoadZFL"].to_numpy(dtype=float),
        "FR": df["cm_RoadZFR"].to_numpy(dtype=float),
        "RL": df["cm_RoadZRL"].to_numpy(dtype=float),
        "RR": df["cm_RoadZRR"].to_numpy(dtype=float),
    }

    # Unsprung positions
    z_u_abs = {
        "FL": df["cm_UnsprungZFL"].to_numpy(dtype=float),
        "FR": df["cm_UnsprungZFR"].to_numpy(dtype=float),
        "RL": df["cm_UnsprungZRL"].to_numpy(dtype=float),
        "RR": df["cm_UnsprungZRR"].to_numpy(dtype=float),
    }

    # Measured Fz
    fz_meas = {
        "FL": df["cm_FzFL"].to_numpy(dtype=float),
        "FR": df["cm_FzFR"].to_numpy(dtype=float),
        "RL": df["cm_FzRL"].to_numpy(dtype=float),
        "RR": df["cm_FzRR"].to_numpy(dtype=float),
    }

    corners = ["FL", "FR", "RL", "RR"]
    models = {corner: SuspensionModel(corner_id=corner, config_path=config_path) for corner in corners}

    # Initialize model states
    dt0 = time[1] - time[0] if len(time) > 1 else dt_default
    if dt0 <= 0.0:
        dt0 = dt_default if dt_default > 0.0 else 0.001
    X_body0 = np.array([
        heave_rel[0], roll_rel[0], pitch_rel[0],
        heave_dot_rel[0], roll_dot_rel[0], pitch_dot_rel[0],
    ])
    for corner in corners:
        z_u_dot0 = (z_u_abs[corner][1] - z_u_abs[corner][0]) / dt0 if len(time) > 1 else 0.0
        _init_model_state(models[corner], X_body0, road_z[corner][0], z_u_abs[corner][0], z_u_dot0)

    n_steps = len(time)
    fz_pred = {corner: np.zeros(n_steps) for corner in corners}
    f_active = {corner: np.zeros(n_steps) for corner in corners}
    delta_f = np.zeros(n_steps)
    delta_r = np.zeros(n_steps)

    # Store initial outputs
    for corner in corners:
        fz_pred[corner][0] = models[corner].state.F_z
        f_active[corner][0] = models[corner].state.F_active

    # ==================== Simulation loop ====================
    for i in range(1, n_steps):
        dt_i = time[i] - time[i - 1]
        if dt_i <= 0.0:
            dt_i = dt_default if dt_default > 0.0 else 0.001

        X_body = np.array([
            heave_rel[i], roll_rel[i], pitch_rel[i],
            heave_dot_rel[i], roll_dot_rel[i], pitch_dot_rel[i],
        ])

        # PD command (per axle)
        delta_f[i] = SIGN_ROLL * (K_ROLL_F * roll_rel[i] + C_ROLL_F * roll_dot_rel[i])
        delta_r[i] = SIGN_ROLL * (K_ROLL_R * roll_rel[i] + C_ROLL_R * roll_dot_rel[i])

        # Corner force split
        F_act = {
            "FL": 0.5 * delta_f[i],
            "FR": -0.5 * delta_f[i],
            "RL": 0.5 * delta_r[i],
            "RR": -0.5 * delta_r[i],
        }

        # Update each corner model
        for corner in corners:
            T_susp = _force_to_torque(models[corner], F_act[corner])
            models[corner].update(dt_i, T_susp, X_body, float(road_z[corner][i]), 0.0)
            fz_pred[corner][i] = models[corner].state.F_z
            f_active[corner][i] = models[corner].state.F_active

    # ==================== Final plots ====================
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    ax_front, ax_rear, ax_cmd = axes

    # Front Fz
    ax_front.plot(time, fz_meas["FL"], label="Fz meas FL", color="tab:blue", alpha=0.5)
    ax_front.plot(time, fz_pred["FL"], label="Fz model FL", color="tab:blue")
    ax_front.plot(time, fz_meas["FR"], label="Fz meas FR", color="tab:orange", alpha=0.5)
    ax_front.plot(time, fz_pred["FR"], label="Fz model FR", color="tab:orange")
    ax_front.set_ylabel("Fz Front [N]")
    ax_front.legend(loc="best")
    ax_front.grid(True, alpha=0.3)

    # Rear Fz
    ax_rear.plot(time, fz_meas["RL"], label="Fz meas RL", color="tab:green", alpha=0.5)
    ax_rear.plot(time, fz_pred["RL"], label="Fz model RL", color="tab:green")
    ax_rear.plot(time, fz_meas["RR"], label="Fz meas RR", color="tab:red", alpha=0.5)
    ax_rear.plot(time, fz_pred["RR"], label="Fz model RR", color="tab:red")
    ax_rear.set_ylabel("Fz Rear [N]")
    ax_rear.legend(loc="best")
    ax_rear.grid(True, alpha=0.3)

    # Command plot
    ax_cmd.plot(time, delta_f, label="DeltaF front", color="tab:purple")
    ax_cmd.plot(time, delta_r, label="DeltaF rear", color="tab:brown")
    ax_cmd.set_ylabel("DeltaF [N]")
    ax_cmd.set_xlabel("Time [s]")
    ax_cmd.legend(loc="best")
    ax_cmd.grid(True, alpha=0.3)

    ax_roll = ax_cmd.twinx()
    ax_roll.plot(time, roll_rel, label="roll (dev)", color="gray", alpha=0.5)
    ax_roll.set_ylabel("roll [rad]")

    # Fixed x limits
    ax_front.set_xlim(time[0], time[-1])

    # Y limits based on measured signals
    fz_front_min = min(fz_meas["FL"].min(), fz_meas["FR"].min())
    fz_front_max = max(fz_meas["FL"].max(), fz_meas["FR"].max())
    fz_rear_min = min(fz_meas["RL"].min(), fz_meas["RR"].min())
    fz_rear_max = max(fz_meas["RL"].max(), fz_meas["RR"].max())
    pad_front = max(100.0, 0.1 * (fz_front_max - fz_front_min))
    pad_rear = max(100.0, 0.1 * (fz_rear_max - fz_rear_min))
    ax_front.set_ylim(fz_front_min - pad_front, fz_front_max + pad_front)
    ax_rear.set_ylim(fz_rear_min - pad_rear, fz_rear_max + pad_rear)

    roll_min = float(np.min(roll_rel))
    roll_max = float(np.max(roll_rel))
    roll_pad = max(0.01, 0.1 * (roll_max - roll_min))
    ax_roll.set_ylim(roll_min - roll_pad, roll_max + roll_pad)

    plt.show()


if __name__ == "__main__":
    main()
