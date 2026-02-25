# Modeling Quick Start

This project provides one beginner entry point for modeling-only simulation.
It supports two input modes:

- `scenario`: built-in demo profile
- `cbnu`: replay references from real-vehicle TXT logs

## 1) Install

```bash
pip install -r requirements.txt
```

## 2) Run

Built-in scenario:

```bash
python run_modeling_quickstart.py
```

Optional scenario duration/dt:

```bash
python run_modeling_quickstart.py --duration 15 --dt 0.01
```

CBNU replay mode:

```bash
python run_modeling_quickstart.py --list-cbnu-files
python run_modeling_quickstart.py --input-source cbnu --cbnu-file 8
```

CBNU options:

```bash
python run_modeling_quickstart.py --input-source cbnu --cbnu-file "4WD Sine-10kmh.txt" --duration 10
python run_modeling_quickstart.py --input-source cbnu --cbnu-file 1 --drivetrain 2wd-front
```

## 3) Output location and files

Each run creates one folder:

`outputs/modeling_quickstart/<run_id>/`

Generated files:

- `signals.csv`: time-series log for key vehicle signals
- `summary.txt`: short run summary
- `trajectory.png`: XY path image
- `trajectory.gif`: XY path animation GIF
- `states.png`: speed/yaw-rate/roll/pitch image
- `dashboard_view.png`: dashboard-style snapshot image
- `input_tracking.png`: reference vs simulation tracking image (CBNU mode only)

## 4) What scenario mode does

- Accelerates the vehicle
- Applies a steering sweep
- Applies braking
- Adds a small front-wheel road bump

This is intended for first-time understanding of model behavior and output data format.

## 5) What CBNU mode does

- Loads one TXT file from `vehicle_sim/Data/*`
- Uses measured speed and steering as reference trajectories
- Generates actuator torques (`T_Drv`, `T_brk`, `T_steer`) to track references
- Saves tracking quality metrics in `summary.txt`
