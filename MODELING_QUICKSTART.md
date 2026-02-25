# Modeling Quick Start

This project provides one beginner entry point for modeling-only simulation.

## 1) Install

```bash
pip install -r requirements.txt
```

## 2) Run

```bash
python run_modeling_quickstart.py
```

Optional:

```bash
python run_modeling_quickstart.py --duration 15 --dt 0.01
```

## 3) Output location and files

Each run creates one folder:

`outputs/modeling_quickstart/<run_id>/`

Generated files:

- `signals.csv`: time-series log for key vehicle signals
- `summary.txt`: short run summary
- `trajectory.png`: XY path image
- `states.png`: speed/yaw-rate/roll/pitch image
- `dashboard_view.png`: dashboard-style snapshot image

## 4) What this scenario does

- Accelerates the vehicle
- Applies a steering sweep
- Applies braking
- Adds a small front-wheel road bump

This is intended for first-time understanding of model behavior and output data format.
