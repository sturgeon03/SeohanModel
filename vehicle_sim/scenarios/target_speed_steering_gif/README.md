# Target Speed + Steering GIF Scenario

Runs a simple closed-loop speed controller and a per-wheel steering feedforward
controller, then exports a GIF using `VehicleVisualizer.animate_trajectory`.

## Run

```bash
python run_target_speed_steering_gif.py --config config.yaml
```

GIF output will be written under `output/` (configurable via `output.save_dir`).

## Key Config Fields

- `targets.speed_mps`: body longitudinal speed target [m/s]
- `targets.steering_deg` or `targets.steering_rad`: per-wheel steering targets
- `targets.steering_profile`: per-wheel sine inputs (set `enabled: true` to use)
- `steering_controller.align_torque_source`:
  - `estimate` (default): estimate aligning torque via slip-angle + linear tire model
  - `true`: use model aligning torque
  - `ignore`: set aligning torque to zero
- `steering_controller.feedback`: optional PID that tracks steering targets
- `speed_controller.gains`: PID gains for `SpeedControllerV2`
- `steering_controller.feedforward`: limits for steering torque FF
- `output.save_gif`: enable/disable GIF export
- `output.save_steering_plot`: save merged steering plot PNG

## Notes

- GIF export uses the Pillow writer (`pillow` package).
- The speed controller uses `vehicle.state.velocity_x` (body longitudinal speed).
