"""Real-time diagnostic plotting for vehicle dynamics simulation."""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import time


class RealtimePlotter:
    """9-panel dashboard with rolling-buffer plots updated during simulation."""

    def __init__(self, figsize: Tuple[int, int] = (16, 10), max_points: int = 500, update_interval: int = 50):
        self.max_points = max_points
        self.update_interval = update_interval

        self.fig, self.axes = plt.subplots(3, 3, figsize=figsize)
        self.fig.suptitle('Vehicle Performance Metrics - Real-time', fontsize=16, fontweight='bold')
        plt.subplots_adjust(hspace=0.35, wspace=0.3)

        self.time_data: Deque[float] = deque(maxlen=max_points)
        self.data_buffers: Dict[str, Deque[float]] = {
            'velocity_x': deque(maxlen=max_points),
            'velocity_y': deque(maxlen=max_points),
            'velocity_total': deque(maxlen=max_points),
            'accel_x': deque(maxlen=max_points),
            'accel_y': deque(maxlen=max_points),
            'accel_total': deque(maxlen=max_points),
            'roll': deque(maxlen=max_points),
            'pitch': deque(maxlen=max_points),
            'yaw': deque(maxlen=max_points),
            'roll_rate': deque(maxlen=max_points),
            'pitch_rate': deque(maxlen=max_points),
            'yaw_rate': deque(maxlen=max_points),
            'tire_fx_fl': deque(maxlen=max_points),
            'tire_fx_fr': deque(maxlen=max_points),
            'tire_fy_fl': deque(maxlen=max_points),
            'tire_fy_fr': deque(maxlen=max_points),
            'susp_fl': deque(maxlen=max_points),
            'susp_fr': deque(maxlen=max_points),
            'susp_rl': deque(maxlen=max_points),
            'susp_rr': deque(maxlen=max_points),
            'steer_fl': deque(maxlen=max_points),
            'steer_fr': deque(maxlen=max_points),
            'slip_angle_fl': deque(maxlen=max_points),
            'slip_ratio_fl': deque(maxlen=max_points),
        }

        self.lines: Dict[str, object] = {}
        self._setup_plots()
        self.axes_twin = None

        self.start_time = time.time()
        self.current_time = 0.0

    def _setup_plots(self) -> None:
        # (0, 0) - Velocity
        ax = self.axes[0, 0]
        ax.set_title('Velocity', fontweight='bold')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Velocity [m/s]')
        ax.grid(True, alpha=0.3)
        self.lines['vx'], = ax.plot([], [], 'r-', label='Vx', linewidth=2)
        self.lines['vy'], = ax.plot([], [], 'b-', label='Vy', linewidth=2)
        self.lines['v_total'], = ax.plot([], [], 'k--', label='Total', linewidth=2)
        ax.legend(loc='upper right', fontsize=8)

        # (0, 1) - Acceleration
        ax = self.axes[0, 1]
        ax.set_title('Acceleration', fontweight='bold')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Acceleration [m/s]')
        ax.grid(True, alpha=0.3)
        self.lines['ax'], = ax.plot([], [], 'r-', label='Ax', linewidth=2)
        self.lines['ay'], = ax.plot([], [], 'b-', label='Ay', linewidth=2)
        self.lines['a_total'], = ax.plot([], [], 'k--', label='Total', linewidth=2)
        ax.legend(loc='upper right', fontsize=8)

        # (0, 2) - Attitude angles
        ax = self.axes[0, 2]
        ax.set_title('Attitude Angles', fontweight='bold')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Angle [deg]')
        ax.grid(True, alpha=0.3)
        self.lines['roll'], = ax.plot([], [], 'r-', label='Roll', linewidth=2)
        self.lines['pitch'], = ax.plot([], [], 'g-', label='Pitch', linewidth=2)
        self.lines['yaw'], = ax.plot([], [], 'b-', label='Yaw', linewidth=2)
        ax.legend(loc='upper right', fontsize=8)

        # (1, 0) - Angular rates
        ax = self.axes[1, 0]
        ax.set_title('Angular Rates', fontweight='bold')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Rate [deg/s]')
        ax.grid(True, alpha=0.3)
        self.lines['roll_rate'], = ax.plot([], [], 'r-', label='Roll rate', linewidth=2)
        self.lines['pitch_rate'], = ax.plot([], [], 'g-', label='Pitch rate', linewidth=2)
        self.lines['yaw_rate'], = ax.plot([], [], 'b-', label='Yaw rate', linewidth=2)
        ax.legend(loc='upper right', fontsize=8)

        # (1, 1) - Tire Fx
        ax = self.axes[1, 1]
        ax.set_title('Tire Longitudinal Forces', fontweight='bold')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Force [N]')
        ax.grid(True, alpha=0.3)
        self.lines['fx_fl'], = ax.plot([], [], 'r-', label='FL', linewidth=2)
        self.lines['fx_fr'], = ax.plot([], [], 'b-', label='FR', linewidth=2)
        ax.legend(loc='upper right', fontsize=8)

        # (1, 2) - Tire Fy
        ax = self.axes[1, 2]
        ax.set_title('Tire Lateral Forces', fontweight='bold')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Force [N]')
        ax.grid(True, alpha=0.3)
        self.lines['fy_fl'], = ax.plot([], [], 'r-', label='FL', linewidth=2)
        self.lines['fy_fr'], = ax.plot([], [], 'b-', label='FR', linewidth=2)
        ax.legend(loc='upper right', fontsize=8)

        # (2, 0) - Suspension stroke
        ax = self.axes[2, 0]
        ax.set_title('Suspension Stroke', fontweight='bold')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Stroke [m]')
        ax.grid(True, alpha=0.3)
        self.lines['susp_fl'], = ax.plot([], [], 'r-', label='FL', linewidth=1.5)
        self.lines['susp_fr'], = ax.plot([], [], 'b-', label='FR', linewidth=1.5)
        self.lines['susp_rl'], = ax.plot([], [], 'g-', label='RL', linewidth=1.5)
        self.lines['susp_rr'], = ax.plot([], [], 'm-', label='RR', linewidth=1.5)
        ax.legend(loc='upper right', fontsize=8)

        # (2, 1) - steering angle
        ax = self.axes[2, 1]
        ax.set_title('Steering Angle', fontweight='bold')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Angle [deg]')
        ax.grid(True, alpha=0.3)
        self.lines['steer_fl'], = ax.plot([], [], 'r-', label='FL', linewidth=2)
        self.lines['steer_fr'], = ax.plot([], [], 'b-', label='FR', linewidth=2)
        ax.legend(loc='upper right', fontsize=8)

        # (2, 2) - slip
        ax = self.axes[2, 2]
        ax.set_title('Tire Slip (FL)', fontweight='bold')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Slip', color='r')
        ax.grid(True, alpha=0.3)
        self.lines['slip_angle'], = ax.plot([], [], 'r-', label='Slip angle [deg]', linewidth=2)
        ax.tick_params(axis='y', labelcolor='r')
        ax.legend(loc='upper left', fontsize=8)

        ax2 = ax.twinx()
        ax2.set_ylabel('Slip Ratio [-]', color='b')
        self.lines['slip_ratio'], = ax2.plot([], [], 'b-', label='Slip ratio', linewidth=2)
        ax2.tick_params(axis='y', labelcolor='b')
        ax2.legend(loc='upper right', fontsize=8)
        self.axes_twin = ax2

    def update_data(self, t: float, vehicle_body, accel: Optional[Tuple[float, float]] = None) -> None:
        """Append one simulation step to all chart buffers."""
        self.current_time = t
        self.time_data.append(t)

        state = vehicle_body.state

        self.data_buffers['velocity_x'].append(state.velocity_x)
        self.data_buffers['velocity_y'].append(state.velocity_y)
        self.data_buffers['velocity_total'].append(np.hypot(state.velocity_x, state.velocity_y))

        if accel is not None:
            ax, ay = accel
        else:
            if len(self.data_buffers['velocity_x']) > 1:
                dt = 0.001  # fallback for simple tests
                vx = self.data_buffers['velocity_x']
                vy = self.data_buffers['velocity_y']
                ax = (vx[-1] - vx[-2]) / dt
                ay = (vy[-1] - vy[-2]) / dt
            else:
                ax, ay = 0.0, 0.0

        self.data_buffers['accel_x'].append(ax)
        self.data_buffers['accel_y'].append(ay)
        self.data_buffers['accel_total'].append(np.hypot(ax, ay))

        self.data_buffers['roll'].append(np.rad2deg(state.roll))
        self.data_buffers['pitch'].append(np.rad2deg(state.pitch))
        self.data_buffers['yaw'].append(np.rad2deg(state.yaw))
        self.data_buffers['roll_rate'].append(np.rad2deg(state.roll_rate))
        self.data_buffers['pitch_rate'].append(np.rad2deg(state.pitch_rate))
        self.data_buffers['yaw_rate'].append(np.rad2deg(state.yaw_rate))

        for label in ['FL', 'FR', 'RL', 'RR']:
            corner = vehicle_body.corners[label]

            if label in ['FL', 'FR']:
                self.data_buffers[f'tire_fx_{label.lower()}'].append(corner.state.F_x_tire)
                self.data_buffers[f'tire_fy_{label.lower()}'].append(corner.state.F_y_tire)

            susp_state = corner.suspension.get_state()
            self.data_buffers[f'susp_{label.lower()}'].append(susp_state['delta_s'])

            if label in ['FL', 'FR']:
                self.data_buffers[f'steer_{label.lower()}'].append(np.rad2deg(corner.state.steering_angle))

        fl_corner = vehicle_body.corners['FL']
        self.data_buffers['slip_angle_fl'].append(np.rad2deg(fl_corner.lateral_tire.state.slip_angle))
        self.data_buffers['slip_ratio_fl'].append(fl_corner.longitudinal_tire.state.slip_ratio)

    def update_plots(self) -> None:
        if len(self.time_data) == 0:
            return

        t_array = np.array(self.time_data)

        self.lines['vx'].set_data(t_array, np.array(self.data_buffers['velocity_x']))
        self.lines['vy'].set_data(t_array, np.array(self.data_buffers['velocity_y']))
        self.lines['v_total'].set_data(t_array, np.array(self.data_buffers['velocity_total']))
        self._auto_scale_axis(self.axes[0, 0], t_array, [
            self.data_buffers['velocity_x'],
            self.data_buffers['velocity_y'],
            self.data_buffers['velocity_total'],
        ])

        self.lines['ax'].set_data(t_array, np.array(self.data_buffers['accel_x']))
        self.lines['ay'].set_data(t_array, np.array(self.data_buffers['accel_y']))
        self.lines['a_total'].set_data(t_array, np.array(self.data_buffers['accel_total']))
        self._auto_scale_axis(self.axes[0, 1], t_array, [
            self.data_buffers['accel_x'],
            self.data_buffers['accel_y'],
            self.data_buffers['accel_total'],
        ])

        self.lines['roll'].set_data(t_array, np.array(self.data_buffers['roll']))
        self.lines['pitch'].set_data(t_array, np.array(self.data_buffers['pitch']))
        self.lines['yaw'].set_data(t_array, np.array(self.data_buffers['yaw']))
        self._auto_scale_axis(self.axes[0, 2], t_array, [
            self.data_buffers['roll'],
            self.data_buffers['pitch'],
            self.data_buffers['yaw'],
        ])

        self.lines['roll_rate'].set_data(t_array, np.array(self.data_buffers['roll_rate']))
        self.lines['pitch_rate'].set_data(t_array, np.array(self.data_buffers['pitch_rate']))
        self.lines['yaw_rate'].set_data(t_array, np.array(self.data_buffers['yaw_rate']))
        self._auto_scale_axis(self.axes[1, 0], t_array, [
            self.data_buffers['roll_rate'],
            self.data_buffers['pitch_rate'],
            self.data_buffers['yaw_rate'],
        ])

        self.lines['fx_fl'].set_data(t_array, np.array(self.data_buffers['tire_fx_fl']))
        self.lines['fx_fr'].set_data(t_array, np.array(self.data_buffers['tire_fx_fr']))
        self._auto_scale_axis(self.axes[1, 1], t_array, [
            self.data_buffers['tire_fx_fl'],
            self.data_buffers['tire_fx_fr'],
        ])

        self.lines['fy_fl'].set_data(t_array, np.array(self.data_buffers['tire_fy_fl']))
        self.lines['fy_fr'].set_data(t_array, np.array(self.data_buffers['tire_fy_fr']))
        self._auto_scale_axis(self.axes[1, 2], t_array, [
            self.data_buffers['tire_fy_fl'],
            self.data_buffers['tire_fy_fr'],
        ])

        self.lines['susp_fl'].set_data(t_array, np.array(self.data_buffers['susp_fl']))
        self.lines['susp_fr'].set_data(t_array, np.array(self.data_buffers['susp_fr']))
        self.lines['susp_rl'].set_data(t_array, np.array(self.data_buffers['susp_rl']))
        self.lines['susp_rr'].set_data(t_array, np.array(self.data_buffers['susp_rr']))
        self._auto_scale_axis(self.axes[2, 0], t_array, [
            self.data_buffers['susp_fl'],
            self.data_buffers['susp_fr'],
            self.data_buffers['susp_rl'],
            self.data_buffers['susp_rr'],
        ])

        self.lines['steer_fl'].set_data(t_array, np.array(self.data_buffers['steer_fl']))
        self.lines['steer_fr'].set_data(t_array, np.array(self.data_buffers['steer_fr']))
        self._auto_scale_axis(self.axes[2, 1], t_array, [
            self.data_buffers['steer_fl'],
            self.data_buffers['steer_fr'],
        ])

        self.lines['slip_angle'].set_data(t_array, np.array(self.data_buffers['slip_angle_fl']))
        self.lines['slip_ratio'].set_data(t_array, np.array(self.data_buffers['slip_ratio_fl']))
        self._auto_scale_axis(self.axes[2, 2], t_array, [self.data_buffers['slip_angle_fl']])

        if len(self.data_buffers['slip_ratio_fl']) > 0:
            data = np.array(self.data_buffers['slip_ratio_fl'])
            y_min, y_max = data.min(), data.max()
            margin = 0.1 * (y_max - y_min) if y_max > y_min else 0.1
            self.axes_twin.set_ylim(y_min - margin, y_max + margin)

        plt.pause(0.001)

    def _auto_scale_axis(self, ax, t_array, data_buffers: List[Deque[float]]) -> None:
        if len(t_array) > 0:
            ax.set_xlim(t_array[0], t_array[-1] + 0.1)

        all_data = []
        for buffer in data_buffers:
            if len(buffer) > 0:
                all_data.extend(buffer)

        if len(all_data) > 0:
            y_min, y_max = min(all_data), max(all_data)
            margin = 0.1 * (y_max - y_min) if y_max > y_min else 0.1
            ax.set_ylim(y_min - margin, y_max + margin)

    def clear_data(self) -> None:
        self.time_data.clear()
        for buffer in self.data_buffers.values():
            buffer.clear()

    def save_figure(self, filename: str) -> None:
        self.fig.savefig(filename, dpi=150, bbox_inches='tight')

    def show(self) -> None:
        plt.show()


