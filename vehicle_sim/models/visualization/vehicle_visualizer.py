"""Vehicle visualization helpers for 2D and 3D rendering.

The class is intentionally lightweight and depends only on the simulation state
returned by :class:`vehicle_sim.models.vehicle_body.vehicle_body.VehicleBody`.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class VehicleVisualizer:
    """Render a simple top-down and/or 3D view of the four-wheel vehicle."""

    def __init__(self, mode: str = '2d', vehicle_params: Optional[Dict] = None):
        """
        Args:
            mode: ``'2d'``, ``'3d'``, ``'both'`` or ``'animation'``
            vehicle_params: Optional visual parameters used for geometry sizing.
        """
        mode = mode.lower()
        if mode not in {'2d', '3d', 'both', 'animation'}:
            raise ValueError("mode must be '2d', '3d', 'both', or 'animation'")

        self.mode = mode
        self.vehicle_params = vehicle_params or self._default_params()

        # Geometry used only for plotting
        self.L = self.vehicle_params.get('L_wheelbase', 2.8)
        self.W = self.vehicle_params.get('L_track', 1.6)
        self.body_length = self.L * 1.5
        self.body_width = self.W * 1.2

        if mode == 'animation':
            self.fig = None
            self.ax = None
            self.ax_3d = None
        elif mode == '2d':
            self.fig, self.ax = plt.subplots(figsize=(12, 10))
            self.ax_3d = None
        elif mode == '3d':
            self.fig = plt.figure(figsize=(12, 10))
            self.ax = None
            self.ax_3d = self.fig.add_subplot(111, projection='3d')
        else:  # both
            self.fig = plt.figure(figsize=(16, 8))
            self.ax = self.fig.add_subplot(121)
            self.ax_3d = self.fig.add_subplot(122, projection='3d')

        self.vehicle_artists: List = []
        self.trajectory_points: List[Tuple[float, float]] = []

        if mode != 'animation':
            self._setup_plots()

    def _default_params(self) -> Dict:
        return {
            'L_wheelbase': 2.8,
            'L_track': 1.6,
            'h_CG': 0.5,
            'R_wheel': 0.316,
        }

    def _setup_plots(self) -> None:
        if self.ax is not None:
            self.ax.set_aspect('equal')
            self.ax.grid(True, alpha=0.3)
            self.ax.set_xlabel('X Position [m]', fontsize=12)
            self.ax.set_ylabel('Y Position [m]', fontsize=12)
            self.ax.set_title('Vehicle 2D View (Top-Down)', fontsize=14, fontweight='bold')

        if self.ax_3d is not None:
            self.ax_3d.set_xlabel('X [m]', fontsize=10)
            self.ax_3d.set_ylabel('Y [m]', fontsize=10)
            self.ax_3d.set_zlabel('Z [m]', fontsize=10)
            self.ax_3d.set_title('Vehicle 3D View', fontsize=14, fontweight='bold')

    def draw_vehicle_2d(
        self,
        x: float,
        y: float,
        yaw: float,
        steering_angles: Dict[str, float],
        vehicle_state: Optional[Dict] = None,
    ) -> None:
        """Draw only 2D view with wheel positions and steering angles."""
        if self.ax is None:
            return

        # Remove last frame artists
        for artist in self.vehicle_artists:
            artist.remove()
        self.vehicle_artists.clear()

        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)

        body_rect = self._draw_vehicle_body_2d(x, y, yaw)
        self.vehicle_artists.append(body_rect)

        windshield = self._draw_windshield_2d(x, y, yaw)
        if windshield is not None:
            self.vehicle_artists.append(windshield)

        wheel_positions = self._get_wheel_positions(x, y, yaw)
        for label, (wx, wy) in wheel_positions.items():
            delta = steering_angles.get(label, 0.0)
            wheel_angle = yaw + delta
            wheel_artists = self._draw_wheel_2d(wx, wy, wheel_angle, label)
            self.vehicle_artists.extend(wheel_artists)

        cg_marker = Circle((x, y), 0.1, color='red', zorder=10)
        self.ax.add_patch(cg_marker)
        self.vehicle_artists.append(cg_marker)

        arrow = self.ax.arrow(
            x,
            y,
            1.0 * cos_yaw,
            1.0 * sin_yaw,
            head_width=0.3,
            head_length=0.2,
            fc='blue',
            ec='blue',
            zorder=10,
        )
        self.vehicle_artists.append(arrow)

        self.trajectory_points.append((x, y))
        if len(self.trajectory_points) > 1:
            traj_x = [p[0] for p in self.trajectory_points]
            traj_y = [p[1] for p in self.trajectory_points]
            traj_line, = self.ax.plot(traj_x, traj_y, 'g--', alpha=0.5, linewidth=1.5)
            self.vehicle_artists.append(traj_line)

        if vehicle_state:
            text_obj = self.ax.text(
                0.02,
                0.98,
                self._create_info_text(vehicle_state),
                transform=self.ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                family='monospace',
            )
            self.vehicle_artists.append(text_obj)

        margin = 5
        self.ax.set_xlim(x - margin, x + margin)
        self.ax.set_ylim(y - margin, y + margin)

    def _draw_vehicle_body_2d(self, x: float, y: float, yaw: float) -> Polygon:
        L = self.body_length
        W = self.body_width

        # A simple stylized body shape centered at x,y in local vehicle frame
        front = L * 0.4
        rear = -L * 0.6
        n_front = 10
        theta_front = np.linspace(-np.pi / 3, np.pi / 3, n_front)
        radius = W * 0.5

        front_x = front + radius * np.cos(theta_front) - radius
        front_y = radius * np.sin(theta_front)
        x_coords = np.concatenate([front_x, [rear, rear], np.flip(front_x)])
        y_coords = np.concatenate([front_y, [W / 2, -W / 2], np.flip(-front_y)])
        local_points = np.column_stack([x_coords, y_coords])

        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        R = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
        global_points = (R @ local_points.T).T + np.array([x, y])

        body = Polygon(
            global_points,
            closed=True,
            facecolor='lightblue',
            edgecolor='darkblue',
            linewidth=2,
            alpha=0.7,
            zorder=5,
        )
        self.ax.add_patch(body)
        return body

    def _draw_windshield_2d(self, x: float, y: float, yaw: float) -> Polygon:
        L = self.body_length
        W = self.body_width

        ws_x_local = L * 0.2
        ws_y_local = W * 0.3

        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)

        ws_points = np.array([
            [ws_x_local, ws_y_local],
            [ws_x_local + 0.3, ws_y_local],
            [ws_x_local + 0.3, -ws_y_local],
            [ws_x_local, -ws_y_local],
        ])

        R = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
        ws_global = (R @ ws_points.T).T + np.array([x, y])

        windshield = Polygon(
            ws_global,
            closed=True,
            facecolor='cyan',
            edgecolor='blue',
            linewidth=1,
            alpha=0.5,
            zorder=6,
        )
        self.ax.add_patch(windshield)
        return windshield

    def _get_wheel_positions(self, x: float, y: float, yaw: float) -> Dict[str, Tuple[float, float]]:
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)

        a = self.L / 2
        b = self.L / 2
        track = self.W / 2

        return {
            'FL': (x + cos_yaw * a - sin_yaw * track, y + sin_yaw * a + cos_yaw * track),
            'FR': (x + cos_yaw * a + sin_yaw * track, y + sin_yaw * a - cos_yaw * track),
            'RL': (x - cos_yaw * b - sin_yaw * track, y - sin_yaw * b + cos_yaw * track),
            'RR': (x - cos_yaw * b + sin_yaw * track, y - sin_yaw * b - cos_yaw * track),
        }

    def _draw_wheel_2d(self, wx: float, wy: float, wheel_angle: float, label: Optional[str] = None,
                       facecolor: str = 'black', edgecolor: str = 'gray', linewidth: float = 1.5,
                       label_color: str = 'yellow'):
        wheel_length = 0.4
        wheel_width = 0.2

        cos_a = np.cos(wheel_angle)
        sin_a = np.sin(wheel_angle)

        half_l = wheel_length / 2
        half_w = wheel_width / 2
        local_corners = np.array([[-half_l, -half_w], [half_l, -half_w], [half_l, half_w], [-half_l, half_w]])

        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        global_corners = (R @ local_corners.T).T + np.array([wx, wy])

        wheel = Polygon(
            global_corners,
            closed=True,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            zorder=7,
        )
        self.ax.add_patch(wheel)

        artists = [wheel]
        if label:
            artists.append(
                self.ax.text(
                    wx,
                    wy,
                    label,
                    fontsize=8,
                    color=label_color,
                    ha='center',
                    va='center',
                    zorder=8,
                    fontweight='bold',
                )
            )
        return artists

    def draw_vehicle_3d(
        self,
        x: float,
        y: float,
        heave: float,
        roll: float,
        pitch: float,
        yaw: float,
        steering_angles: Dict[str, float],
        suspension_states: Optional[Dict] = None,
    ) -> None:
        if self.ax_3d is None:
            return

        self.ax_3d.clear()
        self._draw_vehicle_body_3d(x, y, heave, roll, pitch, yaw)
        self._draw_wheels_3d(
            x,
            y,
            heave,
            roll,
            pitch,
            yaw,
            steering_angles,
            suspension_states,
        )

        margin = 3
        self.ax_3d.set_xlim(x - margin, x + margin)
        self.ax_3d.set_ylim(y - margin, y + margin)
        self.ax_3d.set_zlim(0, 2)
        self.ax_3d.set_xlabel('X [m]')
        self.ax_3d.set_ylabel('Y [m]')
        self.ax_3d.set_zlabel('Z [m]')
        self.ax_3d.set_title('Vehicle 3D View')

    def _draw_vehicle_body_3d(
        self,
        x: float,
        y: float,
        z: float,
        roll: float,
        pitch: float,
        yaw: float,
    ) -> None:
        L = self.body_length
        W = self.body_width
        H = 0.8

        vertices = np.array([
            [-L * 0.6, -W / 2, 0],
            [L * 0.4, -W / 2, 0],
            [L * 0.4, W / 2, 0],
            [-L * 0.6, W / 2, 0],
            [-L * 0.6, -W / 2, H],
            [L * 0.4, -W / 2, H],
            [L * 0.4, W / 2, H],
            [-L * 0.6, W / 2, H],
        ])

        R = self._rotation_matrix_3d(roll, pitch, yaw)
        rotated = (R @ vertices.T).T

        z_abs = self.vehicle_params.get('h_CG', 0.5) + z
        translated = rotated + np.array([x, y, z_abs])

        faces = [
            [translated[0], translated[1], translated[5], translated[4]],
            [translated[2], translated[3], translated[7], translated[6]],
            [translated[1], translated[2], translated[6], translated[5]],
            [translated[3], translated[0], translated[4], translated[7]],
            [translated[4], translated[5], translated[6], translated[7]],
            [translated[0], translated[3], translated[2], translated[1]],
        ]

        body = Poly3DCollection(
            faces,
            facecolors='lightblue',
            edgecolors='darkblue',
            linewidths=1.5,
            alpha=0.7,
        )
        self.ax_3d.add_collection3d(body)

        ws_vertices = np.array([
            [L * 0.2, -W * 0.3, H * 0.5],
            [L * 0.35, -W * 0.3, H * 0.9],
            [L * 0.35, W * 0.3, H * 0.9],
            [L * 0.2, W * 0.3, H * 0.5],
        ])
        ws_rotated = (R @ ws_vertices.T).T + np.array([x, y, z_abs])
        windshield = Poly3DCollection(
            [ws_rotated],
            facecolors='cyan',
            edgecolors='blue',
            linewidths=1,
            alpha=0.6,
        )
        self.ax_3d.add_collection3d(windshield)

    def _draw_wheels_3d(
        self,
        x: float,
        y: float,
        z: float,
        roll: float,
        pitch: float,
        yaw: float,
        steering_angles: Dict[str, float],
        suspension_states: Optional[Dict] = None,
    ) -> None:
        R_body = self._rotation_matrix_3d(roll, pitch, yaw)
        z_abs = self.vehicle_params.get('h_CG', 0.5) + z

        a = self.L / 2
        b = self.L / 2
        track = self.W / 2

        wheel_configs = {
            'FL': (a, track),
            'FR': (a, -track),
            'RL': (-b, track),
            'RR': (-b, -track),
        }

        wheel_radius = self.vehicle_params.get('R_wheel', 0.316)
        wheel_width = 0.2

        for label, (lx, ly) in wheel_configs.items():
            delta_s = 0.0
            if suspension_states and label in suspension_states:
                delta_s = float(suspension_states[label].get('delta_s', 0.0))

            wheel_local = np.array([lx, ly, -self.vehicle_params.get('h_CG', 0.5) + delta_s])
            wheel_global = R_body @ wheel_local + np.array([x, y, z_abs])

            delta = steering_angles.get(label, 0.0)
            wheel_yaw = yaw + delta
            self._draw_single_wheel_3d(wheel_global, wheel_yaw, wheel_radius, wheel_width, label)

    def _draw_single_wheel_3d(
        self,
        position: np.ndarray,
        yaw: float,
        radius: float,
        width: float,
        label: str,
    ) -> None:
        """Draw wheel as a simple cylinder with two edge circles and side faces."""
        wx, wy, wz = position
        theta = np.linspace(0.0, 2.0 * np.pi, 24)

        ax = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        ay = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
        az = np.array([0.0, 0.0, 1.0])

        base = np.array([wx, wy, wz])
        circle_center_right = base + (width / 2) * ay
        circle_center_left = base - (width / 2) * ay

        ring_r = np.outer(np.cos(theta), ax) * radius + np.outer(np.sin(theta), az) * radius
        right_ring = circle_center_right + ring_r
        left_ring = circle_center_left + ring_r

        self.ax_3d.plot(right_ring[:, 0], right_ring[:, 1], right_ring[:, 2], color='k', linewidth=1)
        self.ax_3d.plot(left_ring[:, 0], left_ring[:, 1], left_ring[:, 2], color='k', linewidth=1)

        for i in range(len(theta) - 1):
            face = [
                left_ring[i],
                left_ring[i + 1],
                right_ring[i + 1],
                right_ring[i],
            ]
            self.ax_3d.add_collection3d(
                Poly3DCollection([face], facecolors='black', edgecolors='dimgray', alpha=0.8, linewidths=0.2)
            )

        self.ax_3d.scatter([wx], [wy], [wz], color='red', s=10)
        self.ax_3d.text(wx, wy, wz, label, fontsize=8, color='yellow')

    @staticmethod
    def _rotation_matrix_3d(roll: float, pitch: float, yaw: float) -> np.ndarray:
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        return np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ])

    @staticmethod
    def _create_info_text(state: Dict) -> str:
        vx = state.get('velocity_x', 0.0)
        vy = state.get('velocity_y', 0.0)
        v_total = np.hypot(vx, vy) * 3.6
        yaw_rate = state.get('yaw_rate', 0.0)
        roll = state.get('roll', 0.0)
        pitch = state.get('pitch', 0.0)

        return (
            f"Vehicle State:\n"
            f"Speed: {v_total:6.2f} km/h\n"
            f"Vx:    {vx:6.2f} m/s\n"
            f"Vy:    {vy:6.2f} m/s\n"
            f"Yaw rate: {np.rad2deg(yaw_rate):6.2f} deg/s\n"
            f"Roll:  {np.rad2deg(roll):6.2f} deg\n"
            f"Pitch: {np.rad2deg(pitch):6.2f} deg\n"
        )

    def clear_trajectory(self) -> None:
        self.trajectory_points.clear()

    def update(self, vehicle_body, corner_states: Optional[Dict] = None) -> None:
        """Update plots directly from a :class:`VehicleBody` instance."""
        state = vehicle_body.state

        steering_angles = {
            label: vehicle_body.corners[label].state.steering_angle for label in ['FL', 'FR', 'RL', 'RR']
        }

        vehicle_state = {
            'velocity_x': state.velocity_x,
            'velocity_y': state.velocity_y,
            'yaw_rate': state.yaw_rate,
            'roll': state.roll,
            'pitch': state.pitch,
        }

        if self.ax is not None:
            self.draw_vehicle_2d(state.x, state.y, state.yaw, steering_angles, vehicle_state)

        if self.ax_3d is not None:
            if corner_states is None:
                susp_states = {
                    label: vehicle_body.corners[label].suspension.get_state() for label in ['FL', 'FR', 'RL', 'RR']
                }
            else:
                susp_states = corner_states

            self.draw_vehicle_3d(
                state.x,
                state.y,
                state.heave,
                state.roll,
                state.pitch,
                state.yaw,
                steering_angles,
                susp_states,
            )

        plt.pause(0.001)

    def save_frame(self, filename: str) -> None:
        if self.fig is None:
            raise RuntimeError('No figure created for animation mode')
        self.fig.savefig(filename, dpi=150, bbox_inches='tight')

    def animate_trajectory(
        self,
        time: np.ndarray,
        x_log: np.ndarray,
        y_log: np.ndarray,
        yaw_log: np.ndarray,
        delta_log: Dict[str, np.ndarray],
        velocity_x_log: np.ndarray,
        velocity_y_log: np.ndarray,
        yaw_rate_log: np.ndarray,
        labels: List[str],
        stride: int = 20,
        save_gif: bool = False,
        gif_filename: str = 'vehicle_animation.gif',
        max_gif_frames: int = 500,
    ) -> None:
        if len(time) < 2:
            return

        if self.mode == 'animation':
            raise RuntimeError('animate_trajectory is for 2D/3D display modes')

        if stride < 1:
            stride = 1

        frame_indices = np.arange(0, len(time), stride)
        if save_gif and len(frame_indices) > max_gif_frames:
            stride = max(1, len(time) // max_gif_frames)
            frame_indices = np.arange(0, len(time), stride)

        fig = plt.figure(figsize=(15, 10))
        view_ax = plt.subplot2grid((4, 3), (0, 0), rowspan=4, colspan=2)
        zoom_ax = plt.subplot2grid((4, 3), (0, 2))
        steer_ax = plt.subplot2grid((4, 3), (1, 2), rowspan=2)
        speed_ax = plt.subplot2grid((4, 3), (3, 2))

        # Global view settings
        min_x, max_x = float(np.min(x_log)), float(np.max(x_log))
        min_y, max_y = float(np.min(y_log)), float(np.max(y_log))
        span = max(max_x - min_x, max_y - min_y, 10.0)
        pad = 0.15 * span
        view_ax.set_aspect('equal')
        view_ax.set_xlim(min_x - pad, max_x + pad)
        view_ax.set_ylim(min_y - pad, max_y + pad)
        view_ax.set_xlabel('X [m]')
        view_ax.set_ylabel('Y [m]')
        view_ax.set_title('Vehicle Animation (Global View)', fontsize=14, fontweight='bold')
        view_ax.grid(True, alpha=0.3)

        # Zoom view settings
        vehicle_length = self.L * 1.2
        vehicle_width = self.W * 0.9
        zoom_span = max(vehicle_length * 1.2, vehicle_width * 2.0, 3.0)
        zoom_ax.set_aspect('equal')
        zoom_ax.set_xlim(-zoom_span, zoom_span)
        zoom_ax.set_ylim(-zoom_span, zoom_span)
        zoom_ax.set_xlabel('X [m]', fontsize=9)
        zoom_ax.set_ylabel('Y [m]', fontsize=9)
        zoom_ax.set_title('Zoom View', fontsize=10, fontweight='bold')
        zoom_ax.grid(True, alpha=0.3)

        steer_deg_log = {label: np.rad2deg(delta_log[label]) for label in labels}
        max_steer_deg = max(max(1.0, float(np.max(np.abs(steer_deg_log[label])))) for label in labels)

        steer_ax.set_title('Steering Angles', fontsize=10, fontweight='bold')
        steer_ax.set_xlim(float(time[0]), float(time[-1]))
        steer_ax.set_ylim(-1.1 * max_steer_deg, 1.1 * max_steer_deg)
        steer_ax.set_xlabel('Time [s]', fontsize=9)
        steer_ax.set_ylabel('Steer [deg]', fontsize=9)
        steer_ax.grid(True, alpha=0.3)

        speed_ax.set_title('Vehicle Speed', fontsize=10)
        speed_ax.set_xlim(-1.2, 1.2)
        speed_ax.set_ylim(-1.2, 1.2)
        speed_ax.set_aspect('equal')
        speed_ax.axis('off')

        fig.tight_layout()

        def affine_transform(x_list, y_list, angle, translation=(0.0, 0.0)):
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            x_new = [x * cos_a - y * sin_a + translation[0] for x, y in zip(x_list, y_list)]
            y_new = [x * sin_a + y * cos_a + translation[1] for x, y in zip(x_list, y_list)]
            return x_new, y_new

        def draw_vehicle(ax, center_x, center_y, yaw, steer_map, draw_marker=True):
            artists = []
            vl, vw = vehicle_length, vehicle_width
            body_x = [-0.4 * vl, -0.4 * vl, 0.4 * vl, 0.4 * vl, -0.4 * vl]
            body_y = [-0.5 * vw, 0.5 * vw, 0.5 * vw, -0.5 * vw, -0.5 * vw]
            body_x_g, body_y_g = affine_transform(body_x, body_y, yaw, (center_x, center_y))
            line, = ax.plot(body_x_g, body_y_g, 'black', linewidth=1.5, zorder=3)
            artists.append(line)
            fill = ax.fill(body_x_g, body_y_g, color='lightblue', alpha=0.7, zorder=2)
            artists.extend(fill)

            wl, ww = 0.5, 0.3
            wheel_shape_x = [-0.5 * wl, -0.5 * wl, 0.5 * wl, 0.5 * wl, -0.5 * wl]
            wheel_shape_y = [0.0, 0.5 * ww, 0.5 * ww, -0.5 * ww, -0.5 * ww, 0.0]
            wheel_centers = {
                'FL': (0.35 * vl, 0.35 * vw, steer_map['FL']),
                'FR': (0.35 * vl, -0.35 * vw, steer_map['FR']),
                'RL': (-0.35 * vl, 0.35 * vw, steer_map['RL']),
                'RR': (-0.35 * vl, -0.35 * vw, steer_map['RR']),
            }
            for _, (cx, cy, delta) in wheel_centers.items():
                wx_local, wy_local = affine_transform(wheel_shape_x, wheel_shape_y, delta, (cx, cy))
                wx_global, wy_global = affine_transform(wx_local, wy_local, yaw, (center_x, center_y))
                fill = ax.fill(wx_global, wy_global, color='black', zorder=4)
                artists.extend(fill)

            if draw_marker:
                marker = Circle((center_x, center_y), radius=0.3, fc='red', ec='darkred', linewidth=1, zorder=5)
                artists.append(ax.add_patch(marker))

            return artists

        def draw_yaw_arrow(ax, center_x, center_y, yaw_angle, length, head_w, head_l):
            return ax.arrow(
                center_x,
                center_y,
                length * np.cos(yaw_angle),
                length * np.sin(yaw_angle),
                head_width=head_w,
                head_length=head_l,
                fc='tab:orange',
                ec='tab:orange',
                zorder=6,
            )

        frames = []
        colors = {'FL': 'tab:blue', 'FR': 'tab:orange', 'RL': 'tab:green', 'RR': 'tab:red'}

        for idx in frame_indices:
            frame = []

            x, y, yaw = x_log[idx], y_log[idx], yaw_log[idx]
            vx, vy = velocity_x_log[idx], velocity_y_log[idx]
            v = float(np.hypot(vx, vy))

            steer_map = {
                'FL': float(delta_log['FL'][idx]),
                'FR': float(delta_log['FR'][idx]),
                'RL': float(delta_log['RL'][idx]),
                'RR': float(delta_log['RR'][idx]),
            }

            # full trajectory and vehicle body
            line, = view_ax.plot(x_log[:idx + 1], y_log[:idx + 1], 'purple', linewidth=1.5, alpha=0.7)
            frame.append(line)
            frame.extend(draw_vehicle(view_ax, x, y, yaw, steer_map))
            frame.append(draw_yaw_arrow(view_ax, x, y, yaw, 0.6 * vehicle_length, 0.3, 0.4))

            text = f"t={time[idx]:.2f}s | v={v:.2f}m/s | yaw_rate={yaw_rate_log[idx]:.3f}rad/s"
            txt = view_ax.text(
                0.5,
                0.02,
                text,
                ha='center',
                transform=view_ax.transAxes,
                fontsize=11,
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            )
            frame.append(txt)

            # zoom view
            frame.extend(draw_vehicle(zoom_ax, 0.0, 0.0, yaw, steer_map, draw_marker=False))
            frame.append(draw_yaw_arrow(zoom_ax, 0.0, 0.0, yaw, 0.6 * vehicle_length, 0.2, 0.3))

            # steering history
            for label in labels:
                steer_deg = steer_deg_log[label]
                color = colors.get(label, 'tab:blue')
                line, = steer_ax.plot(time[:idx + 1], steer_deg[:idx + 1], linewidth=1.2, color=color)
                frame.append(line)
                marker = steer_ax.plot(time[idx], steer_deg[idx], 'o', color=color, markersize=3)
                frame.extend(marker)

            max_speed = 25.0
            pie_rate = 0.75
            pie_start = 225
            v_norm = min(v, max_speed)
            pie_obj, _ = speed_ax.pie(
                [v_norm * pie_rate, (max_speed - v_norm) * pie_rate, max_speed * (1 - pie_rate)],
                startangle=pie_start,
                counterclock=False,
                colors=['black', 'lightgray', 'white'],
                wedgeprops={'linewidth': 0, 'edgecolor': 'white', 'width': 0.4},
            )
            frame.extend(pie_obj)
            frame.append(speed_ax.text(0, -1, f"{v:.1f} m/s", size=12, ha='center', va='center', fontfamily='monospace'))
            frames.append(frame)

        dt = float(time[1] - time[0])
        interval_ms = max(1, int(dt * stride * 1000 * 0.5))
        ani = animation.ArtistAnimation(fig, frames, interval=interval_ms, blit=True, repeat=True)

        if save_gif:
            ani.save(gif_filename, writer='pillow', fps=1000 / interval_ms, dpi=60)

        plt.show(block=True)

    def show(self) -> None:
        if self.fig is None:
            raise RuntimeError('No figure created for animation mode')
        plt.show()
