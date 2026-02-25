"""
Create a clean, static coordinate-frame figure for the vehicle.
"""

import argparse

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc


from vehicle_sim.models.vehicle_body.vehicle_body import VehicleBody
from vehicle_sim.models.visualization.vehicle_visualizer import VehicleVisualizer


def _arrow(ax, start: np.ndarray, end: np.ndarray, lw: float = 2.0) -> None:
    ax.annotate(
        '',
        xy=(end[0], end[1]),
        xytext=(start[0], start[1]),
        arrowprops=dict(
            arrowstyle='-|>',
            lw=lw,
            color='black',
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=9,
    )


def _label(ax, pos: np.ndarray, text: str, fontsize: int = 12,
           ha: str = 'center', va: str = 'center') -> None:
    ax.text(
        pos[0],
        pos[1],
        text,
        fontsize=fontsize,
        ha=ha,
        va=va,
        bbox=dict(facecolor='white', edgecolor='none', pad=0.2),
        zorder=10,
        clip_on=False,
    )


def _draw_axis(ax, origin: np.ndarray, direction: np.ndarray, length: float,
               label: str, label_offset: np.ndarray, lw: float = 2.0) -> None:
    end = origin + length * direction
    _arrow(ax, origin, end, lw=lw)
    _label(ax, end + label_offset, label, fontsize=12)


def _draw_body_frame(ax, origin: np.ndarray, axis_len: float) -> None:
    x_dir = np.array([1.0, 0.0])
    y_dir = np.array([0.0, 1.0])

    _draw_axis(
        ax,
        origin,
        x_dir,
        axis_len,
        r'$x_b$',
        label_offset=0.12 * y_dir,
        lw=2.2,
    )
    _draw_axis(
        ax,
        origin,
        y_dir,
        axis_len,
        r'$y_b$',
        label_offset=-0.12 * x_dir,
        lw=2.2,
    )

    ax.plot(origin[0], origin[1], marker='o', color='black', markersize=6, zorder=10)
    _label(ax, origin + np.array([0.12, -0.18]), 'CG', fontsize=11, ha='left', va='top')
    _label(ax, origin + np.array([-0.12, 0.18]), '{b}', fontsize=11, ha='right', va='bottom')


def _draw_wheel_frame(ax, center: np.ndarray, angle: float,
                      axis_len: float, outward_sign: float) -> None:
    x_dir = np.array([np.cos(angle), np.sin(angle)])
    y_dir = np.array([-np.sin(angle), np.cos(angle)])

    x_end = center + axis_len * x_dir
    y_end = center + axis_len * y_dir

    _arrow(ax, center, x_end, lw=1.8)
    _arrow(ax, center, y_end, lw=1.8)

    outward = np.array([0.0, outward_sign])
    x_label_pos = x_end + 0.12 * y_dir + 0.08 * outward
    y_label_pos = y_end - 0.12 * x_dir + 0.08 * outward

    _label(ax, x_label_pos, r'$x_{w_i}$', fontsize=10)
    _label(ax, y_label_pos, r'$y_{w_i}$', fontsize=10)


def _draw_delta_arc(ax, center: np.ndarray, delta: float,
                    radius: float, label: str) -> None:
    if abs(delta) < 1e-6:
        return

    theta1 = 0.0
    theta2 = np.degrees(delta)
    arc = Arc(
        (center[0], center[1]),
        width=2 * radius,
        height=2 * radius,
        angle=0.0,
        theta1=min(theta1, theta2),
        theta2=max(theta1, theta2),
        lw=1.3,
        color='black',
        zorder=8,
    )
    ax.add_patch(arc)

    mid_angle = np.deg2rad((theta1 + theta2) / 2.0)
    label_pos = center + (radius + 0.1) * np.array([np.cos(mid_angle), np.sin(mid_angle)])
    _label(ax, label_pos, label, fontsize=10)


def create_coordinate_frame_figure(out_path: str, dpi: int, steer_deg: float, save_pdf: bool) -> None:
    vehicle = VehicleBody()
    vehicle_params = {
        "L_wheelbase": vehicle.params.L_wheelbase,
        "L_track": vehicle.params.L_track,
    }

    visualizer = VehicleVisualizer(mode='2d', vehicle_params=vehicle_params)
    ax = visualizer.ax
    ax.clear()
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

    body = visualizer._draw_vehicle_body_2d(0.0, 0.0, 0.0)
    body.set_facecolor('white')
    body.set_edgecolor('black')
    body.set_linewidth(2.5)
    body.set_alpha(1.0)

    steer_rad = np.deg2rad(steer_deg)
    steering_angles = {
        'FL': steer_rad,
        'FR': steer_rad,
        'RL': 0.0,
        'RR': 0.0,
    }

    wheel_positions = visualizer._get_wheel_positions(0.0, 0.0, 0.0)
    for label, (wx, wy) in wheel_positions.items():
        wheel_angle = steering_angles.get(label, 0.0)
        visualizer._draw_wheel_2d(
            wx,
            wy,
            wheel_angle,
            label=None,
            facecolor='white',
            edgecolor='black',
            linewidth=2.0,
        )

    _draw_body_frame(ax, np.array([0.0, 0.0]), axis_len=0.9)

    for label, (wx, wy) in wheel_positions.items():
        wheel_angle = steering_angles.get(label, 0.0)
        outward_sign = 1.0 if wy >= 0.0 else -1.0
        _draw_wheel_frame(ax, np.array([wx, wy]), wheel_angle, axis_len=0.6, outward_sign=outward_sign)

    label_offsets = {
        'FL': np.array([0.35, 0.55]),
        'FR': np.array([0.35, -0.55]),
        'RL': np.array([-0.95, 0.55]),
        'RR': np.array([-0.95, -0.55]),
    }
    for label, (wx, wy) in wheel_positions.items():
        offset = label_offsets[label]
        _label(ax, np.array([wx, wy]) + offset, f'{label} $(x_i, y_i)$', fontsize=11)

    _draw_delta_arc(ax, np.array(wheel_positions['FL']), steer_rad, radius=0.35, label=r'$\delta_i$')
    _draw_delta_arc(ax, np.array(wheel_positions['FR']), steer_rad, radius=0.35, label=r'$\delta_i$')

    xs = [pos[0] for pos in wheel_positions.values()]
    ys = [pos[1] for pos in wheel_positions.values()]
    margin_x = 1.8
    margin_y = 1.4
    ax.set_xlim(min(xs) - margin_x, max(xs) + margin_x)
    ax.set_ylim(min(ys) - margin_y, max(ys) + margin_y)

    fig = ax.figure
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight', pad_inches=0.05)
    if save_pdf:
        pdf_path = os.path.splitext(out_path)[0] + '.pdf'
        fig.savefig(pdf_path, bbox_inches='tight', pad_inches=0.05)

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate a vehicle coordinate frame figure.')
    parser.add_argument('--out', type=str, default='vehicle_coordinate_frame.png',
                        help='Output image path (PNG).')
    parser.add_argument('--dpi', type=int, default=300, help='PNG DPI.')
    parser.add_argument('--steer-deg', type=float, default=15.0, help='Front wheel steering angle in degrees.')
    parser.add_argument('--pdf', action='store_true', help='Also save a PDF next to the PNG.')
    args = parser.parse_args()

    create_coordinate_frame_figure(
        out_path=args.out,
        dpi=args.dpi,
        steer_deg=args.steer_deg,
        save_pdf=args.pdf,
    )


if __name__ == '__main__':
    main()

