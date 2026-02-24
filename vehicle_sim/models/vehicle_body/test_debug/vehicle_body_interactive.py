"""
Vehicle body interactive visualizer.

시각화 항목
- 3D 바디 포즈, 2D 궤적(상단평면)
- 차체 상태: vx, vy, roll, pitch, yaw, yaw_rate
- 코너 상태(4바퀴): F_s, F_x, F_y, F_z, steering_angle, omega_wheel,
  slip_ratio, slip_angle, aligning_torque, clamp_force

조작
- 각 바퀴별 슬라이더: T_steer, T_brk, T_Drv, T_susp
- 슬라이더 조정 시 실시간으로 애니메이션 업데이트

실행: python vehicle_sim/models/vehicle_body/test_debug/vehicle_body_interactive.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.widgets import Slider
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle
from matplotlib.transforms import Affine2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from vehicle_sim.models.vehicle_body.vehicle_body import VehicleBody

# Simulation / rendering parameters
SIM_DT = 0.01  # physics integration step [s]
RENDER_FPS_TARGET = 25.0  # render at ~20~30 Hz for responsiveness
SIM_STEPS_PER_FRAME = max(1, int(round((1.0 / RENDER_FPS_TARGET) / SIM_DT)))
RENDER_INTERVAL_MS = int(round(SIM_STEPS_PER_FRAME * SIM_DT * 1000.0))

# History is recorded at render rate (downsampled for speed).
HISTORY_SECONDS = 15.0
HIST_DT = SIM_STEPS_PER_FRAME * SIM_DT
HIST_LEN = int(HISTORY_SECONDS / HIST_DT)

# Slider ranges
STEER_RANGE = (-50.0, 50.0)      # N*m
BRAKE_RANGE = (0.0, 400.0)       # motor torque, clamp internally limited
DRIVE_RANGE = (-800.0, 800.0)    # N*m
SUSP_RANGE = (-4000.0, 4000.0)   # N

# VehicleBody.wheel_labels order is ["FL", "FR", "RR", "RL"].
# Keep the same order here so wheel_idx -> label mapping stays consistent.
WHEEL_LABELS = ["FL", "FR", "RR", "RL"]

# Visualization geometry (approximate, for readability)
BODY_OVERHANG = 0.9      # [m] front/rear overhang beyond axles
BODY_WIDTH_PAD = 0.4     # [m] extra width beyond track
WHEEL_LENGTH = 0.60      # [m]
WHEEL_WIDTH = 0.25       # [m]
CHASSIS_CLEARANCE = 0.25  # [m] ground to chassis bottom (approx)
CHASSIS_HEIGHT = 0.40     # [m] chassis thickness (approx)

ANIM = None  # keep a global reference (prevents GC in some IDE/non-blocking backends)


def build_vehicle():
    return VehicleBody(config_path=None)


def init_history():
    t_axis = np.linspace(-HISTORY_SECONDS, 0, HIST_LEN)
    per_corner = lambda: {lbl: np.zeros(HIST_LEN) for lbl in WHEEL_LABELS}
    return {
        "t": t_axis,
        "vx": np.zeros(HIST_LEN),
        "vy": np.zeros(HIST_LEN),
        "roll": np.zeros(HIST_LEN),
        "pitch": np.zeros(HIST_LEN),
        "yaw": np.zeros(HIST_LEN),
        "yaw_rate": np.zeros(HIST_LEN),
        "F_s": per_corner(),
        "F_x": per_corner(),
        "F_y": per_corner(),
        "F_z": per_corner(),
        "steer": per_corner(),
        "omega": per_corner(),
        "kappa": per_corner(),
        "alpha": per_corner(),
        "M_align": per_corner(),
        "clamp": per_corner(),
        "traj_x": np.zeros(HIST_LEN),
        "traj_y": np.zeros(HIST_LEN),
    }


def roll_history(hist, vehicle):
    for key in hist:
        if key in ("t",):
            continue
        if isinstance(hist[key], dict):
            for lbl in hist[key]:
                hist[key][lbl] = np.roll(hist[key][lbl], -1)
        else:
            hist[key] = np.roll(hist[key], -1)

    hist["vx"][-1] = vehicle.state.velocity_x
    hist["vy"][-1] = vehicle.state.velocity_y
    hist["roll"][-1] = vehicle.state.roll
    hist["pitch"][-1] = vehicle.state.pitch
    hist["yaw"][-1] = vehicle.state.yaw
    hist["yaw_rate"][-1] = vehicle.state.yaw_rate
    hist["traj_x"][-1] = vehicle.state.x
    hist["traj_y"][-1] = vehicle.state.y

    for lbl in WHEEL_LABELS:
        corner = vehicle.corners[lbl]
        hist["F_s"][lbl][-1] = corner.state.F_s
        hist["F_x"][lbl][-1] = corner.state.F_x_tire
        hist["F_y"][lbl][-1] = corner.state.F_y_tire
        hist["F_z"][lbl][-1] = corner.state.F_z
        hist["steer"][lbl][-1] = corner.state.steering_angle
        hist["omega"][lbl][-1] = corner.state.omega_wheel
        hist["kappa"][lbl][-1] = corner.longitudinal_tire.state.slip_ratio
        hist["alpha"][lbl][-1] = corner.lateral_tire.state.slip_angle
        hist["M_align"][lbl][-1] = corner.lateral_tire.state.aligning_torque
        hist["clamp"][lbl][-1] = corner.brake.state.clamp_force


def make_sliders(fig):
    slider_defs = []
    ranges = {
        "T_steer": STEER_RANGE,
        "T_brk": BRAKE_RANGE,
        "T_Drv": DRIVE_RANGE,
        "T_susp": SUSP_RANGE,
    }
    ypos = 0.9
    for key, (vmin, vmax) in ranges.items():
        for lbl in WHEEL_LABELS:
            ax = fig.add_axes([0.80, ypos, 0.18, 0.015])
            slider = Slider(ax, f"{lbl} {key}", vmin, vmax, valinit=0.0, valstep=None)
            slider_defs.append((lbl, key, slider))
            ypos -= 0.02
        ypos -= 0.01  # small gap between groups
    return slider_defs


def collect_inputs(slider_defs):
    corner_inputs = {lbl: {"T_steer": 0.0, "T_brk": 0.0, "T_Drv": 0.0, "T_susp": 0.0} for lbl in WHEEL_LABELS}
    for lbl, key, slider in slider_defs:
        corner_inputs[lbl][key] = slider.val
    return corner_inputs


def body_corners(vehicle):
    L = vehicle.params.L_wheelbase
    W = vehicle.params.L_track
    corners_body = np.array([
        [ L/2,  W/2, -vehicle.params.h_CG],
        [ L/2, -W/2, -vehicle.params.h_CG],
        [-L/2, -W/2, -vehicle.params.h_CG],
        [-L/2,  W/2, -vehicle.params.h_CG],
    ])
    R = vehicle._rotation_matrix()
    pos = np.array([vehicle.state.x, vehicle.state.y, vehicle.params.h_CG + vehicle.state.heave])
    return (R @ corners_body.T).T + pos


def transform_body_to_inertial(vehicle, points_body: np.ndarray) -> np.ndarray:
    """Transform Nx3 points from body(C.G.) frame to inertial frame."""
    R = vehicle._rotation_matrix()
    pos_cg = np.array([vehicle.state.x, vehicle.state.y, vehicle.params.h_CG + vehicle.state.heave], dtype=float)
    return (R @ points_body.T).T + pos_cg


def chassis_box_faces(vehicle):
    """Return 6 faces (quads) for a simple chassis box in inertial frame."""
    front = vehicle.params.a + BODY_OVERHANG
    rear = vehicle.params.b + BODY_OVERHANG
    half_w = 0.5 * (vehicle.params.L_track + BODY_WIDTH_PAD)

    z_bottom = -vehicle.params.h_CG + CHASSIS_CLEARANCE
    z_top = z_bottom + CHASSIS_HEIGHT

    # 8 vertices in body frame (CG at origin)
    v_body = np.array(
        [
            [front, half_w, z_bottom],
            [front, -half_w, z_bottom],
            [-rear, -half_w, z_bottom],
            [-rear, half_w, z_bottom],
            [front, half_w, z_top],
            [front, -half_w, z_top],
            [-rear, -half_w, z_top],
            [-rear, half_w, z_top],
        ],
        dtype=float,
    )
    v = transform_body_to_inertial(vehicle, v_body)

    # faces: bottom, top, front, right, rear, left
    return [
        [v[0], v[1], v[2], v[3]],
        [v[4], v[5], v[6], v[7]],
        [v[0], v[1], v[5], v[4]],
        [v[1], v[2], v[6], v[5]],
        [v[2], v[3], v[7], v[6]],
        [v[3], v[0], v[4], v[7]],
    ]


def maybe_expand_ylim(ax, y_min: float, y_max: float, pad_frac: float = 0.12, min_span: float = 1e-6) -> None:
    """Expand y-limits if new data exceeds current view (never shrinks)."""
    if not np.isfinite(y_min) or not np.isfinite(y_max):
        return
    if y_min > y_max:
        y_min, y_max = y_max, y_min

    span = max(float(y_max - y_min), float(min_span))
    pad = pad_frac * span
    new_min = float(y_min - pad)
    new_max = float(y_max + pad)

    cur_min, cur_max = ax.get_ylim()
    if new_min < cur_min or new_max > cur_max:
        ax.set_ylim(min(cur_min, new_min), max(cur_max, new_max))


def body_outline_2d(vehicle):
    """Return a simple 'car-like' top-view outline in the body frame (CG at origin)."""
    front = vehicle.params.a + BODY_OVERHANG
    rear = vehicle.params.b + BODY_OVERHANG
    half_w = 0.5 * (vehicle.params.L_track + BODY_WIDTH_PAD)

    nose = min(0.8, 0.35 * (front + rear))
    tail = min(0.4, 0.20 * (front + rear))

    # A 6-vertex silhouette with a pointed nose for clear heading.
    return np.array(
        [
            [-rear, half_w],
            [front - nose, half_w],
            [front, 0.0],
            [front - nose, -half_w],
            [-rear, -half_w],
            [-rear + tail, 0.0],
        ],
        dtype=float,
    )


def update_frame(frame_idx, vehicle, hist, slider_defs, artists):
    corner_inputs = collect_inputs(slider_defs)
    for _ in range(SIM_STEPS_PER_FRAME):
        vehicle.update(SIM_DT, corner_inputs, direction=1)
    roll_history(hist, vehicle)

    # 3D pose (simple chassis box + wheel outlines + Fz vectors)
    artists["chassis_poly"].set_verts(chassis_box_faces(vehicle))

    R = vehicle._rotation_matrix()
    half_l = 0.5 * WHEEL_LENGTH
    half_w = 0.5 * WHEEL_WIDTH
    fz_scale = artists["fz_scale"]
    wheel_centers = {}
    for lbl in WHEEL_LABELS:
        idx = WHEEL_LABELS.index(lbl)
        center = vehicle.get_wheel_position(idx)
        wheel_centers[lbl] = center

        delta = vehicle.corners[lbl].state.steering_angle
        u_fwd = R @ np.array([np.cos(delta), np.sin(delta), 0.0], dtype=float)
        u_lat = R @ np.array([-np.sin(delta), np.cos(delta), 0.0], dtype=float)

        p0 = center + u_fwd * half_l + u_lat * half_w
        p1 = center + u_fwd * half_l - u_lat * half_w
        p2 = center - u_fwd * half_l - u_lat * half_w
        p3 = center - u_fwd * half_l + u_lat * half_w
        poly = np.vstack([p0, p1, p2, p3, p0])

        wline = artists["wheel_lines"][lbl]
        wline.set_data(poly[:, 0], poly[:, 1])
        wline.set_3d_properties(poly[:, 2])

        fz = float(vehicle.corners[lbl].state.F_z)
        z0 = center
        z1 = center + np.array([0.0, 0.0, fz * fz_scale], dtype=float)
        fline = artists["fz_vec_lines"][lbl]
        fline.set_data([z0[0], z1[0]], [z0[1], z1[1]])
        fline.set_3d_properties([z0[2], z1[2]])

    artists["traj3d_line"].set_data(hist["traj_x"], hist["traj_y"])
    artists["traj3d_line"].set_3d_properties(np.zeros_like(hist["traj_x"]))

    # Keep a stable 3D follow view window.
    cx, cy = vehicle.state.x, vehicle.state.y
    yaw = vehicle.state.yaw
    speed = float(np.hypot(vehicle.state.velocity_x, vehicle.state.velocity_y))
    win_xy = max(10.0, min(50.0, 0.6 * speed * HISTORY_SECONDS + 12.0))
    lookahead = 0.15 * win_xy
    cx_view = cx + lookahead * np.cos(yaw)
    cy_view = cy + lookahead * np.sin(yaw)
    ax3d = artists["ax3d"]
    ax3d.set_xlim(cx_view - win_xy / 2.0, cx_view + win_xy / 2.0)
    ax3d.set_ylim(cy_view - win_xy / 2.0, cy_view + win_xy / 2.0)
    z_cg = vehicle.params.h_CG + vehicle.state.heave
    ax3d.set_zlim(min(-0.3, z_cg - 1.0), z_cg + CHASSIS_CLEARANCE + CHASSIS_HEIGHT + 0.8)

    # 2D trajectory
    artists["traj_line"].set_data(hist["traj_x"], hist["traj_y"])
    artists["traj_dot"].set_data([vehicle.state.x], [vehicle.state.y])

    # 2D body/wheels (car-like silhouette + heading arrow)
    cx, cy = vehicle.state.x, vehicle.state.y
    yaw = vehicle.state.yaw
    body_patch = artists["body_patch"]
    body_patch.set_transform(Affine2D().rotate(yaw).translate(cx, cy) + body_patch.axes.transData)

    # Wheel patches are defined around (0,0) and moved via transforms.
    wheel_xy = {}
    for lbl, patch in artists["wheel_patches"].items():
        idx = WHEEL_LABELS.index(lbl)
        wp = vehicle.get_wheel_position(idx)
        wheel_xy[lbl] = wp
        delta = vehicle.corners[lbl].state.steering_angle
        theta = yaw + delta
        patch.set_transform(Affine2D().rotate(theta).translate(wp[0], wp[1]) + patch.axes.transData)

        txt = artists["wheel_texts"][lbl]
        txt.set_text(f"{lbl}: {np.rad2deg(delta):+.1f}°")
        txt.set_position((wp[0], wp[1] + 0.45))

    artists["cg_dot"].set_data([cx], [cy])
    if {"FL", "FR", "RL", "RR"}.issubset(wheel_xy):
        artists["front_axle"].set_data(
            [wheel_xy["FL"][0], wheel_xy["FR"][0]],
            [wheel_xy["FL"][1], wheel_xy["FR"][1]],
        )
        artists["rear_axle"].set_data(
            [wheel_xy["RL"][0], wheel_xy["RR"][0]],
            [wheel_xy["RL"][1], wheel_xy["RR"][1]],
        )

    front = vehicle.params.a + BODY_OVERHANG
    hx = cx + front * np.cos(yaw)
    hy = cy + front * np.sin(yaw)
    artists["heading_arrow"].set_positions((cx, cy), (hx, hy))

    # Keep a stable "follow camera" view window.
    speed = float(np.hypot(vehicle.state.velocity_x, vehicle.state.velocity_y))
    win_x = max(12.0, min(60.0, 0.8 * speed * HISTORY_SECONDS + 12.0))
    win_y = 0.70 * win_x
    lookahead = 0.20 * win_x
    cx_view = cx + lookahead * np.cos(yaw)
    cy_view = cy + lookahead * np.sin(yaw)
    ax_xy = body_patch.axes
    ax_xy.set_xlim(cx_view - win_x / 2.0, cx_view + win_x / 2.0)
    ax_xy.set_ylim(cy_view - win_y / 2.0, cy_view + win_y / 2.0)

    # Time histories
    t_axis = hist["t"]
    artists["vx_line"].set_ydata(hist["vx"])
    artists["vy_line"].set_ydata(hist["vy"])
    artists["roll_line"].set_ydata(hist["roll"])
    artists["pitch_line"].set_ydata(hist["pitch"])
    artists["yaw_line"].set_ydata(hist["yaw"])
    artists["yaw_rate_line"].set_ydata(hist["yaw_rate"])

    for lbl in WHEEL_LABELS:
        artists["F_s_lines"][lbl].set_ydata(hist["F_s"][lbl])
        artists["F_x_lines"][lbl].set_ydata(hist["F_x"][lbl])
        artists["F_y_lines"][lbl].set_ydata(hist["F_y"][lbl])
        artists["F_z_lines"][lbl].set_ydata(hist["F_z"][lbl])
        artists["steer_lines"][lbl].set_ydata(np.rad2deg(hist["steer"][lbl]))
        artists["omega_lines"][lbl].set_ydata(hist["omega"][lbl])
        artists["kappa_lines"][lbl].set_ydata(hist["kappa"][lbl])
        artists["alpha_lines"][lbl].set_ydata(hist["alpha"][lbl])
        artists["M_align_lines"][lbl].set_ydata(hist["M_align"][lbl])
        artists["clamp_lines"][lbl].set_ydata(hist["clamp"][lbl])

    # Auto-expand y-limits so changes are visible (matplotlib does not autoscale on set_ydata).
    if frame_idx % 10 == 0:
        ax_v = artists["vx_line"].axes
        maybe_expand_ylim(ax_v, min(float(hist["vx"].min()), float(hist["vy"].min())),
                          max(float(hist["vx"].max()), float(hist["vy"].max())))

        ax_att = artists["roll_line"].axes
        maybe_expand_ylim(ax_att, min(float(hist["roll"].min()), float(hist["pitch"].min())),
                          max(float(hist["roll"].max()), float(hist["pitch"].max())))

        ax_yaw = artists["yaw_line"].axes
        maybe_expand_ylim(ax_yaw, min(float(hist["yaw"].min()), float(hist["yaw_rate"].min())),
                          max(float(hist["yaw"].max()), float(hist["yaw_rate"].max())))

        ax_Fs = next(iter(artists["F_s_lines"].values())).axes
        fs_min = min(float(hist["F_s"][lbl].min()) for lbl in WHEEL_LABELS)
        fs_max = max(float(hist["F_s"][lbl].max()) for lbl in WHEEL_LABELS)
        fz_min = min(float(hist["F_z"][lbl].min()) for lbl in WHEEL_LABELS)
        fz_max = max(float(hist["F_z"][lbl].max()) for lbl in WHEEL_LABELS)
        maybe_expand_ylim(ax_Fs, min(fs_min, fz_min), max(fs_max, fz_max))

        ax_Fxy = next(iter(artists["F_x_lines"].values())).axes
        fx_min = min(float(hist["F_x"][lbl].min()) for lbl in WHEEL_LABELS)
        fx_max = max(float(hist["F_x"][lbl].max()) for lbl in WHEEL_LABELS)
        fy_min = min(float(hist["F_y"][lbl].min()) for lbl in WHEEL_LABELS)
        fy_max = max(float(hist["F_y"][lbl].max()) for lbl in WHEEL_LABELS)
        maybe_expand_ylim(ax_Fxy, min(fx_min, fy_min), max(fx_max, fy_max))

        ax_w = next(iter(artists["omega_lines"].values())).axes
        om_min = min(float(hist["omega"][lbl].min()) for lbl in WHEEL_LABELS)
        om_max = max(float(hist["omega"][lbl].max()) for lbl in WHEEL_LABELS)
        maybe_expand_ylim(ax_w, om_min, om_max)

        ax_slip = next(iter(artists["kappa_lines"].values())).axes
        k_min = min(float(hist["kappa"][lbl].min()) for lbl in WHEEL_LABELS)
        k_max = max(float(hist["kappa"][lbl].max()) for lbl in WHEEL_LABELS)
        a_min = min(float(hist["alpha"][lbl].min()) for lbl in WHEEL_LABELS)
        a_max = max(float(hist["alpha"][lbl].max()) for lbl in WHEEL_LABELS)
        maybe_expand_ylim(ax_slip, min(k_min, a_min), max(k_max, a_max))

        ax_misc = next(iter(artists["steer_lines"].values())).axes
        steer_min = min(float(np.rad2deg(hist["steer"][lbl]).min()) for lbl in WHEEL_LABELS)
        steer_max = max(float(np.rad2deg(hist["steer"][lbl]).max()) for lbl in WHEEL_LABELS)
        ma_min = min(float(hist["M_align"][lbl].min()) for lbl in WHEEL_LABELS)
        ma_max = max(float(hist["M_align"][lbl].max()) for lbl in WHEEL_LABELS)
        maybe_expand_ylim(ax_misc, min(steer_min, ma_min), max(steer_max, ma_max))

        ax_clamp = artists.get("ax_clamp")
        if ax_clamp is not None:
            c_min = min(float(hist["clamp"][lbl].min()) for lbl in WHEEL_LABELS)
            c_max = max(float(hist["clamp"][lbl].max()) for lbl in WHEEL_LABELS)
            maybe_expand_ylim(ax_clamp, min(0.0, c_min), c_max)

    return []


def main():
    vehicle = build_vehicle()
    hist = init_history()

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 4, width_ratios=[1.2, 1, 1, 1], left=0.05, right=0.75, wspace=0.25, hspace=0.35)

    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_xy = fig.add_subplot(gs[0, 1])
    ax_v = fig.add_subplot(gs[0, 2])
    ax_att = fig.add_subplot(gs[0, 3])
    ax_yaw = fig.add_subplot(gs[1, 1])
    ax_Fs = fig.add_subplot(gs[1, 2])
    ax_Fxy = fig.add_subplot(gs[1, 3])
    ax_w = fig.add_subplot(gs[2, 1])
    ax_slip = fig.add_subplot(gs[2, 2])
    ax_misc = fig.add_subplot(gs[2, 3])

    slider_defs = make_sliders(fig)

    # 3D init (ground plane + chassis box + wheel outlines + Fz vectors)
    ground_size = 200.0
    ground = Poly3DCollection(
        [[[-ground_size, -ground_size, 0.0], [ground_size, -ground_size, 0.0], [ground_size, ground_size, 0.0], [-ground_size, ground_size, 0.0]]],
        facecolors="#2ca02c",
        alpha=0.05,
        edgecolors="none",
    )
    ax3d.add_collection3d(ground)

    chassis_poly = Poly3DCollection(
        chassis_box_faces(vehicle),
        facecolors="#bdbdbd",
        alpha=0.65,
        edgecolors="k",
        linewidths=0.8,
    )
    ax3d.add_collection3d(chassis_poly)

    wheel_lines = {}
    fz_vec_lines = {}
    wheel_colors = {"FL": "tab:blue", "FR": "tab:orange", "RR": "tab:red", "RL": "tab:green"}
    for lbl in WHEEL_LABELS:
        (wline,) = ax3d.plot([], [], [], color=wheel_colors.get(lbl, "k"), lw=2.0)
        wheel_lines[lbl] = wline
        (fline,) = ax3d.plot([], [], [], color="tab:purple", lw=2.0, alpha=0.75)
        fz_vec_lines[lbl] = fline

    traj3d_line, = ax3d.plot(
        hist["traj_x"],
        hist["traj_y"],
        np.zeros_like(hist["traj_x"]),
        color="tab:blue",
        alpha=0.35,
        lw=1.2,
    )

    # Visual scale: ~0.25 m per typical wheel normal load
    fz_scale = 1.0 / max(1.0, (vehicle.params.m * vehicle.params.g)) * 2.5
    ax3d.set_xlabel("X [m]"); ax3d.set_ylabel("Y [m]"); ax3d.set_zlabel("Z [m]")
    ax3d.set_title("3D pose")
    ax3d.set_box_aspect([1, 1, 0.45])
    ax3d.view_init(elev=22, azim=-60)

    # Trajectory
    traj_line, = ax_xy.plot(hist["traj_x"], hist["traj_y"], lw=1.5)
    traj_dot, = ax_xy.plot([vehicle.state.x], [vehicle.state.y], "ro")
    ax_xy.set_title("Top view")
    ax_xy.set_xlabel("X [m]")
    ax_xy.set_ylabel("Y [m]")
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.grid(True, linestyle="--", alpha=0.25)

    # Body + wheels patches (defined in body frame around origin)
    body_patch = Polygon(
        body_outline_2d(vehicle),
        closed=True,
        fill=True,
        facecolor="#d9d9d9",
        edgecolor="k",
        linewidth=2,
        alpha=0.9,
        zorder=3,
    )
    ax_xy.add_patch(body_patch)

    heading_arrow = FancyArrowPatch(
        (0.0, 0.0),
        (1.0, 0.0),
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.8,
        color="tab:red",
        zorder=5,
    )
    ax_xy.add_patch(heading_arrow)

    cg_dot, = ax_xy.plot([vehicle.state.x], [vehicle.state.y], "ko", ms=4, zorder=6)
    front_axle, = ax_xy.plot([], [], color="k", lw=1.2, alpha=0.6, zorder=2)
    rear_axle, = ax_xy.plot([], [], color="k", lw=1.2, alpha=0.6, zorder=2)

    wheel_patches = {}
    wheel_texts = {}
    colors = {"FL": "tab:blue", "FR": "tab:orange", "RL": "tab:green", "RR": "tab:red"}
    for lbl in WHEEL_LABELS:
        patch = Rectangle((0, 0), 0.5, 0.25, facecolor=colors.get(lbl, "gray"), alpha=0.8, edgecolor="k")
        ax_xy.add_patch(patch)
        wheel_patches[lbl] = patch
        wheel_texts[lbl] = ax_xy.text(
            0.0,
            0.0,
            f"{lbl}: +0.0°",
            ha="center",
            va="bottom",
            fontsize=8,
            color="tab:red",
            zorder=7,
        )

    # Place wheel rectangles at origin; move them via transforms in update_frame.
    for lbl, patch in wheel_patches.items():
        patch.set_width(WHEEL_LENGTH)
        patch.set_height(WHEEL_WIDTH)
        patch.set_xy((-WHEEL_LENGTH / 2.0, -WHEEL_WIDTH / 2.0))
        patch.set_facecolor("lightgray")
        patch.set_edgecolor(colors.get(lbl, "k"))
        patch.set_linewidth(2.0)
        patch.set_zorder(4)

    t_axis = hist["t"]
    vx_line, = ax_v.plot(t_axis, hist["vx"], label="vx")
    vy_line, = ax_v.plot(t_axis, hist["vy"], label="vy")
    ax_v.legend(); ax_v.set_title("Velocity (body)"); ax_v.grid(True)

    roll_line, = ax_att.plot(t_axis, hist["roll"], label="roll")
    pitch_line, = ax_att.plot(t_axis, hist["pitch"], label="pitch")
    ax_att.legend(); ax_att.set_title("Attitude"); ax_att.grid(True)

    yaw_line, = ax_yaw.plot(t_axis, hist["yaw"], label="yaw")
    yaw_rate_line, = ax_yaw.plot(t_axis, hist["yaw_rate"], label="yaw_rate")
    ax_yaw.legend(); ax_yaw.set_title("Yaw / Yaw rate"); ax_yaw.grid(True)

    F_s_lines, F_z_lines = {}, {}
    for lbl in WHEEL_LABELS:
        (line,) = ax_Fs.plot(t_axis, hist["F_s"][lbl], label=f"{lbl}")
        (lz,) = ax_Fs.plot(t_axis, hist["F_z"][lbl], linestyle="--", label=f"{lbl} Fz")
        F_s_lines[lbl] = line
        F_z_lines[lbl] = lz
    ax_Fs.set_title("F_s (suspension) / F_z (normal)"); ax_Fs.grid(True); ax_Fs.legend(fontsize=8)

    F_x_lines, F_y_lines = {}, {}
    for lbl in WHEEL_LABELS:
        (lx,) = ax_Fxy.plot(t_axis, hist["F_x"][lbl], label=f"{lbl} Fx")
        (ly,) = ax_Fxy.plot(t_axis, hist["F_y"][lbl], linestyle="--", label=f"{lbl} Fy")
        F_x_lines[lbl] = lx; F_y_lines[lbl] = ly
    ax_Fxy.set_title("Tire forces Fx/Fy"); ax_Fxy.grid(True); ax_Fxy.legend(fontsize=8)

    omega_lines = {}
    for lbl in WHEEL_LABELS:
        (line,) = ax_w.plot(t_axis, hist["omega"][lbl], label=lbl)
        omega_lines[lbl] = line
    ax_w.set_title("Wheel speed [rad/s]"); ax_w.grid(True); ax_w.legend(fontsize=8)

    kappa_lines, alpha_lines = {}, {}
    for lbl in WHEEL_LABELS:
        (lk,) = ax_slip.plot(t_axis, hist["kappa"][lbl], label=f"{lbl} κ")
        (la,) = ax_slip.plot(t_axis, hist["alpha"][lbl], linestyle="--", label=f"{lbl} α")
        kappa_lines[lbl] = lk; alpha_lines[lbl] = la
    ax_slip.set_title("Slip ratio / Slip angle"); ax_slip.grid(True); ax_slip.legend(fontsize=8)

    ax_clamp = ax_misc.twinx()
    ax_clamp.grid(False)
    steer_lines, clamp_lines, M_align_lines = {}, {}, {}
    for lbl in WHEEL_LABELS:
        (ls,) = ax_misc.plot(t_axis, np.rad2deg(hist["steer"][lbl]), label=f"{lbl} steer [deg]")
        (lm,) = ax_misc.plot(t_axis, hist["M_align"][lbl], linestyle=":", label=f"{lbl} M_align [N·m]")
        (lc,) = ax_clamp.plot(t_axis, hist["clamp"][lbl], linestyle="--", label=f"{lbl} clamp [N]")
        steer_lines[lbl] = ls
        clamp_lines[lbl] = lc
        M_align_lines[lbl] = lm
    ax_misc.set_title("Steer / Aligning / Clamp")
    ax_misc.set_ylabel("Steer [deg] / M_align [N·m]")
    ax_clamp.set_ylabel("Clamp [N]")
    ax_misc.grid(True)
    h1, l1 = ax_misc.get_legend_handles_labels()
    h2, l2 = ax_clamp.get_legend_handles_labels()
    ax_misc.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")

    artists = {
        "ax3d": ax3d,
        "chassis_poly": chassis_poly,
        "wheel_lines": wheel_lines,
        "fz_vec_lines": fz_vec_lines,
        "traj3d_line": traj3d_line,
        "fz_scale": fz_scale,
        "traj_line": traj_line,
        "traj_dot": traj_dot,
        "body_patch": body_patch,
        "heading_arrow": heading_arrow,
        "cg_dot": cg_dot,
        "front_axle": front_axle,
        "rear_axle": rear_axle,
        "wheel_patches": wheel_patches,
        "wheel_texts": wheel_texts,
        "vx_line": vx_line,
        "vy_line": vy_line,
        "roll_line": roll_line,
        "pitch_line": pitch_line,
        "yaw_line": yaw_line,
        "yaw_rate_line": yaw_rate_line,
        "F_s_lines": F_s_lines,
        "F_z_lines": F_z_lines,
        "F_x_lines": F_x_lines,
        "F_y_lines": F_y_lines,
        "steer_lines": steer_lines,
        "omega_lines": omega_lines,
        "kappa_lines": kappa_lines,
        "alpha_lines": alpha_lines,
        "M_align_lines": M_align_lines,
        "clamp_lines": clamp_lines,
        "ax_clamp": ax_clamp,
    }

    anim = animation.FuncAnimation(
        fig,
        update_frame,
        fargs=(vehicle, hist, slider_defs, artists),
        interval=RENDER_INTERVAL_MS,
        blit=False,
        cache_frame_data=False,
    )
    global ANIM
    ANIM = anim
    fig._anim = anim  # extra guard for interactive IDEs

    plt.show()


if __name__ == "__main__":
    main()
