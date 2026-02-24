"""
VehicleBody 모델 검증 스크립트 (Inverse Dynamics 방식)

서스펜션 테스트와 동일한 방법론:
1. CarMaker 상태(heave/roll/pitch)를 입력
2. 코너 가속도로부터 서스펜션 힘 계산
3. 모델의 가속도 예측과 CarMaker 가속도 비교
"""
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# vehicle_sim을 포함한 경로 추가
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from vehicle_sim.models.vehicle_body.vehicle_body import VehicleBody
from vehicle_sim.utils.config_loader import load_param

# ==================== 데이터 로드 ====================
print("=" * 80)
print("Loading CM_Body.csv...")


print("=" * 80)

data_path = Path(__file__).parent / "CM_Body_6.csv"
df = pd.read_csv(data_path)

time = df["Time"].to_numpy()
n_steps = len(time)
if n_steps < 2:
    raise ValueError("CM_Body.csv에 최소 2개 이상의 샘플이 필요합니다.")
dt_series = np.diff(time)
dt_default = float(dt_series.mean())

print(f"Data loaded: {n_steps} samples, dt ~= {dt_default:.6f} s")

# ==================== CarMaker 입력/출력 ====================
# 4코너 타이어 힘 (Fx/Fy)
F_x_FL = df["cm_FxFL"].to_numpy()
F_x_FR = df["cm_FxFR"].to_numpy()
F_x_RL = df["cm_FxRL"].to_numpy()
F_x_RR = df["cm_FxRR"].to_numpy()

F_y_FL = df["cm_FyFL"].to_numpy()
F_y_FR = df["cm_FyFR"].to_numpy()
F_y_RL = df["cm_FyRL"].to_numpy()
F_y_RR = df["cm_FyRR"].to_numpy()

# 4코너 조향각 (Steering Angle)
steer_FL = df["cm_SteerAngleFL"].to_numpy()
steer_FR = df["cm_SteerAngleFR"].to_numpy()
steer_RL = df["cm_SteerAngleRL"].to_numpy()
steer_RR = df["cm_SteerAngleRR"].to_numpy()

# 차체 상태 (위치/속도)
cm_heave = df["cm_Heave"].to_numpy()
cm_roll = df["cm_Roll"].to_numpy()
cm_pitch = df["cm_Pitch"].to_numpy()
cm_yaw = df["cm_Yaw"].to_numpy()

cm_heave_dot = df["cm_Heave_dot"].to_numpy()
cm_roll_rate = df["cm_Roll_Dot"].to_numpy()
cm_pitch_rate = df["cm_Pitch_Dot"].to_numpy()
cm_yaw_rate = df["cm_YawRate"].to_numpy()

cm_vx = df["cm_vx"].to_numpy()
cm_vy = df["cm_vy"].to_numpy()

# 스프렁 가속도 (각 코너)
sprung_z_ddot_FL = df["cm_SprungZDdotFL"].to_numpy()
sprung_z_ddot_FR = df["cm_SprungZDdotFR"].to_numpy()
sprung_z_ddot_RL = df["cm_SprungZDdotRL"].to_numpy()
sprung_z_ddot_RR = df["cm_SprungZDdotRR"].to_numpy()

# CarMaker 가속도 (비교용)
cm_ax = df["cm_ax"].to_numpy()
cm_ay = df["cm_ay"].to_numpy()

print("\nCarMaker signals loaded (ranges):")
print(f"  Fx FL: [{F_x_FL.min():.2f}, {F_x_FL.max():.2f}] N")
print(f"  Fy FL: [{F_y_FL.min():.2f}, {F_y_FL.max():.2f}] N")
print(f"  SprungZDdot FL: [{sprung_z_ddot_FL.min():.4f}, {sprung_z_ddot_FL.max():.4f}] m/s²")
print(f"  Heave: [{cm_heave.min():.4f}, {cm_heave.max():.4f}] m")
print(f"  Roll:  [{np.rad2deg(cm_roll.min()):.2f}, {np.rad2deg(cm_roll.max()):.2f}] deg")
print(f"  Pitch: [{np.rad2deg(cm_pitch.min()):.2f}, {np.rad2deg(cm_pitch.max()):.2f}] deg")

# ==================== 모델 로드 ====================
print("\n" + "=" * 80)
print("Loading vehicle parameters and model...")
print("=" * 80)

config_path = Path(__file__).parent.parent / "models" / "params" / "vehicle_standard.yaml"
physics_param = load_param("physics", str(config_path))
vehicle_spec_param = load_param("vehicle_spec", str(config_path))

vehicle_body = VehicleBody(config_path=str(config_path))
g = float(physics_param.get("g", vehicle_body.params.g))

# 질량 분배 (YAML front_load_ratio 사용)
geometry_spec = vehicle_spec_param.get("geometry", {})
front_load_ratio = float(geometry_spec.get("front_load_ratio", 0.611))
rear_load_ratio = float(geometry_spec.get("rear_load_ratio", 0.389))

m_total = vehicle_body.params.m
m_s_FL = m_total * front_load_ratio / 2.0
m_s_FR = m_total * front_load_ratio / 2.0
m_s_RL = m_total * rear_load_ratio / 2.0
m_s_RR = m_total * rear_load_ratio / 2.0

print("\nVehicleBody model parameters:")
print(f"  Mass: {vehicle_body.params.m:.2f} kg")
print(f"  Ixx/Iyy/Izz: {vehicle_body.params.Ixx:.2f}, "
      f"{vehicle_body.params.Iyy:.2f}, {vehicle_body.params.Izz:.2f} kg·m²")
print(f"  Wheelbase: {vehicle_body.params.L_wheelbase:.3f} m, "
      f"Track: {vehicle_body.params.L_track:.3f} m, "
      f"CG height: {vehicle_body.params.h_CG:.3f} m")
print(f"  g: {g:.4f} m/s²")
print(f"  Front load ratio: {front_load_ratio:.3f} (m_s_FL/FR = {m_s_FL:.2f} kg)")
print(f"  Rear load ratio: {rear_load_ratio:.3f} (m_s_RL/RR = {m_s_RL:.2f} kg)")

print("\nCorner offsets:")
for label in ["FL", "FR", "RL", "RR"]:
    offset = vehicle_body.corner_offsets[label]
    print(f"  {label}: x={offset['x']:.3f} m, y={offset['y']:.3f} m")

# ==================== 역동역학 시뮬레이션 ====================
print("\n" + "=" * 80)
print("Running Inverse Dynamics Test...")
print("=" * 80)

# 옵션: Fx/Fy 사용 여부 (무조건 사용)
USE_CM_FXY = True

# 결과 저장 배열
pred_heave_ddot = np.zeros(n_steps)
pred_roll_ddot = np.zeros(n_steps)
pred_pitch_ddot = np.zeros(n_steps)
pred_ax = np.zeros(n_steps)
pred_ay = np.zeros(n_steps)
pred_yaw_ddot = np.zeros(n_steps)
pred_vx_int = np.zeros(n_steps)
pred_vy_int = np.zeros(n_steps)
pred_yaw_rate_int = np.zeros(n_steps)
pred_yaw_int = np.zeros(n_steps)
pred_heave_dot_int = np.zeros(n_steps)
pred_roll_rate_int = np.zeros(n_steps)
pred_pitch_rate_int = np.zeros(n_steps)
pred_heave_int = np.zeros(n_steps)
pred_roll_int = np.zeros(n_steps)
pred_pitch_int = np.zeros(n_steps)

# CarMaker 가속도 (비교 대상) - 차체 CG 가속도로 계산 필요
# heave_ddot, roll_ddot, pitch_ddot를 코너 가속도로부터 역산하거나
# 또는 직접 미분해서 계산
# 여기서는 간단히 중앙차분으로 계산
cm_heave_ddot = np.gradient(cm_heave_dot, time, edge_order=2)
cm_roll_ddot = np.gradient(cm_roll_rate, time, edge_order=2)
cm_pitch_ddot = np.gradient(cm_pitch_rate, time, edge_order=2)
cm_yaw_ddot = np.gradient(cm_yaw_rate, time, edge_order=2)

# 적분용 초기값 세팅 (모델 예측)
pred_vx_int[0] = cm_vx[0]
pred_vy_int[0] = cm_vy[0]
pred_yaw_rate_int[0] = cm_yaw_rate[0]
pred_yaw_int[0] = cm_yaw[0]
pred_heave_dot_int[0] = cm_heave_dot[0]
pred_roll_rate_int[0] = cm_roll_rate[0]
pred_pitch_rate_int[0] = cm_pitch_rate[0]
pred_heave_int[0] = cm_heave[0]
pred_roll_int[0] = cm_roll[0]
pred_pitch_int[0] = cm_pitch[0]

for i in range(n_steps):
    if i % 10000 == 0 or i == n_steps - 1:
        print(f"  Step {i}/{n_steps} ({100 * i / n_steps:.1f}%)")

    # CarMaker 상태를 모델에 설정 (inverse dynamics - 상태는 입력)
    vehicle_body.state.heave = float(cm_heave[i])
    vehicle_body.state.roll = float(cm_roll[i])
    vehicle_body.state.pitch = float(cm_pitch[i])
    vehicle_body.state.yaw = float(cm_yaw[i])

    vehicle_body.state.heave_dot = float(cm_heave_dot[i])
    vehicle_body.state.roll_rate = float(cm_roll_rate[i])
    vehicle_body.state.pitch_rate = float(cm_pitch_rate[i])
    vehicle_body.state.yaw_rate = float(cm_yaw_rate[i])

    vehicle_body.state.velocity_x = float(cm_vx[i])
    vehicle_body.state.velocity_y = float(cm_vy[i])

    # 코너 가속도로부터 서스펜션 힘 계산 (F_s = m_s_corner * (g - z_ddot))
    # 타이어 수평력 (Fx, Fy) - CSV에서 휠 프레임으로 읽고, assemble_forces_moments에서 조향각 회전
    if USE_CM_FXY:
        # CSV에서 읽은 Fx/Fy는 휠 프레임 (타이어 좌표계)
        # assemble_forces_moments에서 조향각으로 바디 프레임으로 회전함
        fx_fl, fx_fr, fx_rl, fx_rr = F_x_FL[i], F_x_FR[i], F_x_RL[i], F_x_RR[i]
        fy_fl, fy_fr, fy_rl, fy_rr = F_y_FL[i], F_y_FR[i], F_y_RL[i], F_y_RR[i]
    else:
        fx_fl = fx_fr = fx_rl = fx_rr = 0.0
        fy_fl = fy_fr = fy_rl = fy_rr = 0.0

    # E-Corner에 조향각 설정 (assemble_forces_moments에서 사용)
    vehicle_body.corners["FL"].state.steering_angle = float(steer_FL[i])
    vehicle_body.corners["FR"].state.steering_angle = float(steer_FR[i])
    vehicle_body.corners["RL"].state.steering_angle = float(steer_RL[i])
    vehicle_body.corners["RR"].state.steering_angle = float(steer_RR[i])

    corner_outputs = {
        "FL": (m_s_FL * (g - sprung_z_ddot_FL[i]), fx_fl, fy_fl),
        "FR": (m_s_FR * (g - sprung_z_ddot_FR[i]), fx_fr, fy_fr),
        "RL": (m_s_RL * (g - sprung_z_ddot_RL[i]), fx_rl, fy_rl),
        "RR": (m_s_RR * (g - sprung_z_ddot_RR[i]), fx_rr, fy_rr),
    }

    # 모델로 힘/모멘트 조립 후 가속도 계산
    forces, moments = vehicle_body.assemble_forces_moments(corner_outputs)
    # F_s = m_s*(g - z_ddot)에 중력이 포함되어 있으므로 add_gravity=True
    linear_acc, angular_acc = vehicle_body.calculate_accelerations(forces, moments, add_gravity=True)

    # ax, ay에서 중력 회전 성분 제거 (차량 가속도만 남김)
    # pitch/roll에 의해 중력이 x, y축으로 투영되는 것을 제거
    roll = vehicle_body.state.roll
    pitch = vehicle_body.state.pitch
    g = vehicle_body.params.g
    linear_acc[0] -= g * np.sin(pitch)  # ax에서 pitch 중력 성분 제거
    linear_acc[1] -= -g * np.sin(roll)  # ay에서 roll 중력 성분 제거

    # 디버그: 첫 스텝 확인
    if i == 0:
        print("\n=== First Step Debug ===")
        for label in ["FL", "FR", "RL", "RR"]:
            print(f"  {label} steering_angle: {vehicle_body.corners[label].state.steering_angle:.6f} rad")
        print(f"\n  Corner outputs (F_s, F_x, F_y):")
        for label in ["FL", "FR", "RL", "RR"]:
            print(f"    {label}: {corner_outputs[label]}")
        print(f"\n  Total Forces (Body frame): {forces}")
        print(f"  Total Moments: {moments}")
        print(f"  linear_acc (ax, ay, az): {linear_acc}")
        print(f"  angular_acc (roll_ddot, pitch_ddot, yaw_ddot): {angular_acc}")
        print(f"  CarMaker vx[0]: {cm_vx[0]:.3f} m/s")
        print("======================\n")

    # 가속도 기록 (heave_ddot = linear_acc[2], roll_ddot = angular_acc[0], pitch_ddot = angular_acc[1])
    pred_heave_ddot[i] = linear_acc[2]
    pred_roll_ddot[i] = angular_acc[0]
    pred_pitch_ddot[i] = angular_acc[1]
    pred_ax[i] = linear_acc[0]
    pred_ay[i] = linear_acc[1]
    pred_yaw_ddot[i] = angular_acc[2]

    # 적분하여 속도/요레이트/요 비교용 예측 생성
    if i > 0:
        dt = time[i] - time[i - 1]
        pred_vx_int[i] = pred_vx_int[i - 1] + pred_ax[i] * dt
        pred_vy_int[i] = pred_vy_int[i - 1] + pred_ay[i] * dt
        pred_yaw_rate_int[i] = pred_yaw_rate_int[i - 1] + pred_yaw_ddot[i] * dt
        pred_yaw_int[i] = pred_yaw_int[i - 1] + pred_yaw_rate_int[i] * dt
        pred_heave_dot_int[i] = pred_heave_dot_int[i - 1] + pred_heave_ddot[i] * dt
        pred_roll_rate_int[i] = pred_roll_rate_int[i - 1] + pred_roll_ddot[i] * dt
        pred_pitch_rate_int[i] = pred_pitch_rate_int[i - 1] + pred_pitch_ddot[i] * dt
        pred_heave_int[i] = pred_heave_int[i - 1] + pred_heave_dot_int[i] * dt
        pred_roll_int[i] = pred_roll_int[i - 1] + pred_roll_rate_int[i] * dt
        pred_pitch_int[i] = pred_pitch_int[i - 1] + pred_pitch_rate_int[i] * dt

print("\nInverse dynamics test completed!")

# ==================== 오차 계산 ====================
print("\n" + "=" * 80)
print("Calculating errors...")
print("=" * 80)

err_heave_ddot = pred_heave_ddot - cm_heave_ddot
err_roll_ddot = pred_roll_ddot - cm_roll_ddot
err_pitch_ddot = pred_pitch_ddot - cm_pitch_ddot
err_yaw_ddot = pred_yaw_ddot - cm_yaw_ddot
err_ax = pred_ax - cm_ax
err_ay = pred_ay - cm_ay
err_vx = pred_vx_int - cm_vx
err_vy = pred_vy_int - cm_vy
err_yaw_rate = pred_yaw_rate_int - cm_yaw_rate
err_yaw = pred_yaw_int - cm_yaw
err_heave = pred_heave_int - cm_heave
err_roll = pred_roll_int - cm_roll
err_pitch = pred_pitch_int - cm_pitch

mae_heave_ddot = np.mean(np.abs(err_heave_ddot))
mae_roll_ddot = np.mean(np.abs(err_roll_ddot))
mae_pitch_ddot = np.mean(np.abs(err_pitch_ddot))
mae_yaw_ddot = np.mean(np.abs(err_yaw_ddot))
mae_ax = np.mean(np.abs(err_ax))
mae_ay = np.mean(np.abs(err_ay))
mae_vx = np.mean(np.abs(err_vx))
mae_vy = np.mean(np.abs(err_vy))
mae_yaw_rate = np.mean(np.abs(err_yaw_rate))
mae_yaw = np.mean(np.abs(err_yaw))
mae_heave = np.mean(np.abs(err_heave))
mae_roll = np.mean(np.abs(err_roll))
mae_pitch = np.mean(np.abs(err_pitch))

rmse_heave_ddot = np.sqrt(np.mean(err_heave_ddot ** 2))
rmse_roll_ddot = np.sqrt(np.mean(err_roll_ddot ** 2))
rmse_pitch_ddot = np.sqrt(np.mean(err_pitch_ddot ** 2))
rmse_yaw_ddot = np.sqrt(np.mean(err_yaw_ddot ** 2))
rmse_ax = np.sqrt(np.mean(err_ax ** 2))
rmse_ay = np.sqrt(np.mean(err_ay ** 2))
rmse_vx = np.sqrt(np.mean(err_vx ** 2))
rmse_vy = np.sqrt(np.mean(err_vy ** 2))
rmse_yaw_rate = np.sqrt(np.mean(err_yaw_rate ** 2))
rmse_yaw = np.sqrt(np.mean(err_yaw ** 2))
rmse_heave = np.sqrt(np.mean(err_heave ** 2))
rmse_roll = np.sqrt(np.mean(err_roll ** 2))
rmse_pitch = np.sqrt(np.mean(err_pitch ** 2))

print("\nAcceleration Error Statistics:")
print(f"  Heave_ddot MAE:  {mae_heave_ddot:.6f} m/s²,  RMSE: {rmse_heave_ddot:.6f} m/s²")
print(f"  Roll_ddot MAE:   {mae_roll_ddot:.6f} rad/s², RMSE: {rmse_roll_ddot:.6f} rad/s²")
print(f"  Pitch_ddot MAE:  {mae_pitch_ddot:.6f} rad/s², RMSE: {rmse_pitch_ddot:.6f} rad/s²")
print(f"  Yaw_ddot MAE:    {mae_yaw_ddot:.6f} rad/s², RMSE: {rmse_yaw_ddot:.6f} rad/s²")
print(f"  Ax MAE:          {mae_ax:.6f} m/s²,  RMSE: {rmse_ax:.6f} m/s²")
print(f"  Ay MAE:          {mae_ay:.6f} m/s²,  RMSE: {rmse_ay:.6f} m/s²")
print(f"  Vx MAE:          {mae_vx:.6f} m/s,   RMSE: {rmse_vx:.6f} m/s")
print(f"  Vy MAE:          {mae_vy:.6f} m/s,   RMSE: {rmse_vy:.6f} m/s")
print(f"  Yaw_rate MAE:    {mae_yaw_rate:.6f} rad/s, RMSE: {rmse_yaw_rate:.6f} rad/s")
print(f"  Yaw MAE:         {mae_yaw:.6f} rad,  RMSE: {rmse_yaw:.6f} rad")
print(f"  Heave MAE:       {mae_heave*1000:.3f} mm,  RMSE: {rmse_heave*1000:.3f} mm")
print(f"  Roll MAE:        {mae_roll:.6f} rad, RMSE: {rmse_roll:.6f} rad")
print(f"  Pitch MAE:       {mae_pitch:.6f} rad, RMSE: {rmse_pitch:.6f} rad")

# ==================== 시각화 ====================
print("\n" + "=" * 80)
print("Generating plots...")
print("=" * 80)

skip_time = 1.0  # 초기 1초 스킵
skip_idx = np.searchsorted(time, skip_time)

time_plot = time[skip_idx:]
actual_heave_ddot_plot = cm_heave_ddot[skip_idx:]
pred_heave_ddot_plot = pred_heave_ddot[skip_idx:]
actual_roll_ddot_plot = cm_roll_ddot[skip_idx:]
pred_roll_ddot_plot = pred_roll_ddot[skip_idx:]
actual_pitch_ddot_plot = cm_pitch_ddot[skip_idx:]
pred_pitch_ddot_plot = pred_pitch_ddot[skip_idx:]
err_heave_ddot_plot = err_heave_ddot[skip_idx:]
err_roll_ddot_plot = err_roll_ddot[skip_idx:]
err_pitch_ddot_plot = err_pitch_ddot[skip_idx:]

# 추가 비교용 (yaw_ddot)
actual_yaw_ddot_plot = cm_yaw_ddot[skip_idx:]
pred_yaw_ddot_plot = pred_yaw_ddot[skip_idx:]
err_yaw_ddot_plot = err_yaw_ddot[skip_idx:]

# 상태 변수 plot 준비 (예측값 포함)
cm_yaw_plot = cm_yaw[skip_idx:]
cm_vx_plot = cm_vx[skip_idx:]
cm_vy_plot = cm_vy[skip_idx:]
cm_yaw_rate_plot = cm_yaw_rate[skip_idx:]
pred_vx_plot = pred_vx_int[skip_idx:]
pred_vy_plot = pred_vy_int[skip_idx:]
pred_yaw_rate_plot = pred_yaw_rate_int[skip_idx:]
pred_yaw_plot = pred_yaw_int[skip_idx:]
err_vx_plot = err_vx[skip_idx:]
err_vy_plot = err_vy[skip_idx:]
err_yaw_rate_plot = err_yaw_rate[skip_idx:]
err_yaw_plot = err_yaw[skip_idx:]

# ============ Figure 1: 가속도 (Heave/Roll/Pitch) ============
fig1, axes1 = plt.subplots(3, 3, figsize=(18, 14))
fig1.suptitle(f"VehicleBody Inverse Dynamics Test - Accelerations (initial {skip_time}s skipped)",
              fontsize=18, fontweight="bold")

# Row 1: Heave Acceleration
axes1[0, 0].plot(time_plot, actual_heave_ddot_plot, label="CarMaker", linewidth=2)
axes1[0, 0].plot(time_plot, pred_heave_ddot_plot, label="Model", linewidth=1.5, linestyle="--")
axes1[0, 0].set_xlabel("Time (s)")
axes1[0, 0].set_ylabel("Heave Accel (m/s²)")
axes1[0, 0].set_title("Heave Acceleration")
axes1[0, 0].legend()
axes1[0, 0].grid(True)

axes1[0, 1].plot(time_plot, err_heave_ddot_plot, linewidth=1.5)
axes1[0, 1].axhline(y=0, color="black", linestyle="--", linewidth=1)
axes1[0, 1].fill_between(time_plot, err_heave_ddot_plot, 0, alpha=0.3)
axes1[0, 1].set_xlabel("Time (s)")
axes1[0, 1].set_ylabel("Error (m/s²)")
axes1[0, 1].set_title(f"Heave Accel Error (MAE: {mae_heave_ddot:.6f} m/s²)")
axes1[0, 1].grid(True)

min_val_heave_ddot = min(actual_heave_ddot_plot.min(), pred_heave_ddot_plot.min())
max_val_heave_ddot = max(actual_heave_ddot_plot.max(), pred_heave_ddot_plot.max())
axes1[0, 2].scatter(actual_heave_ddot_plot, pred_heave_ddot_plot, alpha=0.5, s=3)
axes1[0, 2].plot([min_val_heave_ddot, max_val_heave_ddot],
                 [min_val_heave_ddot, max_val_heave_ddot],
                 "r--", linewidth=2, label="Perfect")
axes1[0, 2].set_xlabel("CarMaker Heave Accel (m/s²)")
axes1[0, 2].set_ylabel("Predicted Heave Accel (m/s²)")
axes1[0, 2].set_title("Heave Accel Correlation")
axes1[0, 2].legend()
axes1[0, 2].grid(True)
axes1[0, 2].axis("equal")

# Row 2: Roll Acceleration
axes1[1, 0].plot(time_plot, actual_roll_ddot_plot, label="CarMaker", linewidth=2)
axes1[1, 0].plot(time_plot, pred_roll_ddot_plot, label="Model", linewidth=1.5, linestyle="--")
axes1[1, 0].set_xlabel("Time (s)")
axes1[1, 0].set_ylabel("Roll Accel (rad/s²)")
axes1[1, 0].set_title("Roll Acceleration")
axes1[1, 0].legend()
axes1[1, 0].grid(True)

axes1[1, 1].plot(time_plot, err_roll_ddot_plot, linewidth=1.5)
axes1[1, 1].axhline(y=0, color="black", linestyle="--", linewidth=1)
axes1[1, 1].fill_between(time_plot, err_roll_ddot_plot, 0, alpha=0.3)
axes1[1, 1].set_xlabel("Time (s)")
axes1[1, 1].set_ylabel("Error (rad/s²)")
axes1[1, 1].set_title(f"Roll Accel Error (MAE: {mae_roll_ddot:.6f} rad/s²)")
axes1[1, 1].grid(True)

min_val_roll_ddot = min(actual_roll_ddot_plot.min(), pred_roll_ddot_plot.min())
max_val_roll_ddot = max(actual_roll_ddot_plot.max(), pred_roll_ddot_plot.max())
axes1[1, 2].scatter(actual_roll_ddot_plot, pred_roll_ddot_plot, alpha=0.5, s=3)
axes1[1, 2].plot([min_val_roll_ddot, max_val_roll_ddot],
                 [min_val_roll_ddot, max_val_roll_ddot],
                 "r--", linewidth=2, label="Perfect")
axes1[1, 2].set_xlabel("CarMaker Roll Accel (rad/s²)")
axes1[1, 2].set_ylabel("Predicted Roll Accel (rad/s²)")
axes1[1, 2].set_title("Roll Accel Correlation")
axes1[1, 2].legend()
axes1[1, 2].grid(True)
axes1[1, 2].axis("equal")

# Row 3: Pitch Acceleration
axes1[2, 0].plot(time_plot, actual_pitch_ddot_plot, label="CarMaker", linewidth=2)
axes1[2, 0].plot(time_plot, pred_pitch_ddot_plot, label="Model", linewidth=1.5, linestyle="--")
axes1[2, 0].set_xlabel("Time (s)")
axes1[2, 0].set_ylabel("Pitch Accel (rad/s²)")
axes1[2, 0].set_title("Pitch Acceleration")
axes1[2, 0].legend()
axes1[2, 0].grid(True)

axes1[2, 1].plot(time_plot, err_pitch_ddot_plot, linewidth=1.5)
axes1[2, 1].axhline(y=0, color="black", linestyle="--", linewidth=1)
axes1[2, 1].fill_between(time_plot, err_pitch_ddot_plot, 0, alpha=0.3)
axes1[2, 1].set_xlabel("Time (s)")
axes1[2, 1].set_ylabel("Error (rad/s²)")
axes1[2, 1].set_title(f"Pitch Accel Error (MAE: {mae_pitch_ddot:.6f} rad/s²)")
axes1[2, 1].grid(True)

min_val_pitch_ddot = min(actual_pitch_ddot_plot.min(), pred_pitch_ddot_plot.min())
max_val_pitch_ddot = max(actual_pitch_ddot_plot.max(), pred_pitch_ddot_plot.max())
axes1[2, 2].scatter(actual_pitch_ddot_plot, pred_pitch_ddot_plot, alpha=0.5, s=3)
axes1[2, 2].plot([min_val_pitch_ddot, max_val_pitch_ddot],
                 [min_val_pitch_ddot, max_val_pitch_ddot],
                 "r--", linewidth=2, label="Perfect")
axes1[2, 2].set_xlabel("CarMaker Pitch Accel (rad/s²)")
axes1[2, 2].set_ylabel("Predicted Pitch Accel (rad/s²)")
axes1[2, 2].set_title("Pitch Accel Correlation")
axes1[2, 2].legend()
axes1[2, 2].grid(True)
axes1[2, 2].axis("equal")

fig1.tight_layout()

# ============ Figure 3: 상태 변수 (Yaw/Yaw_rate/Vx/Vy) ============
fig3, axes3 = plt.subplots(4, 3, figsize=(18, 18))
fig3.suptitle(f"VehicleBody States - Yaw/Yaw_rate/Vx/Vy (initial {skip_time}s skipped)",
              fontsize=18, fontweight="bold")

# Row 1: Yaw
axes3[0, 0].plot(time_plot, np.rad2deg(cm_yaw_plot), label="CarMaker", linewidth=2)
axes3[0, 0].plot(time_plot, np.rad2deg(pred_yaw_plot), label="Model", linewidth=1.5, linestyle="--")
axes3[0, 0].set_xlabel("Time (s)")
axes3[0, 0].set_ylabel("Yaw (deg)")
axes3[0, 0].set_title("Yaw Angle")
axes3[0, 0].legend()
axes3[0, 0].grid(True)

axes3[0, 1].plot(time_plot, np.rad2deg(err_yaw_plot), linewidth=1.5)
axes3[0, 1].axhline(y=0, color="black", linestyle="--", linewidth=1)
axes3[0, 1].fill_between(time_plot, np.rad2deg(err_yaw_plot), 0, alpha=0.3)
axes3[0, 1].set_xlabel("Time (s)")
axes3[0, 1].set_ylabel("Error (deg)")
axes3[0, 1].set_title(f"Yaw Error (MAE: {np.rad2deg(mae_yaw):.6f} deg)")
axes3[0, 1].grid(True)

min_val_yaw = min(np.rad2deg(cm_yaw_plot).min(), np.rad2deg(pred_yaw_plot).min())
max_val_yaw = max(np.rad2deg(cm_yaw_plot).max(), np.rad2deg(pred_yaw_plot).max())
axes3[0, 2].scatter(np.rad2deg(cm_yaw_plot), np.rad2deg(pred_yaw_plot), alpha=0.5, s=3)
axes3[0, 2].plot([min_val_yaw, max_val_yaw], [min_val_yaw, max_val_yaw],
                 "r--", linewidth=2, label="Perfect")
axes3[0, 2].set_xlabel("CarMaker Yaw (deg)")
axes3[0, 2].set_ylabel("Predicted Yaw (deg)")
axes3[0, 2].set_title("Yaw Correlation")
axes3[0, 2].legend()
axes3[0, 2].grid(True)
axes3[0, 2].axis("equal")

# Row 2: Yaw_rate
axes3[1, 0].plot(time_plot, cm_yaw_rate_plot, label="CarMaker", linewidth=2)
axes3[1, 0].plot(time_plot, pred_yaw_rate_plot, label="Model", linewidth=1.5, linestyle="--")
axes3[1, 0].set_xlabel("Time (s)")
axes3[1, 0].set_ylabel("Yaw_rate (rad/s)")
axes3[1, 0].set_title("Yaw Rate")
axes3[1, 0].legend()
axes3[1, 0].grid(True)

axes3[1, 1].plot(time_plot, err_yaw_rate_plot, linewidth=1.5)
axes3[1, 1].axhline(y=0, color="black", linestyle="--", linewidth=1)
axes3[1, 1].fill_between(time_plot, err_yaw_rate_plot, 0, alpha=0.3)
axes3[1, 1].set_xlabel("Time (s)")
axes3[1, 1].set_ylabel("Error (rad/s)")
axes3[1, 1].set_title(f"Yaw Rate Error (MAE: {mae_yaw_rate:.6f} rad/s)")
axes3[1, 1].grid(True)

min_val_yaw_rate = min(cm_yaw_rate_plot.min(), pred_yaw_rate_plot.min())
max_val_yaw_rate = max(cm_yaw_rate_plot.max(), pred_yaw_rate_plot.max())
axes3[1, 2].scatter(cm_yaw_rate_plot, pred_yaw_rate_plot, alpha=0.5, s=3)
axes3[1, 2].plot([min_val_yaw_rate, max_val_yaw_rate],
                 [min_val_yaw_rate, max_val_yaw_rate],
                 "r--", linewidth=2, label="Perfect")
axes3[1, 2].set_xlabel("CarMaker Yaw_rate (rad/s)")
axes3[1, 2].set_ylabel("Predicted Yaw_rate (rad/s)")
axes3[1, 2].set_title("Yaw_rate Correlation")
axes3[1, 2].legend()
axes3[1, 2].grid(True)
axes3[1, 2].axis("equal")

# Row 3: Vx
axes3[2, 0].plot(time_plot, cm_vx_plot, label="CarMaker", linewidth=2)
axes3[2, 0].plot(time_plot, pred_vx_plot, label="Model", linewidth=1.5, linestyle="--")
axes3[2, 0].set_xlabel("Time (s)")
axes3[2, 0].set_ylabel("Vx (m/s)")
axes3[2, 0].set_title("Longitudinal Velocity")
axes3[2, 0].legend()
axes3[2, 0].grid(True)

axes3[2, 1].plot(time_plot, err_vx_plot, linewidth=1.5)
axes3[2, 1].axhline(y=0, color="black", linestyle="--", linewidth=1)
axes3[2, 1].fill_between(time_plot, err_vx_plot, 0, alpha=0.3)
axes3[2, 1].set_xlabel("Time (s)")
axes3[2, 1].set_ylabel("Error (m/s)")
axes3[2, 1].set_title(f"Vx Error (MAE: {mae_vx:.6f} m/s)")
axes3[2, 1].grid(True)

min_val_vx = min(cm_vx_plot.min(), pred_vx_plot.min())
max_val_vx = max(cm_vx_plot.max(), pred_vx_plot.max())
axes3[2, 2].scatter(cm_vx_plot, pred_vx_plot, alpha=0.5, s=3)
axes3[2, 2].plot([min_val_vx, max_val_vx], [min_val_vx, max_val_vx],
                 "r--", linewidth=2, label="Perfect")
axes3[2, 2].set_xlabel("CarMaker Vx (m/s)")
axes3[2, 2].set_ylabel("Predicted Vx (m/s)")
axes3[2, 2].set_title("Vx Correlation")
axes3[2, 2].legend()
axes3[2, 2].grid(True)
axes3[2, 2].axis("equal")

# Row 4: Vy
axes3[3, 0].plot(time_plot, cm_vy_plot, label="CarMaker", linewidth=2)
axes3[3, 0].plot(time_plot, pred_vy_plot, label="Model", linewidth=1.5, linestyle="--")
axes3[3, 0].set_xlabel("Time (s)")
axes3[3, 0].set_ylabel("Vy (m/s)")
axes3[3, 0].set_title("Lateral Velocity")
axes3[3, 0].legend()
axes3[3, 0].grid(True)

axes3[3, 1].plot(time_plot, err_vy_plot, linewidth=1.5)
axes3[3, 1].axhline(y=0, color="black", linestyle="--", linewidth=1)
axes3[3, 1].fill_between(time_plot, err_vy_plot, 0, alpha=0.3)
axes3[3, 1].set_xlabel("Time (s)")
axes3[3, 1].set_ylabel("Error (m/s)")
axes3[3, 1].set_title(f"Vy Error (MAE: {mae_vy:.6f} m/s)")
axes3[3, 1].grid(True)

min_val_vy = min(cm_vy_plot.min(), pred_vy_plot.min())
max_val_vy = max(cm_vy_plot.max(), pred_vy_plot.max())
axes3[3, 2].scatter(cm_vy_plot, pred_vy_plot, alpha=0.5, s=3)
axes3[3, 2].plot([min_val_vy, max_val_vy], [min_val_vy, max_val_vy],
                 "r--", linewidth=2, label="Perfect")
axes3[3, 2].set_xlabel("CarMaker Vy (m/s)")
axes3[3, 2].set_ylabel("Predicted Vy (m/s)")
axes3[3, 2].set_title("Vy Correlation")
axes3[3, 2].legend()
axes3[3, 2].grid(True)
axes3[3, 2].axis("equal")

fig3.tight_layout()

fig4, axes4 = plt.subplots(3, 3, figsize=(18, 14))
fig4.suptitle(f"VehicleBody States - Heave/Roll/Pitch (initial {skip_time}s skipped)",
              fontsize=18, fontweight="bold")

# Heave
axes4[0, 0].plot(time_plot, cm_heave[skip_idx:], label="CarMaker", linewidth=2)
axes4[0, 0].plot(time_plot, pred_heave_int[skip_idx:], label="Model", linewidth=1.5, linestyle="--")
axes4[0, 0].set_xlabel("Time (s)")
axes4[0, 0].set_ylabel("Heave (m)")
axes4[0, 0].set_title("Heave")
axes4[0, 0].legend()
axes4[0, 0].grid(True)

axes4[0, 1].plot(time_plot, err_heave[skip_idx:]*1000, linewidth=1.5)
axes4[0, 1].axhline(y=0, color="black", linestyle="--", linewidth=1)
axes4[0, 1].fill_between(time_plot, err_heave[skip_idx:]*1000, 0, alpha=0.3)
axes4[0, 1].set_xlabel("Time (s)")
axes4[0, 1].set_ylabel("Error (mm)")
axes4[0, 1].set_title(f"Heave Error (MAE: {mae_heave*1000:.3f} mm)")
axes4[0, 1].grid(True)

min_val_heave_pos = min(cm_heave[skip_idx:].min(), pred_heave_int[skip_idx:].min())
max_val_heave_pos = max(cm_heave[skip_idx:].max(), pred_heave_int[skip_idx:].max())
axes4[0, 2].scatter(cm_heave[skip_idx:], pred_heave_int[skip_idx:], alpha=0.5, s=3)
axes4[0, 2].plot([min_val_heave_pos, max_val_heave_pos],
                 [min_val_heave_pos, max_val_heave_pos],
                 "r--", linewidth=2, label="Perfect")
axes4[0, 2].set_xlabel("CarMaker Heave (m)")
axes4[0, 2].set_ylabel("Predicted Heave (m)")
axes4[0, 2].set_title("Heave Correlation")
axes4[0, 2].legend()
axes4[0, 2].grid(True)
axes4[0, 2].axis("equal")

# Roll
axes4[1, 0].plot(time_plot, cm_roll[skip_idx:], label="CarMaker", linewidth=2)
axes4[1, 0].plot(time_plot, pred_roll_int[skip_idx:], label="Model", linewidth=1.5, linestyle="--")
axes4[1, 0].set_xlabel("Time (s)")
axes4[1, 0].set_ylabel("Roll (rad)")
axes4[1, 0].set_title("Roll")
axes4[1, 0].legend()
axes4[1, 0].grid(True)

axes4[1, 1].plot(time_plot, err_roll[skip_idx:], linewidth=1.5)
axes4[1, 1].axhline(y=0, color="black", linestyle="--", linewidth=1)
axes4[1, 1].fill_between(time_plot, err_roll[skip_idx:], 0, alpha=0.3)
axes4[1, 1].set_xlabel("Time (s)")
axes4[1, 1].set_ylabel("Error (rad)")
axes4[1, 1].set_title(f"Roll Error (MAE: {mae_roll:.6f} rad)")
axes4[1, 1].grid(True)

min_val_roll_pos = min(cm_roll[skip_idx:].min(), pred_roll_int[skip_idx:].min())
max_val_roll_pos = max(cm_roll[skip_idx:].max(), pred_roll_int[skip_idx:].max())
axes4[1, 2].scatter(cm_roll[skip_idx:], pred_roll_int[skip_idx:], alpha=0.5, s=3)
axes4[1, 2].plot([min_val_roll_pos, max_val_roll_pos],
                 [min_val_roll_pos, max_val_roll_pos],
                 "r--", linewidth=2, label="Perfect")
axes4[1, 2].set_xlabel("CarMaker Roll (rad)")
axes4[1, 2].set_ylabel("Predicted Roll (rad)")
axes4[1, 2].set_title("Roll Correlation")
axes4[1, 2].legend()
axes4[1, 2].grid(True)
axes4[1, 2].axis("equal")

# Pitch
axes4[2, 0].plot(time_plot, cm_pitch[skip_idx:], label="CarMaker", linewidth=2)
axes4[2, 0].plot(time_plot, pred_pitch_int[skip_idx:], label="Model", linewidth=1.5, linestyle="--")
axes4[2, 0].set_xlabel("Time (s)")
axes4[2, 0].set_ylabel("Pitch (rad)")
axes4[2, 0].set_title("Pitch")
axes4[2, 0].legend()
axes4[2, 0].grid(True)

axes4[2, 1].plot(time_plot, err_pitch[skip_idx:], linewidth=1.5)
axes4[2, 1].axhline(y=0, color="black", linestyle="--", linewidth=1)
axes4[2, 1].fill_between(time_plot, err_pitch[skip_idx:], 0, alpha=0.3)
axes4[2, 1].set_xlabel("Time (s)")
axes4[2, 1].set_ylabel("Error (rad)")
axes4[2, 1].set_title(f"Pitch Error (MAE: {mae_pitch:.6f} rad)")
axes4[2, 1].grid(True)

min_val_pitch_pos = min(cm_pitch[skip_idx:].min(), pred_pitch_int[skip_idx:].min())
max_val_pitch_pos = max(cm_pitch[skip_idx:].max(), pred_pitch_int[skip_idx:].max())
axes4[2, 2].scatter(cm_pitch[skip_idx:], pred_pitch_int[skip_idx:], alpha=0.5, s=3)
axes4[2, 2].plot([min_val_pitch_pos, max_val_pitch_pos],
                 [min_val_pitch_pos, max_val_pitch_pos],
                 "r--", linewidth=2, label="Perfect")
axes4[2, 2].set_xlabel("CarMaker Pitch (rad)")
axes4[2, 2].set_ylabel("Predicted Pitch (rad)")
axes4[2, 2].set_title("Pitch Correlation")
axes4[2, 2].legend()
axes4[2, 2].grid(True)
axes4[2, 2].axis("equal")

fig4.tight_layout()

print("\nTest completed! 4 figures generated:")
print("  - Figure 1: 9 subplots (Heave/Roll/Pitch accelerations)")
print("  - Figure 2: 9 subplots (Ax/Ay/Yaw_ddot)")
print("  - Figure 3: 12 subplots (Yaw/Yaw_rate/Vx/Vy)")
print("  - Figure 4: 9 subplots (Heave/Roll/Pitch states)")
plt.show()
