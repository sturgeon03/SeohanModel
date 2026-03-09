"""
레터럴 타이어 모델 검증 테스트 (FL만)
기존 e_corner 타이어 모델을 사용하여 시나리오 데이터 검증
"""
import sys
import os

# vehicle_sim을 포함한 경로 추가
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from vehicle_sim.models.e_corner.tire.lateral.lateral_tire import LateralTireModel

# ==================== 데이터 로드 ====================
print("=" * 80)
print("Loading scenario data...")
print("=" * 80)

df = pd.read_csv(r'c:\CM_Projects\SeohanModel_ver7\SeohanModel\vehicle_sim\Data\CM_Lateral_Steer.csv')
time = df['Time'].values
dt = time[1] - time[0]

print(f"Data points: {len(df)}")
print(f"Time range: {time[0]:.3f} ~ {time[-1]:.3f} seconds")
print(f"Time step (dt): {dt:.5f} seconds")

# ==================== 입력 데이터 준비 (FL만) ====================
wheel_vx_FL = df['cm_vxFL'].values
wheel_vy_FL = df['cm_vyFL'].values
vertical_load_FL = df['cm_FzFL'].values

# 실제 출력 (검증 대상)
actual_Fy_FL = df['cm_FyFL'].values
actual_Mz_FL = - df['cm_TrqAlignFL'].values
actual_slip_angle_FL = df['cm_SlipAngleFL'].values

print(f"\nInput data (FL only):")
print(f"  Wheel Vx: {wheel_vx_FL.min():.2f} ~ {wheel_vx_FL.max():.2f} m/s")
print(f"  Wheel Vy: {wheel_vy_FL.min():.2f} ~ {wheel_vy_FL.max():.2f} m/s")
print(f"  Vertical Load (Fz): {vertical_load_FL.min():.2f} ~ {vertical_load_FL.max():.2f} N")

print(f"\nActual outputs (FL only):")
print(f"  Lateral Force (Fy): {actual_Fy_FL.min():.2f} ~ {actual_Fy_FL.max():.2f} N")
print(f"  Aligning Torque (Mz): {actual_Mz_FL.min():.2f} ~ {actual_Mz_FL.max():.2f} Nm")
print(f"  Slip Angle: {np.rad2deg(actual_slip_angle_FL.min()):.2f} ~ {np.rad2deg(actual_slip_angle_FL.max()):.2f} deg")

# ==================== 타이어 모델 초기화 ====================
print("\n" + "=" * 80)
print("Initializing Lateral Tire Model (FL)...")
print("=" * 80)

# YAML 파일에서 파라미터 로드
config_path = r'c:\CM_Projects\SeohanModel_ver7\SeohanModel\vehicle_sim\models\params\vehicle_standard.yaml'
tire_model_FL = LateralTireModel(config_path=config_path)

print(f"Model parameters (loaded from YAML):")
print(f"  C_alpha: {tire_model_FL.params.C_alpha} N/rad")
print(f"  mu: {tire_model_FL.params.mu}")
print(f"  trail: {tire_model_FL.params.trail} m")

# ==================== 시뮬레이션 실행 ====================
print("\n" + "=" * 80)
print("Running Simulation...")
print("=" * 80)

n_steps = len(time)
predicted_Fy_FL = np.zeros(n_steps)
predicted_Mz_FL = np.zeros(n_steps)
predicted_slip_angle_FL = np.zeros(n_steps)

for i in range(n_steps):
    predicted_Fy_FL[i] = tire_model_FL.update(
        V_wheel_x=wheel_vx_FL[i],
        V_wheel_y=wheel_vy_FL[i],
        F_tire=vertical_load_FL[i]
    )
    predicted_Mz_FL[i] = tire_model_FL.state.aligning_torque
    predicted_slip_angle_FL[i] = tire_model_FL.state.slip_angle

print("Simulation completed!")

# ==================== 오차 분석 ====================
error_Fy_FL = predicted_Fy_FL - actual_Fy_FL
error_Mz_FL = predicted_Mz_FL - actual_Mz_FL
error_slip_FL = predicted_slip_angle_FL - actual_slip_angle_FL

mae_Fy_FL = np.mean(np.abs(error_Fy_FL))
mae_Mz_FL = np.mean(np.abs(error_Mz_FL))
mae_slip_FL = np.mean(np.abs(error_slip_FL))

print("\n" + "=" * 80)
print("Error Analysis")
print("=" * 80)
print(f"FL Lateral Force (Fy): Mean Error={mae_Fy_FL:.2f} N")
print(f"FL Aligning Torque (Mz): Mean Error={mae_Mz_FL:.2f} Nm")
print(f"FL Slip Angle: Mean Error={np.rad2deg(mae_slip_FL):.4f}°")
print("=" * 80)

# ==================== 시각화 ====================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Lateral Tire Model Test - FL Only (E-Corner Model)', fontsize=18, fontweight='bold')

# Row 1, Col 1: 슬립각 비교
axes[0, 0].plot(time, np.rad2deg(actual_slip_angle_FL), label='Actual', linewidth=2)
axes[0, 0].plot(time, np.rad2deg(predicted_slip_angle_FL), label='Model', linewidth=1.5, linestyle='--')
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('Slip Angle (deg)')
axes[0, 0].set_title('FL Slip Angle')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Row 1, Col 2: 횡력 비교
axes[0, 1].plot(time, actual_Fy_FL, label='Actual', linewidth=2)
axes[0, 1].plot(time, predicted_Fy_FL, label='Model', linewidth=1.5, linestyle='--')
axes[0, 1].set_xlabel('Time (s)')
axes[0, 1].set_ylabel('Lateral Force (N)')
axes[0, 1].set_title('FL Lateral Force (Fy)')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Row 1, Col 3: 얼라이닝 토크 비교
axes[0, 2].plot(time, actual_Mz_FL, label='Actual', linewidth=2)
axes[0, 2].plot(time, predicted_Mz_FL, label='Model', linewidth=1.5, linestyle='--')
axes[0, 2].set_xlabel('Time (s)')
axes[0, 2].set_ylabel('Aligning Torque (Nm)')
axes[0, 2].set_title('FL Aligning Torque (Mz)')
axes[0, 2].legend()
axes[0, 2].grid(True)

# Row 2, Col 1: 횡력 오차
axes[1, 0].plot(time, error_Fy_FL, linewidth=1.5)
axes[1, 0].axhline(y=0, color='black', linestyle='--', linewidth=1)
axes[1, 0].fill_between(time, error_Fy_FL, 0, alpha=0.3)
axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].set_ylabel('Error (N)')
axes[1, 0].set_title(f'Fy Error (Mean: {mae_Fy_FL:.2f} N)')
axes[1, 0].grid(True)

# Row 2, Col 2: 얼라이닝 토크 오차
axes[1, 1].plot(time, error_Mz_FL, linewidth=1.5)
axes[1, 1].axhline(y=0, color='black', linestyle='--', linewidth=1)
axes[1, 1].fill_between(time, error_Mz_FL, 0, alpha=0.3)
axes[1, 1].set_xlabel('Time (s)')
axes[1, 1].set_ylabel('Error (Nm)')
axes[1, 1].set_title(f'Mz Error (Mean: {mae_Mz_FL:.2f} Nm)')
axes[1, 1].grid(True)

# Row 2, Col 3: Fy Correlation (Actual vs Model)
min_val = min(actual_Fy_FL.min(), predicted_Fy_FL.min())
max_val = max(actual_Fy_FL.max(), predicted_Fy_FL.max())

axes[1, 2].scatter(actual_Fy_FL, predicted_Fy_FL, alpha=0.5, s=3)
axes[1, 2].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
axes[1, 2].set_xlabel('Actual Fy (N)')
axes[1, 2].set_ylabel('Predicted Fy (N)')
axes[1, 2].set_title('FL Fy Correlation')
axes[1, 2].legend()
axes[1, 2].grid(True)
axes[1, 2].axis('equal')

plt.tight_layout()

print("\nTest completed! 1 figure with 6 subplots generated (FL only).")
plt.show()
