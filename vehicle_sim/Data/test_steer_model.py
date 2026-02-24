"""
스티어 모델 검증 테스트 (FL만)
기존 e_corner 스티어 모델을 사용하여 시나리오 데이터 검증
"""
import sys
import os

# vehicle_sim을 포함한 경로 추가
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from vehicle_sim.models.e_corner.steering.steering_model import SteeringModel

# ==================== 데이터 로드 ====================
print("=" * 80)
print("Loading scenario data...")
print("=" * 80)

df = pd.read_csv(r'c:\CM_Projects\SeohanModel_ver7\SeohanModel\vehicle_sim\Data\CM_Lateral_Steer_data.csv')
time = df['Time'].values
dt = time[1] - time[0]

print(f"Data points: {len(df)}")
print(f"Time range: {time[0]:.3f} ~ {time[-1]:.3f} seconds")
print(f"Time step (dt): {dt:.5f} seconds")

# ==================== 입력 데이터 준비 (FL만) ====================
steering_torque_FL = df['DrivMan_Steering_Trq'].values / 1.0  # 절반만 FL에 입력
align_torque_FL = df['cm_TrqAlignFL'].values
actual_angle_FL = df['cm_SteerAngleFL'].values

print(f"\nInput data (FL only):")
print(f"  Steering Torque (half): {steering_torque_FL.min():.2f} ~ {steering_torque_FL.max():.2f} Nm")
print(f"  Aligning Torque: {align_torque_FL.min():.2f} ~ {align_torque_FL.max():.2f} Nm")
print(f"  Actual Angle: {np.rad2deg(actual_angle_FL.min()):.2f} ~ {np.rad2deg(actual_angle_FL.max()):.2f} deg")

# ==================== 스티어 모델 초기화 ====================
print("\n" + "=" * 80)
print("Initializing Steering Model (FL)...")
print("=" * 80)

# YAML 파일에서 파라미터 로드
config_path = r'c:\CM_Projects\SeohanModel_ver7\SeohanModel\vehicle_sim\models\params\vehicle_standard.yaml'
steer_model_FL = SteeringModel(config_path=config_path, corner_id='FL')

print(f"Model parameters (loaded from YAML):")
print(f"  J_cq: {steer_model_FL.params.J_cq} kg*m^2")
print(f"  B_cq: {steer_model_FL.params.B_cq} N*m*s/rad")
print(f"  K_cq: {steer_model_FL.params.K_cq} N*m/rad")
print(f"  max_angle_pos: {np.rad2deg(steer_model_FL.params.max_angle_pos):.2f} deg")
print(f"  max_angle_neg: {np.rad2deg(steer_model_FL.params.max_angle_neg):.2f} deg")

# ==================== 초기 상태 설정 ====================
print("\n" + "=" * 80)
print("Setting initial state from actual data...")
print("=" * 80)

steer_model_FL.state.steering_angle = actual_angle_FL[0]
steer_model_FL.state.steering_rate = 0.0  # 초기 속도는 0으로 가정

print(f"Initial steering angle: {np.rad2deg(actual_angle_FL[0]):.4f}°")
print(f"Initial steering rate: 0.0 rad/s")

# ==================== 시뮬레이션 실행 ====================
print("\n" + "=" * 80)
print("Running Simulation...")
print("=" * 80)

n_steps = len(time)
predicted_angle_FL = np.zeros(n_steps)

for i in range(n_steps):
    predicted_angle_FL[i] = steer_model_FL.update(
        dt=dt,
        T_str=steering_torque_FL[i],
        T_align=align_torque_FL[i]
    )

print("Simulation completed!")

# ==================== 오차 분석 ====================
error_FL = predicted_angle_FL - actual_angle_FL

mae_FL = np.mean(np.abs(error_FL))

print("\n" + "=" * 80)
print("Error Analysis")
print("=" * 80)
print(f"FL: Mean Error={np.rad2deg(mae_FL):.4f}°")
print("=" * 80)

# ==================== 시각화 ====================
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Steer Model Test - FL Only (E-Corner Model)', fontsize=18, fontweight='bold')

# Row 1, Col 1: 조향각 비교
axes[0, 0].plot(time, np.rad2deg(actual_angle_FL), label='Actual', linewidth=2)
axes[0, 0].plot(time, np.rad2deg(predicted_angle_FL), label='Model', linewidth=1.5, linestyle='--')
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('Steer Angle (deg)')
axes[0, 0].set_title('FL Steer Angle')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Row 1, Col 2: 오차
axes[0, 1].plot(time, np.rad2deg(error_FL), linewidth=1.5)
axes[0, 1].axhline(y=0, color='black', linestyle='--', linewidth=1)
axes[0, 1].fill_between(time, np.rad2deg(error_FL), 0, alpha=0.3)
axes[0, 1].set_xlabel('Time (s)')
axes[0, 1].set_ylabel('Error (deg)')
axes[0, 1].set_title(f'FL Error (Mean: {np.rad2deg(mae_FL):.2f}°)')
axes[0, 1].grid(True)

# Row 2, Col 1: 토크 분석 (Net Torque 제거)
axes[1, 0].plot(time, steering_torque_FL, label='Steering Torque (Input/2)', linewidth=1.5)
axes[1, 0].plot(time, align_torque_FL, label='Aligning Torque (Tire)', linewidth=1.5)
axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].set_ylabel('Torque (Nm)')
axes[1, 0].set_title('FL Input Torques')
axes[1, 0].legend()
axes[1, 0].grid(True)

# Row 2, Col 2: Scatter (Actual vs Model)
min_val = min(np.rad2deg(actual_angle_FL).min(), np.rad2deg(predicted_angle_FL).min())
max_val = max(np.rad2deg(actual_angle_FL).max(), np.rad2deg(predicted_angle_FL).max())

axes[1, 1].scatter(np.rad2deg(actual_angle_FL), np.rad2deg(predicted_angle_FL), alpha=0.5, s=3)
axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
axes[1, 1].set_xlabel('Actual Angle (deg)')
axes[1, 1].set_ylabel('Predicted Angle (deg)')
axes[1, 1].set_title('FL Correlation')
axes[1, 1].legend()
axes[1, 1].grid(True)
axes[1, 1].axis('equal')

plt.tight_layout()

print("\nTest completed! 1 figure with 4 subplots generated (FL only).")
plt.show()
