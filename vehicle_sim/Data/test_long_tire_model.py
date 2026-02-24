"""
종방향 타이어 모델 검증 테스트 (FL만)
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
from vehicle_sim.models.e_corner.tire.longitudinal.longitudinal_tire import LongitudinalTireModel

# ==================== 데이터 로드 ====================
print("=" * 80)
print("Loading scenario data...")
print("=" * 80)

df = pd.read_csv(r'c:\CM_Projects\SeohanModel_ver7\SeohanModel\vehicle_sim\Data\CM_Long_Tire.csv')
time = df['Time'].values
dt = time[1] - time[0]

print(f"Data points: {len(df)}")
print(f"Time range: {time[0]:.3f} ~ {time[-1]:.3f} seconds")
print(f"Time step (dt): {dt:.5f} seconds")

# ==================== 입력 데이터 준비 (FL만) ====================
wheel_speed_FL = df['cm_WheelSpd_FL'].values  # rad/s
wheel_vx_FL = df['cm_vxFL'].values  # m/s
vertical_load_FL = df['cm_FzFL'].values  # N

# 실제 출력 (검증 대상)
actual_Fx_FL = df['cm_FxFL'].values
actual_slip_FL = df['cm_LongSlipFL'].values

print(f"\nInput data (FL only):")
print(f"  Wheel Speed (omega): {wheel_speed_FL.min():.2f} ~ {wheel_speed_FL.max():.2f} rad/s")
print(f"  Wheel Vx: {wheel_vx_FL.min():.2f} ~ {wheel_vx_FL.max():.2f} m/s")
print(f"  Vertical Load (Fz): {vertical_load_FL.min():.2f} ~ {vertical_load_FL.max():.2f} N")

print(f"\nActual outputs (FL only):")
print(f"  Longitudinal Force (Fx): {actual_Fx_FL.min():.2f} ~ {actual_Fx_FL.max():.2f} N")
print(f"  Longitudinal Slip: {actual_slip_FL.min():.4f} ~ {actual_slip_FL.max():.4f}")

# ==================== 타이어 모델 초기화 ====================
print("\n" + "=" * 80)
print("Initializing Longitudinal Tire Model (FL)...")
print("=" * 80)

# YAML 파일에서 파라미터 로드
config_path = r'c:\CM_Projects\SeohanModel_ver7\SeohanModel\vehicle_sim\models\params\vehicle_standard.yaml'
tire_model_FL = LongitudinalTireModel(config_path=config_path)

print(f"Model parameters (loaded from YAML):")
print(f"  C_x: {tire_model_FL.params.C_x}")
print(f"  mu: {tire_model_FL.params.mu}")
print(f"  v_min: {tire_model_FL.params.v_min} m/s")
print(f"  R_eff: {tire_model_FL.params.R_eff} m")

# ==================== 시뮬레이션 실행 ====================
print("\n" + "=" * 80)
print("Running Simulation...")
print("=" * 80)

n_steps = len(time)
predicted_Fx_FL = np.zeros(n_steps)
predicted_slip_FL = np.zeros(n_steps)

for i in range(n_steps):
    # 슬립 비율 계산
    kappa = tire_model_FL.calculate_slip_ratio(
        omega_wheel=wheel_speed_FL[i],
        V_wheel_x=wheel_vx_FL[i]
    )

    # 종방향 힘 계산
    Fx = tire_model_FL.calculate_force(
        kappa=kappa,
        F_z_tire=vertical_load_FL[i]
    )

    predicted_Fx_FL[i] = Fx
    predicted_slip_FL[i] = kappa

print("Simulation completed!")

# ==================== 오차 분석 ====================
error_Fx_FL = predicted_Fx_FL - actual_Fx_FL
error_slip_FL = predicted_slip_FL - actual_slip_FL

mae_Fx_FL = np.mean(np.abs(error_Fx_FL))
mae_slip_FL = np.mean(np.abs(error_slip_FL))

print("\n" + "=" * 80)
print("Error Analysis")
print("=" * 80)
print(f"FL Longitudinal Force (Fx): Mean Error={mae_Fx_FL:.2f} N")
print(f"FL Longitudinal Slip: Mean Error={mae_slip_FL:.4f}")
print("=" * 80)

# ==================== 시각화 ====================
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Longitudinal Tire Model Test - FL Only (E-Corner Model)', fontsize=18, fontweight='bold')

# Row 1, Col 1: 슬립 비율 비교
axes[0, 0].plot(time, actual_slip_FL, label='Actual', linewidth=2)
axes[0, 0].plot(time, predicted_slip_FL, label='Model', linewidth=1.5, linestyle='--')
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('Longitudinal Slip')
axes[0, 0].set_title('FL Longitudinal Slip Ratio')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Row 1, Col 2: 종방향 힘 비교
axes[0, 1].plot(time, actual_Fx_FL, label='Actual', linewidth=2)
axes[0, 1].plot(time, predicted_Fx_FL, label='Model', linewidth=1.5, linestyle='--')
axes[0, 1].set_xlabel('Time (s)')
axes[0, 1].set_ylabel('Longitudinal Force (N)')
axes[0, 1].set_title('FL Longitudinal Force (Fx)')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Row 2, Col 1: Fx 오차
axes[1, 0].plot(time, error_Fx_FL, linewidth=1.5)
axes[1, 0].axhline(y=0, color='black', linestyle='--', linewidth=1)
axes[1, 0].fill_between(time, error_Fx_FL, 0, alpha=0.3)
axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].set_ylabel('Error (N)')
axes[1, 0].set_title(f'Fx Error (Mean: {mae_Fx_FL:.2f} N)')
axes[1, 0].grid(True)

# Row 2, Col 2: Fx Correlation (Actual vs Model)
min_val = min(actual_Fx_FL.min(), predicted_Fx_FL.min())
max_val = max(actual_Fx_FL.max(), predicted_Fx_FL.max())

axes[1, 1].scatter(actual_Fx_FL, predicted_Fx_FL, alpha=0.5, s=3)
axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
axes[1, 1].set_xlabel('Actual Fx (N)')
axes[1, 1].set_ylabel('Predicted Fx (N)')
axes[1, 1].set_title('FL Fx Correlation')
axes[1, 1].legend()
axes[1, 1].grid(True)
axes[1, 1].axis('equal')

plt.tight_layout()

print("\nTest completed! 1 figure with 4 subplots generated (FL only).")
plt.show()
