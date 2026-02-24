"""
서스펜션 모델 정적 평형 검증 테스트 (FL, FR - 전륜)
평형 상태에서 모델이 올바른 초기값을 가지는지 검증
"""
import sys
import os

# vehicle_sim을 포함한 경로 추가
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from vehicle_sim.models.e_corner.suspension.suspension_model import SuspensionModel

# ==================== 데이터 로드 ====================
print("=" * 80)
print("Loading equilibrium scenario data...")
print("=" * 80)

df = pd.read_csv(r'c:\CM_Projects\SeohanModel_ver7\SeohanModel\vehicle_sim\Data\CM_Suspension_equal.csv')
time = df['Time'].values
dt = time[1] - time[0]

print(f"Data points: {len(df)}")
print(f"Time range: {time[0]:.3f} ~ {time[-1]:.3f} seconds")
print(f"Time step (dt): {dt:.5f} seconds")

# ==================== 입력 데이터 준비 (FL, FR) ====================
# CRITICAL: cm_Heave is ABSOLUTE height, not deviation!
# We must convert to deviation coordinates for model comparison
heave_abs = df['cm_Heave'].values
roll_abs = df['cm_Roll'].values
pitch_abs = df['cm_Pitch'].values

print(f"\n[DATA CORRECTION]")
print(f"cm_Heave is absolute height: mean={heave_abs.mean():.6f} m, range={heave_abs.max()-heave_abs.min():.6f} m")
print(f"Converting to deviation coordinates (subtract initial value)")
print(f"WARNING: cm_Heave_dot from CSV is NOT d(cm_Heave)/dt! Using numerical differentiation.")

# Convert to deviation coordinates
heave = heave_abs - heave_abs[0]
roll = roll_abs - roll_abs[0]
pitch = pitch_abs - pitch_abs[0]

# Calculate velocity from numerical differentiation (correct way!)
heave_dot = np.gradient(heave, time)
roll_dot = np.gradient(roll, time)
pitch_dot = np.gradient(pitch, time)

# 노면 높이 (FL, FR)
z_road_FL = df['cm_RoadZFL'].values
z_road_FR = df['cm_RoadZFR'].values

# 서스펜션 토크 (액티브 서스펜션 입력) - 평형이므로 0
T_susp_FL = np.zeros_like(time)
T_susp_FR = np.zeros_like(time)

# 실제 출력 (검증 대상)
actual_F_z_FL = df['cm_FzFL'].values
actual_F_z_FR = df['cm_FzFR'].values

# 언스프렁 절대 높이
if 'cm_UnsprungZFL' in df.columns:
    actual_z_u_abs_FL = df['cm_UnsprungZFL'].values
else:
    actual_z_u_abs_FL = z_road_FL + df['cm_TireDeltaFL'].values

if 'cm_UnsprungZFR' in df.columns:
    actual_z_u_abs_FR = df['cm_UnsprungZFR'].values
else:
    actual_z_u_abs_FR = z_road_FR + df['cm_TireDeltaFR'].values

actual_delta_t_FL = actual_z_u_abs_FL - z_road_FL
actual_delta_t_FR = actual_z_u_abs_FR - z_road_FR

# 차체 언스프렁 위치
actual_z_sprung_FL = df['cm_SprungZFL'].values
actual_z_sprung_FR = df['cm_SprungZFR'].values

print(f"\nInput data (FL, FR) - Should be near equilibrium:")
print(f"  Heave: {heave.min():.6f} ~ {heave.max():.6f} m (should be ~0)")
print(f"  Roll: {np.rad2deg(roll.min()):.6f} ~ {np.rad2deg(roll.max()):.6f} deg (should be ~0)")
print(f"  Pitch: {np.rad2deg(pitch.min()):.6f} ~ {np.rad2deg(pitch.max()):.6f} deg (should be ~0)")

# ==================== 파라미터 수동 설정 ====================
OVERRIDE_PARAMS = False

MANUAL_PARAMS = {
    'K_spring': 44425.0,
    'C_damper_compression': 1595.2,
    'C_damper_rebound': 4475.7,
    'K_t': 406884,
    'C_t': 4608.6,
    'm_sprung': 1806.8,
    'm_unsprung': 74.12,
}

# ==================== 서스펜션 모델 초기화 (FL, FR) ====================
print("\n" + "=" * 80)
print("Initializing Suspension Models (FL, FR)...")
print("=" * 80)

config_path = r'c:\CM_Projects\SeohanModel_ver7\SeohanModel\vehicle_sim\models\params\vehicle_standard.yaml'
suspension_model_FL = SuspensionModel(corner_id='FL', config_path=config_path)
suspension_model_FR = SuspensionModel(corner_id='FR', config_path=config_path)

models = {'FL': suspension_model_FL, 'FR': suspension_model_FR}

if OVERRIDE_PARAMS:
    print("\n*** 파라미터 수동 오버라이드 활성화 ***")

    for corner_id, model in models.items():
        model.params.K_spring = MANUAL_PARAMS['K_spring']
        model.params.C_damper_compression = MANUAL_PARAMS['C_damper_compression']
        model.params.C_damper_rebound = MANUAL_PARAMS['C_damper_rebound']
        model.tire_params.K_t = MANUAL_PARAMS['K_t']
        model.tire_params.C_t = MANUAL_PARAMS['C_t']
        model.unsprung_params.m_u = MANUAL_PARAMS['m_unsprung']

        from vehicle_sim.utils.config_loader import load_param
        physics_param = load_param('physics', config_path)
        vehicle_spec = load_param('vehicle_spec', config_path)

        m_s = MANUAL_PARAMS['m_sprung']
        m_u = MANUAL_PARAMS['m_unsprung']
        g = float(physics_param.get('g', 9.81))
        R_w = float(vehicle_spec.get('wheel', {}).get('R_eff', 0.327))
        K_t = MANUAL_PARAMS['K_t']
        K_spring = MANUAL_PARAMS['K_spring']
        z_CG0 = model.params.z_CG0

        geometry = vehicle_spec.get('geometry', {})
        front_ratio = float(geometry.get('front_load_ratio', 0.611))
        m_s_corner = (m_s * front_ratio) / 2.0

        F_sus_eq = m_s_corner * g
        F_z_eq = (m_s_corner + m_u) * g
        delta_t_eq = F_z_eq / K_t
        z_u_0 = R_w - delta_t_eq
        delta_s_comp = F_sus_eq / K_spring
        delta_s_eq = -delta_s_comp
        F_spring_eq = K_spring * delta_s_comp
        F_tire_eq = K_t * delta_t_eq
        L_s0 = (z_CG0 - z_u_0) + delta_s_comp

        model.params.L_s0 = L_s0
        model._z_u_0 = z_u_0
        model._delta_s_eq = delta_s_eq
        model._F_spring_eq = F_spring_eq
        model._F_sus_eq = F_spring_eq
        model._delta_t_eq = delta_t_eq
        model._F_tire_eq = F_tire_eq

    print(f"  전륜 차체 질량 (한쪽): {m_s_corner:.2f} kg")

from vehicle_sim.utils.config_loader import load_param
physics_param = load_param('physics', config_path)
g = float(physics_param.get('g', 9.81))
m_s_FL = suspension_model_FL.sprung_params.m_s_corner
m_s_FR = suspension_model_FR.sprung_params.m_s_corner

actual_F_sus_FL = m_s_FL * g * np.ones_like(time)
actual_F_sus_FR = m_s_FR * g * np.ones_like(time)

# CarMaker 기반 계산
cm_z_u_abs_FL = actual_z_u_abs_FL
cm_z_u_abs_FR = actual_z_u_abs_FR

cm_delta_t_comp_FL = suspension_model_FL.params.R_w + z_road_FL - cm_z_u_abs_FL
cm_delta_t_comp_FR = suspension_model_FR.params.R_w + z_road_FR - cm_z_u_abs_FR

z_CG0_CM = suspension_model_FL.params.z_CG0
cm_gap_s_to_u_FL = actual_z_sprung_FL - cm_z_u_abs_FL
cm_gap_s_to_u_FR = actual_z_sprung_FR - cm_z_u_abs_FR
cm_gap_ref = z_CG0_CM - suspension_model_FL.params.R_w

cm_delta_s_comp_FL = cm_gap_s_to_u_FL - cm_gap_ref
cm_delta_s_comp_FR = cm_gap_s_to_u_FR - cm_gap_ref

# heave is already deviation coordinate (heave[0] = 0)
heave_for_model = heave
heave_dot_for_model = heave_dot
roll_for_model = roll
roll_dot_for_model = roll_dot
pitch_for_model = pitch
pitch_dot_for_model = pitch_dot

# ==================== 시뮬레이션 실행 ====================
print("\n" + "=" * 80)
print("Running Simulation at 10000Hz...")
print("=" * 80)

dt_model = 0.0001
substeps = int(dt / dt_model)

n_steps = len(time)
predicted_F_sus_FL = np.zeros(n_steps)
predicted_F_z_FL = np.zeros(n_steps)
predicted_F_sus_FR = np.zeros(n_steps)
predicted_F_z_FR = np.zeros(n_steps)

# 초기 상태 설정
suspension_model_FL.reset()
suspension_model_FL.state.z_u_abs = actual_z_u_abs_FL[0]
suspension_model_FL.state.delta_t = cm_delta_t_comp_FL[0]
suspension_model_FL.state.delta_s = cm_delta_s_comp_FL[0]
suspension_model_FL.state.z_body_abs = actual_z_sprung_FL[0]
suspension_model_FL.state.F_s = actual_F_sus_FL[0]
suspension_model_FL.state.F_z = actual_F_z_FL[0]

suspension_model_FR.reset()
suspension_model_FR.state.z_u_abs = actual_z_u_abs_FR[0]
suspension_model_FR.state.delta_t = cm_delta_t_comp_FR[0]
suspension_model_FR.state.delta_s = cm_delta_s_comp_FR[0]
suspension_model_FR.state.z_body_abs = actual_z_sprung_FR[0]
suspension_model_FR.state.F_s = actual_F_sus_FR[0]
suspension_model_FR.state.F_z = actual_F_z_FR[0]

for i in range(n_steps):
    X_body = np.array([
        heave_for_model[i],
        roll_for_model[i],
        pitch_for_model[i],
        heave_dot_for_model[i],
        roll_dot_for_model[i],
        pitch_dot_for_model[i]
    ])

    z_road_FL_current = z_road_FL[i]
    z_road_FR_current = z_road_FR[i]

    if i > 0:
        z_road_dot_FL = (z_road_FL[i] - z_road_FL[i-1]) / dt
        z_road_dot_FR = (z_road_FR[i] - z_road_FR[i-1]) / dt
    else:
        z_road_dot_FL = 0.0
        z_road_dot_FR = 0.0

    for sub in range(substeps):
        F_sus_FL, F_z_FL = suspension_model_FL.update(
            dt=dt_model,
            T_susp=T_susp_FL[i],
            X_body=X_body,
            z_road=z_road_FL_current,
            z_road_dot=z_road_dot_FL
        )

        F_sus_FR, F_z_FR = suspension_model_FR.update(
            dt=dt_model,
            T_susp=T_susp_FR[i],
            X_body=X_body,
            z_road=z_road_FR_current,
            z_road_dot=z_road_dot_FR
        )

    predicted_F_sus_FL[i] = F_sus_FL
    predicted_F_z_FL[i] = F_z_FL
    predicted_F_sus_FR[i] = F_sus_FR
    predicted_F_z_FR[i] = F_z_FR

print("Simulation completed!")

# ==================== 시각화 ====================
plot_skip_time = 0.5
plot_mask = time >= plot_skip_time
plot_time = time[plot_mask]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Suspension Equilibrium Test - Front Wheels (FL, FR)', fontsize=18, fontweight='bold')

# Row 1: FL
axes[0, 0].plot(plot_time, actual_F_sus_FL[plot_mask], label='CarMaker', linewidth=2)
axes[0, 0].plot(plot_time, predicted_F_sus_FL[plot_mask], label='Model', linewidth=1.5, linestyle='--')
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('Force (N)')
axes[0, 0].set_title('FL Suspension Force')
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[0, 1].plot(plot_time, actual_F_z_FL[plot_mask], label='CarMaker', linewidth=2)
axes[0, 1].plot(plot_time, predicted_F_z_FL[plot_mask], label='Model', linewidth=1.5, linestyle='--')
axes[0, 1].set_xlabel('Time (s)')
axes[0, 1].set_ylabel('Force (N)')
axes[0, 1].set_title('FL Tire Vertical Force')
axes[0, 1].legend()
axes[0, 1].grid(True)

categories = ['F_sus', 'F_z']
cm_values_FL = [actual_F_sus_FL.mean(), actual_F_z_FL.mean()]
model_values_FL = [suspension_model_FL._F_sus_eq, suspension_model_FL._F_tire_eq]
x = np.arange(len(categories))
width = 0.35
axes[0, 2].bar(x - width/2, cm_values_FL, width, label='CarMaker')
axes[0, 2].bar(x + width/2, model_values_FL, width, label='Model')
axes[0, 2].set_ylabel('Force (N)')
axes[0, 2].set_title('FL Equilibrium Values')
axes[0, 2].set_xticks(x)
axes[0, 2].set_xticklabels(categories)
axes[0, 2].legend()
axes[0, 2].grid(True, axis='y')

# Row 2: FR
axes[1, 0].plot(plot_time, actual_F_sus_FR[plot_mask], label='CarMaker', linewidth=2)
axes[1, 0].plot(plot_time, predicted_F_sus_FR[plot_mask], label='Model', linewidth=1.5, linestyle='--')
axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].set_ylabel('Force (N)')
axes[1, 0].set_title('FR Suspension Force')
axes[1, 0].legend()
axes[1, 0].grid(True)

axes[1, 1].plot(plot_time, actual_F_z_FR[plot_mask], label='CarMaker', linewidth=2)
axes[1, 1].plot(plot_time, predicted_F_z_FR[plot_mask], label='Model', linewidth=1.5, linestyle='--')
axes[1, 1].set_xlabel('Time (s)')
axes[1, 1].set_ylabel('Force (N)')
axes[1, 1].set_title('FR Tire Vertical Force')
axes[1, 1].legend()
axes[1, 1].grid(True)

cm_values_FR = [actual_F_sus_FR.mean(), actual_F_z_FR.mean()]
model_values_FR = [suspension_model_FR._F_sus_eq, suspension_model_FR._F_tire_eq]
axes[1, 2].bar(x - width/2, cm_values_FR, width, label='CarMaker')
axes[1, 2].bar(x + width/2, model_values_FR, width, label='Model')
axes[1, 2].set_ylabel('Force (N)')
axes[1, 2].set_title('FR Equilibrium Values')
axes[1, 2].set_xticks(x)
axes[1, 2].set_xticklabels(categories)
axes[1, 2].legend()
axes[1, 2].grid(True, axis='y')

plt.tight_layout()
print("\nTest completed! Front wheel (FL, FR) plots generated.")
plt.show()
