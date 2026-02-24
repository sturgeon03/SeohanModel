"""
서스펜션 모델 정적 평형 검증 테스트 (RL, RR - 후륜)
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

# ==================== 입력 데이터 준비 (RL, RR) ====================
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

# 노면 높이 (RL, RR)
z_road_RL = df['cm_RoadZRL'].values
z_road_RR = df['cm_RoadZRR'].values

# 서스펜션 토크 (액티브 서스펜션 입력) - 평형이므로 0
T_susp_RL = np.zeros_like(time)
T_susp_RR = np.zeros_like(time)

# 실제 출력 (검증 대상)
actual_F_z_RL = df['cm_FzRL'].values
actual_F_z_RR = df['cm_FzRR'].values

# 언스프렁 절대 높이
if 'cm_UnsprungZRL' in df.columns:
    actual_z_u_abs_RL = df['cm_UnsprungZRL'].values
else:
    actual_z_u_abs_RL = z_road_RL + df['cm_TireDeltaRL'].values

if 'cm_UnsprungZRR' in df.columns:
    actual_z_u_abs_RR = df['cm_UnsprungZRR'].values
else:
    actual_z_u_abs_RR = z_road_RR + df['cm_TireDeltaRR'].values

actual_delta_t_RL = actual_z_u_abs_RL - z_road_RL
actual_delta_t_RR = actual_z_u_abs_RR - z_road_RR

# 차체 언스프렁 위치
actual_z_sprung_RL = df['cm_SprungZRL'].values
actual_z_sprung_RR = df['cm_SprungZRR'].values

print(f"\nInput data (RL, RR) - Should be near equilibrium:")
print(f"  Heave: {heave.min():.6f} ~ {heave.max():.6f} m (should be ~0)")
print(f"  Roll: {np.rad2deg(roll.min()):.6f} ~ {np.rad2deg(roll.max()):.6f} deg (should be ~0)")
print(f"  Pitch: {np.rad2deg(pitch.min()):.6f} ~ {np.rad2deg(pitch.max()):.6f} deg (should be ~0)")

# ==================== 파라미터 수동 설정 ====================
# True로 설정하면 YAML 대신 아래 값 사용
OVERRIDE_PARAMS = False

# 수동 설정 파라미터 (뒤축용)
MANUAL_PARAMS = {
    # 서스펜션 파라미터
    'K_spring': 44425.0,    # 스프링 강성 [N/m]
    'C_damper_compression': 1595.2,  # 후륜 압축 댐퍼 감쇠 [N*s/m]
    'C_damper_rebound': 4475.7,      # 후륜 리바운드 댐퍼 감쇠 [N*s/m]

    # 타이어 파라미터
    'K_t': 406884,         # 타이어 수직 강성 [N/m]
    'C_t': 4608.6,            # 타이어 수직 댐핑 [N*s/m]

    # 질량 파라미터
    'm_sprung': 1806.8,      # 전체 스프렁 질량 [kg]
    'm_unsprung': 74.12,      # 언스프렁 질량 (한 바퀴) [kg]
}

# ==================== 서스펜션 모델 초기화 (RL, RR) ====================
print("\n" + "=" * 80)
print("Initializing Suspension Models (RL, RR)...")
print("=" * 80)

# YAML 파일에서 파라미터 로드
config_path = r'c:\CM_Projects\SeohanModel_ver7\SeohanModel\vehicle_sim\models\params\vehicle_standard.yaml'
suspension_model_RL = SuspensionModel(corner_id='RL', config_path=config_path)
suspension_model_RR = SuspensionModel(corner_id='RR', config_path=config_path)

models = {'RL': suspension_model_RL, 'RR': suspension_model_RR}

# 파라미터 오버라이드
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
        rear_ratio = float(geometry.get('rear_load_ratio', 0.389))
        m_s_corner = (m_s * rear_ratio) / 2.0

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

    print(f"  후륜 차체 질량 (한쪽): {m_s_corner:.2f} kg")

from vehicle_sim.utils.config_loader import load_param
physics_param = load_param('physics', config_path)
g = float(physics_param.get('g', 9.81))
m_s_RL = suspension_model_RL.sprung_params.m_s_corner
m_s_RR = suspension_model_RR.sprung_params.m_s_corner

actual_F_sus_RL = m_s_RL * g * np.ones_like(time)
actual_F_sus_RR = m_s_RR * g * np.ones_like(time)

# ==================== CarMaker 기반 actual 계산 ====================
z_CG0_CM = suspension_model_RL.params.z_CG0

cm_z_u_abs_RL = actual_z_u_abs_RL
cm_z_u_abs_RR = actual_z_u_abs_RR

cm_delta_t_comp_RL = suspension_model_RL.params.R_w + z_road_RL - cm_z_u_abs_RL
cm_delta_t_comp_RR = suspension_model_RR.params.R_w + z_road_RR - cm_z_u_abs_RR

cm_gap_s_to_u_RL = actual_z_sprung_RL - cm_z_u_abs_RL
cm_gap_s_to_u_RR = actual_z_sprung_RR - cm_z_u_abs_RR
cm_gap_ref = z_CG0_CM - suspension_model_RL.params.R_w

cm_delta_s_comp_RL = cm_gap_s_to_u_RL - cm_gap_ref
cm_delta_s_comp_RR = cm_gap_s_to_u_RR - cm_gap_ref

# heave is already deviation coordinate (heave[0] = 0)
heave_for_model = heave
heave_dot_for_model = heave_dot
roll_for_model = roll
roll_dot_for_model = roll_dot
pitch_for_model = pitch
pitch_dot_for_model = pitch_dot

# ==================== 시뮬레이션 실행 (10000Hz 모델) ====================
print("\n" + "=" * 80)
print("Running Simulation at 10000Hz (Zero-Order Hold from 1000Hz CarMaker)...")
print("=" * 80)

# 모델 시간 스텝 (10000Hz = 0.0001s)
dt_model = 0.0001
substeps = int(dt / dt_model)  # CarMaker 1 스텝당 모델 서브스텝 수

print(f"CarMaker dt: {dt:.5f}s ({1/dt:.0f}Hz)")
print(f"Model dt: {dt_model:.5f}s ({1/dt_model:.0f}Hz)")
print(f"Substeps per CarMaker step: {substeps}")

n_steps = len(time)
predicted_F_sus_RL = np.zeros(n_steps)
predicted_F_z_RL = np.zeros(n_steps)
predicted_F_sus_RR = np.zeros(n_steps)
predicted_F_z_RR = np.zeros(n_steps)

# 초기 상태 설정
suspension_model_RL.reset()
suspension_model_RL.state.z_u_abs = actual_z_u_abs_RL[0]
suspension_model_RL.state.delta_t = cm_delta_t_comp_RL[0]
suspension_model_RL.state.delta_s = cm_delta_s_comp_RL[0]
suspension_model_RL.state.z_body_abs = actual_z_sprung_RL[0]
suspension_model_RL.state.F_s = actual_F_sus_RL[0]
suspension_model_RL.state.F_z = actual_F_z_RL[0]

suspension_model_RR.reset()
suspension_model_RR.state.z_u_abs = actual_z_u_abs_RR[0]
suspension_model_RR.state.delta_t = cm_delta_t_comp_RR[0]
suspension_model_RR.state.delta_s = cm_delta_s_comp_RR[0]
suspension_model_RR.state.z_body_abs = actual_z_sprung_RR[0]
suspension_model_RR.state.F_s = actual_F_sus_RR[0]
suspension_model_RR.state.F_z = actual_F_z_RR[0]

for i in range(n_steps):
    X_body = np.array([
        heave_for_model[i],
        roll_for_model[i],
        pitch_for_model[i],
        heave_dot_for_model[i],
        roll_dot_for_model[i],
        pitch_dot_for_model[i]
    ])

    z_road_RL_current = z_road_RL[i]
    z_road_RR_current = z_road_RR[i]

    if i > 0:
        z_road_dot_RL = (z_road_RL[i] - z_road_RL[i-1]) / dt
        z_road_dot_RR = (z_road_RR[i] - z_road_RR[i-1]) / dt
    else:
        z_road_dot_RL = 0.0
        z_road_dot_RR = 0.0

    for sub in range(substeps):
        F_sus_RL, F_z_RL = suspension_model_RL.update(
            dt=dt_model,
            T_susp=T_susp_RL[i],
            X_body=X_body,
            z_road=z_road_RL_current,
            z_road_dot=z_road_dot_RL
        )

        F_sus_RR, F_z_RR = suspension_model_RR.update(
            dt=dt_model,
            T_susp=T_susp_RR[i],
            X_body=X_body,
            z_road=z_road_RR_current,
            z_road_dot=z_road_dot_RR
        )

    predicted_F_sus_RL[i] = F_sus_RL
    predicted_F_z_RL[i] = F_z_RL
    predicted_F_sus_RR[i] = F_sus_RR
    predicted_F_z_RR[i] = F_z_RR

print("Simulation completed!")

# ==================== 시각화 ====================
plot_skip_time = 0.5
plot_mask = time >= plot_skip_time
plot_time = time[plot_mask]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Suspension Equilibrium Test - Rear Wheels (RL, RR)', fontsize=18, fontweight='bold')

# Row 1: RL
axes[0, 0].plot(plot_time, actual_F_sus_RL[plot_mask], label='CarMaker', linewidth=2)
axes[0, 0].plot(plot_time, predicted_F_sus_RL[plot_mask], label='Model', linewidth=1.5, linestyle='--')
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('Force (N)')
axes[0, 0].set_title('RL Suspension Force')
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[0, 1].plot(plot_time, actual_F_z_RL[plot_mask], label='CarMaker', linewidth=2)
axes[0, 1].plot(plot_time, predicted_F_z_RL[plot_mask], label='Model', linewidth=1.5, linestyle='--')
axes[0, 1].set_xlabel('Time (s)')
axes[0, 1].set_ylabel('Force (N)')
axes[0, 1].set_title('RL Tire Vertical Force')
axes[0, 1].legend()
axes[0, 1].grid(True)

categories = ['F_sus', 'F_z']
cm_values_RL = [actual_F_sus_RL.mean(), actual_F_z_RL.mean()]
model_values_RL = [suspension_model_RL._F_sus_eq, suspension_model_RL._F_tire_eq]
x = np.arange(len(categories))
width = 0.35
axes[0, 2].bar(x - width/2, cm_values_RL, width, label='CarMaker')
axes[0, 2].bar(x + width/2, model_values_RL, width, label='Model')
axes[0, 2].set_ylabel('Force (N)')
axes[0, 2].set_title('RL Equilibrium Values')
axes[0, 2].set_xticks(x)
axes[0, 2].set_xticklabels(categories)
axes[0, 2].legend()
axes[0, 2].grid(True, axis='y')

# Row 2: RR
axes[1, 0].plot(plot_time, actual_F_sus_RR[plot_mask], label='CarMaker', linewidth=2)
axes[1, 0].plot(plot_time, predicted_F_sus_RR[plot_mask], label='Model', linewidth=1.5, linestyle='--')
axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].set_ylabel('Force (N)')
axes[1, 0].set_title('RR Suspension Force')
axes[1, 0].legend()
axes[1, 0].grid(True)

axes[1, 1].plot(plot_time, actual_F_z_RR[plot_mask], label='CarMaker', linewidth=2)
axes[1, 1].plot(plot_time, predicted_F_z_RR[plot_mask], label='Model', linewidth=1.5, linestyle='--')
axes[1, 1].set_xlabel('Time (s)')
axes[1, 1].set_ylabel('Force (N)')
axes[1, 1].set_title('RR Tire Vertical Force')
axes[1, 1].legend()
axes[1, 1].grid(True)

cm_values_RR = [actual_F_sus_RR.mean(), actual_F_z_RR.mean()]
model_values_RR = [suspension_model_RR._F_sus_eq, suspension_model_RR._F_tire_eq]
axes[1, 2].bar(x - width/2, cm_values_RR, width, label='CarMaker')
axes[1, 2].bar(x + width/2, model_values_RR, width, label='Model')
axes[1, 2].set_ylabel('Force (N)')
axes[1, 2].set_title('RR Equilibrium Values')
axes[1, 2].set_xticks(x)
axes[1, 2].set_xticklabels(categories)
axes[1, 2].legend()
axes[1, 2].grid(True, axis='y')

plt.tight_layout()
print("\nTest completed! Rear wheel (RL, RR) plots generated.")
plt.show()
