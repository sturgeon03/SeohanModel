"""
서스펜션 모델 검증 테스트 (RL, RR - 후륜)
기존 e_corner 서스펜션 모델을 사용하여 시나리오 데이터 검증
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
print("Loading scenario data...")
print("=" * 80)

df = pd.read_csv(r'c:\CM_Projects\SeohanModel_ver7\SeohanModel\vehicle_sim\Data\CM_Suspension_data.csv')
time = df['Time'].values
dt = time[1] - time[0]

print(f"Data points: {len(df)}")
print(f"Time range: {time[0]:.3f} ~ {time[-1]:.3f} seconds")
print(f"Time step (dt): {dt:.5f} seconds")

# ==================== 입력 데이터 준비 (RL, RR) ====================
# CRITICAL: cm_Heave is ABSOLUTE height, not deviation!
heave_abs = df['cm_Heave'].values
roll_abs = df['cm_Roll'].values
pitch_abs = df['cm_Pitch'].values

print(f"\n[DATA CORRECTION]")
print(f"cm_Heave is absolute height: mean={heave_abs.mean():.6f} m, range={heave_abs.max()-heave_abs.min():.6f} m")
print(f"Converting to deviation coordinates (subtract initial value)")
print(f"WARNING: cm_Heave_dot from CSV is NOT d(cm_Heave)/dt! Using numerical differentiation.")

# Convert to deviation coordinates
heave_rel = heave_abs - heave_abs[0]
pitch_rel = pitch_abs - pitch_abs[0]
roll_rel = roll_abs - roll_abs[0]

# Calculate velocity from numerical differentiation (correct way!)
heave_dot_rel = np.gradient(heave_rel, time)
pitch_dot_rel = np.gradient(pitch_rel, time)
roll_dot_rel = np.gradient(roll_rel, time)

# 차체 가속도 신호 (스프렁 질량 동역학 계산용)
sprung_z_ddot_RL = df['cm_SprungZDdotRL'].values
sprung_z_ddot_RR = df['cm_SprungZDdotRR'].values

# 노면 높이
z_road_RL = df['cm_RoadZRL'].values
z_road_RR = df['cm_RoadZRR'].values

# 서스펜션 토크 (액티브 서스펜션 입력) - 일단 0으로 가정
T_susp_RL = np.zeros_like(time)
T_susp_RR = np.zeros_like(time)

# 타이어 수직력 (검증용)
actual_F_z_RL = df['cm_FzRL'].values
actual_F_z_RR = df['cm_FzRR'].values

if 'cm_UnsprungZRL' not in df.columns or 'cm_UnsprungZRR' not in df.columns:
    raise ValueError("cm_UnsprungZRL and cm_UnsprungZRR columns are required in CSV!")
actual_z_u_abs_RL = df['cm_UnsprungZRL'].values
actual_z_u_abs_RR = df['cm_UnsprungZRR'].values

actual_z_sprung_RL = df['cm_SprungZRL'].values
actual_z_sprung_RR = df['cm_SprungZRR'].values

# For backward compatibility (keep old variable names)
heave = heave_abs
roll = roll_abs
pitch = pitch_abs

print(f"\nInput data (RL, RR):")
print(f"  Heave: {heave.min():.4f} ~ {heave.max():.4f} m")
print(f"  Roll: {np.rad2deg(roll.min()):.2f} ~ {np.rad2deg(roll.max()):.2f} deg")
print(f"  Pitch: {np.rad2deg(pitch.min()):.2f} ~ {np.rad2deg(pitch.max()):.2f} deg")

print(f"\nCarMaker outputs (RL, RR):")
print(f"  RL Tire Vertical Force (F_z): {actual_F_z_RL.min():.2f} ~ {actual_F_z_RL.max():.2f} N")
print(f"  RR Tire Vertical Force (F_z): {actual_F_z_RR.min():.2f} ~ {actual_F_z_RR.max():.2f} N")

# ==================== 파라미터 수동 설정 ====================
# True로 설정하면 YAML 대신 아래 값 사용
OVERRIDE_PARAMS = False


# 수동 설정 파라미터 (뒤축용)
MANUAL_PARAMS = {
    # 서스펜션 파라미터
    'K_spring': 44425, #67300.0,    # 스프링 강성 [N/m]
    'C_damper_compression': 150.2,  # 후륜 압축 댐퍼 감쇠 [N*s/m]
    'C_damper_rebound': 150.7,      # 후륜 리바운드 댐퍼 감쇠 [N*s/m]

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

config_path = r'c:\CM_Projects\SeohanModel_ver7\SeohanModel\vehicle_sim\models\params\vehicle_standard.yaml'
suspension_model_RL = SuspensionModel(corner_id='RL', config_path=config_path)
suspension_model_RR = SuspensionModel(corner_id='RR', config_path=config_path)

models = {'RL': suspension_model_RL, 'RR': suspension_model_RR}

if OVERRIDE_PARAMS:
    print("\n*** 파라미터 수동 오버라이드 활성화 ***")

    for corner_id, model in models.items():
        model.params.K_spring = MANUAL_PARAMS['K_spring']
        model.params.C_damper_compression = MANUAL_PARAMS['C_damper_compression']
        model.params.C_damper_rebound = MANUAL_PARAMS['C_damper_rebound']
        model.tire_params.K_t = MANUAL_PARAMS['K_t']
        model.tire_params.C_t = MANUAL_PARAMS['C_t']
        model.unsprung_params.m_u = MANUAL_PARAMS['m_unsprung']

    # 평형점 재계산
    from vehicle_sim.utils.config_loader import load_param
    physics_param = load_param('physics', config_path)
    vehicle_spec = load_param('vehicle_spec', config_path)

    m_s = MANUAL_PARAMS['m_sprung']
    m_u = MANUAL_PARAMS['m_unsprung']
    g = float(physics_param.get('g', 9.81))
    R_w = float(vehicle_spec.get('wheel', {}).get('R_eff', 0.327))
    K_t = MANUAL_PARAMS['K_t']
    K_spring = MANUAL_PARAMS['K_spring']
    z_CG0 = suspension_model_RL.params.z_CG0

    # 서스펜션 모델에서 이미 rear_load_ratio를 읽어서 m_s_corner 계산했음
    # 여기서는 검증용으로만 다시 계산
    geometry = vehicle_spec.get('geometry', {})
    rear_ratio = float(geometry.get('rear_load_ratio', 0.389))
    m_s_RL = (m_s * rear_ratio) / 2.0

    # 검증용 계산
    x_CG = float(geometry.get('x_CG', 2.635))
    x_wheel_front = float(geometry.get('x_wheel_front', 3.79))
    x_wheel_rear = float(geometry.get('x_wheel_rear', 0.82))
    L = float(geometry.get('L_wheelbase', 2.97))
    a = x_CG - x_wheel_rear  # 1.815 m
    b = x_wheel_front - x_CG  # 1.155 m

    # 모델 값과 비교
    print(f"\n하중 분배 검증:")
    print(f"  계산된 m_s_RL: {m_s_RL:.2f} kg")
    print(f"  모델 m_s_corner: {suspension_model_RL.sprung_params.m_s_corner:.2f} kg")
    print(f"  일치 여부: {abs(m_s_RL - suspension_model_RL.sprung_params.m_s_corner) < 0.01}")

    # RL 코너 평형 힘
    F_sus_RL_eq = m_s_RL * g
    F_z_RL_eq = (m_s_RL + m_u) * g

    # 타이어 평형 압축
    delta_t_eq = F_z_RL_eq / K_t
    z_u_0 = R_w - delta_t_eq

    # 서스펜션 평형 압축
    delta_s_comp = F_sus_RL_eq / K_spring
    delta_s_eq = -delta_s_comp
    F_spring_eq = K_spring * delta_s_comp
    F_tire_eq = K_t * delta_t_eq

    # 서스펜션 자유 길이
    L_s0 = (z_CG0 - z_u_0) + delta_s_comp

    # 모델 내부 값 업데이트
    suspension_model_RL.params.L_s0 = L_s0
    suspension_model_RL._z_u_0 = z_u_0
    suspension_model_RL._delta_s_eq = delta_s_eq
    suspension_model_RL._F_spring_eq = F_spring_eq
    suspension_model_RL._F_sus_eq = F_spring_eq
    suspension_model_RL._delta_t_eq = delta_t_eq
    suspension_model_RL._F_tire_eq = F_tire_eq

    print(f"오버라이드된 파라미터:")
    print(f"  K_spring: {K_spring:.1f} N/m")
    print(f"  C_damper_compression (Push): {MANUAL_PARAMS['C_damper_compression']:.1f} N*s/m")
    print(f"  C_damper_rebound (Pull): {MANUAL_PARAMS['C_damper_rebound']:.1f} N*s/m")
    print(f"  K_t: {K_t:.1f} N/m")
    print(f"  C_t: {MANUAL_PARAMS['C_t']:.1f} N*s/m")
    print(f"  m_sprung: {m_s:.1f} kg (전체)")
    print(f"  m_unsprung: {m_u:.1f} kg (한 바퀴)")
    print(f"\n하중 분배 (코너 위치 기반):")
    print(f"  휠베이스 L: {L:.3f} m")
    print(f"  CG-뒷축 거리 a: {a:.3f} m")
    print(f"  CG-앞축 거리 b: {b:.3f} m")
    print(f"  뒷축 하중 비율: {rear_ratio:.3f}")
    print(f"  RL 차체 질량: {m_s_RL:.2f} kg")
    print(f"\nRL 코너 평형 힘:")
    print(f"  F_sus: {F_sus_RL_eq:.2f} N")
    print(f"  F_z: {F_z_RL_eq:.2f} N")
    print(f"\n재계산된 평형점:")
    print(f"  delta_t_eq: {delta_t_eq:.4f} m")
    print(f"  delta_s_eq: {delta_s_eq:.4f} m")
    print(f"  F_spring_eq: {F_spring_eq:.2f} N")
    print(f"  F_tire_eq: {F_tire_eq:.2f} N")
    print(f"  L_s0: {L_s0:.4f} m\n")

# ==================== 차량 동역학으로 서스펜션 힘 계산 ====================
print("\n" + "=" * 80)
print("Calculating suspension forces from vehicle dynamics...")
print("=" * 80)

# RL 코너 차량 동역학 힘 계산 (스프렁 질량 동역학 이용)
# F_sus = m_s_corner × (z_sprung_ddot + g)
from vehicle_sim.utils.config_loader import load_param

if OVERRIDE_PARAMS:
    m_s_RL = (m_s * rear_ratio) / 2.0
    m_s_RR = m_s_RL
    physics_param = load_param('physics', config_path)
    g = float(physics_param.get('g', 9.81))
else:
    # YAML에서 읽은 값 사용
    vehicle_spec = load_param('vehicle_spec', config_path)
    geometry = vehicle_spec.get('geometry', {})
    rear_ratio = float(geometry.get('rear_load_ratio', 0.389))
    vehicle_body_param = load_param('vehicle_body', config_path)
    m_s_total = float(vehicle_body_param.get('m', 1806.8))
    m_s_RL = (m_s_total * rear_ratio) / 2.0
    m_s_RR = m_s_RL
    physics_param = load_param('physics', config_path)
    g = float(physics_param.get('g', 9.81))

# CarMaker 서스펜션 힘 (로그에서 직접 읽기)
# cm_SusFrcRL/RR: 서스펜션 총 힘 (스프링 + 댐퍼 + 액티브 + 스태빌라이저)
actual_F_sus_RL = df['cm_SusFrcRL'].values
actual_F_sus_RR = df['cm_SusFrcRR'].values

print(f"RL 코너 차량 동역학 힘 (스프렁 질량 동역학):")
print(f"  m_s_RL: {m_s_RL:.2f} kg")
print(f"  m_s_RR: {m_s_RR:.2f} kg")
print(f"  z_ddot_RL range: {sprung_z_ddot_RL.min():.2f} ~ {sprung_z_ddot_RL.max():.2f} m/s²")
print(f"  z_ddot_RR range: {sprung_z_ddot_RR.min():.2f} ~ {sprung_z_ddot_RR.max():.2f} m/s²")
print(f"  F_sus_RL range: {actual_F_sus_RL.min():.2f} ~ {actual_F_sus_RL.max():.2f} N")
print(f"  F_sus_RR range: {actual_F_sus_RR.min():.2f} ~ {actual_F_sus_RR.max():.2f} N")

print(f"Model parameters (loaded from YAML):")
print(f"  K_spring: {suspension_model_RL.params.K_spring} N/m")
print(f"  C_damper_compression (Push): {suspension_model_RL.params.C_damper_compression} N*s/m")
print(f"  C_damper_rebound (Pull): {suspension_model_RL.params.C_damper_rebound} N*s/m")
print(f"  K_t (tire): {suspension_model_RL.tire_params.K_t} N/m")
print(f"  C_t (tire): {suspension_model_RL.tire_params.C_t} N*s/m")
print(f"  m_u (unsprung): {suspension_model_RL.unsprung_params.m_u} kg")
print(f"  R_w (wheel radius): {suspension_model_RL.params.R_w} m")
print(f"  L_s0 (rest length): {suspension_model_RL.params.L_s0} m")

# 타이어 압축량(R_w 기준) - 모델과 동일한 정의
# delta_t = R_w + z_road - z_u_abs (양수 = 압축)
actual_delta_t_RL = suspension_model_RL.params.R_w + z_road_RL - actual_z_u_abs_RL

# 스프렁-언스프렁 갭과 모델 평형 갭(z_CG0 - R_w) 기준 편차
actual_gap_s_to_u_RL = actual_z_sprung_RL - actual_z_u_abs_RL
actual_gap_ref = suspension_model_RL.params.z_CG0 - suspension_model_RL.params.R_w
actual_delta_s_RL = actual_gap_s_to_u_RL - actual_gap_ref

print(f"\nCalculated actual values:")
print(f"  Unsprung Position (z_u): {actual_z_u_abs_RL.min():.4f} ~ {actual_z_u_abs_RL.max():.4f} m")
print(f"  Suspension Stroke (delta_s): {actual_delta_s_RL.min():.4f} ~ {actual_delta_s_RL.max():.4f} m")

# CarMaker 평형 평균값 (초기 과도 구간 제외: 처음 5초 제외)
skip_time = 5.0  # 초
skip_idx = int(skip_time / dt)
cm_z_u_eq = actual_z_u_abs_RL[skip_idx:].mean()
cm_delta_t_eq = actual_delta_t_RL[skip_idx:].mean()
cm_delta_s_eq = actual_delta_s_RL[skip_idx:].mean()
cm_z_body_eq = actual_z_sprung_RL[skip_idx:].mean()
cm_F_sus_eq = actual_F_sus_RL[skip_idx:].mean()
cm_F_z_eq = actual_F_z_RL[skip_idx:].mean()

cm_gap_s_to_u_eq = actual_gap_s_to_u_RL[skip_idx:].mean()

print(f"\nCarMaker Equilibrium Values:")
print(f"  z_u (unsprung): {cm_z_u_eq:.4f} m")
print(f"  z_body (sprung): {cm_z_body_eq:.4f} m")
print(f"  actual_gap (z_sprung - z_u): {cm_gap_s_to_u_eq:.4f} m")
print(f"  actual_gap_ref (z_CG0 - R_w): {actual_gap_ref:.4f} m")
print(f"  delta_t: {cm_delta_t_eq:.4f} m")
print(f"  delta_s (actual_gap - gap_ref): {cm_delta_s_eq:.4f} m")
print(f"  F_sus: {cm_F_sus_eq:.2f} N")
print(f"  F_z: {cm_F_z_eq:.2f} N")

print(f"\nModel Equilibrium Values:")
print(f"  z_u_0: {suspension_model_RL._z_u_0:.4f} m")
print(f"  z_CG0: {suspension_model_RL.params.z_CG0:.4f} m")
print(f"  delta_t_eq: {suspension_model_RL._delta_t_eq:.4f} m")
print(f"  delta_s_eq: {suspension_model_RL._delta_s_eq:.4f} m")
print(f"  F_spring_eq: {suspension_model_RL._F_spring_eq:.2f} N")
print(f"  F_tire_eq: {suspension_model_RL._F_tire_eq:.2f} N")

# 평형점 차이 계산
delta_z_u = cm_z_u_eq - suspension_model_RL._z_u_0
delta_z_body = cm_z_body_eq - suspension_model_RL.params.z_CG0
print(f"\nEquilibrium Differences (CM - Model):")
print(f"  Δz_u: {delta_z_u:.4f} m")
print(f"  Δz_body: {delta_z_body:.4f} m")
print(f"  ΔF_sus: {cm_F_sus_eq - suspension_model_RL._F_sus_eq:.2f} N")
print(f"  ΔF_z: {cm_F_z_eq - suspension_model_RL._F_tire_eq:.2f} N")

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
predicted_delta_s_RL = np.zeros(n_steps)
predicted_delta_t_RL = np.zeros(n_steps)
predicted_z_u_abs_RL = np.zeros(n_steps)

# 초기 상태를 CarMaker 첫 시점 값으로 세팅
suspension_model_RL.reset()
suspension_model_RL.state.z_u_abs = actual_z_u_abs_RL[0]
suspension_model_RL.state.z_u_dot = 0.0
suspension_model_RL.state.delta_t = actual_delta_t_RL[0]
suspension_model_RL.state.delta_s = actual_delta_s_RL[0]
suspension_model_RL.state.z_body_abs = actual_z_sprung_RL[0]
suspension_model_RL.state.F_spring = actual_F_sus_RL[0]
suspension_model_RL.state.F_damper = 0.0
suspension_model_RL.state.F_s = actual_F_sus_RL[0]
suspension_model_RL.state.F_z = actual_F_z_RL[0]

print(f"\nInitial state set to CarMaker first time step:")
print(f"  z_u_abs[0]: {suspension_model_RL.state.z_u_abs:.4f} m")
print(f"  z_body_abs[0]: {suspension_model_RL.state.z_body_abs:.4f} m")
print(f"  delta_s[0]: {suspension_model_RL.state.delta_s:.4f} m")
print(f"  delta_t[0]: {suspension_model_RL.state.delta_t:.4f} m")
print(f"  F_s[0]: {suspension_model_RL.state.F_s:.2f} N")
print(f"  F_z[0]: {suspension_model_RL.state.F_z:.2f} N")

for i in range(n_steps):
    # CarMaker 입력값 (Zero-Order Hold) - 위치, 속도만
    X_body = np.array([
        heave_rel[i],
        roll_rel[i],
        pitch_rel[i],
        heave_dot_rel[i],
        roll_dot_rel[i],
        pitch_dot_rel[i]
    ])

    z_road_current = z_road_RL[i]
    T_susp_current = T_susp_RL[i]

    # 노면 속도 (CarMaker 스텝 기준)
    if i > 0:
        z_road_dot = (z_road_RL[i] - z_road_RL[i-1]) / dt
    else:
        z_road_dot = 0.0

    # 10000Hz로 서브스텝 실행
    for sub in range(substeps):
        F_sus, F_z = suspension_model_RL.update(
            dt=dt_model,
            T_susp=T_susp_current,
            X_body=X_body,
            z_road=z_road_current,
            z_road_dot=z_road_dot
        )

    # CarMaker 스텝 끝에서 결과 저장
    predicted_F_sus_RL[i] = F_sus
    predicted_F_z_RL[i] = F_z
    predicted_delta_s_RL[i] = suspension_model_RL.state.delta_s
    predicted_delta_t_RL[i] = suspension_model_RL.state.delta_t
    predicted_z_u_abs_RL[i] = suspension_model_RL.state.z_u_abs

print("Simulation completed!")

# ==================== 오차 분석 ====================
# 오차는 Stabi 제외된 순수 서스펜션 힘으로 계산
error_F_sus_RL = predicted_F_sus_RL - actual_F_sus_RL
error_F_z_RL = predicted_F_z_RL - actual_F_z_RL
error_delta_t_RL = predicted_delta_t_RL - actual_delta_t_RL
error_delta_s_RL = predicted_delta_s_RL - actual_delta_s_RL
error_z_u_abs_RL = predicted_z_u_abs_RL - actual_z_u_abs_RL

mae_F_sus_RL = np.mean(np.abs(error_F_sus_RL))
mae_F_z_RL = np.mean(np.abs(error_F_z_RL))
mae_delta_t_RL = np.mean(np.abs(error_delta_t_RL))
mae_delta_s_RL = np.mean(np.abs(error_delta_s_RL))
mae_z_u_abs_RL = np.mean(np.abs(error_z_u_abs_RL))

print("\n" + "=" * 80)
print("Error Analysis (errors computed with pure suspension force, no Stabi)")
print("=" * 80)
print(f"RL Suspension Force (F_sus): Mean Error={mae_F_sus_RL:.2f} N")
print(f"RL Tire Vertical Force (F_z): Mean Error={mae_F_z_RL:.2f} N")
print(f"RL Tire Deflection (delta_t): Mean Error={mae_delta_t_RL:.4f} m")
print(f"RL Suspension Stroke (delta_s): Mean Error={mae_delta_s_RL:.4f} m")
print(f"RL Unsprung Position (z_u): Mean Error={mae_z_u_abs_RL:.4f} m")
print("=" * 80)

# ==================== 시각화 ====================
# 초기 과도 구간 제외하고 그래프 그리기
time_plot = time[skip_idx:]
actual_F_sus_RL_plot = actual_F_sus_RL[skip_idx:]
predicted_F_sus_RL_plot = predicted_F_sus_RL[skip_idx:]
actual_F_z_RL_plot = actual_F_z_RL[skip_idx:]
predicted_F_z_RL_plot = predicted_F_z_RL[skip_idx:]
actual_delta_t_RL_plot = actual_delta_t_RL[skip_idx:]
predicted_delta_t_RL_plot = predicted_delta_t_RL[skip_idx:]
actual_delta_s_RL_plot = actual_delta_s_RL[skip_idx:]
predicted_delta_s_RL_plot = predicted_delta_s_RL[skip_idx:]
actual_z_u_abs_RL_plot = actual_z_u_abs_RL[skip_idx:]
predicted_z_u_abs_RL_plot = predicted_z_u_abs_RL[skip_idx:]
error_F_sus_RL_plot = error_F_sus_RL[skip_idx:]
error_F_z_RL_plot = error_F_z_RL[skip_idx:]
error_delta_t_RL_plot = error_delta_t_RL[skip_idx:]
error_delta_s_RL_plot = error_delta_s_RL[skip_idx:]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(f'Suspension Model Test - RL Only (initial {skip_time}sec skip)', fontsize=18, fontweight='bold')

# Row 1, Col 1: 서스펜션 힘 비교
axes[0, 0].plot(time_plot, actual_F_sus_RL_plot, label='Logged (no Stabi)', linewidth=2)
axes[0, 0].plot(time_plot, predicted_F_sus_RL_plot, label='Model Predicted', linewidth=1.5, linestyle='--')
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('Force (N)')
axes[0, 0].set_title('RL Suspension Force')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Row 1, Col 2: F_sus 오차
axes[0, 1].plot(time_plot, error_F_sus_RL_plot, linewidth=1.5)
axes[0, 1].axhline(y=0, color='black', linestyle='--', linewidth=1)
axes[0, 1].fill_between(time_plot, error_F_sus_RL_plot, 0, alpha=0.3)
axes[0, 1].set_xlabel('Time (s)')
axes[0, 1].set_ylabel('Error (N)')
axes[0, 1].set_title(f'F_sus Error (Mean: {mae_F_sus_RL:.2f} N)')
axes[0, 1].grid(True)

# Row 1, Col 3: F_sus Correlation
min_val_sus = min(actual_F_sus_RL_plot.min(), predicted_F_sus_RL_plot.min())
max_val_sus = max(actual_F_sus_RL_plot.max(), predicted_F_sus_RL_plot.max())
axes[0, 2].scatter(actual_F_sus_RL_plot, predicted_F_sus_RL_plot, alpha=0.5, s=3)
axes[0, 2].plot([min_val_sus, max_val_sus], [min_val_sus, max_val_sus], 'r--', linewidth=2, label='Perfect')
axes[0, 2].set_xlabel('Logged F_sus (N)')
axes[0, 2].set_ylabel('Predicted F_sus (N)')
axes[0, 2].set_title('RL F_sus Correlation')
axes[0, 2].legend()
axes[0, 2].grid(True)
axes[0, 2].axis('equal')

# Row 2, Col 1: 타이어 수직력 비교
axes[1, 0].plot(time_plot, actual_F_z_RL_plot, label='Actual', linewidth=2)
axes[1, 0].plot(time_plot, predicted_F_z_RL_plot, label='Model', linewidth=1.5, linestyle='--')
axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].set_ylabel('Force (N)')
axes[1, 0].set_title('RL Tire Vertical Force (F_z)')
axes[1, 0].legend()
axes[1, 0].grid(True)

# Row 2, Col 2: F_z 오차
axes[1, 1].plot(time_plot, error_F_z_RL_plot, linewidth=1.5)
axes[1, 1].axhline(y=0, color='black', linestyle='--', linewidth=1)
axes[1, 1].fill_between(time_plot, error_F_z_RL_plot, 0, alpha=0.3)
axes[1, 1].set_xlabel('Time (s)')
axes[1, 1].set_ylabel('Error (N)')
axes[1, 1].set_title(f'F_z Error (Mean: {mae_F_z_RL:.2f} N)')
axes[1, 1].grid(True)

# Row 2, Col 3: F_z Correlation
min_val_fz = min(actual_F_z_RL_plot.min(), predicted_F_z_RL_plot.min())
max_val_fz = max(actual_F_z_RL_plot.max(), predicted_F_z_RL_plot.max())
axes[1, 2].scatter(actual_F_z_RL_plot, predicted_F_z_RL_plot, alpha=0.5, s=3)
axes[1, 2].plot([min_val_fz, max_val_fz], [min_val_fz, max_val_fz], 'r--', linewidth=2, label='Perfect')
axes[1, 2].set_xlabel('Actual F_z (N)')
axes[1, 2].set_ylabel('Predicted F_z (N)')
axes[1, 2].set_title('RL F_z Correlation')
axes[1, 2].legend()
axes[1, 2].grid(True)
axes[1, 2].axis('equal')

plt.tight_layout()

print("\nTest completed! 1 figure with 6 subplots generated (RL only).")
plt.show()
