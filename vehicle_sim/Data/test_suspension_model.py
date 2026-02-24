"""
서스펜션 모델 검증 테스트 (FL만)
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

# ==================== 입력 데이터 준비 (FL만) ====================
# 차체 상태 (X_body)
heave = df['cm_Heave'].values
roll = df['cm_Roll'].values
pitch = df['cm_Pitch'].values
heave_dot = df['cm_Heave_dot'].values
roll_dot = df['cm_Roll_Dot'].values
pitch_dot = df['cm_Pitch_Dot'].values

# 차체 가속도 신호 (이제 사용하지 않음)
# sprung_z_ddot_FL_input = df['cm_SprungZDdotFL'].values

# 노면 높이
z_road_FL = df['cm_RoadZFL'].values

# 서스펜션 토크 (액티브 서스펜션 입력) - 일단 0으로 가정
T_susp_FL = np.zeros_like(time)

# Stabilizer 힘
stab_FL = df['cm_StabFL'].values

# 실제 출력 (검증 대상)
# 로그 서스펜션 힘 (참조용)
logged_F_sus_with_stab_FL = df['cm_SusFrcFL'].values  # Stabi 포함
logged_F_sus_FL = logged_F_sus_with_stab_FL - stab_FL  # Stabi 제외
actual_F_z_FL = df['cm_FzFL'].values
# 언스프렁 절대 위치: cm_UnsprungZFL 필수
if 'cm_UnsprungZFL' not in df.columns:
    raise ValueError("cm_UnsprungZFL column is required in CSV!")
actual_z_u_abs_FL = df['cm_UnsprungZFL'].values

# 차체 언스프렁 위치 (SprungZ는 차체의 각 코너 높이)
actual_z_sprung_FL = df['cm_SprungZFL'].values

# 모델 입력을 초기값 0 기준 편차로 변환
# CarMaker의 heave는 초기 평형점 기준 상대값으로 해석
heave_rel = heave - heave[0]
pitch_rel = pitch - pitch[0]
heave_dot_rel = heave_dot - heave_dot[0]
pitch_dot_rel = pitch_dot - pitch_dot[0]
roll_rel = roll - roll[0]
roll_dot_rel = roll_dot - roll_dot[0]

print(f"\nInput data (FL only):")
print(f"  Heave: {heave.min():.4f} ~ {heave.max():.4f} m")
print(f"  Roll: {np.rad2deg(roll.min()):.2f} ~ {np.rad2deg(roll.max()):.2f} deg")
print(f"  Pitch: {np.rad2deg(pitch.min()):.2f} ~ {np.rad2deg(pitch.max()):.2f} deg")
print(f"  Road Height (FL): {z_road_FL.min():.4f} ~ {z_road_FL.max():.4f} m")
print(f"  Stabilizer Force (FL): {stab_FL.min():.2f} ~ {stab_FL.max():.2f} N")

print(f"\nCarMaker outputs (FL only):")
print(f"  Tire Vertical Force (F_z): {actual_F_z_FL.min():.2f} ~ {actual_F_z_FL.max():.2f} N")
print(f"  Unsprung Position (z_u): {actual_z_u_abs_FL.min():.4f} ~ {actual_z_u_abs_FL.max():.4f} m")

# ==================== 파라미터 수동 설정 ====================
# True로 설정하면 YAML 대신 아래 값 사용
OVERRIDE_PARAMS = False


# 수동 설정 파라미터
MANUAL_PARAMS = {
    # 서스펜션 파라미터
    'K_spring': 44425, #67300.0,    # 스프링 강성 [N/m]
    'C_damper_compression': 150.2,  # 전륜 압축 댐퍼 감쇠 [N*s/m]
    'C_damper_rebound': 150.7,      # 전륜 리바운드 댐퍼 감쇠 [N*s/m]

    # 타이어 파라미터
    'K_t': 406884,         # 타이어 수직 강성 [N/m]
    'C_t': 4608.6,            # 타이어 수직 댐핑 [N*s/m]

    # 질량 파라미터
    'm_sprung': 1806.8,      # 전체 스프렁 질량 [kg]
    'm_unsprung': 74.12,      # 언스프렁 질량 (한 바퀴) [kg]
}


# ==================== 서스펜션 모델 초기화 ====================
print("\n" + "=" * 80)
print("Initializing Suspension Model (FL)...")
print("=" * 80)

# YAML 파일에서 파라미터 로드
config_path = r'c:\CM_Projects\SeohanModel_ver7\SeohanModel\vehicle_sim\models\params\vehicle_standard.yaml'
suspension_model_FL = SuspensionModel(corner_id='FL', config_path=config_path)

# 파라미터 오버라이드
if OVERRIDE_PARAMS:
    print("\n*** 파라미터 수동 오버라이드 활성화 ***")

    # 서스펜션 파라미터 수정
    suspension_model_FL.params.K_spring = MANUAL_PARAMS['K_spring']
    suspension_model_FL.params.C_damper_compression = MANUAL_PARAMS['C_damper_compression']
    suspension_model_FL.params.C_damper_rebound = MANUAL_PARAMS['C_damper_rebound']

    # 타이어 파라미터 수정
    suspension_model_FL.tire_params.K_t = MANUAL_PARAMS['K_t']
    suspension_model_FL.tire_params.C_t = MANUAL_PARAMS['C_t']

    # 언스프렁 질량 수정
    suspension_model_FL.unsprung_params.m_u = MANUAL_PARAMS['m_unsprung']

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
    z_CG0 = suspension_model_FL.params.z_CG0

    # 서스펜션 모델에서 이미 front_load_ratio를 읽어서 m_s_corner 계산했음
    # 여기서는 검증용으로만 다시 계산
    geometry = vehicle_spec.get('geometry', {})
    front_ratio = float(geometry.get('front_load_ratio', 0.611))
    m_s_FL = (m_s * front_ratio) / 2.0

    # 검증용 계산
    x_CG = float(geometry.get('x_CG', 2.635))
    x_wheel_front = float(geometry.get('x_wheel_front', 3.79))
    x_wheel_rear = float(geometry.get('x_wheel_rear', 0.82))
    L = float(geometry.get('L_wheelbase', 2.97))
    a = x_CG - x_wheel_rear  # 1.815 m
    b = x_wheel_front - x_CG  # 1.155 m

    # 모델 값과 비교
    print(f"\n하중 분배 검증:")
    print(f"  계산된 m_s_FL: {m_s_FL:.2f} kg")
    print(f"  모델 m_s_corner: {suspension_model_FL.sprung_params.m_s_corner:.2f} kg")
    print(f"  일치 여부: {abs(m_s_FL - suspension_model_FL.sprung_params.m_s_corner) < 0.01}")

    # FL 코너 평형 힘
    F_sus_FL_eq = m_s_FL * g
    F_z_FL_eq = (m_s_FL + m_u) * g

    # 타이어 평형 압축
    delta_t_eq = F_z_FL_eq / K_t
    z_u_0 = R_w - delta_t_eq

    # 서스펜션 평형 압축
    delta_s_comp = F_sus_FL_eq / K_spring
    delta_s_eq = -delta_s_comp
    F_spring_eq = K_spring * delta_s_comp
    F_tire_eq = K_t * delta_t_eq

    # 서스펜션 자유 길이
    L_s0 = (z_CG0 - z_u_0) + delta_s_comp

    # 모델 내부 값 업데이트
    suspension_model_FL.params.L_s0 = L_s0
    suspension_model_FL._z_u_0 = z_u_0
    suspension_model_FL._delta_s_eq = delta_s_eq
    suspension_model_FL._F_spring_eq = F_spring_eq
    suspension_model_FL._F_sus_eq = F_spring_eq
    suspension_model_FL._delta_t_eq = delta_t_eq
    suspension_model_FL._F_tire_eq = F_tire_eq

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
    print(f"  앞축 하중 비율: {front_ratio:.3f}")
    print(f"  FL 차체 질량: {m_s_FL:.2f} kg")
    print(f"\nFL 코너 평형 힘:")
    print(f"  F_sus: {F_sus_FL_eq:.2f} N")
    print(f"  F_z: {F_z_FL_eq:.2f} N")
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

# FL 코너 차량 동역학 힘 계산 (관성력 제거, 로그된 힘 사용)
if OVERRIDE_PARAMS:
    # 로그된 서스펜션 힘 사용 (Stabi 제외)
    actual_F_sus_FL = logged_F_sus_FL

    print(f"FL 코너 차량 동역학 힘:")
    print(f"  m_s_FL: {m_s_FL:.2f} kg")
    print(f"  Using logged F_sus_FL range: {actual_F_sus_FL.min():.2f} ~ {actual_F_sus_FL.max():.2f} N")
else:
    # YAML 파라미터 사용 시에도 로그된 힘 사용
    actual_F_sus_FL = logged_F_sus_FL
    print("Using logged suspension force")

print(f"Model parameters (loaded from YAML):")
print(f"  K_spring: {suspension_model_FL.params.K_spring} N/m")
print(f"  C_damper_compression (Push): {suspension_model_FL.params.C_damper_compression} N*s/m")
print(f"  C_damper_rebound (Pull): {suspension_model_FL.params.C_damper_rebound} N*s/m")
print(f"  K_t (tire): {suspension_model_FL.tire_params.K_t} N/m")
print(f"  C_t (tire): {suspension_model_FL.tire_params.C_t} N*s/m")
print(f"  m_u (unsprung): {suspension_model_FL.unsprung_params.m_u} kg")
print(f"  R_w (wheel radius): {suspension_model_FL.params.R_w} m")
print(f"  L_s0 (rest length): {suspension_model_FL.params.L_s0} m")

# 타이어 압축량(R_w 기준) - 모델과 동일한 정의
# delta_t = R_w + z_road - z_u_abs (양수 = 압축)
actual_delta_t_FL = suspension_model_FL.params.R_w + z_road_FL - actual_z_u_abs_FL

# 스프렁-언스프렁 갭과 모델 평형 갭(z_CG0 - R_w) 기준 편차
actual_gap_s_to_u_FL = actual_z_sprung_FL - actual_z_u_abs_FL
actual_gap_ref = suspension_model_FL.params.z_CG0 - suspension_model_FL.params.R_w
actual_delta_s_FL = actual_gap_s_to_u_FL - actual_gap_ref

print(f"\nCalculated actual values:")
print(f"  Unsprung Position (z_u): {actual_z_u_abs_FL.min():.4f} ~ {actual_z_u_abs_FL.max():.4f} m")
print(f"  Suspension Stroke (delta_s): {actual_delta_s_FL.min():.4f} ~ {actual_delta_s_FL.max():.4f} m")

# CarMaker 평형 평균값 (초기 과도 구간 제외: 처음 5초 제외)
skip_time = 5.0  # 초
skip_idx = int(skip_time / dt)
cm_z_u_eq = actual_z_u_abs_FL[skip_idx:].mean()
cm_delta_t_eq = actual_delta_t_FL[skip_idx:].mean()
cm_delta_s_eq = actual_delta_s_FL[skip_idx:].mean()
cm_z_body_eq = actual_z_sprung_FL[skip_idx:].mean()
cm_F_sus_eq = actual_F_sus_FL[skip_idx:].mean()
cm_F_z_eq = actual_F_z_FL[skip_idx:].mean()

cm_gap_s_to_u_eq = actual_gap_s_to_u_FL[skip_idx:].mean()

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
print(f"  z_u_0: {suspension_model_FL._z_u_0:.4f} m")
print(f"  z_CG0: {suspension_model_FL.params.z_CG0:.4f} m")
print(f"  delta_t_eq: {suspension_model_FL._delta_t_eq:.4f} m")
print(f"  delta_s_eq: {suspension_model_FL._delta_s_eq:.4f} m")
print(f"  F_spring_eq: {suspension_model_FL._F_spring_eq:.2f} N")
print(f"  F_tire_eq: {suspension_model_FL._F_tire_eq:.2f} N")

# 평형점 차이 계산
delta_z_u = cm_z_u_eq - suspension_model_FL._z_u_0
delta_z_body = cm_z_body_eq - suspension_model_FL.params.z_CG0
print(f"\nEquilibrium Differences (CM - Model):")
print(f"  Δz_u: {delta_z_u:.4f} m")
print(f"  Δz_body: {delta_z_body:.4f} m")
print(f"  ΔF_sus: {cm_F_sus_eq - suspension_model_FL._F_sus_eq:.2f} N")
print(f"  ΔF_z: {cm_F_z_eq - suspension_model_FL._F_tire_eq:.2f} N")

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
predicted_F_sus_FL = np.zeros(n_steps)
predicted_F_z_FL = np.zeros(n_steps)
predicted_delta_s_FL = np.zeros(n_steps)
predicted_delta_t_FL = np.zeros(n_steps)
predicted_z_u_abs_FL = np.zeros(n_steps)

# 초기 상태를 CarMaker 첫 시점 값으로 세팅
suspension_model_FL.reset()
suspension_model_FL.state.z_u_abs = actual_z_u_abs_FL[0]
suspension_model_FL.state.z_u_dot = 0.0
suspension_model_FL.state.delta_t = actual_delta_t_FL[0]
suspension_model_FL.state.delta_s = actual_delta_s_FL[0]
suspension_model_FL.state.z_body_abs = actual_z_sprung_FL[0]
suspension_model_FL.state.F_spring = actual_F_sus_FL[0]
suspension_model_FL.state.F_damper = 0.0
suspension_model_FL.state.F_s = actual_F_sus_FL[0]
suspension_model_FL.state.F_z = actual_F_z_FL[0]

print(f"\nInitial state set to CarMaker first time step:")
print(f"  z_u_abs[0]: {suspension_model_FL.state.z_u_abs:.4f} m")
print(f"  z_body_abs[0]: {suspension_model_FL.state.z_body_abs:.4f} m")
print(f"  delta_s[0]: {suspension_model_FL.state.delta_s:.4f} m")
print(f"  delta_t[0]: {suspension_model_FL.state.delta_t:.4f} m")
print(f"  F_s[0]: {suspension_model_FL.state.F_s:.2f} N")
print(f"  F_z[0]: {suspension_model_FL.state.F_z:.2f} N")

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

    z_road_current = z_road_FL[i]
    T_susp_current = T_susp_FL[i]

    # 노면 속도 (CarMaker 스텝 기준)
    if i > 0:
        z_road_dot = (z_road_FL[i] - z_road_FL[i-1]) / dt
    else:
        z_road_dot = 0.0

    # 10000Hz로 서브스텝 실행
    for sub in range(substeps):
        F_sus, F_z = suspension_model_FL.update(
            dt=dt_model,
            T_susp=T_susp_current,
            X_body=X_body,
            z_road=z_road_current,
            z_road_dot=z_road_dot
        )

    # CarMaker 스텝 끝에서 결과 저장
    predicted_F_sus_FL[i] = F_sus
    predicted_F_z_FL[i] = F_z
    predicted_delta_s_FL[i] = suspension_model_FL.state.delta_s
    predicted_delta_t_FL[i] = suspension_model_FL.state.delta_t
    predicted_z_u_abs_FL[i] = suspension_model_FL.state.z_u_abs

print("Simulation completed!")

# ==================== 오차 분석 ====================
# 오차는 Stabi 제외된 순수 서스펜션 힘으로 계산
error_F_sus_FL = predicted_F_sus_FL - actual_F_sus_FL
error_F_z_FL = predicted_F_z_FL - actual_F_z_FL
error_delta_t_FL = predicted_delta_t_FL - actual_delta_t_FL
error_delta_s_FL = predicted_delta_s_FL - actual_delta_s_FL
error_z_u_abs_FL = predicted_z_u_abs_FL - actual_z_u_abs_FL

mae_F_sus_FL = np.mean(np.abs(error_F_sus_FL))
mae_F_z_FL = np.mean(np.abs(error_F_z_FL))
mae_delta_t_FL = np.mean(np.abs(error_delta_t_FL))
mae_delta_s_FL = np.mean(np.abs(error_delta_s_FL))
mae_z_u_abs_FL = np.mean(np.abs(error_z_u_abs_FL))

print("\n" + "=" * 80)
print("Error Analysis (errors computed with pure suspension force, no Stabi)")
print("=" * 80)
print(f"FL Suspension Force (F_sus): Mean Error={mae_F_sus_FL:.2f} N")
print(f"FL Tire Vertical Force (F_z): Mean Error={mae_F_z_FL:.2f} N")
print(f"FL Tire Deflection (delta_t): Mean Error={mae_delta_t_FL:.4f} m")
print(f"FL Suspension Stroke (delta_s): Mean Error={mae_delta_s_FL:.4f} m")
print(f"FL Unsprung Position (z_u): Mean Error={mae_z_u_abs_FL:.4f} m")
print("=" * 80)

# ==================== 시각화 ====================
# 초기 과도 구간 제외하고 그래프 그리기
time_plot = time[skip_idx:]
actual_F_sus_FL_plot = actual_F_sus_FL[skip_idx:]
predicted_F_sus_FL_plot = predicted_F_sus_FL[skip_idx:]
actual_F_z_FL_plot = actual_F_z_FL[skip_idx:]
predicted_F_z_FL_plot = predicted_F_z_FL[skip_idx:]
actual_delta_t_FL_plot = actual_delta_t_FL[skip_idx:]
predicted_delta_t_FL_plot = predicted_delta_t_FL[skip_idx:]
actual_delta_s_FL_plot = actual_delta_s_FL[skip_idx:]
predicted_delta_s_FL_plot = predicted_delta_s_FL[skip_idx:]
actual_z_u_abs_FL_plot = actual_z_u_abs_FL[skip_idx:]
predicted_z_u_abs_FL_plot = predicted_z_u_abs_FL[skip_idx:]
error_F_sus_FL_plot = error_F_sus_FL[skip_idx:]
error_F_z_FL_plot = error_F_z_FL[skip_idx:]
error_delta_t_FL_plot = error_delta_t_FL[skip_idx:]
error_delta_s_FL_plot = error_delta_s_FL[skip_idx:]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(f'Suspension Model Test - FL Only (initial {skip_time}sec skip)', fontsize=18, fontweight='bold')

# Row 1, Col 1: 서스펜션 힘 비교
axes[0, 0].plot(time_plot, actual_F_sus_FL_plot, label='Logged (no Stabi)', linewidth=2)
axes[0, 0].plot(time_plot, predicted_F_sus_FL_plot, label='Model Predicted', linewidth=1.5, linestyle='--')
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('Force (N)')
axes[0, 0].set_title('FL Suspension Force')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Row 1, Col 2: F_sus 오차
axes[0, 1].plot(time_plot, error_F_sus_FL_plot, linewidth=1.5)
axes[0, 1].axhline(y=0, color='black', linestyle='--', linewidth=1)
axes[0, 1].fill_between(time_plot, error_F_sus_FL_plot, 0, alpha=0.3)
axes[0, 1].set_xlabel('Time (s)')
axes[0, 1].set_ylabel('Error (N)')
axes[0, 1].set_title(f'F_sus Error (Mean: {mae_F_sus_FL:.2f} N)')
axes[0, 1].grid(True)

# Row 1, Col 3: F_sus Correlation
min_val_sus = min(actual_F_sus_FL_plot.min(), predicted_F_sus_FL_plot.min())
max_val_sus = max(actual_F_sus_FL_plot.max(), predicted_F_sus_FL_plot.max())
axes[0, 2].scatter(actual_F_sus_FL_plot, predicted_F_sus_FL_plot, alpha=0.5, s=3)
axes[0, 2].plot([min_val_sus, max_val_sus], [min_val_sus, max_val_sus], 'r--', linewidth=2, label='Perfect')
axes[0, 2].set_xlabel('Logged F_sus (N)')
axes[0, 2].set_ylabel('Predicted F_sus (N)')
axes[0, 2].set_title('FL F_sus Correlation')
axes[0, 2].legend()
axes[0, 2].grid(True)
axes[0, 2].axis('equal')

# Row 2, Col 1: 타이어 수직력 비교
axes[1, 0].plot(time_plot, actual_F_z_FL_plot, label='Actual', linewidth=2)
axes[1, 0].plot(time_plot, predicted_F_z_FL_plot, label='Model', linewidth=1.5, linestyle='--')
axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].set_ylabel('Force (N)')
axes[1, 0].set_title('FL Tire Vertical Force (F_z)')
axes[1, 0].legend()
axes[1, 0].grid(True)

# Row 2, Col 2: F_z 오차
axes[1, 1].plot(time_plot, error_F_z_FL_plot, linewidth=1.5)
axes[1, 1].axhline(y=0, color='black', linestyle='--', linewidth=1)
axes[1, 1].fill_between(time_plot, error_F_z_FL_plot, 0, alpha=0.3)
axes[1, 1].set_xlabel('Time (s)')
axes[1, 1].set_ylabel('Error (N)')
axes[1, 1].set_title(f'F_z Error (Mean: {mae_F_z_FL:.2f} N)')
axes[1, 1].grid(True)

# Row 2, Col 3: F_z Correlation
min_val_fz = min(actual_F_z_FL_plot.min(), predicted_F_z_FL_plot.min())
max_val_fz = max(actual_F_z_FL_plot.max(), predicted_F_z_FL_plot.max())
axes[1, 2].scatter(actual_F_z_FL_plot, predicted_F_z_FL_plot, alpha=0.5, s=3)
axes[1, 2].plot([min_val_fz, max_val_fz], [min_val_fz, max_val_fz], 'r--', linewidth=2, label='Perfect')
axes[1, 2].set_xlabel('Actual F_z (N)')
axes[1, 2].set_ylabel('Predicted F_z (N)')
axes[1, 2].set_title('FL F_z Correlation')
axes[1, 2].legend()
axes[1, 2].grid(True)
axes[1, 2].axis('equal')

plt.tight_layout()

print("\nTest completed! 1 figure with 6 subplots generated (FL only).")
plt.show()
