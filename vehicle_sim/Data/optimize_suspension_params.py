"""
서스펜션 파라미터 최적화
CarMaker 데이터에 맞는 K_spring, C_damper, K_t, C_t 값을 역산
"""
import sys
import os

# vehicle_sim을 포함한 경로 추가
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
from scipy.optimize import minimize, differential_evolution
from vehicle_sim.models.e_corner.suspension.suspension_model import SuspensionModel

# ==================== 데이터 로드 ====================
print("=" * 80)
print("Loading CarMaker data...")
print("=" * 80)

df = pd.read_csv(r'c:\CM_Projects\SeohanModel_ver7\SeohanModel\vehicle_sim\Data\CM_Suspension_data.csv')
time = df['Time'].values
dt = time[1] - time[0]

# 입력 데이터
heave = df['cm_Heave'].values
roll = df['cm_Roll'].values
pitch = df['cm_Pitch'].values
heave_dot = df['cm_Heave_dot'].values
roll_dot = df['cm_Roll_Dot'].values
pitch_dot = df['cm_Pitch_Dot'].values
z_road_FL = df['cm_RoadZFL'].values
stab_FL = df['cm_StabFL'].values

# 실제 출력 (목표값)
actual_F_sus_FL = df['cm_SusFrcFL'].values - stab_FL
actual_F_z_FL = df['cm_FzFL'].values

# 언스프렁 위치
if 'cm_UnsprungZFL' in df.columns:
    actual_z_u_abs_FL = df['cm_UnsprungZFL'].values
else:
    actual_z_u_abs_FL = z_road_FL + df['cm_TireDeltaFL'].values

actual_z_sprung_FL = df['cm_SprungZFL'].values

# 초기값 기준 상대 편차
heave_rel = heave - heave[0]
pitch_rel = pitch - pitch[0]
heave_dot_rel = heave_dot - heave_dot[0]
pitch_dot_rel = pitch_dot - pitch_dot[0]
roll_rel = roll - roll[0]
roll_dot_rel = roll_dot - roll_dot[0]

print(f"Data loaded: {len(df)} points, dt={dt:.5f}s")

# ==================== 목적 함수 정의 ====================
def objective_function(params, verbose=False):
    """
    목적 함수: 예측값과 실제값의 RMSE

    params: [K_spring, C_damper, K_t, C_t]
    """
    K_spring, C_damper, K_t, C_t = params

    # 파라미터 범위 체크
    if K_spring <= 0 or C_damper <= 0 or K_t <= 0 or C_t < 0:
        return 1e10

    try:
        # YAML 설정 로드 및 파라미터 오버라이드
        config_path = r'c:\CM_Projects\SeohanModel_ver7\SeohanModel\vehicle_sim\models\params\vehicle_standard.yaml'
        model = SuspensionModel(corner_id='FL', config_path=config_path)

        # 파라미터 업데이트
        model.params.K_spring = K_spring
        model.params.C_damper = C_damper
        model.tire_params.K_t = K_t
        model.tire_params.C_t = C_t

        # 평형점 재계산 (K_spring, K_t 변경으로 인해)
        from vehicle_sim.utils.config_loader import load_param
        physics_param = load_param('physics', config_path)
        vehicle_body = load_param('vehicle_body', config_path)
        unsprung_param = load_param('unsprung', config_path)
        vehicle_spec = load_param('vehicle_spec', config_path)

        m_s = float(vehicle_body.get('m', 1827.0))
        m_u = float(unsprung_param.get('m_u', 65.4))
        g = float(physics_param.get('g', 9.81))
        R_w = float(vehicle_spec.get('wheel', {}).get('R_eff', 0.327))
        z_CG0 = float(model.params.z_CG0)

        # 타이어 평형 압축
        delta_t_eq = ((m_s / 4.0) + m_u) * g / K_t
        z_u_0 = R_w - delta_t_eq

        # 서스펜션 평형 압축
        delta_s_comp = (m_s / 4.0) * g / K_spring
        delta_s_eq = -delta_s_comp
        F_spring_eq = K_spring * delta_s_comp
        F_tire_eq = K_t * delta_t_eq

        # 초기 상태를 CarMaker 첫 시점 값으로 설정
        model.reset()
        model.state.z_u_abs = actual_z_u_abs_FL[0]
        model.state.z_u_dot = 0.0
        model.state.delta_t = R_w + z_road_FL[0] - actual_z_u_abs_FL[0]
        model.state.delta_s = (actual_z_sprung_FL[0] - actual_z_u_abs_FL[0]) - model.params.L_s0
        model.state.z_body_abs = actual_z_sprung_FL[0]
        model.state.F_spring = actual_F_sus_FL[0]
        model.state.F_damper = 0.0
        model.state.F_s = actual_F_sus_FL[0]
        model.state.F_z = actual_F_z_FL[0]

        # 시뮬레이션
        n_steps = len(time)
        pred_F_sus = np.zeros(n_steps)
        pred_F_z = np.zeros(n_steps)

        for i in range(n_steps):
            X_body = np.array([
                heave_rel[i], roll_rel[i], pitch_rel[i],
                heave_dot_rel[i], roll_dot_rel[i], pitch_dot_rel[i]
            ])

            if i > 0:
                z_road_dot = (z_road_FL[i] - z_road_FL[i-1]) / dt
            else:
                z_road_dot = 0.0

            F_sus, F_z = model.update(
                dt=dt,
                T_susp=0.0,
                X_body=X_body,
                z_road=z_road_FL[i],
                z_road_dot=z_road_dot
            )

            pred_F_sus[i] = F_sus
            pred_F_z[i] = F_z

        # RMSE 계산 (F_sus와 F_z 모두 고려)
        rmse_F_sus = np.sqrt(np.mean((pred_F_sus - actual_F_sus_FL)**2))
        rmse_F_z = np.sqrt(np.mean((pred_F_z - actual_F_z_FL)**2))

        # 가중 평균 (F_sus를 더 중요하게)
        total_error = 0.7 * rmse_F_sus + 0.3 * rmse_F_z

        if verbose:
            print(f"K_spring={K_spring:.1f}, C_damper={C_damper:.1f}, K_t={K_t:.1f}, C_t={C_t:.1f}")
            print(f"  RMSE F_sus={rmse_F_sus:.2f} N, F_z={rmse_F_z:.2f} N, Total={total_error:.2f}")

        return total_error

    except Exception as e:
        print(f"Error in simulation: {e}")
        return 1e10

# ==================== 초기값 설정 ====================
# YAML에서 로드한 초기값
initial_params = np.array([
    128400.0,  # K_spring
    750.0,     # C_damper
    469500.0,  # K_t
    608.6      # C_t
])

print("\n" + "=" * 80)
print("Initial parameters (from YAML):")
print("=" * 80)
print(f"K_spring: {initial_params[0]:.1f} N/m")
print(f"C_damper: {initial_params[1]:.1f} N*s/m")
print(f"K_t: {initial_params[2]:.1f} N/m")
print(f"C_t: {initial_params[3]:.1f} N*s/m")

initial_error = objective_function(initial_params, verbose=True)
print(f"\nInitial total error: {initial_error:.2f}")

# ==================== 최적화 실행 ====================
print("\n" + "=" * 80)
print("Starting optimization (Differential Evolution)...")
print("=" * 80)

# 파라미터 범위 설정
bounds = [
    (50000, 200000),   # K_spring [N/m]
    (100, 3000),       # C_damper [N*s/m]
    (200000, 600000),  # K_t [N/m]
    (100, 2000)        # C_t [N*s/m]
]

# Differential Evolution (전역 최적화)
result = differential_evolution(
    objective_function,
    bounds,
    strategy='best1bin',
    maxiter=50,
    popsize=15,
    tol=0.01,
    mutation=(0.5, 1),
    recombination=0.7,
    seed=42,
    callback=lambda xk, convergence: print(f"Iteration: convergence={convergence:.6f}"),
    polish=True,
    disp=True
)

optimal_params = result.x

print("\n" + "=" * 80)
print("Optimization completed!")
print("=" * 80)
print(f"\nOptimal parameters:")
print(f"  K_spring: {optimal_params[0]:.1f} N/m (초기값: {initial_params[0]:.1f})")
print(f"  C_damper: {optimal_params[1]:.1f} N*s/m (초기값: {initial_params[1]:.1f})")
print(f"  K_t: {optimal_params[2]:.1f} N/m (초기값: {initial_params[2]:.1f})")
print(f"  C_t: {optimal_params[3]:.1f} N*s/m (초기값: {initial_params[3]:.1f})")

print(f"\nFinal error: {result.fun:.2f}")
print(f"Improvement: {((initial_error - result.fun) / initial_error * 100):.1f}%")

# ==================== 최적 파라미터로 검증 ====================
print("\n" + "=" * 80)
print("Validation with optimal parameters...")
print("=" * 80)

final_error = objective_function(optimal_params, verbose=True)

# YAML 업데이트 가이드
print("\n" + "=" * 80)
print("Update vehicle_standard.yaml with these values:")
print("=" * 80)
print(f"""
suspension:
  K_spring: {optimal_params[0]:.1f}
  C_damper: {optimal_params[1]:.1f}

tire:
  vertical:
    K_t: {optimal_params[2]:.1f}
    C_t: {optimal_params[3]:.1f}
""")

print("\nOptimization completed!")
