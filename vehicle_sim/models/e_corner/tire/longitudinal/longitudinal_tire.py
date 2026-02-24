#!/bin/python3
"""
종방향 타이어 동역학 모델
입력: κ (slip ratio), F_z_tire (타이어 수직력)
출력: F_x_tire (종방향 타이어 힘)
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass

from vehicle_sim.utils.config_loader import load_param


@dataclass
class LongitudinalTireParameters:
    """종방향 타이어 파라미터"""
    C_x: float = 10.0       # 종방향 타이어 강성 계수 (정규화) [-]
    mu: float = 0.9         # 마찰 계수 (건조 노면) [-]
    v_min: float = 0.1      # 최소 속도 (슬립 계산용) [m/s]
    R_eff: float = 0.3      # 타이어 유효 반지름 [m]


@dataclass
class LongitudinalTireState:
    """종방향 타이어 상태 변수"""
    slip_ratio: float = 0.0          # κ: 슬립 비율 [-]
    longitudinal_force: float = 0.0  # F_x_tire: 종방향 타이어 힘 [N]


class LongitudinalTireModel:
    """
    종방향 타이어 동역학 모델
    입력: κ (slip ratio), F_z_tire (타이어 수직력)
    출력: F_x_tire (종방향 타이어 힘)

    모델:
    F_x_tire = C_x × F_z_tire × κ
    단, |F_x_tire| ≤ μ × F_z_tire  (마찰 한계)
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        종방향 타이어 모델 초기화

        Args:
            config_path: YAML 설정 파일 경로. None이면 기본 vehicle_standard.yaml 사용
        """
        # 타이어 파라미터 로드
        tire_param = load_param('tire', config_path)
        long_param = tire_param.get('longitudinal', {})

        # 차량 스펙에서 휠 반지름 로드
        vehicle_spec = load_param('vehicle_spec', config_path)
        wheel_spec = vehicle_spec.get('wheel', {})

        self.params = LongitudinalTireParameters(
            C_x=float(long_param.get('C_x', 10.0)),
            mu=float(long_param.get('mu', tire_param.get('mu', 0.9))),  # 공통 타이어 마찰 계수 fallback
            v_min=float(long_param.get('v_min', 0.1)),
            R_eff=float(wheel_spec.get('R_eff', 0.3))
        )
        self.state = LongitudinalTireState()

    def calculate_slip_ratio(self, omega_wheel: float, V_wheel_x: float) -> float:
        """
        슬립 비율 계산 (저속 데드밴드 적용)

        입력:
            - omega_wheel: 휠 각속도 [rad/s]
            - V_wheel_x: 휠 중심 종방향 속도 [m/s]

        출력:
            - κ: 슬립 비율 [-]

        공식:
            κ = (V_wheel - V_x) / |V_x|

        여기서:
            - V_wheel = ω × R_eff (휠 둘레 속도)
            - κ > 0: 구동 슬립 (가속)
            - κ < 0: 제동 슬립 (감속)
            - κ = 0: 완전 구름 (no slip)

        저속 안정화:
            - 정지/근접 저속에서 κ를 0으로 강제 (데드밴드)
            - 등속 영역(예: 5m/s 이상)에는 영향 없음
        """
        # 휠 둘레 속도 (R_eff는 YAML에서 로드됨)
        V_wheel = omega_wheel * self.params.R_eff

        # (1) 정지 근처 데드밴드 (저속 안정화)
        # CRITICAL: 정지 조건은 "절대 속도가 모두 작을 때만" 적용
        # 등속 주행(V_wheel ≈ V_wheel_x)과 구분하기 위해 절대값 기준 사용
        eps_v = 0.5       # 절대 속도 임계값 [m/s] (정지 판정)
        v_rel = V_wheel - V_wheel_x

        # 정지 판정: 차체 속도와 휠 속도가 모두 eps_v 이하
        is_stopped = (abs(V_wheel_x) < eps_v and abs(V_wheel) < eps_v)

        if is_stopped:
            # 정지 상태: 슬립 강제로 0 (수치 안정화)
            kappa = 0.0
        else:
            # (2) 일반 영역: 차체 속도 기준으로 정규화
            # κ = (V_wheel - V_wheel_x) / |V_wheel_x|
            # 분모는 차체 속도만 사용 (표준 슬립 정의)
            denom = max(abs(V_wheel_x), self.params.v_min)
            kappa = v_rel / denom

        # 상태 업데이트
        self.state.slip_ratio = kappa

        return kappa

    def calculate_force(self, kappa: float, F_z_tire: float) -> float:
        """
        종방향 타이어 힘 계산

        입력:
            - kappa: 슬립 비율 [-]
            - F_z_tire: 타이어 수직력 [N]

        출력:
            - F_x_tire: 종방향 타이어 힘 [N]

        모델:
            1. 선형 영역: F_x_tire = C_x × F_z_tire × κ
            2. 마찰 한계: |F_x_tire| ≤ μ × F_z_tire

        특징:
            - 수직력에 비례하는 강성 (정규화된 C_x)
            - F_z_tire가 클수록 더 큰 종방향 힘 발생
        """
        # 1. 수직력 비례 모델
        F_x_tire = self.params.C_x * kappa

        # 2. 마찰 한계 적용 (포화)
        F_x_tire_max = self.params.mu * abs(F_z_tire)
        F_x_tire = np.clip(F_x_tire, -F_x_tire_max, F_x_tire_max)

        # 상태 업데이트
        self.state.longitudinal_force = F_x_tire

        return F_x_tire

    def reset(self) -> None:
        """타이어 모델 상태 리셋"""
        self.state = LongitudinalTireState()

    def get_state(self) -> Dict:
        """현재 종방향 타이어 상태 조회"""
        return {
            "slip_ratio": self.state.slip_ratio,
            "longitudinal_force": self.state.longitudinal_force,
        }
