"""
Active Stabilizer Controller
Roll/Roll rate 기반 PD 제어로 좌우 서스펜션 힘 차이를 생성하여 롤 안정화
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class ActiveStabilizerGains:
    """Active Stabilizer PD 게인"""
    K_roll_front: float = 1.68e5    # 전륜 롤 비례 게인 [N/rad]
    C_roll_front: float = 8.2e3     # 전륜 롤 미분 게인 [N*s/rad]
    K_roll_rear: float = 1.16e5     # 후륜 롤 비례 게인 [N/rad]
    C_roll_rear: float = 3.15e3     # 후륜 롤 미분 게인 [N*s/rad]
    sign_roll: float = 1.0          # 롤 부호 (1.0: 정방향, -1.0: 역방향)
    max_force: Optional[float] = None  # 최대 액티브 힘 [N] (None이면 서스펜션 모델 제한 사용)


class ActiveStabilizerController:
    """
    Active Stabilizer Controller

    롤과 롤 레이트를 기반으로 PD 제어를 수행하여 좌우 서스펜션 힘 차이를 생성.
    각 코너의 서스펜션 액추에이터 토크로 변환하여 출력.

    제어 법칙:
        ΔF_front = sign_roll * (K_roll_front * roll + C_roll_front * roll_dot)
        ΔF_rear = sign_roll * (K_roll_rear * roll + C_roll_rear * roll_dot)

        F_FL = +0.5 * ΔF_front
        F_FR = -0.5 * ΔF_front
        F_RL = +0.5 * ΔF_rear
        F_RR = -0.5 * ΔF_rear

    입력: roll [rad], roll_rate [rad/s]
    출력: T_susp 딕셔너리 {corner: torque [N*m]}
    """

    def __init__(self, gains: Optional[ActiveStabilizerGains] = None):
        """
        Active Stabilizer Controller 초기화

        Args:
            gains: PD 게인 설정. None이면 기본값 사용
        """
        self.gains = gains if gains is not None else ActiveStabilizerGains()

        # 내부 상태 (필요 시 확장 가능)
        self.delta_f: float = 0.0  # 전륜 힘 차이 [N]
        self.delta_r: float = 0.0  # 후륜 힘 차이 [N]

    def reset(self) -> None:
        """제어기 상태 리셋"""
        self.delta_f = 0.0
        self.delta_r = 0.0

    def update(
        self,
        roll: float,
        roll_rate: float,
        vehicle_body=None,
    ) -> Dict[str, float]:
        """
        롤 상태 기반 서스펜션 토크 계산

        Args:
            roll: 차체 롤 각도 [rad] (편차 좌표)
            roll_rate: 차체 롤 레이트 [rad/s]
            vehicle_body: VehicleBody 인스턴스 (서스펜션 파라미터 접근용, 선택)

        Returns:
            T_susp: 코너별 서스펜션 액추에이터 토크 [N*m]
                    {"FL": T_FL, "FR": T_FR, "RL": T_RL, "RR": T_RR}
        """
        # PD 제어 - 축별 힘 차이 계산
        self.delta_f = self.gains.sign_roll * (
            self.gains.K_roll_front * roll + self.gains.C_roll_front * roll_rate
        )
        self.delta_r = self.gains.sign_roll * (
            self.gains.K_roll_rear * roll + self.gains.C_roll_rear * roll_rate
        )

        # 코너별 힘 분배 (좌: +, 우: -)
        F_act = {
            "FL": 0.5 * self.delta_f,
            "FR": -0.5 * self.delta_f,
            "RL": 0.5 * self.delta_r,
            "RR": -0.5 * self.delta_r,
        }

        # 힘 → 토크 변환
        T_susp = {}
        for corner, F in F_act.items():
            # vehicle_body가 제공되면 해당 코너의 서스펜션 파라미터 사용
            if vehicle_body is not None:
                susp_params = vehicle_body.corners[corner].suspension.params
                T_susp[corner] = self._force_to_torque(F, susp_params)
            else:
                # vehicle_body가 없으면 기본 파라미터 사용 (lead=0.01, efficiency=0.9)
                T_susp[corner] = self._force_to_torque_default(F)

        return T_susp

    def _force_to_torque(self, F_act: float, susp_params) -> float:
        """
        액티브 힘을 서스펜션 액추에이터 토크로 변환

        Args:
            F_act: 액티브 힘 [N]
            susp_params: 서스펜션 파라미터 (lead, efficiency, F_active_max)

        Returns:
            T_susp: 서스펜션 토크 [N*m]
        """
        # 힘 제한 (서스펜션 모델 제한 사용)
        max_force = self.gains.max_force
        if max_force is None and hasattr(susp_params, 'F_active_max'):
            max_force = susp_params.F_active_max

        if max_force is not None:
            F_act = float(np.clip(F_act, -max_force, max_force))

        # T = F * lead / (2π * efficiency)
        # lead: 리드 [m/rev], efficiency: 효율 [-]
        lead = susp_params.lead if hasattr(susp_params, 'lead') else 0.01
        efficiency = susp_params.efficiency if hasattr(susp_params, 'efficiency') else 0.9

        T_susp = F_act * lead / (2.0 * np.pi * efficiency)
        return float(T_susp)

    def _force_to_torque_default(self, F_act: float) -> float:
        """
        기본 파라미터로 힘을 토크로 변환

        Args:
            F_act: 액티브 힘 [N]

        Returns:
            T_susp: 서스펜션 토크 [N*m]
        """
        # 힘 제한
        if self.gains.max_force is not None:
            F_act = float(np.clip(F_act, -self.gains.max_force, self.gains.max_force))

        # 기본값: lead=0.01 m/rev, efficiency=0.9
        lead = 0.01
        efficiency = 0.9
        T_susp = F_act * lead / (2.0 * np.pi * efficiency)
        return float(T_susp)

    def get_state(self) -> Dict[str, float]:
        """
        현재 제어기 상태 조회

        Returns:
            state: {"delta_f": 전륜 힘차 [N], "delta_r": 후륜 힘차 [N]}
        """
        return {
            "delta_f": self.delta_f,
            "delta_r": self.delta_r,
        }
