#얼라인 토크 잘볼것
"""
조향 액추에이터 모델
능동 조향 및 액추에이터 동역학 구현
"""
# Dynamics: J_cq * ddot(delta) + B_cq * dot(delta) = T_str * gear_ratio - T_align

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass

from vehicle_sim.utils.config_loader import load_param


@dataclass
class SteeringParameters:
    """조향 액추에이터 파라미터"""
    J_cq: float = 0.05          # 등가 관성 [kg*m^2]
    B_cq: float = 0.5           # 점성 댐핑 [N*m*s/rad]
    gear_ratio: float = 118.0   # 모터축 -> 조향축 감속비 [-]
    max_angle_pos: float = 0.0   # 양(+) 방향 최대 조향 각도 [rad] (CCW, config로 주입)
    max_angle_neg: float = 0.0   # 음(-) 방향 최대 조향 각도 [rad] (CW, config로 주입)
    max_rate: float = np.deg2rad(360.0)  # 최대 조향 속도 [rad/s]


@dataclass
class SteeringState:
    """조향 액추에이터 상태 변수"""
    steering_angle: float = 0.0          # 실제 조향 각도 [rad]
    steering_rate: float = 0.0           # 조향 각속도 [rad/s]
    steering_torque: float = 0.0         # 입력 조향 토크 [N*m]
    self_aligning_torque: float = 0.0    # 타이어로부터의 셀프 얼라이닝 토크 [N*m]


class SteeringModel:
    """
    능동 조향 액추에이터 모델
    E-corner용 스티어 바이 와이어 또는 능동 조향 시스템 모델링
    """

    def __init__(self, config: Optional[Dict] = None,
                 config_path: Optional[str] = None,
                 corner_id: Optional[str] = None,
                 side: Optional[str] = None):
        """
        조향 모델 초기화

        Args:
            config: 파라미터 딕셔너리. None이면 config_path를 통해 로드
            config_path: YAML 설정 파일 경로. None이면 기본 vehicle_standard.yaml 사용
            corner_id/side: 좌/우 바퀴 구분 (YAML 구조가 steering.left/right일 때 필요)
        """
        steering_param = config if config is not None else load_param('steering', config_path)
        self.params = SteeringParameters()
        self.params.J_cq = float(steering_param.get('J_cq', self.params.J_cq))
        self.params.B_cq = float(steering_param.get('B_cq', self.params.B_cq))
        self.params.gear_ratio = float(steering_param.get('gear_ratio', self.params.gear_ratio))
        self.params.max_rate = float(steering_param.get('max_rate', self.params.max_rate))

        # YAML이 steering.left/right 트리 구조일 때를 대비해 좌/우를 선택해 각도 제한을 읽어온다.
        side_key = None
        if side:
            side_key = 'left' if side.lower().startswith('l') else 'right'
        elif corner_id:
            side_key = 'left' if corner_id in ['FL', 'RL'] else 'right'

        def _get_angle_limit(key: str) -> float:
            # config가 이미 max_angle_*를 펼쳐 넣은 형태라면 그대로 사용
            if isinstance(steering_param, dict) and key in steering_param:
                return float(steering_param.get(key, 0.0))
            # nested 형태면 side별 블록에서 가져온다
            if side_key and isinstance(steering_param, dict):
                side_cfg = steering_param.get(side_key, {})
                if isinstance(side_cfg, dict) and key in side_cfg:
                    return float(side_cfg.get(key, 0.0))
            return 0.0

        self.params.max_angle_pos = _get_angle_limit('max_angle_pos')
        self.params.max_angle_neg = _get_angle_limit('max_angle_neg')

        if self.params.max_angle_pos == 0.0 and self.params.max_angle_neg == 0.0:
            raise ValueError("Steering max_angle_pos/max_angle_neg must be provided via config")

        self.state = SteeringState()

    def update(self, dt: float, T_str: float,
               T_align: float = 0.0) -> float:
        """
        조향 액추에이터 상태 업데이트
        J_cq * ddot(delta) + B_cq * dot(delta) = T_str * gear_ratio - T_align
        """
        # 입력/상태 업데이트
        self.state.steering_torque = float(T_str) * self.params.gear_ratio
        self.state.self_aligning_torque = T_align

        # 2차 시스템 가속도 계산 (스프링 항 제거)
        delta = self.state.steering_angle
        delta_dot = self.state.steering_rate
        numerator = float(T_str) * self.params.gear_ratio - T_align - self.params.B_cq * delta_dot
        delta_ddot = numerator / self.params.J_cq

        # 경계 체크 → 미분(속도/가속도) 투영 → 적분 순서로 처리
        lower = min(self.params.max_angle_neg, self.params.max_angle_pos)
        upper = max(self.params.max_angle_neg, self.params.max_angle_pos)

        # 경계에 붙어 있고 바깥으로 더 가려는 경우, 미분을 0으로 투영
        at_upper_and_outward = delta >= upper and delta_dot >= 0.0 and delta_ddot >= 0.0
        at_lower_and_outward = delta <= lower and delta_dot <= 0.0 and delta_ddot <= 0.0
        if at_upper_and_outward or at_lower_and_outward:
            delta_ddot = 0.0
            delta_dot = 0.0

        # 각속도 적분 + 속도 제한
        desired_rate = delta_dot + dt * delta_ddot
        limited_rate = self.apply_rate_limits(desired_rate)

        # 각도 적분 + 각도 제한 (안전하게 한 번 더 클립)
        desired_angle = delta + dt * limited_rate
        limited_angle = self.apply_angle_limits(desired_angle)

        # 상태 저장
        self.state.steering_rate = limited_rate
        self.state.steering_angle = limited_angle

        return self.state.steering_angle

    def apply_angle_limits(self, angle: float) -> float:
        """조향 각도 제한 적용"""
        lower = min(self.params.max_angle_neg, self.params.max_angle_pos)
        upper = max(self.params.max_angle_neg, self.params.max_angle_pos)
        return float(np.clip(angle, lower, upper))

    def apply_rate_limits(self, desired_rate: float) -> float:
        """조향 속도 제한 적용"""
        return float(np.clip(desired_rate, -self.params.max_rate, self.params.max_rate))


    def get_state(self) -> Dict:
        """현재 조향 상태 조회"""
        return {
            "steering_angle": self.state.steering_angle,
            "steering_rate": self.state.steering_rate,
            "steering_torque": self.state.steering_torque,
            "self_aligning_torque": self.state.self_aligning_torque,
        }
    
    def reset(self) -> None:
        """조향 상태 리셋"""
        self.state = SteeringState()
