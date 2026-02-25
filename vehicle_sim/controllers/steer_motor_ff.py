"""
스티어링 모터 토크 피드포워드 컨트롤러 (얼라이닝 토크 보상 포함).
"""

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Union

import numpy as np


@dataclass
class SteeringMotorTorqueFeedforwardOptions:
    max_accel: Optional[float] = None
    torque_limit: Optional[float] = None


class SteeringMotorTorqueFeedforwardController:
    """
    액추에이터 모델로 바퀴별 스티어링 모터 토크 피드포워드를 계산한다:
        T_axis = J_cq * delta_ddot + B_cq * delta_dot + T_align
    셀프 얼라이닝 토크 보상을 포함해 계산한다.
    계산된 축 토크를 기어비로 나누어 모터 토크로 출력한다.
    """

    def __init__(
        self,
        dt: float,
        options: Optional[SteeringMotorTorqueFeedforwardOptions] = None,
    ) -> None:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self.dt = float(dt)
        self.options = options or SteeringMotorTorqueFeedforwardOptions()
        self.prev_delta_cmd: Dict[str, float] = {}
        self.prev_delta_dot_cmd: Dict[str, float] = {}

    def reset(self) -> None:
        """내부 커맨드 히스토리를 리셋한다."""
        self.prev_delta_cmd.clear()
        self.prev_delta_dot_cmd.clear()

    def compute_torque(
        self,
        vehicle_body,
        delta_cmd: Union[float, Dict[str, float]],
        aligning_torque_cmd: Union[float, Dict[str, float]],
        delta_dot_cmd: Optional[Union[float, Dict[str, float]]] = None,
        delta_ddot_cmd: Optional[Union[float, Dict[str, float]]] = None,
    ) -> Dict[str, float]:
        """
        인자:
            vehicle_body: VehicleBody 인스턴스.
            delta_cmd: 목표 조향각 [rad].
            aligning_torque_cmd: 바퀴별 얼라이닝 토크 [N*m] (필수).
            delta_dot_cmd: 목표 조향각 속도 [rad/s] (선택).
            delta_ddot_cmd: 목표 조향각 가속도 [rad/s^2] (선택).
        """
        labels = list(vehicle_body.wheel_labels)
        # 입력을 바퀴별 dict로 정규화
        delta_cmd_map = self._normalize_command(delta_cmd, labels)
        delta_dot_map = self._normalize_command(delta_dot_cmd, labels) if delta_dot_cmd is not None else {}
        delta_ddot_map = self._normalize_command(delta_ddot_cmd, labels) if delta_ddot_cmd is not None else {}
        align_map = self._normalize_command(aligning_torque_cmd, labels)

        torques: Dict[str, float] = {}
        for label in labels:
            corner = vehicle_body.corners[label]
            params = corner.steering.params

            # 조향각 명령(바퀴별) → 각도 제한 적용
            delta_cmd_i = float(delta_cmd_map.get(label, 0.0))
            delta_cmd_i = self._clip_angle(delta_cmd_i, params.max_angle_neg, params.max_angle_pos)

            prev_delta = self.prev_delta_cmd.get(label, delta_cmd_i)
            has_prev_delta_dot = label in self.prev_delta_dot_cmd
            prev_delta_dot = self.prev_delta_dot_cmd.get(label, 0.0)

            # 속도 명령이 없으면 이전 명령으로 미분해 계산
            if delta_dot_cmd is None:
                delta_dot_cmd_i = (delta_cmd_i - prev_delta) / self.dt
            else:
                delta_dot_cmd_i = float(delta_dot_map.get(label, 0.0))
            delta_dot_cmd_i = float(np.clip(delta_dot_cmd_i, -params.max_rate, params.max_rate))

            # 가속도 명령이 없으면 속도 명령을 미분해 계산
            if delta_ddot_cmd is None:
                if not has_prev_delta_dot:
                    delta_ddot_cmd_i = 0.0
                else:
                    delta_ddot_cmd_i = (delta_dot_cmd_i - prev_delta_dot) / self.dt
            else:
                delta_ddot_cmd_i = float(delta_ddot_map.get(label, 0.0))
            if self.options.max_accel is not None:
                delta_ddot_cmd_i = float(
                    np.clip(delta_ddot_cmd_i, -self.options.max_accel, self.options.max_accel)
                )

            # 얼라이닝 토크 보상 포함
            t_align = float(align_map.get(label, 0.0))
            # 조향축 토크(축 기준) 계산
            torque_axis = (
                params.J_cq * delta_ddot_cmd_i
                + params.B_cq * delta_dot_cmd_i
                + t_align
            )

            # 축 토크 제한 적용
            if self.options.torque_limit is not None:
                torque_axis = float(
                    np.clip(torque_axis, -self.options.torque_limit, self.options.torque_limit)
                )

            # 축 토크 → 모터 토크 변환
            torques[label] = float(self._axis_to_motor_torque(torque_axis, params))
            self.prev_delta_cmd[label] = delta_cmd_i
            self.prev_delta_dot_cmd[label] = delta_dot_cmd_i

        return torques

    @staticmethod
    def _axis_to_motor_torque(torque_axis: float, params) -> float:
        scale = float(params.gear_ratio)
        if abs(scale) < 1e-6:
            return float(torque_axis)
        return float(torque_axis) / scale

    @staticmethod
    def _normalize_command(
        value: Optional[Union[float, Dict[str, float]]],
        labels: Iterable[str],
    ) -> Dict[str, float]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return {label: float(value.get(label, 0.0)) for label in labels}
        return {label: float(value) for label in labels}

    @staticmethod
    def _clip_angle(angle: float, neg: float, pos: float) -> float:
        lower = min(neg, pos)
        upper = max(neg, pos)
        return float(np.clip(angle, lower, upper))
