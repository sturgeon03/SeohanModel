"""
각 휠의 횡력(Fy) 명령을 조향각(steering angle)으로 변환하는 피드포워드 매핑.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class SteeringFeedforwardOptions:
    clamp_fy: bool = True
    unwrap_delta: bool = True


class SteeringFeedforwardController:
    """
    휠 프레임 횡력(Fy_wheel_cmd) 명령을 타이어 역모델로 슬립각으로 바꾸고,
    현재 슬립각(alpha_now)을 휠 프레임 속도에서 얻어 조향각 명령을 만든다:

        alpha_cmd = -Fy_wheel_cmd / C_alpha
        alpha_now = atan2(Vy_wheel, Vx_wheel)
        delta_cmd = delta_now + alpha_now - alpha_cmd
    """

    def __init__(self, options: Optional[SteeringFeedforwardOptions] = None) -> None:
        self.options = options or SteeringFeedforwardOptions()
        self._prev_delta_cmd: Dict[str, float] = {}

    def reset(self) -> None:
        """내부 명령 히스토리를 초기화한다(예: 시나리오 시작 시)."""
        self._prev_delta_cmd.clear()

    @staticmethod
    def _wrap_to_pi(angle: float) -> float:
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    @classmethod
    def _unwrap_near(cls, angle: float, reference: float) -> float:
        return float(reference + cls._wrap_to_pi(angle - reference))

    def compute_delta_cmd(
        self,
        vehicle_body,
        fy_wheel_cmd: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Args:
            vehicle_body: VehicleBody 인스턴스(상태 + 타이어 파라미터).
            fy_wheel_cmd: 휠별 횡력(Fy) 명령 [N] (휠 로컬 프레임, +y가 좌측).

        Returns:
            휠별 조향각 명령 [rad].
        """
        delta_cmd: Dict[str, float] = {}
        labels = list(vehicle_body.wheel_labels)
        if set(self._prev_delta_cmd.keys()) != set(labels):
            self._prev_delta_cmd = {}

        for idx, label in enumerate(labels):
            corner = vehicle_body.corners[label]
            fy_in = float(fy_wheel_cmd.get(label, 0.0))
            delta_ref = float(corner.state.steering_angle)
            ref = float(self._prev_delta_cmd.get(label, delta_ref))

            def _clamp_fy_wheel(fy_wheel_value: float) -> float:
                if not self.options.clamp_fy:
                    return float(fy_wheel_value)
                fz = float(corner.state.F_z)
                if fz <= 0.0:
                    return float(fy_wheel_value)
                mu = float(corner.lateral_tire.params.mu)
                return float(np.clip(fy_wheel_value, -mu * abs(fz), mu * abs(fz)))

            c_alpha = float(corner.lateral_tire.params.C_alpha)
            if c_alpha == 0.0:
                raise ValueError("C_alpha must be non-zero for feedforward steering")

            v_wheel = vehicle_body.get_wheel_velocity(idx, frame="wheel")
            v_wx = float(v_wheel[0])
            v_wy = float(v_wheel[1])

            # 운동학적 항등식(슬립각 정의, 휠 프레임):
            #   alpha_now = atan2(Vy_wheel, Vx_wheel)
            # 원하는 슬립각(alpha_cmd)을 만들기 위한 조향 명령:
            #   delta_cmd = delta_now + alpha_now - alpha_cmd
            alpha_now = float(np.arctan2(v_wy, v_wx))
            fy_wheel = _clamp_fy_wheel(float(fy_in))
            alpha_cmd = -fy_wheel / c_alpha
            delta_raw = float(delta_ref + alpha_now - alpha_cmd)

            if self.options.unwrap_delta:
                # atan2() 래핑(wrap-around)으로 Vx가 0을 넘거나 Vy 부호가 바뀔 때 생기는 2π 점프를 방지한다.
                delta_raw = self._unwrap_near(delta_raw, ref)

            delta_cmd[label] = float(delta_raw)
            self._prev_delta_cmd[label] = float(delta_raw)

        return delta_cmd
