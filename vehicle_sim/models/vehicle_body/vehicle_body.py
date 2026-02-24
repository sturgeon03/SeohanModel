"""
차체 동역학 모델
차체 6자유도 강체 동역학을 처리
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from ..e_corner.e_corner import ECorner
from vehicle_sim.utils.config_loader import load_param

@dataclass
class VehicleBodyState:
    """차체 상태 변수"""
    # 관성 좌표계 위치
    x: float = 0.0     # 종방향 위치 [m]
    y: float = 0.0     # 횡방향 위치 [m]
    heave: float = 0.0 # 수직 변위 (편차 좌표) [m]

    # 자세 (오일러 각)
    roll: float = 0.0   # 롤 각 [rad]
    pitch: float = 0.0  # 피치 각 [rad]
    yaw: float = 0.0    # 요 각 [rad]

    # 차체 좌표계 속도
    velocity_x: float = 0.0  # 종방향 속도 [m/s]
    velocity_y: float = 0.0  # 횡방향 속도 [m/s]
    heave_dot: float = 0.0   # 수직 속도 (heave rate) [m/s]

    # 차체 좌표계 각속도
    roll_rate: float = 0.0   # 롤 속도 [rad/s]
    pitch_rate: float = 0.0  # 피치 속도 [rad/s]
    yaw_rate: float = 0.0    # 요 속도 [rad/s]

    # 이전 스텝 가속도 (관성 모멘트 계산용)
    ax_prev: float = 0.0     # 이전 종방향 가속도 [m/s²]
    ay_prev: float = 0.0     # 이전 횡방향 가속도 [m/s²]


@dataclass
class VehicleBodyParameters:
    """차체 물리 파라미터"""
    # 질량
    m: float = 1500.0           # 스프렁 질량 [kg]
    m_total: float = 1500.0     # 전체 차량 질량 (스프렁 + 언스프렁) [kg]

    # 관성 모멘트 (CG 기준)
    Ixx: float = 500.0          # 롤 관성 모멘트 [kg·m²]
    Iyy: float = 2500.0         # 피치 관성 모멘트 [kg·m²]
    Izz: float = 2800.0         # 요 관성 모멘트 [kg·m²]
    Ixz: float = 0.0            # 교차 관성적 (xz 평면) [kg·m²]

    # 기하 정보
    h_CG: float = 0.5           # CG 높이 [m]
    a: float = 1.4              # CG → 전축 거리 [m]
    b: float = 1.4              # CG → 후축 거리 [m]
    L_track: float = 1.6        # 트랙 폭 [m]
    L_wheelbase: float = 2.8    # 휠베이스 [m]

    # 물리 상수
    g: float = 9.81             # 중력가속도 [m/s²]


class VehicleBody:
    """
    차체 동역학 모델
    6자유도 강체 차체 운동을 시뮬레이션
    """

    def __init__(self, parameters: VehicleBodyParameters = None, config_path: Optional[str] = None):
        """
        차체 모델 초기화

        Args:
            parameters: 차체 물리 파라미터 (None이면 YAML에서 로드)
            config_path: YAML 설정 파일 경로 (None이면 기본 경로 사용)
        """
        if parameters is None:
            # YAML에서 파라미터 로드
            vehicle_body = load_param('vehicle_body', config_path)
            vehicle_spec = load_param('vehicle_spec', config_path)
            physics_param = load_param('physics', config_path)
            susp_param = load_param('suspension', config_path)
            unsprung_param = load_param('unsprung', config_path)

            geometry = vehicle_spec.get('geometry', {})
            inertia = vehicle_body.get('inertia', {})

            # corner_offsets 로드 (CG 기준 각 코너 위치)
            corner_offsets = geometry.get('corner_offsets', {})

            # a, b (CG→전축/후축 거리) - front는 양수, rear는 양수 크기로 저장
            front_x = float(corner_offsets.get('FL', {}).get('x', 1.155))
            rear_x = float(corner_offsets.get('RL', {}).get('x', -1.815))
            a = abs(front_x)  # CG → 전축 거리
            b = abs(rear_x)   # CG → 후축 거리

            # 언스프렁 질량 로드 및 전체 질량 계산
            m_sprung = float(vehicle_body.get('m', 1500.0))
            m_u_front = float(unsprung_param.get('m_u_front', 69.12))
            m_u_rear = float(unsprung_param.get('m_u_rear', 54.995))
            m_unsprung_total = 2 * m_u_front + 2 * m_u_rear  # 4바퀴 전체
            m_total = m_sprung + m_unsprung_total

            self.params = VehicleBodyParameters(
                m=m_sprung,
                m_total=m_total,
                Ixx=float(inertia.get('Ixx', 500.0)),
                Iyy=float(inertia.get('Iyy', 2500.0)),
                Izz=float(inertia.get('Izz', 2800.0)),
                Ixz=float(inertia.get('Ixz', 0.0)),
                h_CG=float(susp_param.get('z_CG0', 0.5)),
                a=a,
                b=b,
                L_track=float(geometry.get('L_track', 1.6)),
                L_wheelbase=float(geometry.get('L_wheelbase', 2.8)),
                g=float(physics_param.get('g', 9.81))
            )

            # corner_offsets 저장 (실제 코너 위치 계산에 사용)
            self.corner_offsets = {
                "FL": {"x": float(corner_offsets.get('FL', {}).get('x', 1.155)),
                       "y": float(corner_offsets.get('FL', {}).get('y', 0.817))},
                "FR": {"x": float(corner_offsets.get('FR', {}).get('x', 1.155)),
                       "y": float(corner_offsets.get('FR', {}).get('y', -0.817))},
                "RL": {"x": float(corner_offsets.get('RL', {}).get('x', -1.815)),
                       "y": float(corner_offsets.get('RL', {}).get('y', 0.817))},
                "RR": {"x": float(corner_offsets.get('RR', {}).get('x', -1.815)),
                       "y": float(corner_offsets.get('RR', {}).get('y', -0.817))}
            }
        else:
            self.params = parameters
            # 파라미터로 초기화할 경우 corner_offsets 계산
            self.corner_offsets = {
                "FL": {"x": parameters.b, "y": parameters.L_track / 2.0},
                "FR": {"x": parameters.b, "y": -parameters.L_track / 2.0},
                "RL": {"x": -parameters.a, "y": parameters.L_track / 2.0},
                "RR": {"x": -parameters.a, "y": -parameters.L_track / 2.0}
            }

        self.state = VehicleBodyState()
        self.wheel_labels: List[str] = ["FL", "FR", "RR", "RL"]  # 4개 바퀴 고정
        self.corners: Dict[str, ECorner] = {
            label: ECorner(corner_id=label, config_path=config_path) for label in self.wheel_labels
        }

        # 코너별 부호 정의 (서스펜션 모델과 동일)
    def _rotation_matrix(self) -> np.ndarray:
        """바디→관성 회전 행렬 (ZYX Euler)"""
        phi, theta, psi = self.state.roll, self.state.pitch, self.state.yaw

        c_phi, s_phi = np.cos(phi), np.sin(phi)
        c_theta, s_theta = np.cos(theta), np.sin(theta)
        c_psi, s_psi = np.cos(psi), np.sin(psi)

        return np.array([
            [c_psi * c_theta, c_psi * s_theta * s_phi - s_psi * c_phi, c_psi * s_theta * c_phi + s_psi * s_phi],
            [s_psi * c_theta, s_psi * s_theta * s_phi + c_psi * c_phi, s_psi * s_theta * c_phi - c_psi * s_phi],
            [-s_theta,        c_theta * s_phi,                          c_theta * c_phi]
        ])

    def update(self, dt: float, corner_inputs: Dict[str, Dict[str, float]], direction: int = 1) -> None:
        """
        전체 차량 업데이트 (통합 메서드)

        E-Corner 4개 + 차체 동역학을 통합하여 한 번에 처리

        Args:
            dt: 시간 간격 [s]
            corner_inputs: 코너별 입력
                {
                    "FL": {"T_steer": ..., "T_brk": ..., "T_Drv": ..., "T_susp": ..., "z_road": ...},
                    "FR": {...},
                    "RL": {...},
                    "RR": {...}
                }
            direction: 전진/후진 (1: 전진, -1: 후진)
        """
        # 1. 현재 차체 상태로부터 E-Corner 입력 계산
        # (roll, pitch, heave, roll_dot, pitch_dot, heave_dot, V_wheel_x, V_wheel_y)
        corner_body_inputs = self.get_corner_inputs()

        # 2. 4개 E-Corner 업데이트
        corner_outputs = {}
        for label in self.wheel_labels:
            corner = self.corners[label]
            inputs = corner_inputs.get(label, {})

            # 액추에이터 토크
            T_steer = inputs.get("T_steer", 0.0)
            T_brk = inputs.get("T_brk", 0.0)
            T_Drv = inputs.get("T_Drv", 0.0)
            T_susp = inputs.get("T_susp", 0.0)

            # 노면 높이 및 속도
            z_road = inputs.get("z_road", 0.0)
            z_road_dot = inputs.get("z_road_dot", 0.0)

            # 휠 속도
            V_wheel_x = corner_body_inputs["wheel_velocities"][label][0]
            V_wheel_y = corner_body_inputs["wheel_velocities"][label][1]

            # X_body 배열 생성 [heave, roll, pitch, heave_dot, roll_rate, pitch_rate]
            X_body = np.array([
                corner_body_inputs["heave"],
                corner_body_inputs["roll"],
                corner_body_inputs["pitch"],
                corner_body_inputs["heave_rate"],
                corner_body_inputs["roll_rate"],
                corner_body_inputs["pitch_rate"]
            ])

            # E-Corner update
            F_s, F_x, F_y = corner.update(
                dt,
                T_steer, T_brk, T_Drv, T_susp,
                V_wheel_x, V_wheel_y,
                X_body,
                z_road,
                z_road_dot,
                direction
            )
            corner_outputs[label] = (F_s, F_x, F_y)

        # 3. 코너 힘 합산 (이전 스텝 가속도를 이용한 관성 모멘트 포함)
        forces, moments = self.assemble_forces_moments(
            corner_outputs,
            ax=self.state.ax_prev,
            ay=self.state.ay_prev
        )

        # 4. 차체 동역학 업데이트
        self._update_dynamics(dt, forces, moments)

    def _update_dynamics(self, dt: float, forces: np.ndarray, moments: np.ndarray) -> None:
        """차체 동역학 업데이트 (내부 메서드)

        Args:
            dt: 시간 간격 [s]
            forces: 차체에 작용하는 총 힘 [Fx, Fy, Fz] (바디 좌표계) [N]
            moments: 차체에 작용하는 총 모멘트 [Mx, My, Mz] (바디 좌표계) [N·m]
        """
        # 1. 가속도 계산
        linear_acc, angular_acc = self.calculate_accelerations(forces, moments)

        # 다음 스텝을 위해 현재 가속도 저장
        self.state.ax_prev = float(linear_acc[0])
        self.state.ay_prev = float(linear_acc[1])

        # 2. 바디 좌표계 속도 적분
        self.state.velocity_x += linear_acc[0] * dt
        self.state.velocity_y += linear_acc[1] * dt
        self.state.heave_dot += linear_acc[2] * dt

        # 3. 바디 좌표계 각속도 적분
        self.state.roll_rate += angular_acc[0] * dt
        self.state.pitch_rate += angular_acc[1] * dt
        self.state.yaw_rate += angular_acc[2] * dt

        # 4. 자세 적분 (오일러 각 미분 방정식)
        # [phi_dot, theta_dot, psi_dot] = T(phi, theta) @ [roll_rate, pitch_rate, yaw_rate]
        phi, theta = self.state.roll, self.state.pitch
        c_phi, s_phi = np.cos(phi), np.sin(phi)
        c_theta, s_theta = np.cos(theta), np.sin(theta)
        t_theta = np.tan(theta)

        # Euler angle rates transformation matrix
        # https://en.wikipedia.org/wiki/Euler_angles#Relationship_with_angular_velocity
        T = np.array([
            [1.0, s_phi * t_theta, c_phi * t_theta],
            [0.0, c_phi, -s_phi],
            [0.0, s_phi / c_theta, c_phi / c_theta]
        ])

        euler_rates = T @ np.array([self.state.roll_rate, self.state.pitch_rate, self.state.yaw_rate])

        self.state.roll += euler_rates[0] * dt
        self.state.pitch += euler_rates[1] * dt
        self.state.yaw += euler_rates[2] * dt

        # 5. 위치 적분 (바디 속도를 관성 좌표계로 변환)
        # 회전 행렬 (바디 → 관성)
        R = self._rotation_matrix()

        v_body = np.array([self.state.velocity_x, self.state.velocity_y, self.state.heave_dot])
        v_inertial = R @ v_body

        self.state.x += v_inertial[0] * dt
        self.state.y += v_inertial[1] * dt
        self.state.heave += v_inertial[2] * dt  # heave는 관성계 z 성분으로 적분

    def calculate_accelerations(self, forces: np.ndarray,
                                moments: np.ndarray,
                                add_gravity=True) -> Tuple[np.ndarray, np.ndarray]:
        """선형 및 각 가속도 계산 (Newton-Euler 방정식)

        Args:
            forces: 차체에 작용하는 총 힘 [Fx, Fy, Fz] (바디 좌표계) [N]
            moments: 차체에 작용하는 총 모멘트 [Mx, My, Mz] (바디 좌표계) [N·m]
            add_gravity: True이면 중력(-m*g)을 바디 좌표계로 변환해 forces에 더함
                         (이미 입력 forces에 중력항이 포함돼 있으면 False로 설정)

        Returns:
            linear_acc: 선형 가속도 [ax, ay, az] (바디 좌표계) [m/s²]
            angular_acc: 각 가속도 [wx_dot, wy_dot, wz_dot] (바디 좌표계) [rad/s²]
        """
        # 회전 좌표계에서 Newton-Euler 방정식
        # F = m × (a + ω × v)  =>  a = F/m - ω × v
        # M = I × α + ω × (I × ω)  =>  α = I^-1 × (M - ω × (I × ω))

        omega = np.array([self.state.roll_rate, self.state.pitch_rate, self.state.yaw_rate])
        v = np.array([self.state.velocity_x, self.state.velocity_y, self.state.heave_dot])

        # 1. 선형 가속도 (중력 포함 여부 선택)
        F_total = forces.copy()
        if add_gravity:
            R = self._rotation_matrix()
            # 중력은 스프렁 질량에만 작용 (언스프렁은 서스펜션에서 별도 처리)
            F_gravity_inertial = np.array([0.0, 0.0, -self.params.m * self.params.g])
            F_gravity_body = R.T @ F_gravity_inertial
            F_total = F_total + F_gravity_body

        # Coriolis/centrifugal force 보정
        # IMPORTANT: F_total은 4개 코너 힘의 총합이므로 전체 질량 사용
        coriolis_term = np.cross(omega, v)
        linear_acc = np.array([
            F_total[0] / self.params.m_total - coriolis_term[0],  # ax: 전체 질량
            F_total[1] / self.params.m_total - coriolis_term[1],  # ay: 전체 질량
            F_total[2] / self.params.m - coriolis_term[2]   # az: 전체 질량 (수정!)
        ])

        # 2. 각 가속도
        # 관성 텐서 (대각 + Ixz)
        I = np.array([
            [self.params.Ixx, 0.0, -self.params.Ixz],
            [0.0, self.params.Iyy, 0.0],
            [-self.params.Ixz, 0.0, self.params.Izz]
        ])

        # I × ω
        I_omega = I @ omega

        # 자이로스코픽 모멘트: ω × (I × ω)
        gyro_moment = np.cross(omega, I_omega)

        # α = I^-1 × (M - ω × (I × ω))
        angular_acc = np.linalg.solve(I, moments - gyro_moment)

        return linear_acc, angular_acc

    def get_wheel_position(self, wheel_idx: int) -> np.ndarray:
        """휠 위치 조회 (관성 좌표계)

        Args:
            wheel_idx: 휠 인덱스 (0=FL, 1=FR, 2=RR, 3=RL)

        Returns:
            position: 휠 중심의 위치 [X, Y, Z] (관성 좌표계) [m]
        """
        label = self.wheel_labels[wheel_idx]

        # corner_offsets에서 직접 x, y 값 사용
        x_i = self.corner_offsets[label]["x"]
        y_i = self.corner_offsets[label]["y"]
        r_body = np.array([x_i, y_i, -self.params.h_CG])

        # 회전 행렬 (바디 → 관성 좌표계, ZYX Euler 각 사용)
        R = self._rotation_matrix()

        # 관성 좌표계에서 휠 위치
        r_inertial = R @ r_body
        # heave는 편차 좌표이므로 절대 높이는 h_CG + heave
        z_abs = self.params.h_CG + self.state.heave
        pos_cg = np.array([self.state.x, self.state.y, z_abs])

        return pos_cg + r_inertial

    def get_wheel_velocity(self, wheel_idx: int, frame: str = "body") -> np.ndarray:
        """휠 속도 조회

        Args:
            wheel_idx: 휠 인덱스 (0=FL, 1=FR, 2=RR, 3=RL)
            frame: "body" 또는 "wheel" (조향각 적용 프레임)

        Returns:
            velocity: 휠 중심 속도 [vx, vy, vz] (요청한 좌표계) [m/s]
        """
        label = self.wheel_labels[wheel_idx]

        # corner_offsets에서 직접 x, y 값 사용
        x_i = self.corner_offsets[label]["x"]
        y_i = self.corner_offsets[label]["y"]
        r = np.array([x_i, y_i, -self.params.h_CG])

        # 휠 속도: V_wheel = V_CG + ω × r
        v_cg = np.array([self.state.velocity_x, self.state.velocity_y, self.state.heave_dot])
        omega = np.array([self.state.roll_rate, self.state.pitch_rate, self.state.yaw_rate])

        v_wheel = v_cg + np.cross(omega, r)

        if frame == "wheel":
            delta = self.corners[label].state.steering_angle
            c, s = np.cos(delta), np.sin(delta)
            v_wx = c * v_wheel[0] + s * v_wheel[1]
            v_wy = -s * v_wheel[0] + c * v_wheel[1]
            return np.array([v_wx, v_wy, v_wheel[2]])

        return v_wheel


    def get_state_vector(self) -> np.ndarray:
        """현재 상태를 벡터로 반환"""
        return np.array([
            self.state.x, self.state.y, self.state.heave,
            self.state.roll, self.state.pitch, self.state.yaw,
            self.state.velocity_x, self.state.velocity_y, self.state.heave_dot,
            self.state.roll_rate, self.state.pitch_rate, self.state.yaw_rate,
        ], dtype=float)

    def set_state_vector(self, state_vector: np.ndarray) -> None:
        """벡터로부터 상태 설정

        Args:
            state_vector: 12차원 상태 벡터 [x, y, heave, roll, pitch, yaw, velocity_x, velocity_y, heave_dot, roll_rate, pitch_rate, yaw_rate]
        """
        if len(state_vector) != 12:
            raise ValueError(f"State vector must have 12 elements, got {len(state_vector)}")

        self.state.x = float(state_vector[0])
        self.state.y = float(state_vector[1])
        self.state.heave = float(state_vector[2])
        self.state.roll = float(state_vector[3])
        self.state.pitch = float(state_vector[4])
        self.state.yaw = float(state_vector[5])
        self.state.velocity_x = float(state_vector[6])
        self.state.velocity_y = float(state_vector[7])
        self.state.heave_dot = float(state_vector[8])
        self.state.roll_rate = float(state_vector[9])
        self.state.pitch_rate = float(state_vector[10])
        self.state.yaw_rate = float(state_vector[11])

    def reset(self) -> None:
        """차체 상태 초기화 (평형 상태로 리셋)"""
        self.state = VehicleBodyState()

        # E-Corner도 평형으로 리셋
        for corner in self.corners.values():
            corner.suspension.reset()

        # 리셋 후 한 번 업데이트하여 평형 상태의 힘을 계산
        dummy_inputs = {
            label: {
                "T_steer": 0.0,
                "T_brk": 0.0,
                "T_Drv": 0.0,
                "T_susp": 0.0,
                "z_road": 0.0
            }
            for label in self.wheel_labels
        }
        self.update(dt=0.001, corner_inputs=dummy_inputs)

    def get_corner_inputs(self) -> Dict:
        """
        E-Corner로 전달할 입력 생성 (4바퀴 고정: FL, FR, RR, RL)
        출력: 바퀴당 휠 속도(x, y, 바디 프레임), heave/roll/pitch 및 roll_dot/pitch_dot/heave_dot
        """
        wheel_velocities = {
            label: self.get_wheel_velocity(idx, frame="body")[:2]
            for idx, label in enumerate(self.wheel_labels)
        }

        # heave는 관성계 z 편차, heave_rate는 바디 속도를 관성계로 변환한 z 성분
        R = self._rotation_matrix()
        v_body = np.array([self.state.velocity_x, self.state.velocity_y, self.state.heave_dot])
        heave_rate_inertial = float((R @ v_body)[2])

        return {
            "wheel_velocities": wheel_velocities,
            "roll": self.state.roll,
            "pitch": self.state.pitch,
            "heave": self.state.heave,  # 이미 편차 좌표
            "roll_rate": self.state.roll_rate,
            "pitch_rate": self.state.pitch_rate,
            "heave_rate": heave_rate_inertial,
        }

    def get_outputs(self) -> Dict:
        """
        차체 외부로 제공할 출력 생성 (4바퀴 고정: FL, FR, RR, RL)
        포함: Vx, Vy, roll, pitch, yaw, heave, yaw_rate, 바퀴당 휠 속도(x, y, 휠 프레임)
        """
        wheel_velocities = {
            label: self.get_wheel_velocity(idx, frame="wheel")[:2]
            for idx, label in enumerate(self.wheel_labels)
        }

        return {
            "velocity_x": self.state.velocity_x,
            "velocity_y": self.state.velocity_y,
            "roll": self.state.roll,
            "pitch": self.state.pitch,
            "yaw": self.state.yaw,
            "heave": self.state.heave,  # 이미 편차 좌표
            "yaw_rate": self.state.yaw_rate,
            "wheel_velocities": wheel_velocities,
        }

    def assemble_forces_moments(self, corner_outputs: Dict[str, Tuple[float, float, float]],
                                ax: float = 0.0, ay: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        4개 코너 힘을 차체 좌표계 forces, moments로 합산 + 가속도 기반 관성 모멘트 추가

        Args:
            corner_outputs: 코너별 출력 {label: (F_s, F_x, F_y), ...}
                - F_s: 서스펜션 힘 (z 방향) [N]
                - F_x: 종방향 타이어 힘 (휠 좌표계) [N]
                - F_y: 횡방향 타이어 힘 (휠 좌표계) [N]
            ax: 종방향 가속도 [m/s²] (이전 스텝 값)
            ay: 횡방향 가속도 [m/s²] (이전 스텝 값)

        Returns:
            forces: 총 힘 [Fx, Fy, Fz] (바디 좌표계) [N]
            moments: 총 모멘트 [Mx, My, Mz] (CG 기준, 바디 좌표계) [N·m]

        좌표계:
            - 바디 좌표계: x=전방, y=좌측, z=상방
            - 모멘트: Mx=롤, My=피치, Mz=요

        합산 공식:
            F_total = Σ F_i
            Mx (Roll)  = Σ (y_i × F_s)           # 좌우 수직력 차이
            My (Pitch) = Σ (-x_i × F_s)          # 전후 수직력 차이
            Mz (Yaw)   = Σ (x_i × F_y - y_i × F_x) # 수평 힘의 모멘트

        관성 모멘트 (하중 이동):
            M_x_acc = m* × h_CG × ay   # 롤 모멘트 (선회 시)
            M_y_acc = -m* × h_CG × ax  # 피치 모멘트 (가속/제동 시)
        """
        # 초기화
        F_total = np.zeros(3)  # [Fx, Fy, Fz]
        M_total = np.zeros(3)  # [Mx, My, Mz]

        for label in self.wheel_labels:
            if label not in corner_outputs:
                continue

            F_s, F_x, F_y = corner_outputs[label]
            # 조향각으로 휠→바디 회전
            delta = self.corners[label].state.steering_angle
            c, s = np.cos(delta), np.sin(delta)
            F_x_body = c * F_x - s * F_y
            F_y_body = s * F_x + c * F_y

            # 1. 코너 위치 (CG 기준, 바디 좌표계) - corner_offsets에서 직접 사용
            x_i = self.corner_offsets[label]["x"]
            y_i = self.corner_offsets[label]["y"]

            # 2. 힘 합산
            F_corner = np.array([F_x_body, F_y_body, F_s])
            F_total += F_corner

            # 3. 모멘트 합산 (물리적 의미에 맞게 직접 계산)
            # Roll/Pitch는 수직력(F_s)만, Yaw는 수평력(F_x, F_y)만
            M_roll = y_i * F_s                    # 좌우 위치 × 수직력
            M_pitch = -x_i * F_s                  # 전후 위치 × 수직력
            M_yaw = x_i * F_y_body - y_i * F_x_body  # 수평면 모멘트

            M_corner = np.array([M_roll, M_pitch, M_yaw])
            M_total += M_corner

        # 4. 가속도 기반 관성 모멘트 추가 (하중 이동 효과)
        # 동적 CG 높이 사용: h_CG = z_CG0 + heave
        h_CG_dynamic = self.params.h_CG + self.state.heave
        m_star = self.params.m  # 스프렁 질량

        # 관성 모멘트 계산
        M_x_acc = m_star * h_CG_dynamic * ay   # 롤 모멘트 (선회 시, +ay → +roll)
        M_y_acc = -m_star * h_CG_dynamic * ax  # 피치 모멘트 (가속 시, +ax → -pitch/nose-up)

        # 모멘트에 추가
        M_total[0] += M_x_acc
        M_total[1] += M_y_acc

        return F_total, M_total
