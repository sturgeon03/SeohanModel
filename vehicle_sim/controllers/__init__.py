"""
Vehicle control modules
"""

try:
    from .steering_inner_loop_controller import SteeringInnerLoopController, SteeringInnerLoopGains
except ModuleNotFoundError as exc:
    if exc.name != "vehicle_sim.controllers.steering_inner_loop_controller":
        raise
    SteeringInnerLoopController = None
    SteeringInnerLoopGains = None
from .pid_controller import PIDController, PIDGains
from .steer_angle_ff import SteeringFeedforwardController, SteeringFeedforwardOptions
from .steer_motor_ff import (
    SteeringMotorTorqueFeedforwardController,
    SteeringMotorTorqueFeedforwardOptions,
)
from .lateral_force_estimator import (
    LateralForceEstimator,
    LateralForceEstimatorOptions,
)
from .slip_angle_estimator import SlipAngleEstimator, SlipAngleEstimatorOptions
from .tire_lateral_force_estimator import (
    TireLateralForceEstimator,
    TireLateralForceEstimatorOptions,
)
from .yaw_moment_feedforward_controller import (
    YawMomentFeedforwardController,
    YawMomentFeedforwardOptions,
)
from .yaw_moment_allocator import YawMomentAllocator
from .yaw_rate_to_steer_torque_feedforward_controller import (
    YawRateToSteerTorqueFeedforwardController,
)

__all__ = [
    'PIDController',
    'PIDGains',
    'SteeringFeedforwardController',
    'SteeringFeedforwardOptions',
    'SteeringMotorTorqueFeedforwardController',
    'SteeringMotorTorqueFeedforwardOptions',
    'LateralForceEstimator',
    'LateralForceEstimatorOptions',
    'SlipAngleEstimator',
    'SlipAngleEstimatorOptions',
    'TireLateralForceEstimator',
    'TireLateralForceEstimatorOptions',
    'YawMomentFeedforwardController',
    'YawMomentFeedforwardOptions',
    'YawMomentAllocator',
    'YawRateToSteerTorqueFeedforwardController',
]

if SteeringInnerLoopController is not None and SteeringInnerLoopGains is not None:
    __all__.extend([
        'SteeringInnerLoopController',
        'SteeringInnerLoopGains',
    ])
