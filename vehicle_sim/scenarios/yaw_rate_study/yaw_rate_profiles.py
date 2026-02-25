#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple

import numpy as np


_EPS = 1e-12


def _to_float(cfg: Mapping[str, object], key: str, default: float) -> float:
    value = cfg.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _smoothstep01(x: float) -> Tuple[float, float]:
    """Return smoothstep value and derivative for x in [0, 1]."""
    x = float(np.clip(x, 0.0, 1.0))
    y = x * x * (3.0 - 2.0 * x)
    dy_dx = 6.0 * x * (1.0 - x)
    return y, dy_dx


def _ramp_with_derivative(t: float, start: float, ramp_time: float) -> Tuple[float, float]:
    if ramp_time <= _EPS:
        return (1.0, 0.0) if t >= start else (0.0, 0.0)
    if t <= start:
        return 0.0, 0.0
    if t >= start + ramp_time:
        return 1.0, 0.0
    x = (t - start) / ramp_time
    y, dy_dx = _smoothstep01(x)
    return y, dy_dx / ramp_time


class YawProfileBase:
    def evaluate(self, t: float) -> Tuple[float, float]:
        raise NotImplementedError


@dataclass
class SineProfile(YawProfileBase):
    amp: float
    freq_hz: float
    start_delay: float
    ramp_time: float

    def evaluate(self, t: float) -> Tuple[float, float]:
        t = float(t)
        local = max(0.0, t - self.start_delay)
        ramp, ramp_dot = _ramp_with_derivative(t, self.start_delay, self.ramp_time)
        w = 2.0 * np.pi * self.freq_hz
        phase = w * local
        s = np.sin(phase)
        c = np.cos(phase)
        yaw_rate = self.amp * ramp * s
        yaw_accel = self.amp * (ramp_dot * s + ramp * w * c)
        return float(yaw_rate), float(yaw_accel)


@dataclass
class CircleProfile(YawProfileBase):
    yaw_rate: float
    start_delay: float
    ramp_time: float

    def evaluate(self, t: float) -> Tuple[float, float]:
        ramp, ramp_dot = _ramp_with_derivative(float(t), self.start_delay, self.ramp_time)
        return float(self.yaw_rate * ramp), float(self.yaw_rate * ramp_dot)


@dataclass
class StepProfile(YawProfileBase):
    value: float
    start_time: float
    ramp_time: float

    def evaluate(self, t: float) -> Tuple[float, float]:
        ramp, ramp_dot = _ramp_with_derivative(float(t), self.start_time, self.ramp_time)
        return float(self.value * ramp), float(self.value * ramp_dot)


@dataclass
class SquareProfile(YawProfileBase):
    amp: float
    freq_hz: float
    start_delay: float
    ramp_time: float
    transition_time: float
    start_sign: int

    def evaluate(self, t: float) -> Tuple[float, float]:
        t = float(t)
        local = max(0.0, t - self.start_delay)
        ramp, ramp_dot = _ramp_with_derivative(t, self.start_delay, self.ramp_time)

        w = max(_EPS, 2.0 * np.pi * self.freq_hz)
        phase = w * local
        sin_term = np.sin(phase)
        cos_term = np.cos(phase)

        # Smaller eps means a sharper edge.
        edge = max(_EPS, 0.25 * w * max(_EPS, self.transition_time))
        shape = np.tanh(sin_term / edge)
        dshape_dt = (1.0 - shape * shape) * (w * cos_term / edge)

        signed_shape = float(np.sign(self.start_sign)) * shape
        signed_dshape_dt = float(np.sign(self.start_sign)) * dshape_dt

        yaw_rate = self.amp * ramp * signed_shape
        yaw_accel = self.amp * (ramp_dot * signed_shape + ramp * signed_dshape_dt)
        return float(yaw_rate), float(yaw_accel)


@dataclass
class ChirpProfile(YawProfileBase):
    amp: float
    f0_hz: float
    f1_hz: float
    t1: float
    start_delay: float
    ramp_time: float

    def _phase_and_freq(self, local_t: float) -> Tuple[float, float]:
        local_t = max(0.0, float(local_t))
        t1 = max(_EPS, self.t1)
        k = (self.f1_hz - self.f0_hz) / t1

        if local_t <= t1:
            phase = 2.0 * np.pi * (self.f0_hz * local_t + 0.5 * k * local_t * local_t)
            freq = self.f0_hz + k * local_t
            return phase, freq

        phase_t1 = 2.0 * np.pi * (self.f0_hz * t1 + 0.5 * k * t1 * t1)
        phase = phase_t1 + 2.0 * np.pi * self.f1_hz * (local_t - t1)
        return phase, self.f1_hz

    def evaluate(self, t: float) -> Tuple[float, float]:
        t = float(t)
        local = max(0.0, t - self.start_delay)
        ramp, ramp_dot = _ramp_with_derivative(t, self.start_delay, self.ramp_time)
        phase, freq = self._phase_and_freq(local)

        s = np.sin(phase)
        c = np.cos(phase)
        phase_dot = 2.0 * np.pi * freq

        yaw_rate = self.amp * ramp * s
        yaw_accel = self.amp * (ramp_dot * s + ramp * phase_dot * c)
        return float(yaw_rate), float(yaw_accel)


def build_profile(cfg: Mapping[str, object] | None) -> YawProfileBase:
    cfg = cfg or {}
    profile_type = str(cfg.get("type", "sine")).strip().lower()

    if profile_type == "sine":
        c = cfg.get("sine", {}) if isinstance(cfg.get("sine", {}), Mapping) else {}
        return SineProfile(
            amp=_to_float(c, "amp", 0.3),
            freq_hz=_to_float(c, "freq_hz", 0.5),
            start_delay=_to_float(c, "start_delay", 1.0),
            ramp_time=_to_float(c, "ramp_time", 1.0),
        )

    if profile_type == "circle":
        c = cfg.get("circle", {}) if isinstance(cfg.get("circle", {}), Mapping) else {}
        return CircleProfile(
            yaw_rate=_to_float(c, "yaw_rate", 0.2),
            start_delay=_to_float(c, "start_delay", 1.0),
            ramp_time=_to_float(c, "ramp_time", 1.0),
        )

    if profile_type == "step":
        c = cfg.get("step", {}) if isinstance(cfg.get("step", {}), Mapping) else {}
        return StepProfile(
            value=_to_float(c, "value", 0.25),
            start_time=_to_float(c, "start_time", 1.0),
            ramp_time=_to_float(c, "ramp_time", 0.5),
        )

    if profile_type == "square":
        c = cfg.get("square", {}) if isinstance(cfg.get("square", {}), Mapping) else {}
        start_sign = int(_to_float(c, "start_sign", 1.0))
        if start_sign == 0:
            start_sign = 1
        return SquareProfile(
            amp=_to_float(c, "amp", 0.25),
            freq_hz=_to_float(c, "freq_hz", 0.5),
            start_delay=_to_float(c, "start_delay", 1.0),
            ramp_time=_to_float(c, "ramp_time", 1.0),
            transition_time=_to_float(c, "transition_time", 0.1),
            start_sign=start_sign,
        )

    if profile_type == "chirp":
        c = cfg.get("chirp", {}) if isinstance(cfg.get("chirp", {}), Mapping) else {}
        return ChirpProfile(
            amp=_to_float(c, "amp", 0.25),
            f0_hz=_to_float(c, "f0_hz", 0.2),
            f1_hz=_to_float(c, "f1_hz", 1.5),
            t1=_to_float(c, "t1", 8.0),
            start_delay=_to_float(c, "start_delay", 1.0),
            ramp_time=_to_float(c, "ramp_time", 1.0),
        )

    supported = ["sine", "circle", "step", "square", "chirp"]
    raise ValueError(f"Unknown yaw_profile.type='{profile_type}'. Supported: {supported}")
