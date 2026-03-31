from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .quaternion import upsample_quaternions


@dataclass
class PointingData:
    time_us: np.ndarray
    quat_us: np.ndarray
    flag_ext1: np.ndarray
    flag_ext3: np.ndarray
    sampling_rate_hz: float


@dataclass
class NativePointing:
    time_native: np.ndarray
    quat_native: np.ndarray
    flag_native: np.ndarray


def reconstruct_native_time(time_us: np.ndarray, sampling_rate_hz: float) -> np.ndarray:
    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive")
    time_us = np.asarray(time_us, dtype=np.float64)
    if time_us.ndim != 1 or time_us.size < 2:
        raise ValueError("time_us must be a 1D array with at least 2 samples")
    if np.any(np.diff(time_us) <= 0):
        raise ValueError("time_us must be strictly increasing")

    dt_ns = 1e9 / sampling_rate_hz
    n_samp = int(np.floor((time_us[-1] - time_us[0]) / dt_ns + 0.5)) + 1
    return time_us[0] + dt_ns * np.arange(n_samp, dtype=np.float64)


def _expand_flag_nearest(flag_us: np.ndarray, n_native: int) -> np.ndarray:
    if flag_us.size == n_native:
        return flag_us.astype(np.int8, copy=False)
    idx = np.linspace(0, flag_us.size - 1, num=n_native)
    idx = np.rint(idx).astype(np.int64)
    return flag_us[idx].astype(np.int8, copy=False)


def reconstruct_native_pointing(pointing: PointingData, angular_eps: float = 1e-10) -> NativePointing:
    time_native = reconstruct_native_time(pointing.time_us, pointing.sampling_rate_hz)
    quat_native = upsample_quaternions(pointing.time_us, pointing.quat_us, time_native, eps=angular_eps)

    # Keep bad samples marked; ext1 and ext3 flags are OR-combined.
    f1 = _expand_flag_nearest(pointing.flag_ext1, time_native.size)
    f3 = _expand_flag_nearest(pointing.flag_ext3, time_native.size)
    flag_native = np.bitwise_or(f1, f3).astype(np.int8)

    return NativePointing(time_native=time_native, quat_native=quat_native, flag_native=flag_native)
