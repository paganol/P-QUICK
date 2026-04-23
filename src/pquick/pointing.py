from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .quaternion import normalize_quaternion, quat_mul


@dataclass
class PointingData:
    """Undersampled (coarse) pointing data loaded from an NPZ file.

    Attributes:
        time_us: Timestamps in nanoseconds at the coarse sampling rate.
        quat_us: Unit quaternions ``(x, y, z, w)`` at the coarse rate, shape ``(N, 4)``.
        flag: Quality flags at the **native** (full-rate) sampling rate.
        sampling_rate_hz: Native (full-rate) detector sampling frequency in Hz.
        original_indices: Optional mapping from coarse samples to native-rate indices,
            enabling non-uniform undersampling.
    """

    time_us: np.ndarray
    quat_us: np.ndarray
    flag: np.ndarray
    sampling_rate_hz: float
    original_indices: np.ndarray | None = None


@dataclass
class NativePointing:
    """Full-rate reconstructed pointing: time grid, quaternions, and combined flags."""

    time_native: np.ndarray
    quat_native: np.ndarray
    flag_native: np.ndarray


@dataclass
class PointingInterpolator:
    """Wraps a ``ducc0.PointingProvider`` with the metadata needed for chunked TOD generation.

    Attributes:
        provider: Underlying ``ducc0.pointingprovider.PointingProvider`` instance.
        coarse_t0_ns: Start timestamp of the coarse time grid in nanoseconds.
        native_rate_hz: Full detector sampling rate in Hz.
        n_native: Total number of native-rate samples in this OD.
        flag_native: Combined quality flags at the native rate, shape ``(n_native,)``.
        frame_rotation: Fixed quaternion applied to convert native pointing into the
            requested sky frame.
    """

    provider: object
    coarse_t0_ns: float
    native_rate_hz: float
    n_native: int
    flag_native: np.ndarray
    frame_rotation: np.ndarray

    def get_detector_quaternions(self, detector_quat: np.ndarray, start: int, count: int) -> np.ndarray:
        """Compose boresight with detector offset quaternion for a chunk of native samples.

        Args:
            detector_quat: Fixed detector offset quaternion ``(x, y, z, w)``, shape ``(4,)``.
            start: Index of the first native sample in the chunk.
            count: Number of native samples to evaluate.

        Returns:
            Float64 array of shape ``(count, 4)`` containing the rotated unit quaternions.
        """
        t0_s = float(start) / float(self.native_rate_hz)
        detq = normalize_quaternion(np.asarray(detector_quat, dtype=np.float64))
        q = np.asarray(
            self.provider.get_rotated_quaternions(
                t0_s,
                float(self.native_rate_hz),
                detq,
                int(count),
                False,
            ),
            dtype=np.float64,
        )
        return normalize_quaternion(quat_mul(self.frame_rotation, q))


def _rotation_matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError("rotation must have shape (3, 3)")

    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quat = np.array(
            [
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
                0.25 * scale,
            ],
            dtype=np.float64,
        )
        return normalize_quaternion(quat)

    diag = np.diag(rotation)
    idx = int(np.argmax(diag))
    if idx == 0:
        scale = 2.0 * np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
        quat = np.array(
            [
                0.25 * scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
            ],
            dtype=np.float64,
        )
    elif idx == 1:
        scale = 2.0 * np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
        quat = np.array(
            [
                (rotation[0, 1] + rotation[1, 0]) / scale,
                0.25 * scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
            ],
            dtype=np.float64,
        )
    else:
        scale = 2.0 * np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
        quat = np.array(
            [
                (rotation[0, 2] + rotation[2, 0]) / scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
                0.25 * scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ],
            dtype=np.float64,
        )
    return normalize_quaternion(quat)


def _frame_rotation_quaternion(coordinate_system: str) -> np.ndarray:
    frame = coordinate_system.strip().lower()
    if frame == "ecliptic":
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    if frame != "galactic":
        raise ValueError(
            f"Unsupported coordinate_system={coordinate_system!r}; expected 'ecliptic' or 'galactic'"
        )

    from astropy.coordinates import BarycentricMeanEcliptic, CartesianRepresentation, Galactic, SkyCoord
    from astropy import units as u

    basis = np.eye(3, dtype=np.float64)
    cols = []
    for axis in basis:
        coord = SkyCoord(
            CartesianRepresentation(*(axis * u.one)),
            frame=BarycentricMeanEcliptic(),
        ).transform_to(Galactic())
        cols.append(coord.cartesian.xyz.to_value(u.one))
    rotation = np.column_stack(cols)
    return _rotation_matrix_to_quaternion(rotation)


def _normalize_original_indices(original_indices: np.ndarray | None, n_us: int) -> np.ndarray | None:
    if original_indices is None:
        return None
    idx = np.asarray(original_indices, dtype=np.int64)
    if idx.ndim != 1 or idx.size != n_us:
        raise ValueError("original_indices must be 1D and match undersampled array length")
    if np.any(np.diff(idx) <= 0):
        raise ValueError("original_indices must be strictly increasing")
    return idx - int(idx[0])


def reconstruct_native_time(
    time_us: np.ndarray,
    sampling_rate_hz: float,
    original_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Reconstruct a uniformly spaced native-rate time grid from undersampled timestamps.

    When *original_indices* is provided, the grid spans from the first undersampled
    timestamp to the native index of the last sample; otherwise the span is inferred
    from the median time step.

    Args:
        time_us: Coarse timestamps in nanoseconds, strictly increasing.
        sampling_rate_hz: Native detector sampling rate in Hz.
        original_indices: Optional mapping from coarse to native sample indices.

    Returns:
        1-D float64 array of native-rate timestamps in nanoseconds.

    Raises:
        ValueError: If inputs are inconsistent or non-monotone.
    """
    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive")
    time_us = np.asarray(time_us, dtype=np.float64)
    if time_us.ndim != 1 or time_us.size < 2:
        raise ValueError("time_us must be a 1D array with at least 2 samples")
    if np.any(np.diff(time_us) <= 0):
        raise ValueError("time_us must be strictly increasing")

    dt_ns = 1e9 / sampling_rate_hz
    norm_idx = _normalize_original_indices(original_indices, time_us.size)
    if norm_idx is not None:
        n_samp = int(norm_idx[-1]) + 1
    else:
        n_samp = int(np.floor((time_us[-1] - time_us[0]) / dt_ns + 0.5)) + 1
    return time_us[0] + dt_ns * np.arange(n_samp, dtype=np.float64)


def _estimate_coarse_rate_hz(
    time_us: np.ndarray,
    native_rate_hz: float,
    original_indices: np.ndarray | None = None,
) -> float:
    norm_idx = _normalize_original_indices(original_indices, np.asarray(time_us).size)
    if norm_idx is not None and norm_idx.size > 1:
        step = float(np.median(np.diff(norm_idx)))
        if step <= 0:
            raise ValueError("Invalid original_indices step for coarse-rate estimate")
        return float(native_rate_hz / step)

    dt_ns = np.diff(np.asarray(time_us, dtype=np.float64))
    if np.any(dt_ns <= 0):
        raise ValueError("undersampled time grid must be strictly increasing")
    return float(1e9 / np.median(dt_ns))


def reconstruct_native_pointing(
    pointing: PointingData,
    angular_eps: float = 1e-10,
    coordinate_system: str = "galactic",
) -> NativePointing:
    """Upsample boresight quaternions to the native rate via ``ducc0.PointingProvider``.

    Uses native-rate quality flags from :class:`PointingData` directly.

    Args:
        pointing: Undersampled pointing data from :func:`~pquick.io.load_pointing_npz`.
        angular_eps: Unused; kept for API compatibility.
        coordinate_system: Output sky frame for the returned quaternions. Supported
            values are ``"ecliptic"`` and ``"galactic"``.

    Returns:
        A :class:`NativePointing` instance at the full detector sampling rate.
    """
    del angular_eps
    try:
        from ducc0.pointingprovider import PointingProvider
    except Exception as exc:  # pragma: no cover
        raise ImportError("ducc0 is required for pointing interpolation") from exc

    time_native = reconstruct_native_time(pointing.time_us, pointing.sampling_rate_hz, pointing.original_indices)
    coarse_rate_hz = _estimate_coarse_rate_hz(pointing.time_us, pointing.sampling_rate_hz, pointing.original_indices)

    pp = PointingProvider(0.0, coarse_rate_hz, pointing.quat_us)
    quat_native = np.asarray(
        pp.get_rotated_quaternions(
            0.0,
            float(pointing.sampling_rate_hz),
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            int(time_native.size),
            False,
        ),
        dtype=np.float64,
    )
    quat_native = normalize_quaternion(quat_mul(_frame_rotation_quaternion(coordinate_system), quat_native))

    # Keep bad samples marked; flags are already at native rate.
    flag_native = np.asarray(pointing.flag, dtype=np.int8)

    return NativePointing(time_native=time_native, quat_native=quat_native, flag_native=flag_native)


def build_pointing_interpolator(
    pointing: PointingData,
    coordinate_system: str = "galactic",
) -> PointingInterpolator:
    """Create a :class:`PointingInterpolator` backed by ``ducc0.PointingProvider``.

    Reconstructs the native time grid, merges quality flags, and initialises the
    ``PointingProvider`` at the coarse rate for subsequent on-the-fly interpolation
    during chunked TOD generation.

    Args:
        pointing: Undersampled pointing data from :func:`~pquick.io.load_pointing_npz`.
        coordinate_system: Output sky frame for detector quaternions. Supported
            values are ``"ecliptic"`` and ``"galactic"``.

    Returns:
        A :class:`PointingInterpolator` ready for use in the pipeline loop.
    """
    try:
        from ducc0.pointingprovider import PointingProvider
    except Exception as exc:  # pragma: no cover
        raise ImportError("ducc0 is required for pointing interpolation") from exc

    time_native = reconstruct_native_time(pointing.time_us, pointing.sampling_rate_hz, pointing.original_indices)
    coarse_rate_hz = _estimate_coarse_rate_hz(pointing.time_us, pointing.sampling_rate_hz, pointing.original_indices)

    # Flags are already at native rate.
    flag_native = np.asarray(pointing.flag, dtype=np.int8)

    pp = PointingProvider(0.0, coarse_rate_hz, pointing.quat_us)
    return PointingInterpolator(
        provider=pp,
        coarse_t0_ns=float(pointing.time_us[0]),
        native_rate_hz=float(pointing.sampling_rate_hz),
        n_native=int(flag_native.size),
        flag_native=flag_native,
        frame_rotation=_frame_rotation_quaternion(coordinate_system),
    )
