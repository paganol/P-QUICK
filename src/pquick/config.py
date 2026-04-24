from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DetectorSelection:
    """Filters for selecting a subset of detectors from the full focal plane.

    Attributes:
        channel: Optional channel prefix (e.g. ``"100"``); keeps only detectors whose
            name starts with this string.
        detectors: Explicit allowlist of detector names; if non-empty, only these pass.
    """

    channel: str | None = None
    detectors: list[str] = field(default_factory=list)


@dataclass
class PointingConfig:
    """Configuration for locating and selecting input pointing NPZ files.

    Attributes:
        input_root: Common path prefix for pointing NPZ files, e.g.
            ``"inputs/pointings/processed_od_"``.
        mission_length: OD selector (e.g. ``"full"``, ``"survey 2"``,
            or explicit ``"91-99"``). Defaults to ``"full"`` when omitted.
        use_flag: If ``True`` (default), flagged samples are excluded from the
            convolution and map accumulation. Set to ``False`` to ignore the
            flag array and process all samples.
    """

    input_root: str
    mission_length: str | None = None
    use_flag: bool = True


@dataclass
class ResamplingConfig:
    """Parameters controlling native-rate pointing reconstruction.

    Attributes:
        coordinate_system: Output sky frame for reconstructed pointing. Supported
            values are ``"ecliptic"`` and ``"galactic"``.
    """

    coordinate_system: str = "galactic"


@dataclass
class ConvolutionConfig:
    """Parameters controlling the ``ducc0`` total-convolution step.

    Attributes:
        lmax: Maximum multipole ℓ for both sky and beam ALMs.
        mmax: Maximum azimuthal order *m* of the beam (``kmax`` in ducc0 notation).
        epsilon: Interpolation accuracy target for the ducc0 gridder.
        chunks: Number of equal-length interpolation calls per OD (1 = whole OD at once).
    """

    lmax: int
    mmax: int
    epsilon: float = 1e-5
    chunks: int = 1


@dataclass
class MapConfig:
    """HEALPix output map parameters: resolution, pixel ordering, and filename prefix."""

    nside: int
    nest: bool = False
    output_prefix: str = "pquick"


@dataclass
class InputsConfig:
    """Paths to all pipeline inputs: sky ALMs, beam directory, RIMO, and pointings."""

    sky_alm: str
    beams_dir: str
    rimo_file: str
    pointing: PointingConfig


@dataclass
class OutputConfig:
    """Specifies the directory where output FITS maps are written."""

    output_dir: str = "outputs"


@dataclass
class PipelineConfig:
    """Top-level configuration object aggregating all sub-configs for a pipeline run.

    Attributes:
        nthreads: Number of threads for ducc0 and numba.
            ``0`` = read ``OMP_NUM_THREADS`` from the environment (fallback 1).
    """

    inputs: InputsConfig
    detector_selection: DetectorSelection
    resampling: ResamplingConfig
    convolution: ConvolutionConfig
    map: MapConfig
    output: OutputConfig
    verbose: bool = False
    nthreads: int = 0


def _to_dataclass(data: dict[str, Any]) -> PipelineConfig:
    detsel = data.get("detector_selection") or {}
    channel = detsel.get("channel")
    detectors = list(detsel.get("detectors") or [])
    if channel and detectors:
        raise ValueError("Specify only one of detector_selection.channel or detector_selection.detectors")

    return PipelineConfig(
        inputs=InputsConfig(
            sky_alm=data["inputs"]["sky_alm"],
            beams_dir=data["inputs"]["beams_dir"],
            rimo_file=str(data["inputs"]["rimo_file"]),
            pointing=PointingConfig(
                input_root=data["inputs"]["pointing"]["input_root"],
                mission_length=data["inputs"]["pointing"].get("mission_length"),
                use_flag=bool(data["inputs"]["pointing"].get("use_flag", True)),
            ),
        ),
        detector_selection=DetectorSelection(
            channel=channel,
            detectors=detectors,
        ),
        resampling=ResamplingConfig(
            coordinate_system=str(data.get("resampling", {}).get("coordinate_system", "galactic")),
        ),
        convolution=ConvolutionConfig(
            lmax=int(data["convolution"]["lmax"]),
            mmax=int(data["convolution"].get("mmax", data["convolution"].get("kmax"))),
            epsilon=float(data.get("convolution", {}).get("epsilon", 1e-5)),
            chunks=int(data.get("convolution", {}).get("chunks", 1)),
        ),
        map=MapConfig(
            nside=int(data["map"]["nside"]),
            nest=bool(data.get("map", {}).get("nest", False)),
            output_prefix=str(data.get("map", {}).get("output_prefix", "pquick")),
        ),
        output=OutputConfig(output_dir=str(data.get("output", {}).get("output_dir", "outputs"))),
        verbose=bool(data.get("verbose", False)),
        nthreads=int(data.get("nthreads", 0)),
    )


def load_config(path: str | Path) -> PipelineConfig:
    """Read a YAML file and deserialise it into a :class:`PipelineConfig`.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        A fully populated :class:`PipelineConfig` instance.

    Raises:
        ValueError: If the YAML root element is not a mapping.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a YAML mapping")
    return _to_dataclass(raw)
