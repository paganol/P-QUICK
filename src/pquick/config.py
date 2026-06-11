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
class InputsConfig:
    """Paths to all pipeline inputs.

    Attributes:
        sky_alm: Sky ALM file path.
        beams_dir: Directory with detector beam ALM FITS files.
        rimo_file: RIMO FITS path.
        mission_length: OD selector (e.g. ``"full"``, ``"survey 2"``,
            or explicit ``"91-99"``). Defaults to ``"full"`` when omitted.
        pointings: Prefix for pointing NPZ files. The pipeline appends
            ``od_{OD:04d}.npz``.
            Example: ``inputs/pointings/pointing_`` ->
            ``inputs/pointings/pointing_od_0091.npz``.
        flags: Prefix for flags NPZ files. The pipeline appends
            ``{FREQ:03d}ghz_od_{OD:04d}.npz``.
            Example: ``inputs/flags/flags_`` ->
            ``inputs/flags/flags_100ghz_od_0091.npz``.
            When ``None`` (default) or when the per-OD file is absent, flags
            are not applied and all samples are treated as good.
        bad_rings_file: Optional path to a TOAST/NPIPE-style bad-ring interval
            text file with rows ``<det_or_ALL> <tstart_s> <tstop_s>``.
            Intervals are applied on top of existing flags.
    """

    sky_alm: str
    beams_dir: str
    rimo_file: str
    mission_length: str | None = None
    pointings: str = "inputs/pointings/pointing_"
    flags: str | None = None
    bad_rings_file: str | None = None


@dataclass
class ResamplingConfig:
    """Parameters controlling native-rate pointing reconstruction.

    Attributes:
        coordinate_system: Output sky frame for reconstructed pointing. Supported
            values are ``"ecliptic"`` and ``"galactic"``.
        center_pointing: If ``True``, snap each resampled detector direction to
            the center of a HEALPix pixel before convolution. This suppresses
            subpixel effects by enforcing one representative direction per pixel.
            Pixel centers are always taken at ``map.nside``.
    """

    coordinate_system: str = "galactic"
    center_pointing: bool = False


@dataclass
class ConvolutionConfig:
    """Parameters controlling the ``ducc0`` total-convolution step.

    Attributes:
        lmax: Maximum multipole ℓ for both sky and beam ALMs.
        mmax: Maximum azimuthal order *m* of the beam (``kmax`` in ducc0 notation).
        epsilon: Interpolation accuracy target for the ducc0 gridder.
        chunks: Number of equal-length interpolation calls per OD (1 = whole OD at once).
        beam_normalization: Beam scaling mode before convolution. ``"unit_integral"``
            divides by ``sqrt(4 pi) b_00`` so a constant-sky input remains constant
            after convolution. ``"raw"`` uses the beam coefficients exactly as stored
            in the FITS file.
        polarized_beam: If ``True`` (default), synthesise a proper spin-2 polarised
            beam ``[T, E, B]`` from the scalar co-polar Planck blm so the convolution
            captures the detector polarisation response. If ``False``, use the
            intensity beam only (``[b, 0, 0]``); Q/U recover ~0.
        extra_psi_deg: Constant offset (degrees) added to the convolution psi only
            (the beam orientation), not to the map-making psi. Diagnostic knob for
            an asymmetric-beam orientation error: sweep it and look for the value
            that flattens the transfer-function ratio. ``0`` for production.
    """

    lmax: int
    mmax: int
    epsilon: float = 1e-5
    chunks: int = 1
    beam_normalization: str = "unit_integral"
    polarized_beam: bool = True
    extra_psi_deg: float = 0.0


@dataclass
class MapConfig:
    """HEALPix output map parameters.

    Attributes:
        nside: HEALPix resolution parameter of the output map.
        nest: If ``True``, write maps in NESTED ordering instead of the default RING ordering.
        use_cross_pol: If ``True`` (default), weight the map-making polarisation response
            by the per-detector ``rho = (1 - eps)/(1 + eps)`` from the RIMO (matches
            qp_planck ``rhohit: IMO``). If ``False``, assume ideal detectors (``rho = 1``,
            qp_planck ``rhohit: Ideal``). Does not affect the temperature map.
    """

    nside: int
    nest: bool = False
    use_cross_pol: bool = True


@dataclass
class OutputConfig:
    """Output location for generated products.

    Attributes:
        output_dir: Directory where output FITS maps are written.
        output_prefix: Filename stem used when constructing the output FITS map name.
    """

    output_dir: str = "outputs"
    output_prefix: str = "pquick"


@dataclass
class PipelineConfig:
    """Top-level configuration object aggregating all sub-configs for a pipeline run.

    Attributes:
        inputs: File paths and mission-selection options for the run.
        detector_selection: Rules for choosing which detectors participate in the pipeline.
        resampling: Native-rate pointing reconstruction options.
        convolution: Harmonic convolution parameters passed to ``ducc0``.
        map: HEALPix output map settings.
        output: Destination directory for generated files.
        verbose: If ``True``, emit more detailed progress logging while the pipeline runs.
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
    map_cfg = data.get("map", {}) or {}
    output_cfg = data.get("output", {}) or {}
    channel = detsel.get("channel")
    detectors = list(detsel.get("detectors") or [])
    if channel and detectors:
        raise ValueError("Specify only one of detector_selection.channel or detector_selection.detectors")

    return PipelineConfig(
        inputs=InputsConfig(
            sky_alm=data["inputs"]["sky_alm"],
            beams_dir=data["inputs"]["beams_dir"],
            rimo_file=str(data["inputs"]["rimo_file"]),
            mission_length=(str(data["inputs"].get("mission_length")) if data["inputs"].get("mission_length") is not None else None),
            pointings=str(data["inputs"].get("pointings", "inputs/pointings/pointing_")),
            flags=(str(data["inputs"]["flags"]) if data["inputs"].get("flags") is not None else None),
            bad_rings_file=(
                str(data["inputs"]["bad_rings_file"])
                if data["inputs"].get("bad_rings_file") is not None
                else None
            ),
        ),
        detector_selection=DetectorSelection(
            channel=channel,
            detectors=detectors,
        ),
        resampling=ResamplingConfig(
            coordinate_system=str(data.get("resampling", {}).get("coordinate_system", "galactic")),
            center_pointing=bool(data.get("resampling", {}).get("center_pointing", False)),
        ),
        convolution=ConvolutionConfig(
            lmax=int(data["convolution"]["lmax"]),
            mmax=int(data["convolution"].get("mmax", data["convolution"].get("kmax"))),
            epsilon=float(data.get("convolution", {}).get("epsilon", 1e-5)),
            chunks=int(data.get("convolution", {}).get("chunks", 1)),
            beam_normalization=str(data.get("convolution", {}).get("beam_normalization", "unit_integral")),
            polarized_beam=bool(data.get("convolution", {}).get("polarized_beam", True)),
            extra_psi_deg=float(data.get("convolution", {}).get("extra_psi_deg", 0.0)),
        ),
        map=MapConfig(
            nside=int(data["map"]["nside"]),
            nest=bool(map_cfg.get("nest", False)),
            use_cross_pol=bool(map_cfg.get("use_cross_pol", True)),
        ),
        output=OutputConfig(
            output_dir=str(output_cfg.get("output_dir", "outputs")),
            output_prefix=str(output_cfg.get("output_prefix", map_cfg.get("output_prefix", "pquick"))),
        ),
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
