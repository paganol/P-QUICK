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
        rescale: Per-component multipliers ``(x, y, z)`` applied to the input sky
            ``(almT, almE, almB)`` before convolution. ``None`` (default) means
            ``(1, 1, 1)`` (no rescaling); a scalar ``s`` means ``(s, s, s)``. Useful
            for isolating components, e.g. ``[1, 0, 0]`` -> T-only, ``[0, 1, 0]`` ->
            E-only.
        weights: Detector map-weight set. ``"NPIPE"`` (default) uses the per-horn
            qp_planck/NPIPE weight table; ``"PR3"`` uses the SRoll per-detector
            ``(calib/NEP)^2`` weights for HFI and the Planck-2018 per-horn
            ``2/(sigma_M^2 + sigma_S^2)`` weights for LFI.
    """

    sky_alm: str
    beams_dir: str
    rimo_file: str
    mission_length: str | None = None
    pointings: str = "inputs/pointings/pointing_"
    flags: str | None = None
    bad_rings_file: str | None = None
    rescale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    weights: str = "NPIPE"


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
        cache_interpolator: If ``True`` (default), build each detector's ducc0
            convolution cube once and reuse it across every OD/chunk on the rank. The
            cube depends only on the sky, beam, ``lmax``, ``mmax`` and ``epsilon`` — not
            on the pointing — so this removes a redundant per-OD rebuild (the dominant
            convolution cost) at the price of holding one cube per detector resident
            (~0.4 GB at lmax=1024/mmax=6, ~1-2 GB at lmax=2048). Set ``False`` to rebuild
            per OD (lower memory, slower).

    The scalar Planck blm is always synthesised into a spin-2 ``[T, E, B]`` beam
    (:func:`~pquick.io.build_polarized_beam_alm`): the ellipse is carried Dxx -> Pxx
    by ``psi_uv`` so a horn's two PSB arms co-orient in the common Pxx frame, and the
    beam, convolution and map-making all share that frame.
    """

    lmax: int
    mmax: int
    epsilon: float = 1e-5
    chunks: int = 1
    beam_normalization: str = "unit_integral"
    cache_interpolator: bool = True


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
        extended_outputs: If ``True``, also write the ``_hits``, ``_wpol`` and
            ``_nobs00`` diagnostic maps alongside ``_iqu``. Default ``False`` (only
            the T/Q/U map is written).
    """

    output_dir: str = "outputs"
    output_prefix: str = "pquick"
    extended_outputs: bool = False


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


def _parse_rescale(value: Any) -> tuple[float, float, float]:
    """Parse ``inputs.rescale`` into a (T, E, B) multiplier triple.

    ``None`` -> ``(1, 1, 1)``; a scalar ``s`` -> ``(s, s, s)``; a 3-element
    sequence ``[x, y, z]`` -> ``(x, y, z)``.
    """
    if value is None:
        return (1.0, 1.0, 1.0)
    if isinstance(value, (int, float)):
        s = float(value)
        return (s, s, s)
    seq = list(value)
    if len(seq) != 3:
        raise ValueError(f"inputs.rescale must be null, a scalar, or 3 numbers; got {value!r}")
    return tuple(float(x) for x in seq)  # type: ignore[return-value]


def _parse_weights(value: Any) -> str:
    """Parse ``inputs.weights`` into ``"NPIPE"`` or ``"PR3"`` (default ``"NPIPE"``).

    ``"PR4"`` is accepted as an alias for ``"NPIPE"`` (NPIPE is the PR4 release).
    """
    if value is None:
        return "NPIPE"
    w = str(value).strip().upper()
    if w == "PR4":
        return "NPIPE"
    if w not in ("NPIPE", "PR3"):
        raise ValueError(f"inputs.weights must be 'NPIPE', 'PR4', or 'PR3'; got {value!r}")
    return w


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
            rescale=_parse_rescale(data["inputs"].get("rescale")),
            weights=_parse_weights(data["inputs"].get("weights")),
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
            cache_interpolator=bool(data.get("convolution", {}).get("cache_interpolator", True)),
        ),
        map=MapConfig(
            nside=int(data["map"]["nside"]),
            nest=bool(map_cfg.get("nest", False)),
            use_cross_pol=bool(map_cfg.get("use_cross_pol", True)),
        ),
        output=OutputConfig(
            output_dir=str(output_cfg.get("output_dir", "outputs")),
            output_prefix=str(output_cfg.get("output_prefix", map_cfg.get("output_prefix", "pquick"))),
            extended_outputs=bool(output_cfg.get("extended_outputs", False)),
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
