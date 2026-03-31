from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DetectorSelection:
    channel: str | None = None
    detectors: list[str] = field(default_factory=list)
    include_regex: list[str] = field(default_factory=list)
    exclude_regex: list[str] = field(default_factory=list)


@dataclass
class PointingConfig:
    npz_glob: str


@dataclass
class ResamplingConfig:
    angular_eps: float = 1e-10


@dataclass
class ConvolutionConfig:
    lmax: int
    kmax: int
    nthreads: int = 0
    separate: bool = False
    epsilon: float = 1e-5


@dataclass
class MapConfig:
    nside: int
    nest: bool = False
    output_prefix: str = "pquick"


@dataclass
class InputsConfig:
    sky_alm: str
    beams_dir: str
    rimo_files: list[str]
    pointing: PointingConfig


@dataclass
class OutputConfig:
    output_dir: str = "outputs"


@dataclass
class PipelineConfig:
    inputs: InputsConfig
    detector_selection: DetectorSelection
    resampling: ResamplingConfig
    convolution: ConvolutionConfig
    map: MapConfig
    output: OutputConfig


def _to_dataclass(data: dict[str, Any]) -> PipelineConfig:
    return PipelineConfig(
        inputs=InputsConfig(
            sky_alm=data["inputs"]["sky_alm"],
            beams_dir=data["inputs"]["beams_dir"],
            rimo_files=list(data["inputs"]["rimo_files"]),
            pointing=PointingConfig(npz_glob=data["inputs"]["pointing"]["npz_glob"]),
        ),
        detector_selection=DetectorSelection(
            channel=data.get("detector_selection", {}).get("channel"),
            detectors=list(data.get("detector_selection", {}).get("detectors", [])),
            include_regex=list(data.get("detector_selection", {}).get("include_regex", [])),
            exclude_regex=list(data.get("detector_selection", {}).get("exclude_regex", [])),
        ),
        resampling=ResamplingConfig(
            angular_eps=float(data.get("resampling", {}).get("angular_eps", 1e-10))
        ),
        convolution=ConvolutionConfig(
            lmax=int(data["convolution"]["lmax"]),
            kmax=int(data["convolution"]["kmax"]),
            nthreads=int(data.get("convolution", {}).get("nthreads", 0)),
            separate=bool(data.get("convolution", {}).get("separate", False)),
            epsilon=float(data.get("convolution", {}).get("epsilon", 1e-5)),
        ),
        map=MapConfig(
            nside=int(data["map"]["nside"]),
            nest=bool(data.get("map", {}).get("nest", False)),
            output_prefix=str(data.get("map", {}).get("output_prefix", "pquick")),
        ),
        output=OutputConfig(output_dir=str(data.get("output", {}).get("output_dir", "outputs"))),
    )


def load_config(path: str | Path) -> PipelineConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a YAML mapping")
    return _to_dataclass(raw)
