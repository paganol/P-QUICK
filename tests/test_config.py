from pathlib import Path

from pquick.config import load_config


def test_load_config_rejects_channel_and_detectors_together(tmp_path: Path):
    cfg_path = tmp_path / "invalid.yaml"
    cfg_path.write_text(
        """
inputs:
  sky_alm: inputs/sky/alms_cmb0000.fits
  beams_dir: inputs/beams
  mission_length: 91-99
  pointings: inputs/pointings/pointing_
  flags: inputs/flags/flags_

detector_selection:
  channel: 100ghz
  detectors:
    - 100-1a

resampling:
  angular_eps: 1.0e-10
  coordinate_system: galactic

convolution:
  lmax: 16
  mmax: 6

map:
  nside: 8

output:
  output_dir: outputs

verbose: false
""".strip(),
        encoding="utf-8",
    )

    try:
        load_config(cfg_path)
    except ValueError as exc:
        assert "only one" in str(exc)
    else:
        raise AssertionError("Expected ValueError when both channel and detectors are set")


def test_load_config_parses_resampling_centering_options(tmp_path: Path):
    cfg_path = tmp_path / "centering.yaml"
    cfg_path.write_text(
        """
inputs:
  sky_alm: inputs/sky/alms_cmb0000.fits
  beams_dir: inputs/beams

detector_selection:
  channel: null
  detectors: []

resampling:
  coordinate_system: galactic
  center_pointing: true

convolution:
  lmax: 16
  mmax: 6

map:
  nside: 512

output:
  output_dir: outputs
""".strip(),
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)
    assert cfg.resampling.center_pointing is True


def test_load_config_reads_output_prefix_from_output_section(tmp_path: Path):
    cfg_path = tmp_path / "output_prefix.yaml"
    cfg_path.write_text(
        """
inputs:
  sky_alm: inputs/sky/alms_cmb0000.fits
  beams_dir: inputs/beams

detector_selection:
  channel: null
  detectors: []

convolution:
  lmax: 16
  mmax: 6

map:
  nside: 512

output:
  output_dir: outputs
  output_prefix: custom_prefix
""".strip(),
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)
    assert cfg.output.output_prefix == "custom_prefix"


def test_load_config_extended_outputs(tmp_path: Path):
    base = """
inputs:
  sky_alm: inputs/sky/alms_cmb0000.fits
  beams_dir: inputs/beams

detector_selection:
  channel: null
  detectors: []

convolution:
  lmax: 16
  mmax: 6

map:
  nside: 512

output:
  output_dir: outputs
"""
    default_cfg = tmp_path / "default.yaml"
    default_cfg.write_text(base.strip(), encoding="utf-8")
    assert load_config(default_cfg).output.extended_outputs is False

    on_cfg = tmp_path / "on.yaml"
    on_cfg.write_text((base + "  extended_outputs: true\n").strip(), encoding="utf-8")
    assert load_config(on_cfg).output.extended_outputs is True


def test_load_config_defaults_beam_normalization_to_unit_integral(tmp_path: Path):
    cfg_path = tmp_path / "beam_norm.yaml"
    cfg_path.write_text(
        """
inputs:
  sky_alm: inputs/sky/alms_cmb0000.fits
  beams_dir: inputs/beams

detector_selection:
  channel: null
  detectors: []

convolution:
  lmax: 16
  mmax: 6

map:
  nside: 512

output:
  output_dir: outputs
""".strip(),
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)
    assert cfg.convolution.beam_normalization == "unit_integral"
