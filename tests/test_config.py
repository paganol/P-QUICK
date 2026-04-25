from pathlib import Path

from pquick.config import load_config


def test_load_config_rejects_channel_and_detectors_together(tmp_path: Path):
    cfg_path = tmp_path / "invalid.yaml"
    cfg_path.write_text(
        """
inputs:
  sky_alm: inputs/sky/alms_cmb0000.fits
  beams_dir: inputs/beams
  rimo_file: inputs/RIMOs/RIMO_HFI_npipe5v16_symmetrized.fits
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
