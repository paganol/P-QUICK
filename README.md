# P-QUICK
Planck QUick Integrated Convolution Kit

P-QUICK is an MPI-ready Python pipeline that:

1. Reads undersampled Planck boresight quaternion NPZ files.
2. Reconstructs native-rate pointing using `ducc0.pointingprovider.PointingProvider`.
3. Loads Planck RIMO metadata and detector beam ALMs.
4. Convolves sky ALMs with beams using `ducc0.totalconvolve`.
5. Bins convolved detector timelines into simple weighted HEALPix I/Q/U maps.


## Install

```bash
python -m pip install -e .
```

## Configure

Use the template in `configs/default.yaml`.

Important fields:

1. `inputs.sky_alm`: sky ALM input (`.fits`, `.npy`, `.npz`).
2. `inputs.rimo_file`: path to the RIMO FITS file.
3. `inputs.pointing.input_root`: common path prefix for pointing NPZ files (e.g. `inputs/pointings/processed_od_`); files are resolved as `{input_root}{od:04d}.npz`.
4. `inputs.pointing.mission_length`: OD selector (`full`, `survey 1` ... `survey 5`, or explicit `91-99`). Defaults to `full`.
4. `detector_selection`: choose either a channel/detset alias or an explicit detector list.
5. `convolution.lmax` / `convolution.mmax`: harmonic limits.

The per-detector `psi_uv` is always removed from the beam-shape convolution orientation (kept only in the map-making polarization angle), so a horn's two PSB arms convolve their near-identical beams co-oriented on the sky instead of 90° apart — the scan-relative frame qp_planck uses. Without it the orthogonal arms cancel the channel beam ellipticity and the temperature window is wrong.
7. `map.nside`: output HEALPix map resolution.
8. `map.use_cross_pol`: `true` (default) weights the map-making polarisation by the per-detector `rho = (1-eps)/(1+eps)` from the RIMO (= qp_planck `rhohit: IMO`); `false` assumes ideal detectors (`rho = 1`, qp_planck `rhohit: Ideal`). Temperature is unaffected.
9. `resampling.coordinate_system`: pointing frame (`ecliptic` or `galactic`).

## Run

Serial:

```bash
pquick-run --config configs/default.yaml
```

MPI:

```bash
mpirun -n 4 pquick-run --config configs/default.yaml
```

Outputs are written under `output.output_dir` as:

1. `<prefix>_iqu.fits`
2. `<prefix>_hits.fits`
3. `<prefix>_wpol.fits`
4. `<prefix>_nobs00.fits`

## Input assumptions for pointing NPZ

Each NPZ must contain:

1. `time` (undersampled OBT timestamps)
2. `qx`, `qy`, `qz`, `qs` (undersampled quaternions, scalar last)
3. `flag` (native-rate quality flag array)
4. `sampling_rate_hz` (native sampling rate)

Optional field:

1. `original_indices` (indices of undersampled quaternions in the original native-rate stream)
2. `inputs.pointing.mission_length` can restrict ODs after glob discovery using:
	- `full` -> OD91 to OD974
	- `survey 1` -> OD91 to OD270
	- `survey 2` -> OD271 to OD456
	- `survey 3` -> OD457 to OD636
	- `survey 4` -> OD637 to OD807
	- `survey 5` -> OD808 to OD974
	- explicit range string, e.g. `91-99`

Notes:

1. In the new single-flag schema, `flag != 0` marks bad native-rate samples.
2. Native-rate boresight interpolation is performed by `ducc0.pointingprovider.PointingProvider`.
3. Native pointing is defined in the ecliptic frame.
4. Supported output pointing frames are `ecliptic` and `galactic`.

## Detector weighting

Map accumulation uses the detector weights defined in qp_planck utilities.

1. HFI polarized arms map to horn weights (example: 100-1a and 100-1b use 100-1).
2. LFI M/S arms map to horn weights (example: LFI27M and LFI27S use LFI27).
3. Unknown detectors fall back to weight 1.0.

## Repository layout

1. `src/pquick/config.py`: typed YAML config loading.
2. `src/pquick/io.py`: pointing, ALM, RIMO, beam I/O.
3. `src/pquick/quaternion.py`: quaternion math and SLERP.
4. `src/pquick/pointing.py`: native-rate reconstruction.
5. `src/pquick/convolution.py`: ducc0 wrapper.
6. `src/pquick/mapmaking.py`: simple I/Q/U binning.
7. `src/pquick/utilities.py`: detector weights and mission-length helpers.
8. `src/pquick/pipeline.py`: MPI-aware orchestrator.

## Tests

```bash
pytest -q
```
