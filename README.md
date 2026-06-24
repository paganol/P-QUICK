# P-QUICK
Planck QUick Integrated Convolution Kit

P-QUICK is an MPI-ready Python pipeline that:

1. Reads undersampled Planck boresight quaternion NPZ files.
2. Reconstructs native-rate pointing using `ducc0.pointingprovider.PointingProvider`.
3. Loads Planck RIMO metadata and detector beam ALMs.
4. Convolves sky ALMs with beams using `ducc0.totalconvolve`.
5. Accumulates the convolved detector timelines into per-pixel polarised normal equations and solves them for condition-masked HEALPix T/Q/U maps.


## Install

```bash
python -m pip install -e .
```

## Configure

Use the template in `configs/default.yaml`.

Important fields:

1. `inputs.sky_alm`: sky ALM input (`.fits`, `.npy`, `.npz`).
2. `inputs.beams_dir`: directory with per-detector beam ALM FITS files.
3. `inputs.rimo_file`: path to the RIMO FITS file.
4. `inputs.pointings`: common path prefix for pointing NPZ files (e.g. `inputs/pointings/pointing_`); files are resolved as `{pointings}od_{od:04d}.npz`.
5. `inputs.mission_length`: OD selector (`full`, `survey 1` ... `survey 5`, or explicit `91-99`). Defaults to `full`.
6. `inputs.flags`: optional prefix for per-OD flag NPZ files, resolved as `{flags}{freq:03d}ghz_od_{od:04d}.npz`; when unset or a file is missing, all samples are treated as good.
7. `inputs.bad_rings_file`: optional TOAST/NPIPE-style bad-ring interval text file (`<det_or_ALL> <tstart_s> <tstop_s>` rows), applied on top of the flags.
8. `inputs.rescale`: optional per-component multipliers for the input sky `(almT, almE, almB)` — `null` = `(1, 1, 1)`, a scalar `s` = `(s, s, s)`, or `[x, y, z]`. Useful for isolating components (e.g. `[0, 1, 0]` = E-only).
9. `detector_selection`: choose either a channel/detset alias or an explicit detector list.
10. `convolution.lmax` / `convolution.mmax`: harmonic limits.
11. `convolution.cache_interpolator`: `true` (default) builds each detector's `ducc0` convolution cube once and reuses it across every OD/chunk on a rank (the cube depends only on the sky, beam, `lmax`, `mmax` and `epsilon` — not on the pointing), removing the dominant per-OD rebuild. It keeps one cube resident per detector (~0.4 GB at lmax=1024/mmax=6, ~1–2 GB at lmax=2048); set `false` to rebuild per OD for lower memory.
12. `nthreads`: thread count for `ducc0` and the numba-parallel resampling / map-making kernels (`0` = all available cores). Trade off against the number of MPI ranks per node.
13. `map.nside`: output HEALPix map resolution.
14. `map.use_cross_pol`: `true` (default) weights the map-making polarisation by the per-detector `rho = (1-eps)/(1+eps)` from the RIMO (= qp_planck `rhohit: IMO`); `false` assumes ideal detectors (`rho = 1`, qp_planck `rhohit: Ideal`). Temperature is unaffected.
15. `resampling.coordinate_system`: pointing frame (`ecliptic` or `galactic`).

The per-detector `psi_uv` is always removed from the beam-shape convolution orientation (kept only in the map-making polarization angle), so a horn's two PSB arms convolve their near-identical beams co-oriented on the sky instead of 90° apart — the scan-relative frame qp_planck uses. Without it the orthogonal arms cancel the channel beam ellipticity and the temperature window is wrong.

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
2. `inputs.mission_length` can restrict ODs after glob discovery using:
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
6. `src/pquick/mapmaking.py`: polarised normal-equation (T/Q/U) accumulation and solve.
7. `src/pquick/utilities.py`: detector weights and mission-length helpers.
8. `src/pquick/pipeline.py`: MPI-aware orchestrator.

## Tests

```bash
pytest -q
```
