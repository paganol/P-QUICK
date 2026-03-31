# P-QUICK
Planck QUick Integrated Convolution Kit

P-QUICK is an MPI-ready Python pipeline that:

1. Reads undersampled Planck boresight quaternion NPZ files.
2. Reconstructs native-rate pointing using quaternion SLERP.
3. Loads Planck RIMO metadata and detector beam ALMs.
4. Convolves sky ALMs with beams using `ducc0.totalconvolve`.
5. Bins convolved detector timelines into simple weighted HEALPix I/Q/U maps.

## Current status

Initial end-to-end scaffold is implemented:

1. NPZ pointing ingestion and validation.
2. Native-rate quaternion reconstruction.
3. Detector-specific pointing from RIMO quaternions.
4. ducc0 interpolation wrapper for timeline convolution.
5. Detector selection from YAML with explicit lists and regex.
6. Map-level detector weighting using qp_planck utility weights.
7. Simple map binning (no destriping in v1).

## Install

```bash
python -m pip install -e .
```

## Configure

Use the template in `configs/default.yaml`.

Important fields:

1. `inputs.sky_alm`: sky ALM input (`.fits`, `.npy`, `.npz`).
2. `inputs.pointing.npz_glob`: glob to your undersampled NPZ files.
3. `detector_selection`: choose explicit detector lists and/or regex.
4. `convolution.lmax` / `convolution.kmax`: harmonic limits.
5. `map.nside`: output HEALPix map resolution.

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

## Input assumptions for pointing NPZ

Each NPZ must contain:

1. `time` (undersampled OBT timestamps)
2. `qx`, `qy`, `qz`, `qs` (undersampled quaternions, scalar last)
3. `flag_ext1`, `flag_ext3`
4. `sampling_rate_hz` (native sampling rate)

If flags are not at native length, nearest-neighbor expansion is used.

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
7. `src/pquick/pipeline.py`: MPI-aware orchestrator.

## Tests

```bash
pytest -q
```
