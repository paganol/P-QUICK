# User guide

## Install

```bash
python -m pip install -e .
```

This installs the `pquick-run` console script.

## Inputs

| Input | Config key | Notes |
|-------|-----------|-------|
| Sky `a_ℓm` | `inputs.sky_alm` | `.fits`, `.npy`, or `.npz`; `[T, E, B]`. |
| Beam `a_ℓm` | `inputs.beams_dir` | One scalar-beam FITS per detector; resolved by detector name. |
| Focal plane | `inputs.data_version` | Built-in per-release tables provide `phi_uv`/`theta_uv`/`psi_uv`/`psi_pol`, `ε` and map weights; no RIMO FITS is read. |
| Pointing | `inputs.pointings` | Per-OD NPZ, resolved as `{pointings}od_{OD:04d}.npz`. |
| Flags | `inputs.flags` | Optional per-OD NPZ, `{flags}{freq:03d}ghz_od_{OD:04d}.npz`. |
| Bad rings | `inputs.bad_rings_file` | Optional TOAST/NPIPE interval text file. |

### Preparing pointing + flag NPZ

`scripts/extract_planck_pointing_and_flags.py` builds the pointing and flag NPZ
files from Planck pointing/attitude FITS. The pointing NPZ stores undersampled
**boresight** quaternions (the satellite attitude, `QUATERNION_X/Y/Z/S`), the
native sampling rate, and the undersampling indices; the pipeline reconstructs
the native rate by SLERP. Each pointing NPZ must contain:

- `t0_ns`, `qx`, `qy`, `qz`, `qs` (scalar-last quaternions)
- `sampling_rate_hz`, and the `idx_first`/`idx_last`/`idx_step` undersampling
  indices

## Configuration reference

Start from `configs/default.yaml`. Full key list:

### `inputs`
- `sky_alm`, `beams_dir` — required paths.
- `data_version` — Planck data release to emulate: `NPIPE` (default; `PR4` is an
  alias) or `PR3`. Selects the focal-plane geometry, the polarisation
  efficiency `ε`, and the detector map weights together, from built-in tables
  (`pquick.utilities.FOCAL_PLANE_NPIPE`/`FOCAL_PLANE_PR3` and the weight
  tables); the beams are common to both releases, so no beam or RIMO file
  changes. NPIPE tables come from the npipe-symmetrized RIMOs (= `R4.00`); PR3
  tables from `HFI_RIMO_R2.00` + `LFI_RIMO_R2.50`, with R2.00's orientation
  (stored there in `PSI_POL`) mapped into `psi_uv` and no pol-axis fine offset,
  matching the PR3 pipeline. See [methodology](methodology.md#detector-weighting).
- `mission_length` — OD selector: `full`, `survey 1`…`survey 5`, or an explicit
  range like `91-99`. Default `full`.
- `pointings` — prefix; pipeline appends `od_{OD:04d}.npz`.
- `flags` — prefix; appends `{freq:03d}ghz_od_{OD:04d}.npz`. `null` (or a missing
  file) ⇒ all samples treated as good.
- `bad_rings_file` — optional bad-ring interval file, applied on top of flags.
- `rescale` — per-component `(almT, almE, almB)` multipliers. `null` = `(1,1,1)`,
  scalar `s` = `(s,s,s)`, or `[x,y,z]`. E.g. `[0,1,0]` = E-only.

### `detector_selection`
- `channel` — keep detectors whose name starts with this prefix (e.g. `"100"`), or
  a detset alias (`100ds1`, `143dsA`, `143swb`, `100ghz`, …; case-insensitive), **or**
- `detectors` — explicit allowlist. Specify only one of the two.

### `resampling`
- `coordinate_system` — `galactic` (default) or `ecliptic`.
- `center_pointing` — snap each sample to its HEALPix pixel centre (at
  `map.nside`) before convolution. Default `false`.

### `convolution`
- `lmax` — max multipole (sky and beam). Required.
- `mmax` — max beam azimuthal order *m* (`kmax`). Required.
- `epsilon` — ducc0 interpolation accuracy. Default `1e-5`.
- `chunks` — interpolation calls per OD (1 = whole OD at once). Default `1`.
- `beam_normalization` — `unit_integral` (default; constant sky stays constant)
  or `raw`.
- `cache_interpolator` — build each detector's convolution cube once and reuse it
  across all ODs/chunks (the cube is pointing-independent), removing the dominant
  per-OD rebuild. Default `true`. One cube is held resident **per detector, per
  rank** (~0.4 GB at lmax=1024, ~1–2 GB at lmax=2048), so cache RAM ≈ `(selected
  detectors) × cube` per rank (e.g. ~11–22 GB for a full 143 GHz channel at
  lmax=2048), on top of the map-making accumulator. Set `false` to rebuild per OD
  for lower memory, slower runs.

### `map`
- `nside` — output HEALPix resolution. Required.
- `nest` — NESTED ordering instead of RING. Default `false`.
- `use_cross_pol` — weight Q/U by RIMO `ρ = (1−ε)/(1+ε)` (`true`, = qp_planck
  `rhohit: IMO`) or use the ideal PSB flag (`false`, = qp_planck `rhohit: Ideal`:
  `ρ = 1` for PSBs, `ρ = 0` for unpolarised SWBs). T is unaffected.

### top level
- `nthreads` — threads for ducc0 + numba. `0` = all available cores. Map-making
  memory scales with `nthreads` (per-thread accumulator, ~3.6 GB/thread at
  nside 2048); lower it if a rank is RAM-bound.
- `verbose` — detailed per-OD progress logging.
- `output.output_dir` / `output.output_prefix` — where products go and their
  filename stem.
- `output.extended_outputs` — also write the `_hits`/`_wpol`/`_nobs00` diagnostic
  maps. Default `false` (only `_iqu`).

## Run

```bash
# serial
pquick-run --config configs/default.yaml

# MPI (ODs distributed across ranks)
mpirun -n 4 pquick-run --config configs/default.yaml
```

## Outputs

Written under `output.output_dir` with stem `output.output_prefix`:

- `<prefix>_iqu.fits` — the T/Q/U map (always written).

With `output.extended_outputs: true` (default `false`) the diagnostic maps are also written:

- `<prefix>_hits.fits` — per-pixel hit count.
- `<prefix>_wpol.fits`, `<prefix>_nobs00.fits` — polarisation weight / `AᵀA[0,0]`
  diagnostics.

## Reading the timing summary

With `verbose`, each OD prints a `[OD timing]` line and the run ends with a
`[Timing summary]`: `setup / resamp / conv / macc / flag / prep / pix / od_other
/ reduce / solve / write / unaccounted / wall`. `conv` is usually the largest;
`pix` only appears when `center_pointing` is on. Tune `epsilon`, `mmax`, or
`nthreads` if a bucket dominates.
