# Architecture

## Modules

| Module | Responsibility |
|--------|----------------|
| `config.py` | Typed YAML config (`PipelineConfig` and sub-dataclasses), `load_config`. |
| `io.py` | Load pointing NPZ, sky `a_ℓm`, RIMO, beam `a_ℓm`; build the spin-2 beam. |
| `quaternion.py` | Quaternion math, SLERP, and the boresight⊗detector → (θ, φ, ψ) kernels. |
| `pointing.py` | Native-rate pointing reconstruction (`PointingInterpolator` over ducc0). |
| `convolution.py` | Thin `ducc0.totalconvolve` wrapper: build the cube, evaluate at pointings. |
| `mapmaking.py` | Per-pixel polarised normal-equation accumulation and the 3×3 solve. |
| `utilities.py` | Detector weight sets (`NPIPE_DETECTOR_WEIGHTS` / `PR3_DETECTOR_WEIGHTS`), detset aliases, mission-length parsing, MPI/thread/memory helpers. |
| `pipeline.py` | MPI-aware orchestrator tying it all together; CLI entry `main`. |

## Data flow

```
config (YAML)
   │  load_config
   ▼
PipelineConfig ──► detector selection (io.select_detectors)
   │                     │
   │                     ▼
   │              per-detector: RIMO offset quat + spin-2 beam a_ℓm (io)
   │                     │                         convolution cube (convolution, cached)
   ▼                     ▼
OD list (utilities) ──► [OD loop, distributed across MPI ranks]
                            │
                            ├─ load pointing NPZ (io)
                            ├─ build PointingInterpolator (pointing)  ── SLERP upsample
                            ├─ load flags / bad rings
                            ├─ skip OD if no detector has a good sample
                            └─ [chunk loop]
                                  ├─ boresight quaternions for the chunk (once)
                                  └─ [detector loop]
                                        ├─ mask flagged/ring-bad samples
                                        ├─ q_bore ⊗ det_quat → (θ, φ, ψ)   (quaternion)
                                        ├─ (optional) snap to pixel centres
                                        ├─ evaluate convolution → TOD       (convolution)
                                        └─ accumulate into normal eqns      (mapmaking)
                            ▼
                     per-rank (npix,3,3) matrix + hits
                            │  MPI reduce
                            ▼
                     solve T/Q/U (mapmaking) ──► write FITS (rank 0)
```

## Key design points

- **Convolution cube is cached per detector** (`cache_interpolator: true`). The
  cube depends only on `(sky, beam, lmax, mmax, epsilon)` — not pointing — so it
  is built once per detector per rank and reused across every OD/chunk,
  removing the dominant per-OD rebuild. Memory cost: ~0.4 GB/detector at
  lmax 1024.

- **Boresight is interpolated once per chunk**, detector-independent. Each
  detector's line of sight is the cheap composition `q_bore ⊗ det_quat`. The
  `*_masked` kernel variant reads only the good-sample indices to skip a copy.

- **Map-making uses a per-thread `(nthreads, npix, 3, 3)` accumulator** so the
  scatter parallelises without a lock; the slices are summed after the OD loop.
  On real (spatially local) scan data this beats a single serial-scatter matrix
  by ~4×, at `nthreads ×` the memory. Per-pixel hits use a serial scatter into a
  persistent buffer (no per-call `npix` allocation). See
  [`mapmaking`](api-reference.md#mapmaking).

- **Whole-OD skip**: before the chunk loop, if no detector has a single good
  sample (common flag | horn flag | bad-ring interval), the OD is skipped — no
  boresight interpolation, no convolution.

- **MPI**: ODs are distributed across ranks; each rank accumulates its own
  normal-equation matrix and hit map, reduced to rank 0 for the solve and write.

- **Detector selection is filtered to working detectors**: a selected detector
  with no entry in the chosen weight set (143-8, 545-3) or no beam file is skipped
  with a warning. The solve auto-switches to **temperature-only** when ≤ 2 PSBs are
  present (Q/U unconstrained), recovering I and leaving Q/U unseen.

## Pointing

Undersampled boresight quaternions (scalar-last `x, y, z, w`) and a native
sampling rate come in via the pointing NPZ. `build_pointing_interpolator` wraps
a `ducc0.pointingprovider.PointingProvider` that SLERP-interpolates to the
native rate; `get_boresight_quaternions` evaluates a chunk with an identity
detector quaternion and applies the fixed frame rotation (ecliptic→galactic if
requested). The detector line of sight is then `q_bore ⊗ det_quat`, with
`det_quat` built from the RIMO `(phi_uv, theta_uv, psi_uv, psi_pol)`.

See [methodology](methodology.md#frame-conventions) for what the angles mean.

## Tests

`pytest -q` — unit tests cover quaternion math, the convolution wrapper, the
spin-2 beam construction (against an analytic Gaussian), and the map-making
accumulation/solve.
