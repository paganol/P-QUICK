# API reference

Public functions and classes per module. See the docstrings in the source for
full argument/return details; this page is the map.

## config

- `load_config(path) -> PipelineConfig` — read a YAML file into the typed config.
- `PipelineConfig` and sub-dataclasses: `InputsConfig`, `DetectorSelection`,
  `ResamplingConfig`, `ConvolutionConfig`, `MapConfig`, `OutputConfig`.

## io

- `load_pointing_npz(path) -> PointingData` — undersampled boresight quaternions.
- `load_horn_flag_npz(path, horn, n_samples=None) -> ndarray` — per-horn flags.
- `load_sky_alm(path) -> ndarray` — `[T, E, B]` sky `a_ℓm`.
- `infer_lmax_from_alm(alm) -> int`, `truncate_alm(alm, lmax_src, lmax_dst)`.
- `load_rimo_detectors(rimo_path) -> dict` — per-detector `phi_uv`/`theta_uv`/
  `psi_uv`/`psi_pol`/`ε` and the offset quaternion.
- `select_detectors(all_detectors, selection) -> list[str]`.
- `detector_to_beam_file(beams_dir, detector) -> Path`, `load_beam_alm(...)`.
- `build_polarized_beam_alm(...)` — scalar `b_ℓm` → spin-2 `[T, E, B]` beam,
  carrying the ellipse Dxx→Pxx by `psi_uv`. See
  [methodology](methodology.md#the-beam-scalar-blm--spin-2-t-e-b).
- `normalize_beam_alm(beam_alm, mode="unit_integral") -> ndarray`.

## quaternion

- `normalize_quaternion`, `quat_mul`, `quat_conj`, `quat_rotate_vec`,
  `frame_rotate_normalize` — quaternion algebra (scalar-last `x, y, z, w`).
- `slerp(q0, q1, t)`, `upsample_quaternions(...)` — interpolation.
- `bore_det_to_angles(...)`, `bore_det_to_ptg(...)`,
  `bore_det_to_ptg_masked(...)` — compose `q_bore ⊗ det_quat` and extract
  `(θ, φ, ψ)`; the masked variant reads only good-sample indices.
- `quaternion_to_thetaphipsi(q) -> (theta, phi, psi)`.

## pointing

- `PointingData`, `NativePointing`, `PointingInterpolator` — data/wrapper classes.
- `build_pointing_interpolator(point_us, coordinate_system) -> PointingInterpolator`.
- `PointingInterpolator.get_boresight_quaternions(start, count)` — frame-rotated
  boresight quaternions for a chunk (identity detector).
- `reconstruct_native_time(...)` — native-rate time grid.

## convolution

- `build_convolution_interpolator(sky_alm, beam_alm, lmax, mmax, ...) -> Interpolator`
  — build the `ducc0.totalconvolve` cube (pointing-independent, cacheable).
- `evaluate_convolution(interp, ptg_thetaphipsi) -> ndarray` — evaluate the cube
  at a pointing chunk; returns a 1-D float64 TOD.

## mapmaking

- `init_map_matrix(nside) -> ndarray` — zeroed `(npix, 3, 3)` accumulator.
- `accumulate_tqu_matrix(matrix, pix, psi, tod, det_weight, rho=1.0)` — serial
  single-matrix accumulation (response model `d = I + ρ(Q cos2ψ + U sin2ψ)`).
- `accumulate_tqu_local(local, pix, psi, tod, det_weight, rho=1.0)` — parallel
  per-thread `(nthreads, npix, 3, 3)` variant; sum over axis 0 after the loop.
- `add_hits(hits, pix)` — serial scatter of per-sample hit counts into a
  persistent buffer.
- `solve_tqu_from_matrix(matrix, cond_threshold=1e10) -> (t, q, u)` — per-pixel
  3×3 solve with condition-number masking.
- `accumulate_simple_iqu(...)` / `finalize_simple_iqu(acc)` — simpler
  binned-IQU path (numerator/denominator accumulators).

## utilities

- `detector_map_weight(detector, default=1.0) -> float` — qp_planck horn weight.
- `parse_mission_length(value) -> (od_start, od_end)`,
  `filter_pointing_files_by_mission_length(files, mission_length)`,
  `build_pointing_file_paths(prefix, od_start, od_end)`,
  `extract_od_from_pointing_filename(path) -> int`.
- `print_mpi_distribution(...)`, `suggest_tasks_per_node(...)`,
  `estimate_memory_per_rank_mb(nside, lmax=0, mmax=0)`,
  `resolve_nthreads(nthreads) -> int`.

## pipeline

- `run_pipeline(config) -> Path | None` — the orchestrator (uses `MPI.COMM_WORLD`; returns the output map path on rank 0).
- `main()` — CLI entry point (`pquick-run --config <yaml>`).
