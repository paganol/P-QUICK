# Methodology

P-QUICK measures the effect of realistic, asymmetric beams on Planck-like maps
by **direct time-domain convolution** along the real scan, rather than through a
precomputed beam window. It complements the QuickPol beam-window formalism
(Hivon et al. 2017): QuickPol predicts the effective `B_ℓ` and T→P / P→P leakage
analytically from beam spin-moments, while P-QUICK produces the actual scanned,
map-made T/Q/U that those windows are meant to describe — so the two can be
compared against each other.

## Pipeline in one line

For each detector, the sky `a_ℓm` is convolved with the beam `a_ℓm` at every
pointing sample to make a timeline `d(t)`, and the timelines are binned into
per-pixel polarised normal equations and solved for T/Q/U.

```
sky a_ℓm  ⊗  beam a_ℓm   →  d(t) per detector   →  Σ per-pixel (AᵀA, Aᵀd)   →  solve T/Q/U
        (ducc0 totalconvolve)        (scan pointing)         (map-making)
```

## The detector response model

A polarisation-sensitive detector pointing at pixel `p` with polarisation angle
`ψ` measures

```
d = I + ρ (Q cos2ψ + U sin2ψ),   ρ = (1 − ε)/(1 + ε)
```

where `ε` is the cross-polar leakage from the RIMO. `ρ = 1` is an ideal detector
(`map.use_cross_pol: false`); `ρ` from the RIMO matches qp_planck's
`rhohit: IMO`. Note `ρ` does **not** affect the recovered temperature — the I–I
element of the normal matrix is `ρ`-independent — only Q/U. This model is
implemented in [`accumulate_tqu_matrix`](api-reference.md#mapmaking).

The per-pixel normal equations stack the weighted pointing matrix `AᵀN⁻¹A`
(upper 2×2 block, `A = [1, ρcos2ψ, ρsin2ψ]`) and the weighted RHS `AᵀN⁻¹d`,
then solve the 3×3 system per pixel. Pixels with too few crossing angles are
rejected by a condition-number cut (`solve_tqu_from_matrix`).

## Frame conventions

The beam shape and the polarisation angle live in two related frames. Getting
this right is the subtle part of the pipeline; the conventions below are
validated and load-bearing.

| Frame | Meaning |
|-------|---------|
| **Dxx** | Beam frame — orientation in which the beam `a_ℓm` are stored. |
| **Pxx** | Polarisation frame — the scan-relative frame qp_planck uses. |
| `psi_uv` | The Dxx → Pxx rotation (per detector, from the RIMO). |
| `psi_pol` | The Pxx → polarisation-axis angle (per detector, from the RIMO). |

Two rules follow:

1. **`psi_uv` is applied to the beam, not the convolution orientation.** The
   beam ellipse is carried from Dxx into Pxx by multiplying `b_ℓm` by
   `exp(i m · psi_uv)` (see [`build_polarized_beam_alm`](api-reference.md#io)).
   The convolution then runs at the Pxx angle directly. This makes a horn's two
   PSB arms co-orient on the sky instead of sitting 90° apart — otherwise the
   orthogonal arms cancel the channel's beam ellipticity and the temperature
   window comes out wrong.

2. **`psi_pol` enters the map-making angle, not the beam shape.** The
   polarisation axis stays at the Pxx x-axis for the beam; `psi_pol` is added to
   the pointing `ψ` used in the response model above.

## The beam: scalar blm → spin-2 [T, E, B]

Planck beams are stored as a **scalar** intensity `b_ℓm`. P-QUICK synthesises a
full spin-2 `[T, E, B]` beam from it (`build_polarized_beam_alm`): T is the
intensity beam; E/B carry the polarised response at `m = 2` with the Challinor
spin-2 factor, with `B = i·E` for an ideal co-polar beam (no circular
polarisation). This is an exact spin-2 construction (`d^ℓ_{m,2}` radial
functions), which differs from a scalar `ndb = 1` beam by the irreducible
spin-2-vs-spin-0 term — the residual seen when comparing TB/BB/EB against a
qp_planck scalar-blm run is exactly this construction difference, not a bug.

## What is and isn't approximated

- **Pointing** is reconstructed at the native sample rate by SLERP-interpolating
  the undersampled boresight quaternions (`ducc0.pointingprovider`), then
  composing with each detector's fixed RIMO offset. See
  [architecture](architecture.md#pointing).
- **`center_pointing`** (optional) snaps each sample to its HEALPix pixel centre
  before convolution, suppressing subpixel pointing scatter at the cost of one
  representative direction per pixel.
- **`rescale`** lets you isolate input components (e.g. `[0, 1, 0]` = E-only) to
  study T→P / E→B leakage in isolation.

## References

- Hivon, Mottet & Ponthieu (2017), *QuickPol: Fast calculation of effective beam
  matrices for CMB polarization*, A&A 598, A25.
- ducc0 `totalconvolve` / `pointingprovider` — the convolution and SLERP engines.
