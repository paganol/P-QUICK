# P-QUICK documentation

Planck QUick Integrated Convolution Kit — an MPI-ready pipeline that convolves a
sky with per-detector beams along the Planck scan and bins the result into
condition-masked HEALPix T/Q/U maps.

| Doc | For |
|-----|-----|
| [Methodology](methodology.md) | The physics: beam-window context, frame conventions (Dxx/Pxx, `psi_uv`/`psi_pol`), the spin-2 beam, the map-making response model. |
| [Architecture](architecture.md) | Module-by-module design and the pointing → convolution → map-making data flow. For working on the code. |
| [User guide](user-guide.md) | Installing, preparing inputs, the full config reference, running (serial/MPI), and outputs. |
| [API reference](api-reference.md) | Public functions and classes per module. |

Start with the [README](../README.md) for a one-page overview, then the
[user guide](user-guide.md) to run it.
