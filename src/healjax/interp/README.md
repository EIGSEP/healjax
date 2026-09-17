# healjax.interp

JAX-accelerated HEALPix map interpolation. One module, `interpolation.py`,
exporting two entry points:

- `interpolate_map` — interpolate a HEALPix map at arbitrary sky positions.
- `rotate_interpolate_and_sum` — rotate a map (or set of maps) and
  accumulate an interpolated sum, the operation beam-mapping and
  sky-simulation code actually calls in the hot path.

## Recent changes

- 2026-09-15 (`software-engineer`): added this file (README-convention
  retrofit, fleet-wide consolidation pass).
