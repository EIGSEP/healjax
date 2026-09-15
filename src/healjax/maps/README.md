# healjax.maps

Object layer for HEALPix maps — the JAX-accelerated, aipy-free replacement
for the map containers other EIGSEP packages used to get from `aipy`/
`healpy` directly.

## Layout

| Module | Purpose |
|---|---|
| `hpm.py` | `HPM` — the class most callers want: a HEALPix map container with JAX-accelerated pixel lookup and interpolation (via `healjax.interp`), no `aipy` dependency. |
| `base.py` | `HealpixBase` — pixel-scheme bookkeeping for the HEALPix sphere; the foundation `HPM`/`HealpixMap` build on. |
| `healpix_map.py` | `HealpixMap` — a data array living on a HEALPix sphere. |
| `alm.py` | Spherical-harmonic coefficients (`Alm`) and the `sph_fit`/`x_to_alm` fitting entry points. |
| `_optional.py` | Lazy accessors for the optional third-party dependencies this package uses (import-on-first-use, so `healjax.maps` doesn't hard-require everything). |

`__init__.py` re-exports `HealpixBase`, `Alm`, `HealpixMap`, and `HPM` as the
public surface.

## Recent changes

- 2026-09-15 (`software-engineer`): added this file (README-convention
  retrofit, fleet-wide consolidation pass).
