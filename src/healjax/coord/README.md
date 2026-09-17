# healjax.coord

Coordinate transforms for spherical astronomy, shared across the EIGSEP
packages (`eigsep_sim`, `eigsep_terrain`, and others use these rather than
each carrying their own copy — see `eigsep_sim`/`eigsep_terrain`'s
`feature/use-healjax-eigsep-base` history for the migration).

`transforms.py` is the only module: rotation matrices, coordinate-frame
conversions (topocentric ↔ equatorial ↔ RA/Dec, alt/az ↔ topocentric,
lat/long ↔ XYZ), and the general `angles_to_coord` entry point. See the
module docstring and individual function docstrings for conventions
(frame handedness, angle units).

## Recent changes

- 2026-09-15 (`software-engineer`): added this file (README-convention
  retrofit, fleet-wide consolidation pass).
