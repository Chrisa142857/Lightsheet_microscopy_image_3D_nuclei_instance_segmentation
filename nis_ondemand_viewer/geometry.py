"""Pure-Python geometry for on-demand zoom cubes.

No third-party dependencies live here on purpose: this module holds the tricky
index math (resolution remapping, cube windowing, run-length offsets, tile
overlap) so it can be unit-tested without numpy/torch/nibabel installed.

Coordinate frames
-----------------
* **map**   - the low-res density map the frontend draws as the "main view"
              (isotropic ``map_res`` micron voxels, e.g. 25 um). Crosshair
              clicks arrive here, snapped to a ``(cuben, cuben, cuben_z)`` grid.
* **orig**  - the original imaging resolution (anisotropic ``orig_res`` =
              (z, x, y) micron, e.g. (4, .75, .75)). The cube we emit lives here
              so the viewer reflects the true acquisition resolution.
* **nis**   - the resolution the NIS C++ tool hard-codes for segmentation
              (``nis_res`` = (z, x, y) micron, e.g. (2.5, .75, .75)). Per-stack
              voxel/centre coordinates are stored in this frame.

All public functions take/return plain ints or small dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple


@dataclass(frozen=True)
class ResolutionConfig:
    """Voxel spacings (micron) and the zoom grid step.

    ``*_res`` are ``(z, x, y)`` tuples to match the NIS C++ ``(z, x, y)`` sort.
    """

    map_res: float = 25.0
    orig_res: Tuple[float, float, float] = (4.0, 0.75, 0.75)
    nis_res: Tuple[float, float, float] = (2.5, 0.75, 0.75)
    cuben: int = 5          # grid step in map voxels along x and y
    cuben_z: int = 3        # grid step in map voxels along z

    @property
    def z_nis_to_orig(self) -> float:
        """Scale a NIS z-index onto the original-resolution z-axis (``2.5/4``)."""
        return self.nis_res[0] / self.orig_res[0]

    def cube_size_orig(self) -> Tuple[int, int, int]:
        """Cube extent in original-resolution voxels, as ``(x, y, z)``.

        Mirrors ``cube_size = [int(map*cuben/ox), int(map*cuben/oy),
        int(map*cuben_z/oz)]`` from the offline ``prepare_nis_to_aws.py``.
        """
        oz, ox, oy = self.orig_res
        return (
            int(self.map_res * self.cuben / ox),
            int(self.map_res * self.cuben / oy),
            int(self.map_res * self.cuben_z / oz),
        )


def snap_to_grid(map_x: int, map_y: int, map_z: int, cfg: ResolutionConfig) -> Tuple[int, int, int]:
    """Snap a map-frame click to the zoom grid (floor to the cube origin).

    Matches the frontend's ``floor(coord/cubesz)*cubesz`` so the same click maps
    to the same cube regardless of which side issues the request.
    """
    return (
        (map_x // cfg.cuben) * cfg.cuben,
        (map_y // cfg.cuben) * cfg.cuben,
        (map_z // cfg.cuben_z) * cfg.cuben_z,
    )


def map_to_orig(map_x: int, map_y: int, map_z: int, cfg: ResolutionConfig) -> Tuple[int, int, int]:
    """Map-voxel index -> original-resolution voxel index ``(x, y, z)``.

    ``orig = map_index * map_res / orig_res`` (i.e. ``x / (ox/map)`` in the
    offline script).
    """
    oz, ox, oy = cfg.orig_res
    return (
        int(map_x * cfg.map_res / ox),
        int(map_y * cfg.map_res / oy),
        int(map_z * cfg.map_res / oz),
    )


@dataclass(frozen=True)
class CubeWindow:
    """A half-open axis-aligned box in original-resolution voxels (global)."""

    x1: int
    x2: int
    y1: int
    y2: int
    z1: int
    z2: int

    @property
    def shape_xyz(self) -> Tuple[int, int, int]:
        return (self.x2 - self.x1, self.y2 - self.y1, self.z2 - self.z1)


def cube_window(orig_x: int, orig_y: int, orig_z: int, cfg: ResolutionConfig) -> CubeWindow:
    """Centre a cube of ``cube_size_orig`` on an original-resolution point."""
    sx, sy, sz = cfg.cube_size_orig()
    x1 = orig_x - sx // 2
    y1 = orig_y - sy // 2
    z1 = orig_z - sz // 2
    return CubeWindow(x1, x1 + sx, y1, y1 + sy, z1, z1 + sz)


# --------------------------------------------------------------------------- #
# Run-length decoding of per-cell voxel membership.
#
# The NIS C++ tool stores, per stack:
#   coordinate    N_voxel x 3   (all cells' voxels concatenated, (z, x, y))
#   center        N_cell  x 3
#   instance_label N_cell x 1
#   nis_volume    N_cell  x 1   (voxel count per cell == the run lengths)
#
# Cell ``i`` owns ``coordinate[start_i:end_i]`` with
#   start_i = cumsum(nis_volume)[i-1]   (0 for i == 0)
#   end_i   = cumsum(nis_volume)[i]
# --------------------------------------------------------------------------- #
def rle_bounds(volumes: Sequence[int]) -> List[Tuple[int, int]]:
    """Return ``[(start_i, end_i), ...]`` for every cell from its voxel counts."""
    bounds: List[Tuple[int, int]] = []
    start = 0
    for v in volumes:
        end = start + int(v)
        bounds.append((start, end))
        start = end
    return bounds


# --------------------------------------------------------------------------- #
# Tile geometry. A tile sits at ``(x_off, y_off)`` in the stitched/original
# frame with pixel size ``(width, height)`` and ``depth`` planes.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LocalWindow:
    """A cube window expressed in a single tile's local pixel coordinates.

    ``z1/z2`` stay in global original-resolution plane indices (the raw TIFFs are
    one file per global plane); ``x*/y*`` are clamped into the tile.
    """

    x1: int
    x2: int
    y1: int
    y2: int
    z1: int
    z2: int

    @property
    def shape_xyz(self) -> Tuple[int, int, int]:
        return (self.x2 - self.x1, self.y2 - self.y1, self.z2 - self.z1)


def tile_local_window(
    window: CubeWindow,
    x_off: int,
    y_off: int,
    width: int,
    height: int,
    depth: int,
) -> LocalWindow | None:
    """Clip ``window`` into a tile; ``None`` if they do not overlap in x/y.

    Reproduces the offline check (centre inside the tile, local coords clamped to
    ``>= 0`` and ``z2`` capped at tile depth) but generalised to a true overlap
    test so seam-straddling cubes are handled per tile.
    """
    lx1 = max(window.x1 - x_off, 0)
    ly1 = max(window.y1 - y_off, 0)
    lx2 = min(window.x2 - x_off, width)
    ly2 = min(window.y2 - y_off, height)
    if lx1 >= lx2 or ly1 >= ly2:
        return None
    z1 = max(window.z1, 0)
    z2 = min(window.z2, depth)
    if z1 >= z2:
        return None
    return LocalWindow(lx1, lx2, ly1, ly2, z1, z2)
