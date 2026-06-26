"""Build a raw-image cube on demand by cropping per-plane TIFFs.

For a tile-local window, open only the plane files in ``[z1, z2)`` and crop the
``[x1:x2, y1:y2]`` sub-window from each. Missing planes become zeros (matching
the offline behaviour of skipping unreadable planes). The result axis order is
``(z, x, y)`` to match :func:`nis_cube.build_nis_cube`.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Dict

import numpy as np

from .geometry import LocalWindow
from .tiff_reader import build_plane_index, read_region


def build_raw_cube(
    local: LocalWindow,
    tile_dir: str,
    channel: str,
    max_workers: int = 8,
) -> np.ndarray:
    """Return a ``(z, x, y)`` raw-image cube for one tile's local window."""
    sx, sy, sz = local.shape_xyz
    cube_zxy = np.zeros((sz, sx, sy), dtype=np.uint16)

    plane_index: Dict[int, str] = build_plane_index(tile_dir, channel)

    def fill(plane: int) -> None:
        path = plane_index.get(plane)
        if path is None:
            return
        region = read_region(path, local.x1, local.y1, local.x2, local.y2)
        if region is None:
            return
        # read_region returns (h=y, w=x); store as (x, y).
        region = np.asarray(region)
        if region.ndim == 3:
            region = region[..., 0]
        ry, rx = region.shape[:2]
        cube_zxy[plane - local.z1, : min(rx, sx), : min(ry, sy)] = region.T[: min(rx, sx), : min(ry, sy)]

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        list(ex.map(fill, range(local.z1, local.z2)))

    return cube_zxy
