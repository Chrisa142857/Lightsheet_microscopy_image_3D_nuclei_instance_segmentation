"""Build a NIS label cube on demand from per-stack torch tensors.

Implements the layout you specified, with no pre-materialised dense volumes:

* Each stack stores ``coordinate`` (N_voxel x 3, ``(z, x, y)``), ``center``
  (N_cell x 3), ``instance_label`` (N_cell x 1), ``nis_volume`` (N_cell x 1).
* Per-voxel label = ``instance_label`` run-length-expanded by ``nis_volume``
  (equivalently ``coordinate[start_i:end_i]`` for cell ``i`` via
  :func:`geometry.rle_bounds`).
* Global z = within-stack ``z + zmin``; original-resolution z =
  ``global_z * (nis_res_z / orig_res_z)``. x/y already share the NIS and
  original pixel size, so they pass through unchanged.

Only stacks whose z-extent overlaps the requested cube are touched.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch

from .geometry import LocalWindow, ResolutionConfig

# Default NIS C++ filenames. ``instance_center`` is the discovery key; the other
# tensors are the same name with the field swapped (matching prepare_nis_to_aws).
_CENTER_KEY = "instance_center"
_FIELD_NAMES = {
    "coordinate": "instance_coordinate",
    "volume": "instance_volume",
    "label": "instance_label",
}
_ZMIN_RE = re.compile(r"_zmin(\d+)")


@dataclass
class Stack:
    name: str          # the instance_center filename
    zmin: int
    depth: Optional[int] = None


class StackTensorStore:
    """Lazily loads per-stack tensors for one tile, with light caching."""

    def __init__(self, tile_dir: str, center_key: str = _CENTER_KEY) -> None:
        self.tile_dir = tile_dir
        self.center_key = center_key
        self._cache: dict[str, torch.Tensor] = {}

    def list_stacks(self) -> List[Stack]:
        stacks: List[Stack] = []
        if not os.path.isdir(self.tile_dir):
            return stacks
        for fn in os.listdir(self.tile_dir):
            if self.center_key not in fn:
                continue
            m = _ZMIN_RE.search(fn)
            if not m:
                continue
            stacks.append(Stack(name=fn, zmin=int(m.group(1))))
        stacks.sort(key=lambda s: s.zmin)
        return stacks

    def _load(self, stack: Stack, field: str) -> torch.Tensor:
        fn = stack.name.replace(self.center_key, _FIELD_NAMES[field]) if field in _FIELD_NAMES else stack.name
        path = os.path.join(self.tile_dir, fn)
        if path not in self._cache:
            self._cache[path] = torch.load(path, map_location="cpu")
        return self._cache[path]

    def coordinate(self, stack: Stack) -> torch.Tensor:
        return self._load(stack, "coordinate")

    def volume(self, stack: Stack) -> torch.Tensor:
        return self._load(stack, "volume")

    def label(self, stack: Stack) -> torch.Tensor:
        return self._load(stack, "label")


def _stack_overlaps_z(stack: Stack, z1: int, z2: int, zr: float, default_depth: int) -> bool:
    depth = stack.depth if stack.depth is not None else default_depth
    oz_lo = stack.zmin * zr
    oz_hi = (stack.zmin + depth) * zr
    return oz_hi > z1 and oz_lo < z2


def build_nis_cube(
    local: LocalWindow,
    store: StackTensorStore,
    cfg: ResolutionConfig,
    default_stack_depth: int = 256,
) -> np.ndarray:
    """Return a ``(z, x, y)`` int32 label cube for one tile's local window.

    The axis order matches ``prepare_nis_to_aws.py`` (build ``[x, y, z]`` then
    ``transpose(2, 0, 1)``) so cubes are byte-compatible with the offline ones.
    """
    sx, sy, sz = local.shape_xyz
    cube_xyz = np.zeros((sx, sy, sz), dtype=np.int32)
    zr = cfg.z_nis_to_orig

    for stack in store.list_stacks():
        if not _stack_overlaps_z(stack, local.z1, local.z2, zr, default_stack_depth):
            continue

        coord = store.coordinate(stack)            # (Nv, 3) int, (z, x, y)
        vol = store.volume(stack).reshape(-1)       # (Ncell,)
        lab = store.label(stack).reshape(-1)        # (Ncell,)
        if coord.numel() == 0 or vol.numel() == 0:
            continue

        # Per-voxel label via run-length expansion (== coordinate[start_i:end_i]).
        vox_label = torch.repeat_interleave(lab, vol)
        n = min(vox_label.shape[0], coord.shape[0])
        coord = coord[:n]
        vox_label = vox_label[:n]

        gz = coord[:, 0].to(torch.float64) + stack.zmin
        oz = torch.round(gz * zr).to(torch.int64).numpy()
        ox = coord[:, 1].to(torch.int64).numpy()
        oy = coord[:, 2].to(torch.int64).numpy()
        vlab = vox_label.to(torch.int64).numpy()

        m = (
            (oz >= local.z1) & (oz < local.z2)
            & (ox >= local.x1) & (ox < local.x2)
            & (oy >= local.y1) & (oy < local.y2)
        )
        if not m.any():
            continue
        cube_xyz[ox[m] - local.x1, oy[m] - local.y1, oz[m] - local.z1] = vlab[m]

    return np.ascontiguousarray(cube_xyz.transpose(2, 0, 1))  # -> (z, x, y)


def extract_cell_voxels(coord: torch.Tensor, volume: torch.Tensor, i: int) -> torch.Tensor:
    """Voxels of cell ``i``: ``coordinate[start_i:end_i]`` (your RLE formula).

    Provided for targeted single-instance queries (mesh/point export).
    """
    csum = torch.cumsum(volume.reshape(-1), 0)
    start = int(csum[i - 1]) if i > 0 else 0
    end = int(csum[i])
    return coord[start:end]
