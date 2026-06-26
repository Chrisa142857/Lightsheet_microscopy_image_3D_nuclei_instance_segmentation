"""Partial reads of per-plane TIFFs -- no pre-conversion of the raw data.

The raw layout is already chunked along z (one 2D multichannel TIFF per plane,
with the z index and a channel tag in the filename), so a zoom cube only needs
to open the ~dozen plane files in its z-range and crop the x/y sub-window from
each. We never stack a whole tile.

Backends, best first:
* pyvips  -- libvips reads only the strips/tiles covering the crop (lowest mem).
* tifffile + zarr -- true partial decode for tiled TIFFs.
* PIL     -- decodes a single 2D plane then crops (fine as a fallback).
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Dict, Optional

import numpy as np

_Z_RE = re.compile(r" Z(\d{3,5})")


def build_plane_index(tile_dir: str, channel: str) -> Dict[int, str]:
    """Map ``{z_index: filepath}`` for one tile + channel.

    z is parsed from the ``' Z####'`` token (as in ``fn.split(' Z')[1][:4]``).
    """
    index: Dict[int, str] = {}
    if not os.path.isdir(tile_dir):
        return index
    for fn in os.listdir(tile_dir):
        if channel not in fn:
            continue
        m = _Z_RE.search(fn)
        if not m:
            continue
        index[int(m.group(1))] = os.path.join(tile_dir, fn)
    return index


# --------------------------------------------------------------------------- #
# Region readers. Each returns a (h, w) array for the half-open box
# [y1:y2, x1:x2]; callers transpose as needed.
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _backend() -> str:
    try:
        import pyvips  # noqa: F401
        return "pyvips"
    except Exception:
        pass
    try:
        import tifffile  # noqa: F401
        import zarr  # noqa: F401
        return "tifffile"
    except Exception:
        pass
    try:
        from PIL import Image  # noqa: F401
        return "pil"
    except Exception:
        pass
    return "none"


def read_region(path: str, x1: int, y1: int, x2: int, y2: int) -> Optional[np.ndarray]:
    """Read ``[y1:y2, x1:x2]`` from a 2D TIFF plane, or ``None`` on failure."""
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None
    backend = _backend()
    try:
        if backend == "pyvips":
            import pyvips
            img = pyvips.Image.new_from_file(path, access="random")
            region = img.crop(x1, y1, w, h)
            arr = np.ndarray(
                buffer=region.write_to_memory(),
                dtype=_VIPS_DTYPE.get(region.format, np.uint16),
                shape=[region.height, region.width, region.bands],
            )
            return arr[..., 0] if arr.shape[-1] == 1 else arr.mean(axis=-1)
        if backend == "tifffile":
            import tifffile
            import zarr
            with tifffile.TiffFile(path) as tf:
                z = zarr.open(tf.aszarr(), mode="r")
                a = z[0] if isinstance(z, zarr.hierarchy.Group) else z
                return np.asarray(a[y1:y2, x1:x2])
        if backend == "pil":
            from PIL import Image
            with Image.open(path) as im:
                return np.asarray(im.crop((x1, y1, x2, y2)))
    except Exception:
        return None
    return None


_VIPS_DTYPE = {
    "uchar": np.uint8,
    "char": np.int8,
    "ushort": np.uint16,
    "short": np.int16,
    "uint": np.uint32,
    "int": np.int32,
    "float": np.float32,
    "double": np.float64,
}


def backend_name() -> str:
    """Expose the selected backend (for /healthz and logging)."""
    return _backend()
