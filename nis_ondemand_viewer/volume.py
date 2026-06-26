"""Encode a ``(z, x, y)`` cube as gzip-compressed NIfTI at original resolution.

The header ``pixdim`` is set to the original acquisition spacing so NiiVue draws
the cube at true physical scale (NIS results are remapped onto this frame before
they reach here -- see :mod:`nis_cube`).
"""

from __future__ import annotations

import gzip

import nibabel as nib
import numpy as np

from .geometry import ResolutionConfig


def cube_to_nifti_gz(cube_zxy: np.ndarray, cfg: ResolutionConfig) -> bytes:
    """Return ``.nii.gz`` bytes for a ``(z, x, y)`` array.

    Axis 0 = z (``orig_res[0]``), axis 1 = x, axis 2 = y -- matching the
    ``transpose(2, 0, 1)`` cubes produced offline.
    """
    oz, ox, oy = cfg.orig_res
    affine = np.diag([float(oz), float(ox), float(oy), 1.0]).astype(np.float64)
    img = nib.Nifti1Image(np.ascontiguousarray(cube_zxy), affine)
    hdr = img.header
    hdr.set_zooms((float(oz), float(ox), float(oy)))
    raw = img.to_bytes()
    return gzip.compress(raw, compresslevel=4)
