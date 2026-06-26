"""On-demand NIS zoom viewer backend.

Serves zoom cubes (NIS labels + raw image) cropped on request from the native
NIS C++ tensors and per-plane TIFFs -- replacing the offline whole-brain
MinIO pre-materialisation in ``prepare_nis_to_aws.py``.

The dependency-free geometry/stitch core (``geometry``, ``stitch``) is unit
tested; the array/I/O layers (``nis_cube``, ``raw_cube``, ``volume``) require
numpy/torch/nibabel and the TIFF backends.
"""

from .geometry import ResolutionConfig

__all__ = ["ResolutionConfig"]
