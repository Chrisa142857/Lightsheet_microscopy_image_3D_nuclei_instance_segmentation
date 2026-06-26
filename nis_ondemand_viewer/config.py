"""Service configuration, populated from environment variables.

Paths reflect the NIS C++ / Lightsheet layout but are all overridable so the
service can point at any deployment without code edits.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from .geometry import ResolutionConfig


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    return float(v) if v not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v not in (None, "") else default


@dataclass
class ServiceConfig:
    # --- data roots ---------------------------------------------------------
    # Per-tile per-stack NIS C++ tensors: <nis_root>/<P_tag>/<pair>/<brain>/<tile>/
    nis_root: str = field(default_factory=lambda: os.environ.get(
        "NIS_ROOT", "/scheibel/ACMUSERS/ziquanw/Lightsheet/results"))
    # Raw imaging: <raw_root>/<P_tag>/<pair>/<brain>/<tile>/<...Z####...C##...tif>
    raw_root: str = field(default_factory=lambda: os.environ.get(
        "RAW_ROOT", "/cajal/ACMUSERS/ziquanw/Lightsheet/image_before_stitch"))
    # Stitch params: <stitch_root>/<brain>.xml (Imaris example) -- see stitch.py
    stitch_root: str = field(default_factory=lambda: os.environ.get(
        "STITCH_ROOT", "/scheibel/ACMUSERS/ziquanw/Lightsheet/stitch_params"))

    p_tag: str = field(default_factory=lambda: os.environ.get("P_TAG", "P14"))
    channel: str = field(default_factory=lambda: os.environ.get("RAW_CHANNEL", "C01"))

    # --- resolutions / cube grid -------------------------------------------
    resolution: ResolutionConfig = field(default_factory=lambda: ResolutionConfig(
        map_res=_env_float("MAP_RES", 25.0),
        orig_res=(_env_float("ORIG_RES_Z", 4.0), _env_float("ORIG_RES_X", 0.75),
                  _env_float("ORIG_RES_Y", 0.75)),
        nis_res=(_env_float("NIS_RES_Z", 2.5), _env_float("NIS_RES_X", 0.75),
                 _env_float("NIS_RES_Y", 0.75)),
        cuben=_env_int("CUBEN", 5),
        cuben_z=_env_int("CUBEN_Z", 3),
    ))

    # --- grid fallback (used when no stitch XML is present) ------------------
    grid_cols: int = field(default_factory=lambda: _env_int("GRID_COLS", 1))
    grid_rows: int = field(default_factory=lambda: _env_int("GRID_ROWS", 1))
    tile_width: int = field(default_factory=lambda: _env_int("TILE_WIDTH", 2048))
    tile_height: int = field(default_factory=lambda: _env_int("TILE_HEIGHT", 2048))
    tile_depth: int = field(default_factory=lambda: _env_int("TILE_DEPTH", 2000))
    tile_overlap: int = field(default_factory=lambda: _env_int("TILE_OVERLAP", 0))

    # --- caching ------------------------------------------------------------
    cache_capacity: int = field(default_factory=lambda: _env_int("CACHE_CAPACITY", 256))

    # --- public origin advertised in /zoom/info URLs ------------------------
    public_origin: str = field(default_factory=lambda: os.environ.get(
        "PUBLIC_ORIGIN", ""))  # empty -> relative URLs
