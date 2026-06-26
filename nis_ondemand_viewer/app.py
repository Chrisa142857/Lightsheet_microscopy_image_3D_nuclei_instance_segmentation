"""FastAPI service: on-demand zoom cubes, no MinIO pre-materialisation.

Endpoints
---------
* ``GET /healthz``                  -- liveness + selected TIFF backend.
* ``GET /zoom/info``                -- ``{imgUrl, nisUrl}`` (same shape the
                                       frontend already parses), one entry per
                                       tile covering the clicked cube.
* ``GET /zoom/nis/{filename}``      -- gzip NIfTI label cube for one tile.
* ``GET /zoom/raw/{filename}``      -- gzip NIfTI raw-image cube for one tile.

The ``info`` URLs carry a descriptive ``filename`` (so NiiVue sniffs ``.nii.gz``
and the frontend derives the volume key) plus query params identifying the
brain, the snapped map coordinate, the channel, and the tile.
"""

from __future__ import annotations

import os
import re
from typing import List, Optional
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware

from .cache import LRUBytesCache
from .config import ServiceConfig
from .geometry import CubeWindow, cube_window, map_to_orig, snap_to_grid
from .nis_cube import StackTensorStore, build_nis_cube
from .raw_cube import build_raw_cube
from .stitch import TilePlacement, build_stitch_transform
from .tiff_reader import backend_name
from .volume import cube_to_nifti_gz

_SANITISE = re.compile(r"[^A-Za-z0-9]+")


def _tiletok(tile_id: str) -> str:
    """Filesystem/URL-safe token for a tile id (e.g. 'UltraII[00 x 00]' -> 'UltraII00x00')."""
    return _SANITISE.sub("", tile_id)


def make_app(cfg: Optional[ServiceConfig] = None) -> FastAPI:
    cfg = cfg or ServiceConfig()
    app = FastAPI(title="NIS on-demand zoom viewer")
    # The viewer is a separate origin; allow it to fetch cubes.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "HEAD"],
        allow_headers=["*"],
    )
    cache = LRUBytesCache(cfg.cache_capacity)

    # ---- data-path resolution (override to match your deployment) ----------
    def resolve_pair(brain: str) -> Optional[str]:
        base = os.path.join(cfg.nis_root, cfg.p_tag)
        if not os.path.isdir(base):
            return None
        for pair in os.listdir(base):
            if os.path.isdir(os.path.join(base, pair, brain)):
                return pair
        return None

    def nis_tile_dir(pair: str, brain: str, tile_id: str) -> str:
        return os.path.join(cfg.nis_root, cfg.p_tag, pair, brain, tile_id)

    def raw_tile_dir(pair: str, brain: str, tile_id: str) -> str:
        return os.path.join(cfg.raw_root, cfg.p_tag, pair, brain, tile_id)

    def window_for(map_x: int, map_y: int, map_z: int) -> CubeWindow:
        sx, sy, sz = snap_to_grid(map_x, map_y, map_z, cfg.resolution)
        ox, oy, oz = map_to_orig(sx, sy, sz, cfg.resolution)
        return cube_window(ox, oy, oz, cfg.resolution)

    def find_placement(brain: str, window: CubeWindow, tiletok: str) -> Optional[tuple[TilePlacement, "object"]]:
        stitch = build_stitch_transform(brain, cfg)
        for placement, local in stitch.tiles_covering(window):
            if _tiletok(placement.tile_id) == tiletok:
                return placement, local
        return None

    def url(kind: str, brain: str, placement: TilePlacement, x: int, y: int, z: int, channel: str) -> str:
        # Query-free path so NiiVue sniffs the .nii.gz extension cleanly. The
        # trailing filename also carries the volume key the frontend derives.
        tok = _tiletok(placement.tile_id)
        token = "rawImage" if kind == "raw" else "nis3d"
        fname = f"{brain}_{tok}_zoomin_{token}_{x:03d}_{y:03d}_{z:03d}.nii.gz"
        return (
            f"{cfg.public_origin}/zoom/{kind}/{quote(brain)}/{tok}/{quote(channel)}"
            f"/{x}/{y}/{z}/{fname}"
        )

    # ---- endpoints ---------------------------------------------------------
    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok", "tiff_backend": backend_name(), "cached": len(cache)}

    @app.get("/zoom/info")
    def zoom_info(
        brain: str = Query(...),
        x: int = Query(...),
        y: int = Query(...),
        z: int = Query(...),
        channel: Optional[str] = Query(None),
    ) -> dict:
        channel = channel or cfg.channel
        sx, sy, sz = snap_to_grid(x, y, z, cfg.resolution)
        window = window_for(x, y, z)
        stitch = build_stitch_transform(brain, cfg)
        img_urls: List[str] = []
        nis_urls: List[str] = []
        for placement, _local in stitch.tiles_covering(window):
            img_urls.append(url("raw", brain, placement, sx, sy, sz, channel))
            nis_urls.append(url("nis", brain, placement, sx, sy, sz, channel))
        return {"imgUrl": img_urls, "nisUrl": nis_urls}

    @app.get("/zoom/nis/{brain}/{tiletok}/{channel}/{x}/{y}/{z}/{filename}")
    def zoom_nis(brain: str, tiletok: str, channel: str, x: int, y: int, z: int, filename: str) -> Response:
        key = f"nis:{brain}:{tiletok}:{x}:{y}:{z}"
        cached = cache.get(key)
        if cached is None:
            pair = resolve_pair(brain)
            if pair is None:
                raise HTTPException(404, f"brain {brain} not found")
            found = find_placement(brain, window_for(x, y, z), tiletok)
            if found is None:
                raise HTTPException(404, "tile does not cover this window")
            placement, local = found
            store = StackTensorStore(nis_tile_dir(pair, brain, placement.tile_id))
            cube = build_nis_cube(local, store, cfg.resolution, cfg.tile_depth)
            cached = cube_to_nifti_gz(cube, cfg.resolution)
            cache.put(key, cached)
        return Response(content=cached, media_type="application/gzip")

    @app.get("/zoom/raw/{brain}/{tiletok}/{channel}/{x}/{y}/{z}/{filename}")
    def zoom_raw(brain: str, tiletok: str, channel: str, x: int, y: int, z: int, filename: str) -> Response:
        channel = channel or cfg.channel
        key = f"raw:{brain}:{tiletok}:{channel}:{x}:{y}:{z}"
        cached = cache.get(key)
        if cached is None:
            pair = resolve_pair(brain)
            if pair is None:
                raise HTTPException(404, f"brain {brain} not found")
            found = find_placement(brain, window_for(x, y, z), tiletok)
            if found is None:
                raise HTTPException(404, "tile does not cover this window")
            placement, local = found
            cube = build_raw_cube(local, raw_tile_dir(pair, brain, placement.tile_id), channel)
            cached = cube_to_nifti_gz(cube, cfg.resolution)
            cache.put(key, cached)
        return Response(content=cached, media_type="application/gzip")

    return app


app = make_app()
