"""Stitch-parameter abstraction.

The stitch param *format* is intentionally pluggable -- different labs export
tile placements differently (Imaris stitcher XML, TeraStitcher, BigStitcher,
a custom ``parse_stitch_tform`` accessor, ...). Everything downstream only needs
a list of :class:`TilePlacement` for a given z, so we depend on the
:class:`StitchTransform` interface and ship one concrete example
(:class:`ImarisXmlStitchTransform`) plus a dependency-free
:class:`GridStitchTransform` so the service runs end-to-end before you wire your
real params.

To plug in your real format, subclass :class:`StitchTransform` and implement
``placements_at_z`` (and optionally ``tile_image_dir``); nothing else changes.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

from .geometry import CubeWindow, LocalWindow, tile_local_window


@dataclass(frozen=True)
class TilePlacement:
    """One stitched tile in original-resolution pixel coordinates."""

    tile_id: str          # e.g. "UltraII[00 x 00]" -- used to locate the image folder
    col: int
    row: int
    x_off: int            # left edge in the stitched/original frame (pixels)
    y_off: int            # top edge (pixels)
    width: int            # tile pixel width  (x)
    height: int           # tile pixel height (y)
    depth: int            # number of z planes


class StitchTransform(ABC):
    """Maps a cube window to the tiles that cover it."""

    @abstractmethod
    def placements_at_z(self, z: int) -> List[TilePlacement]:
        """All tile placements valid at original-resolution plane ``z``.

        Formats with per-z drift can vary the offsets with ``z``; static formats
        (Imaris) ignore it and return the same list every time.
        """

    def tiles_covering(self, window: CubeWindow) -> List[tuple[TilePlacement, LocalWindow]]:
        """Tiles overlapping ``window`` paired with the per-tile local crop box."""
        out: List[tuple[TilePlacement, LocalWindow]] = []
        for p in self.placements_at_z(window.z1):
            local = tile_local_window(window, p.x_off, p.y_off, p.width, p.height, p.depth)
            if local is not None:
                out.append((p, local))
        return out

    def tile_image_dir(self, brain_root: str, placement: TilePlacement) -> str:
        """Folder holding a tile's per-plane TIFFs. Override if your layout differs.

        Default matches the NIS C++ raw layout: ``<brain_root>/<tile_id>/``.
        """
        return os.path.join(brain_root, placement.tile_id)


class GridStitchTransform(StitchTransform):
    """Dependency-free placeholder: a regular grid with optional overlap.

    Useful for local testing and for single-tile brains. Not a substitute for
    real stitch params when tiles have measured sub-pixel/stage offsets.
    """

    def __init__(
        self,
        n_cols: int,
        n_rows: int,
        tile_w: int,
        tile_h: int,
        depth: int,
        overlap: int = 0,
        tile_id_fmt: str = "UltraII[{col:02d} x {row:02d}]",
    ) -> None:
        self._placements: List[TilePlacement] = []
        step_x = tile_w - overlap
        step_y = tile_h - overlap
        for col in range(n_cols):
            for row in range(n_rows):
                self._placements.append(
                    TilePlacement(
                        tile_id=tile_id_fmt.format(col=col, row=row),
                        col=col,
                        row=row,
                        x_off=col * step_x,
                        y_off=row * step_y,
                        width=tile_w,
                        height=tile_h,
                        depth=depth,
                    )
                )

    def placements_at_z(self, z: int) -> List[TilePlacement]:
        return list(self._placements)


class ImarisXmlStitchTransform(StitchTransform):
    """Example parser for an Imaris-stitcher style XML export.

    The exact schema varies by exporter, so this targets a representative layout
    and is easy to adapt. Expected shape::

        <ImarisStitcher>
          <ImageSize X="2048" Y="2048" Z="1850"/>       <!-- per-tile pixels -->
          <VoxelSize X="0.75" Y="0.75" Z="4.0" Unit="um"/>
          <Tiles>
            <Tile Name="UltraII[00 x 00]" Column="0" Row="0">
              <!-- stage position; micron by default, or pixels if PositionUnit="px" -->
              <Position X="0.0" Y="0.0" Z="0.0"/>
            </Tile>
            ...
          </Tiles>
        </ImarisStitcher>

    Positions in micron are converted to pixel offsets with ``VoxelSize`` (or the
    ``voxel_size`` override). Offsets are normalised so the minimum tile sits at
    origin.
    """

    def __init__(
        self,
        xml_path: str,
        voxel_size: Optional[tuple[float, float, float]] = None,
    ) -> None:
        root = ET.parse(xml_path).getroot()

        img = root.find("ImageSize")
        if img is None:
            raise ValueError("ImarisXml: missing <ImageSize>")
        tile_w = int(round(float(img.get("X"))))
        tile_h = int(round(float(img.get("Y"))))
        depth = int(round(float(img.get("Z"))))

        if voxel_size is not None:
            # ResolutionConfig.orig_res is (z, x, y).
            vx, vy = voxel_size[1], voxel_size[2]
        else:
            vx = vy = None
            vs = root.find("VoxelSize")
            if vs is not None:
                vx = float(vs.get("X"))
                vy = float(vs.get("Y"))
        # Fall back to 1.0 (positions already in pixels) when unknown.
        vx = vx or 1.0
        vy = vy or 1.0

        tiles_node = root.find("Tiles")
        if tiles_node is None:
            raise ValueError("ImarisXml: missing <Tiles>")

        raw: List[tuple[str, int, int, float, float]] = []
        for t in tiles_node.findall("Tile"):
            name = t.get("Name") or t.get("FileName") or t.get("Id") or ""
            col = int(t.get("Column", t.get("Col", "0")))
            row = int(t.get("Row", "0"))
            pos = t.find("Position")
            if pos is None:
                px = float(t.get("X", "0"))
                py = float(t.get("Y", "0"))
                unit = t.get("PositionUnit", "um")
            else:
                px = float(pos.get("X", "0"))
                py = float(pos.get("Y", "0"))
                unit = pos.get("Unit", t.get("PositionUnit", "um"))
            if unit.lower() in ("px", "pixel", "pixels"):
                ox, oy = px, py
            else:
                ox, oy = px / vx, py / vy
            raw.append((name, col, row, ox, oy))

        if not raw:
            raise ValueError("ImarisXml: no <Tile> entries found")

        min_x = min(r[3] for r in raw)
        min_y = min(r[4] for r in raw)
        self._placements = [
            TilePlacement(
                tile_id=name,
                col=col,
                row=row,
                x_off=int(round(ox - min_x)),
                y_off=int(round(oy - min_y)),
                width=tile_w,
                height=tile_h,
                depth=depth,
            )
            for (name, col, row, ox, oy) in raw
        ]

    def placements_at_z(self, z: int) -> List[TilePlacement]:
        return list(self._placements)


def build_stitch_transform(brain_id: str, cfg: "object") -> StitchTransform:
    """Factory stub -- wire this to your real per-brain stitch params.

    Resolution order is deliberately simple and documented so you can replace it:

    1. ``<stitch_root>/<brain_id>.xml`` parsed as Imaris XML, else
    2. a :class:`GridStitchTransform` from ``cfg`` defaults (single/grid tiles).

    Replace the body with your ``parse_stitch_tform``-equivalent and return any
    :class:`StitchTransform` subclass.
    """
    from .config import ServiceConfig  # local import to avoid a cycle

    assert isinstance(cfg, ServiceConfig)
    xml_path = os.path.join(cfg.stitch_root, f"{brain_id}.xml")
    if os.path.exists(xml_path):
        return ImarisXmlStitchTransform(xml_path, voxel_size=cfg.resolution.orig_res)
    # Fallback: assume a single tile sized to the configured grid defaults.
    return GridStitchTransform(
        n_cols=cfg.grid_cols,
        n_rows=cfg.grid_rows,
        tile_w=cfg.tile_width,
        tile_h=cfg.tile_height,
        depth=cfg.tile_depth,
        overlap=cfg.tile_overlap,
    )
