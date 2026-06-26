"""Dependency-free tests for the geometry + stitch core.

Run with: ``python -m unittest discover -s nis_ondemand_viewer/tests``
These avoid numpy/torch/nibabel so they pass on a bare interpreter.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from nis_ondemand_viewer.geometry import (  # noqa: E402
    ResolutionConfig,
    cube_window,
    map_to_orig,
    rle_bounds,
    snap_to_grid,
    tile_local_window,
)
from nis_ondemand_viewer.stitch import (  # noqa: E402
    GridStitchTransform,
    ImarisXmlStitchTransform,
)


class TestGeometry(unittest.TestCase):
    def setUp(self):
        self.cfg = ResolutionConfig()  # map=25, orig=(4,.75,.75), nis=(2.5,.75,.75)

    def test_rle_bounds(self):
        # cell volumes -> concatenated voxel slices
        self.assertEqual(rle_bounds([3, 2, 4]), [(0, 3), (3, 5), (5, 9)])
        self.assertEqual(rle_bounds([]), [])
        self.assertEqual(rle_bounds([1]), [(0, 1)])

    def test_rle_matches_cumsum_formula(self):
        vols = [5, 1, 7, 2]
        bounds = rle_bounds(vols)
        csum = []
        run = 0
        for v in vols:
            run += v
            csum.append(run)
        for i, (s, e) in enumerate(bounds):
            self.assertEqual(s, csum[i - 1] if i > 0 else 0)
            self.assertEqual(e, csum[i])

    def test_z_scale(self):
        self.assertAlmostEqual(self.cfg.z_nis_to_orig, 2.5 / 4.0)

    def test_cube_size_orig(self):
        # 25*5/.75 = 166 (int), z: 25*3/4 = 18
        self.assertEqual(self.cfg.cube_size_orig(), (166, 166, 18))

    def test_map_to_orig(self):
        # x: 5 * 25/.75 = 166 ; z: 3 * 25/4 = 18
        self.assertEqual(map_to_orig(5, 5, 3, self.cfg), (166, 166, 18))

    def test_snap_to_grid(self):
        self.assertEqual(snap_to_grid(7, 12, 4, self.cfg), (5, 10, 3))

    def test_cube_window_centered(self):
        w = cube_window(100, 100, 50, self.cfg)
        sx, sy, sz = self.cfg.cube_size_orig()
        self.assertEqual(w.shape_xyz, (sx, sy, sz))
        self.assertEqual(w.x1, 100 - sx // 2)
        self.assertEqual(w.x2, w.x1 + sx)

    def test_tile_local_window_inside(self):
        w = cube_window(100, 100, 50, self.cfg)
        local = tile_local_window(w, x_off=0, y_off=0, width=2048, height=2048, depth=200)
        self.assertIsNotNone(local)
        self.assertEqual(local.x1, w.x1)        # offset 0, not clamped
        self.assertEqual(local.z2, min(w.z2, 200))

    def test_tile_local_window_clamped_and_seam(self):
        w = cube_window(10, 10, 5, self.cfg)     # near origin -> negative start clamped
        local = tile_local_window(w, 0, 0, 2048, 2048, 200)
        self.assertIsNotNone(local)
        self.assertEqual(local.x1, 0)
        self.assertEqual(local.y1, 0)

    def test_tile_local_window_no_overlap(self):
        w = cube_window(100, 100, 50, self.cfg)
        # tile far away in x
        self.assertIsNone(tile_local_window(w, 10000, 0, 2048, 2048, 200))


class TestStitch(unittest.TestCase):
    def test_grid_placements_and_coverage(self):
        st = GridStitchTransform(n_cols=2, n_rows=1, tile_w=1000, tile_h=1000, depth=100)
        self.assertEqual(len(st.placements_at_z(0)), 2)
        cfg = ResolutionConfig()
        # window centered at x=1000 straddles the seam between tile col0 and col1
        w = cube_window(1000, 500, 50, cfg)
        covering = st.tiles_covering(w)
        self.assertEqual(len(covering), 2)  # both tiles contribute

    def test_imaris_xml(self):
        xml = """<?xml version='1.0'?>
        <ImarisStitcher>
          <ImageSize X="1000" Y="800" Z="120"/>
          <VoxelSize X="0.75" Y="0.75" Z="4.0" Unit="um"/>
          <Tiles>
            <Tile Name="UltraII[00 x 00]" Column="0" Row="0">
              <Position X="0.0" Y="0.0" Z="0.0" Unit="um"/>
            </Tile>
            <Tile Name="UltraII[01 x 00]" Column="1" Row="0">
              <Position X="750.0" Y="0.0" Z="0.0" Unit="um"/>
            </Tile>
          </Tiles>
        </ImarisStitcher>"""
        with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as f:
            f.write(xml)
            path = f.name
        try:
            st = ImarisXmlStitchTransform(path)
            ps = st.placements_at_z(0)
            self.assertEqual(len(ps), 2)
            self.assertEqual(ps[0].width, 1000)
            self.assertEqual(ps[0].height, 800)
            # 750um / 0.75um-per-px = 1000 px offset for the second tile
            offsets = sorted(p.x_off for p in ps)
            self.assertEqual(offsets, [0, 1000])
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
