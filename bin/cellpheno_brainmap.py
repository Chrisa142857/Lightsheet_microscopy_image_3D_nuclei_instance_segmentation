#!/usr/bin/env python3
"""cellpheno/brainmap — fuse stitched NIS tiles into a downsampled whole-brain map.

Thin CLI around ``brainmap_nogui.Brainmap`` for one brain (replaces the directory walk
in ``gen_brain_map.py``). Applies the stitch transform, de-duplicates NIS IDs across
tile overlaps ("undouble"), and writes ``fused:<map-type>_<btag>.nii.gz`` to ``--save-root``.
"""
import argparse
import os
import sys

_HOME = os.environ.get("CELLPHENO_HOME") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_HOME, os.path.join(_HOME, "image_stitch"), os.path.join(_HOME, "coloc")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from brainmap_nogui import Brainmap  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Generate a downsampled whole-brain map from NIS results")
    ap.add_argument("--nis-result-path", required=True, help="NIS results dir for this brain (UltraII[col x row]/)")
    ap.add_argument("--stitch-root", required=True, help="Root containing <ptag>/<btag>/NIS_tranform/*.json")
    ap.add_argument("--map-type", default="cell count", choices=["avg volume", "cell count"])
    ap.add_argument("--ncol", type=int, default=2, help="Number of tile columns")
    ap.add_argument("--nrow", type=int, default=2, help="Number of tile rows")
    ap.add_argument("--overlap-r", type=float, default=0.2, help="Tile overlap ratio used for stitching")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--channel-layer", default=None, help="Optional channel layer from coloc")
    ap.add_argument("--stitch-type", default="Refine", choices=["Refine", "Manual"])
    ap.add_argument("--manual-stitch-path", default=None)
    ap.add_argument("--save-root", default=".", help="Where to write the fused .nii.gz")
    ap.add_argument("--temp-root", default="./tmp", help="Scratch dir for de-duplicated NIS IDs")
    args = ap.parse_args()

    os.makedirs(args.temp_root, exist_ok=True)
    os.makedirs(args.save_root, exist_ok=True)

    bm = Brainmap(
        cpp_result_root=args.nis_result_path,
        device=args.device,
        channel_layer=args.channel_layer,
        maptype=args.map_type,
        ncol=args.ncol,
        nrow=args.nrow,
        org_overlap_r=args.overlap_r,
        btag_split=False,
    )
    bm.arrange_brainmap_layer(args.stitch_root, stitch_type=args.stitch_type, manual_stitch_path=args.manual_stitch_path)
    bm.run_fusion(save_root=args.temp_root)
    bm.save_fused_image(save_root=args.save_root)
    print(f"wrote fused map to {args.save_root}")


if __name__ == "__main__":
    main()
