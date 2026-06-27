#!/usr/bin/env python3
"""cellpheno/stitch — NIS-guided phase-correlation tile-stitching transform for one brain.

Thin CLI around ``image_stitch.phase_correlation_stitch.get_stitch_tform`` (replaces the
hard-coded ``image_stitch/stitch_main.py``). Expects the brain's NIS results laid out as
``<nis-result-path>/UltraII[%02d x %02d]/...`` and the raw light-sheet images under
``<ls-image-root>``; writes ``<save-path>/NIS_tranform/<btag>_tform_refine.json``.
"""
import argparse
import os
import sys

_HOME = os.environ.get("CELLPHENO_HOME") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_HOME, os.path.join(_HOME, "image_stitch"), os.path.join(_HOME, "coloc")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from phase_correlation_stitch import get_stitch_tform  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="NIS-guided tile stitching transform")
    ap.add_argument("--ptag", required=True, help="Pair tag (e.g. test_pair)")
    ap.add_argument("--btag", required=True, help="Brain tag (e.g. test_brain)")
    ap.add_argument("--ls-image-root", required=True, help="Raw light-sheet image dir for this brain")
    ap.add_argument("--nis-result-path", required=True, help="NIS results dir (contains UltraII[col x row]/)")
    ap.add_argument("--save-path", default=".", help="Output dir (NIS_tranform/ is created inside)")
    ap.add_argument("--overlap-r", type=float, default=0.2, help="Tile overlap ratio")
    ap.add_argument("--btag-split", action="store_true", help="Split btag on '_' (legacy naming)")
    args = ap.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    get_stitch_tform(
        args.ptag,
        args.btag,
        args.ls_image_root,
        args.save_path,
        args.nis_result_path,
        overlap_r=args.overlap_r,
        btag_split=args.btag_split,
    )
    print(f"wrote {args.save_path}/NIS_tranform/{args.btag}_tform_refine.json")


if __name__ == "__main__":
    main()
