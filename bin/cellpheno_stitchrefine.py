#!/usr/bin/env python3
"""cellpheno/stitchrefine — point-registration refinement of the tile-stitch transform.

Wraps image_stitch.ptreg_stitch.stitch_by_ptreg (README step 3.5). It recomputes a finer
per-slice tile alignment from the NIS instances and writes
``<btag>_tform_refine_ptreg.json``, which the brain-map step applies in preference to the
phase-correlation ``_tform_refine.json`` when present.

The research code names the file with ``btag.split('_')[1]``; this wrapper renames it to the
full ``<btag>`` so brainmap_nogui (which looks up the full btag) finds it.
"""
import argparse
import glob
import os
import sys

_HOME = os.environ.get("CELLPHENO_HOME") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_HOME, os.path.join(_HOME, "image_stitch"), os.path.join(_HOME, "coloc")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ptreg_stitch import stitch_by_ptreg  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Point-registration refinement of the stitch transform")
    ap.add_argument("--ptag", required=True)
    ap.add_argument("--btag", required=True)
    ap.add_argument("--nis-result-path", required=True, help="NIS results dir (contains UltraII[col x row]/)")
    ap.add_argument("--save-path", default=".", help="Output dir (NIS_tranform/ created inside)")
    ap.add_argument("--ncol", type=int, required=True)
    ap.add_argument("--nrow", type=int, required=True)
    ap.add_argument("--overlap-r", type=float, default=0.2)
    ap.add_argument("--zrange", type=int, default=15, help="Z window for point registration")
    ap.add_argument("--slice-lo", type=int, default=0)
    ap.add_argument("--slice-hi", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.save_path, "NIS_tranform"), exist_ok=True)
    stitch_tile_ij = [[i, j] for i in range(args.ncol) for j in range(args.nrow)]
    stitch_slice_ranges = [[args.slice_lo, args.slice_hi] for _ in stitch_tile_ij]

    stitch_by_ptreg(
        stitch_tile_ij,
        stitch_slice_ranges,
        ptag=args.ptag,
        btag=args.btag,
        device=args.device,
        save_path=args.save_path,
        result_path=args.nis_result_path,
        overlap_r=args.overlap_r,
        ZRANGE=args.zrange,
    )

    # Rename the research-code output (btag.split('_')[1]) to the full-btag name brainmap expects.
    want = os.path.join(args.save_path, "NIS_tranform", f"{args.btag}_tform_refine_ptreg.json")
    produced = glob.glob(os.path.join(args.save_path, "NIS_tranform", "*_tform_refine_ptreg.json"))
    if produced and os.path.abspath(produced[0]) != os.path.abspath(want):
        os.replace(produced[0], want)
    print(f"wrote {want}")


if __name__ == "__main__":
    main()
