#!/usr/bin/env python3
"""cellpheno/stitch — NIS-guided phase-correlation tile-stitching transform for one brain.

Thin CLI around ``image_stitch.phase_correlation_stitch.get_stitch_tform`` (replaces the
hard-coded ``image_stitch/stitch_main.py``). Expects the brain's NIS results laid out as
``<nis-result-path>/UltraII[%02d x %02d]/...`` and the raw light-sheet images under
``<ls-image-root>``; writes ``<save-path>/NIS_tranform/<btag>_tform_refine.json``.

By default get_stitch_tform auto-derives the per-slice filename template from the first
file in each tile dir, assuming ``<prefix>_<z>_<mid>_xyz-Table Z<z>.ome.tif``. Pass
``--slice-pattern`` to support any slice filename: this wrapper symlinks each slice to the
canonical name that auto-derivation reproduces, leaving the research code untouched.
"""
import argparse
import os
import re
import sys

_HOME = os.environ.get("CELLPHENO_HOME") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_HOME, os.path.join(_HOME, "image_stitch"), os.path.join(_HOME, "coloc")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from phase_correlation_stitch import get_stitch_tform  # noqa: E402


def _printf_to_regex(pattern):
    """Convert a printf-style slice pattern (e.g. 'L_C1_Z%04d.ome.tif') into a regex that
    captures the integer z index as group 1."""
    esc = re.escape(pattern)
    esc = re.sub(r"%0?\d*d", lambda m: r"(\d+)", esc)
    return re.compile("^" + esc + "$")


def _canonicalize_slices(ls_root, slice_pattern, out_root):
    """For every tile dir under ls_root, symlink each slice file (named per slice_pattern,
    carrying the z index) to the canonical name get_stitch_tform's auto-derivation
    reproduces. Returns out_root, to be passed as the new ls-image-root."""
    rx = _printf_to_regex(slice_pattern)
    tiles = [d for d in os.listdir(ls_root)
             if d.startswith("UltraII") and os.path.isdir(os.path.join(ls_root, d))]
    n = 0
    for sub in tiles:
        srcdir = os.path.join(ls_root, sub)
        dstdir = os.path.join(out_root, sub)
        os.makedirs(dstdir, exist_ok=True)
        for f in sorted(os.listdir(srcdir)):
            m = rx.match(f)
            if not m:
                continue
            z = int(m.group(1))
            canon = f"LSslice_{z:04d}_C1_xyz-Table Z{z:04d}.ome.tif"
            dst = os.path.join(dstdir, canon)
            if not os.path.exists(dst):
                os.symlink(os.path.abspath(os.path.join(srcdir, f)), dst)
                n += 1
    if n == 0:
        sys.exit(f"--slice-pattern {slice_pattern!r} matched no files under {ls_root}/UltraII*/")
    print(f"canonicalized {n} slice files via pattern {slice_pattern!r}")
    return out_root


def main():
    ap = argparse.ArgumentParser(description="NIS-guided tile stitching transform")
    ap.add_argument("--ptag", required=True, help="Pair tag (e.g. test_pair)")
    ap.add_argument("--btag", required=True, help="Brain tag (e.g. test_brain)")
    ap.add_argument("--ls-image-root", required=True, help="Raw light-sheet image dir for this brain")
    ap.add_argument("--nis-result-path", required=True, help="NIS results dir (contains UltraII[col x row]/)")
    ap.add_argument("--save-path", default=".", help="Output dir (NIS_tranform/ is created inside)")
    ap.add_argument("--overlap-r", type=float, default=0.2, help="Tile overlap ratio")
    ap.add_argument("--slice-pattern", default=None,
                    help="printf pattern of the raw 2D slice filenames carrying the z index, "
                         "e.g. 'L_C1_Z%%04d.ome.tif'. Default: auto-derive from the files.")
    ap.add_argument("--btag-split", action="store_true", help="Split btag on '_' (legacy naming)")
    args = ap.parse_args()

    ls_root = args.ls_image_root
    if args.slice_pattern:
        ls_root = _canonicalize_slices(args.ls_image_root, args.slice_pattern, os.path.abspath("_ls_canon"))

    os.makedirs(args.save_path, exist_ok=True)
    get_stitch_tform(
        args.ptag,
        args.btag,
        ls_root,
        args.save_path,
        args.nis_result_path,
        overlap_r=args.overlap_r,
        btag_split=args.btag_split,
    )
    print(f"wrote {args.save_path}/NIS_tranform/{args.btag}_tform_refine.json")


if __name__ == "__main__":
    main()
