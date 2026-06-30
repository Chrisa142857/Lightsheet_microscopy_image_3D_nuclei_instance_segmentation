#!/usr/bin/env python3
"""cellpheno/coordtobbox — derive per-instance 3D bounding boxes from NIS output.

Thin CLI around ``coord_to_bbox.coord_to_bbox`` so the step runs per NIS tile inside
Nextflow instead of the hard-coded directory walk in ``coord_to_bbox.py``. For every
``*instance_center*.zip`` stack in ``--tile-dir`` it loads the matching
``instance_coordinate``/``instance_volume`` tensors and writes ``*instance_bbox*.zip``.
"""
import argparse
import os
import sys

# Make the research modules at the repo root importable both when this script is run
# from the pipeline ``bin/`` (repo checkout) and from the bundled postproc container
# (``CELLPHENO_HOME=/opt/cellpheno``).
_HOME = os.environ.get("CELLPHENO_HOME") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_HOME, os.path.join(_HOME, "image_stitch"), os.path.join(_HOME, "coloc")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402
from coord_to_bbox import coord_to_bbox  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Generate NIS instance bounding boxes")
    ap.add_argument("--tile-dir", required=True, help="Directory of NIS result .zip tensors for one tile")
    ap.add_argument("--out-dir", default=".", help="Where to write *instance_bbox*.zip")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    centers = sorted(f for f in os.listdir(args.tile_dir) if "instance_center" in f and f.endswith(".zip"))
    if not centers:
        sys.exit(f"No *instance_center*.zip found in {args.tile_dir}")

    for name in centers:
        coordfn = os.path.join(args.tile_dir, name.replace("instance_center", "instance_coordinate"))
        volfn = os.path.join(args.tile_dir, name.replace("instance_center", "instance_volume"))
        bboxfn = os.path.join(args.out_dir, name.replace("instance_center", "instance_bbox"))
        if not (os.path.exists(coordfn) and os.path.exists(volfn)):
            print(f"skip {name}: missing instance_coordinate/instance_volume", file=sys.stderr)
            continue
        vol = torch.load(volfn).long()
        coord = torch.load(coordfn)
        bbox = coord_to_bbox(coord, vol.to(args.device), args.device).cpu()
        torch.save(bbox, bboxfn)
        print(f"wrote {bboxfn} ({len(bbox)} bboxes)")


if __name__ == "__main__":
    main()
