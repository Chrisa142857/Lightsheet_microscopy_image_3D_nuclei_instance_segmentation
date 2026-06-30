#!/usr/bin/env python3
"""cellpheno/morphometry — per-nucleus 3D shape morphometry (paper stage 12).

For every NIS instance in a tile, build its binary voxel mask and measure the
ellipsoid with SimpleITK's LabelShapeStatisticsImageFilter — the same primitive the
paper uses ("SimpleITK estimates three principal axes of each single instance
segmentation mask to measure the ellipsoid", cf. analysis_nis_shape.py:get_pa/itk_pa).

Writes ``*instance_pa*.zip`` per NIS stack: a dict with per-instance principal axes
(3x3), equivalent-ellipsoid diameters (3), principal moments (3), volume and label.

The original analysis_nis_shape.py / nis_shape_featuremap.py can't be imported (module
-level argparse + hard-coded /cajal lab paths, and an undefined-variable WIP map step),
so this re-implements just the morphometry computation cleanly on the pipeline's NIS
output. Aggregating these into a 25 um whole-brain morphology map is a further step.
"""
import argparse
import os
import sys

_HOME = os.environ.get("CELLPHENO_HOME") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_HOME, os.path.join(_HOME, "image_stitch"), os.path.join(_HOME, "coloc")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import SimpleITK as sitk  # noqa: E402


def _instance_morphometry(coord_np):
    """coord_np: (n_voxels, 3) int coords of one instance -> (PA[9], ellipsoid[3], moments[3])."""
    pts = coord_np - coord_np.min(0)
    shape = (pts.max(0) + 1).tolist()
    frame = np.zeros(shape, dtype=np.uint8)
    frame[pts[:, 0], pts[:, 1], pts[:, 2]] = 1
    f = sitk.LabelShapeStatisticsImageFilter()
    f.Execute(sitk.GetImageFromArray(frame))
    if not f.HasLabel(1):
        return np.zeros(9), np.zeros(3), np.zeros(3)
    return (np.asarray(f.GetPrincipalAxes(1), dtype=np.float32),
            np.asarray(f.GetEquivalentEllipsoidDiameter(1), dtype=np.float32),
            np.asarray(f.GetPrincipalMoments(1), dtype=np.float32))


def main():
    ap = argparse.ArgumentParser(description="Per-nucleus 3D shape morphometry (SimpleITK)")
    ap.add_argument("--tile-dir", required=True, help="Directory of NIS result .zip tensors for one tile")
    ap.add_argument("--out-dir", default=".", help="Where to write *instance_pa*.zip")
    ap.add_argument("--vol-min", type=int, default=1, help="Skip instances with volume < this")
    ap.add_argument("--vol-max", type=int, default=0, help="Skip instances with volume > this (0 = no max)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    centers = sorted(f for f in os.listdir(args.tile_dir) if "instance_center" in f and f.endswith(".zip"))
    if not centers:
        sys.exit(f"No *instance_center*.zip found in {args.tile_dir}")

    for name in centers:
        coordfn = os.path.join(args.tile_dir, name.replace("instance_center", "instance_coordinate"))
        volfn = os.path.join(args.tile_dir, name.replace("instance_center", "instance_volume"))
        pafn = os.path.join(args.out_dir, name.replace("instance_center", "instance_pa"))
        if not (os.path.exists(coordfn) and os.path.exists(volfn)):
            print(f"skip {name}: missing instance_coordinate/volume", file=sys.stderr)
            continue
        vol = torch.load(volfn).long().numpy()
        coord = torch.load(coordfn).long().numpy()
        splits = np.concatenate([[0], np.cumsum(vol)])

        pa_all, ell_all, mom_all, lab_all, vol_all = [], [], [], [], []
        for i in range(len(vol)):
            v = int(vol[i])
            if v < args.vol_min or (args.vol_max and v > args.vol_max):
                continue
            inst = coord[splits[i]:splits[i + 1]]
            pa, ell, mom = _instance_morphometry(inst)
            pa_all.append(pa); ell_all.append(ell); mom_all.append(mom)
            lab_all.append(i); vol_all.append(v)

        torch.save({
            "PA": torch.tensor(np.stack(pa_all)) if pa_all else torch.zeros(0, 9),
            "ellipsoid": torch.tensor(np.stack(ell_all)) if ell_all else torch.zeros(0, 3),
            "principal_moments": torch.tensor(np.stack(mom_all)) if mom_all else torch.zeros(0, 3),
            "label": torch.tensor(lab_all, dtype=torch.long),
            "volume": torch.tensor(vol_all, dtype=torch.long),
        }, pafn)
        print(f"wrote {pafn} ({len(lab_all)} nuclei measured)")


if __name__ == "__main__":
    main()
