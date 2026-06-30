#!/usr/bin/env python3
"""cellpheno/coloc — NIS-guided multi-channel co-localization classifier (inference).

Thin CLI around ``coloc/coloc_classifier-multiclassBboxLoc_inferNIS.py:infer_model``
(the file name contains a '-', so it is loaded via importlib rather than imported).

NOTE: this is the least portable step of the chain. The underlying dataset
(``InferNisPatchColocDataset``) and the default model-weight discovery resolve several
data-root paths relative to ``coloc/`` (e.g. ``model_weights/``, ``saver/``) and, when
``--ptag`` is omitted, an absolute lab path. Point ``--weights-path``/``--ptag`` at your
own data, or treat this step as optional (``--run_coloc`` in the pipeline).
"""
import argparse
import glob
import importlib.util
import os
import shutil
import sys

_HOME = os.environ.get("CELLPHENO_HOME") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_HOME, os.path.join(_HOME, "image_stitch"), os.path.join(_HOME, "coloc")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load_infer():
    path = os.path.join(_HOME, "coloc", "coloc_classifier-multiclassBboxLoc_inferNIS.py")
    spec = importlib.util.spec_from_file_location("coloc_infer", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["coloc_infer"] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(description="NIS-guided co-localization inference")
    ap.add_argument("--btag", required=True, help="Brain tag")
    ap.add_argument("--ptag", default=None, help="Pair tag (auto-discovered if omitted)")
    ap.add_argument("--gtag", default="P4", help="Growth-stage tag (P4 or P14)")
    ap.add_argument("--weights-path", default=None, help="Explicit classifier checkpoint")
    ap.add_argument("--weights-tag", default="16brain-layer23")
    ap.add_argument("--mtag", default="resnet50ChAlign")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--zmin", type=int, default=200)
    ap.add_argument("--zmax", type=int, default=400)
    ap.add_argument("--xyexpand", type=int, default=3)
    a = ap.parse_args()

    ns = argparse.Namespace(
        ptag=a.ptag, btag=a.btag, gtag=a.gtag,
        weights_path=a.weights_path, weights_tag=a.weights_tag, mtag=a.mtag,
        device=a.device, batch_size=a.batch_size,
        zmin=a.zmin, zmax=a.zmax, xyexpand=a.xyexpand,
    )
    workdir = os.getcwd()
    mod = _load_infer()
    # The script writes to a hard-coded absolute `saver` ({saver}/{gtag}/*_results_*.csv).
    # Redirect it to the Nextflow work dir so the result is captured, then lift the
    # CSV out of the {gtag}/ subdir to the work-dir root for the module's output glob.
    mod.saver = workdir
    os.makedirs(os.path.join(workdir, a.gtag), exist_ok=True)
    # infer_model auto-discovers model_weights/ relative to coloc/ when --weights-path
    # is omitted; run it from there, but `saver` (absolute) still points at workdir.
    os.chdir(os.path.join(_HOME, "coloc"))
    try:
        mod.infer_model(ns)
    finally:
        os.chdir(workdir)
    for csv in glob.glob(os.path.join(workdir, a.gtag, "*_results_*.csv")):
        shutil.move(csv, os.path.join(workdir, os.path.basename(csv)))


if __name__ == "__main__":
    main()
