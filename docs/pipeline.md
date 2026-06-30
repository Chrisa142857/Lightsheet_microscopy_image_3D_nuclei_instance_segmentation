# Whole-brain NIS pipeline (one `nextflow run`)

The manual, multi-command workflow from the repo `README.md` /
[`cpp/README.md`](../cpp/README.md) is wired into a single Nextflow run:

```
NIS (per tile)  ─►  coord_to_bbox (per tile)  ─►  [group tiles by brain]
                                              ─►  stitch  ─►  brain map  ─►  (optional) coloc
```

| Step | Module | Wraps | Granularity |
|------|--------|-------|-------------|
| Segment nuclei | `NIS_SEGMENT` (`modules/local/nis`) | C++/LibTorch `main` | per tile |
| Bounding boxes | `CELLPHENO_COORDTOBBOX` | `coord_to_bbox.py` | per tile |
| Morphometry (optional) | `CELLPHENO_MORPHOMETRY` | SimpleITK ellipsoid (cf. `analysis_nis_shape.py`) | per tile |
| Tile stitching | `CELLPHENO_STITCH` | `image_stitch/phase_correlation_stitch.py` | per brain |
| Stitch refine (optional) | `CELLPHENO_STITCHREFINE` | `image_stitch/ptreg_stitch.py` | per brain |
| Whole-brain map | `CELLPHENO_BRAINMAP` | `brainmap_nogui.Brainmap` | per brain |
| Co-localization | `CELLPHENO_COLOC` (optional) | `coloc/...inferNIS.py` | per brain |

The `cellpheno/*` postprocessing modules call thin CLI wrappers in [`bin/`](../bin)
(`cellpheno_coordtobbox.py`, `cellpheno_stitch.py`, `cellpheno_brainmap.py`,
`cellpheno_coloc.py`) which import the original research code unchanged.

## Quick start

```bash
# Wiring smoke-test — runs the whole chain with stub outputs (no GPU/data/container):
nextflow run . -profile test -stub

# Real run:
nextflow run . -profile docker \
    --input samplesheet.csv \
    --models /path/to/NIS/torchscript/models \
    --outdir results
```

Use `-profile singularity` for Apptainer/Singularity. Add `--run_coloc` to enable the
optional co-localization step.

## Samplesheet (one row per tile)

```csv
brain,pair,tile_x,tile_y,tile_dir,image_dir,device
test_brain,test_pair,0,0,data/test_pair/test_brain/UltraII[00 x 00],data/test_pair/test_brain,cuda:0
test_brain,test_pair,0,1,data/test_pair/test_brain/UltraII[00 x 01],data/test_pair/test_brain,cuda:0
test_brain,test_pair,1,0,data/test_pair/test_brain/UltraII[01 x 00],data/test_pair/test_brain,cuda:0
test_brain,test_pair,1,1,data/test_pair/test_brain/UltraII[01 x 01],data/test_pair/test_brain,cuda:0
```

| Column | Required | Meaning |
|--------|----------|---------|
| `brain` | yes | Brain tag; groups tiles for stitch / brain map / coloc |
| `pair` | no | Pair tag (default `pair`) |
| `tile_x`, `tile_y` | no | Tile grid column/row (default 0); the brain's `ncol`/`nrow` are derived from the max |
| `tile_dir` | yes | Directory of raw 2D slices for that tile (NIS input) |
| `image_dir` | yes for stitch | Raw light-sheet image dir for the brain |
| `device` | no | CUDA device (default `cuda:0`) |

Tiles flow between processes as a `tar` of the literal `UltraII[col x row]/` directory,
because that name (spaces + `[ ]`) cannot be represented by Nextflow path globs. NIS runs
per tile but is prefixed with the **brain** tag (`ext.prefix` in `conf/modules.config`) so
each tile yields `{brain}_NIScpp_results_*` — the naming the stitch/brain-map code expects.

## Key parameters (`nextflow.config`)

| Param | Default | Description |
|-------|---------|-------------|
| `overlap_ratio` | `0.2` | Tile overlap ratio (stitch + fusion) |
| `map_type` | `cell count` | `cell count` or `avg volume` brain map |
| `num_column` / `num_row` | `2` | Tile-grid fallback (overridden per brain) |
| `image_tile_pattern` | `UltraII[%02d x %02d]` | printf pattern of the raw-image tile sub-folders under `image_dir` (column, then row) |
| `slice_filename_pattern` | `null` (auto-derive) | printf pattern of the 2D slice filenames carrying the z index, e.g. `L_C1_Z%04d.ome.tif` |
| `run_morphometry` | `false` | Run per-nucleus 3D shape morphometry (ellipsoid principal axes; paper stage 12) |
| `morph_vol_min` / `morph_vol_max` | `1` / `0` | Volume filter for morphometry (`0` max = no cap) |
| `run_stitchrefine` | `false` | Run the optional point-registration stitch refinement (README 3.5); brainmap prefers its transform |
| `ptreg_zrange` | `15` | Z window for point-registration refinement |
| `run_coloc` | `false` | Run the optional coloc step |
| `coloc_gtag` | `P4` | Growth-stage tag for coloc (`P4`/`P14`) |
| `coloc_weights` | `null` | Optional coloc classifier checkpoint |

## Containers

Two GPU container images back the pipeline (built/pushed like the nf-core module):

```bash
bash containers/nis/build_and_push.sh      ghcr.io/chrisa142857/cellpheno-nis      1.0.0  # C++ NIS binary
bash containers/postproc/build_and_push.sh ghcr.io/chrisa142857/cellpheno-postproc 1.0.0  # Python downstream
```

## Caveats

- **postproc container**: built + pushed to `ghcr.io/chrisa142857/cellpheno-postproc:1.0.0`
  (torch `2.5.1+cu121` + `torch_scatter` from the PyG wheel index; imports of all deps
  and the research modules, plus a real `coord_to_bbox` run, were validated). It is a
  private ghcr package — make it public for anonymous pulls, or stay logged in to ghcr.
  Rebuild with `containers/postproc/build_and_push.sh` if dependencies change.
- **coloc → brain-map channel is not wired** (upstream limitation): `brainmap_nogui`'s
  `channel_layer` reads per-tile `instance_<channel>.zip` masks, but no committed script
  produces them (the writer is commented out in `coloc_fgmask_to_label.py`). So `coloc`
  emits its classification CSV as a standalone output; a coloc-annotated brain map would
  need that producer implemented upstream first.
- **coloc portability**: the coloc dataset resolves several data-root paths and a
  hard-coded absolute `saver` inside `coloc/...inferNIS.py`; the wrapper redirects
  `saver` to the work dir, but the input data roots + model weights still need to point
  at your data. Kept optional (`--run_coloc`); `--coloc_weights` is effectively required.
- **stitch inputs**: `image_dir` must be the brain's raw-image **root** containing one
  sub-directory of `.ome.tif` slices per tile (not a flat dir). The sub-folder naming is
  set by `--image_tile_pattern` (default `UltraII[%02d x %02d]`; the stitch module
  symlinks them into the layout the research code expects). The brain must have a
  **complete rectangular tile grid** — `get_stitch_tform` asserts `len(tiles) == ncol*nrow`.
  Stitch is CPU-only (no GPU needed).
- The real (non-stub) downstream steps require the `cellpheno-postproc` image; NIS,
  coord_to_bbox, brain map and coloc need a GPU (stitch does not).
