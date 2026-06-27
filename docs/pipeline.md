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
| Tile stitching | `CELLPHENO_STITCH` | `image_stitch/phase_correlation_stitch.py` | per brain |
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
- **coloc portability**: the coloc dataset resolves several data-root paths and a
  hard-coded absolute `saver` inside `coloc/...inferNIS.py`; the wrapper redirects
  `saver` to the work dir, but the input data roots + model weights still need to point
  at your data. Kept optional (`--run_coloc`); `--coloc_weights` is effectively required.
- The real (non-stub) downstream steps require the `cellpheno-postproc` image and a GPU.
