# NIS nf-core module

This directory documents the Nextflow/nf-core packaging of the NIS (Nuclei
Instance Segmentation) C++ executable that lives in [`cpp/`](../cpp). It lets the
whole-brain segmentation step run as a portable, containerised,
resource-managed Nextflow process instead of the hand-written
`build_main/main ...` invocations in [`cpp/README.md`](../cpp/README.md).

## Why a *local* module

nf-core modules normally wrap a tool published on Bioconda/BioContainers. NIS is
a bespoke LibTorch + OpenCV + CUDA binary built from this repository, so it has
no conda package. The idiomatic nf-core pattern for that situation is a
**`modules/local/` module backed by a purpose-built container image**
([`containers/nis/Dockerfile`](../containers/nis/Dockerfile)). Everything else
follows nf-core DSL2 conventions: `tag`/`label` directives, `task.ext.args`,
`versions.yml`, a `meta.yml` schema, an nf-test, and a `-stub` run for CI.

## Layout

```
modules/local/nis/
├── main.nf            # NIS_SEGMENT process
├── meta.yml           # nf-core module metadata / IO schema
├── environment.yml    # build toolchain (execution uses the container)
└── tests/
    ├── main.nf.test   # nf-test (stub)
    └── tags.yml
containers/nis/Dockerfile   # reproducible build of the `main` binary
workflows/nis.nf            # samplesheet -> NIS_SEGMENT
main.nf, nextflow.config    # runnable pipeline entry point
conf/{base,modules}.config  # resources + per-module args/publishing
assets/samplesheet.csv      # example input
```

## Module IO

`NIS_SEGMENT` mirrors the executable's CLI (`cpp/main.cpp`):

| Channel | Type | Maps to NIS flag |
| --- | --- | --- |
| input `tuple val(meta), path(tile_dir)` | one tile's ordered slices | `--data_root`, `--brain_tag = meta.id`, `--device = meta.device` |
| input `path models` | TorchScript `.pt` model dir | `--model_root` |
| output `nis` | `*_NIScpp_results_zmin*_*.zip` | written to `--save_root .` |
| output `remap` (optional) | `*_remap.zip` | GNN gap-stitch remaps |
| output `versions` | `versions.yml` | — |

Other flags (`--batch_size`, `--chunk_depth`, `--cellprob_threshold`,
`--filename_tag`, `--no_foreground_detection`, `--isotropic`) are forwarded from
`params` via `ext.args` in [`conf/modules.config`](../conf/modules.config).

## Build the container

```bash
docker build -f containers/nis/Dockerfile -t ghcr.io/chrisa142857/lightsheet-nis:1.0.0 .
```

The image compiles `cpp/` against a CUDA build of LibTorch and ships the binary
as `main` on the `PATH`. It must run with GPU access (`--gpus all` for Docker,
`--nv` for Singularity); both profiles in `nextflow.config` set this.

## Run

```bash
# Real run (needs a GPU, the container, and the .pt models)
nextflow run main.nf -profile docker \
    --input assets/samplesheet.csv \
    --models downloads/resource \
    --outdir results

# Logic-only smoke test, no GPU/data required
nextflow run main.nf -profile docker -stub \
    --input assets/samplesheet.csv \
    --models downloads/resource
```

## Test

```bash
nf-test test modules/local/nis/tests/main.nf.test
```

The test runs in `-stub` mode, so it validates the process wiring and outputs
without requiring a GPU, the container, or input data.
