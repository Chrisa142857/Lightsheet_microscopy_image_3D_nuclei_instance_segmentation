# NIS nf-core module

This directory documents the Nextflow/nf-core packaging of the NIS (Nuclei
Instance Segmentation) C++ executable that lives in [`cpp/`](../cpp). It lets the
whole-brain segmentation step run as a portable, containerised,
resource-managed Nextflow process instead of the hand-written
`build_main/main ...` invocations in [`cpp/README.md`](../cpp/README.md).

## Two module variants

| Path | Purpose | Status |
| --- | --- | --- |
| [`modules/local/nis`](../modules/local/nis) | Runs in this repo's own pipeline (`main.nf`). Pragmatic: `versions.yml` heredoc, runnable stub anywhere. | ✅ Working, stub-validated |
| [`modules/nf-core/nis`](../modules/nf-core/nis) | Submission candidate for [nf-core/modules](https://github.com/nf-core/modules) (issue [#11311](https://github.com/nf-core/modules/issues/11311)). Follows the **`parabricks/*` GPU pattern**: single vendor container, no conda, `topic: versions` + `eval`. | ✅ `nf-core modules lint`: **41/41 pass** (1 expected GPU exception) |

### nf-core submission status (`modules/nf-core/nis`)

Generated with `nf-core/tools` 4.0.2 (`nf-core modules create nis`), then aligned to
the pattern used by the **merged `nf-core/parabricks/*` GPU modules** — the
canonical precedent for a GPU tool with **no Conda package, shipped as a vendor
container**. Current `nf-core modules lint nis` result:

```
[✔]  41 Tests Passed
[!]   3 Test Warnings
[✗]   1 Test Failed
```

**No maintainer decision is required** — the remaining failure and warnings are the
*identical, accepted* lint output that the merged `parabricks/fq2bam` module also
produces (verified by linting it side by side). They are not code defects:

1. **`container_links` (the 1 failure) + the `405` warning.** Lint tries to *fetch*
   the container image to verify it exists; it can't, because the image hasn't been
   **published** yet (and isn't reachable from CI). `parabricks` fails this check
   the same way. Resolves once
   `ghcr.io/chrisa142857/lightsheet-nis:1.0.0` is built and pushed — see
   [`containers/nis/Dockerfile`](../containers/nis/Dockerfile). Not a conda/Bioconda
   requirement: nf-core accepts container-only GPU modules (parabricks ships from
   `nvcr.io`, no conda). For the upstream PR the cleanest option is to host the
   image under the nf-core org namespace, **`quay.io/nf-core/nis:<tag>`** — the
   pattern `numorph/3dunet` uses (`quay.io/nf-core/numorph-3dunet:1.0.9`) — which
   also satisfies the registry-prefix check without an allowlist entry.
2. **`process_gpu` "non-standard label" warning.** Emitted for *every* GPU module,
   including `parabricks`; it is the accepted GPU label. Tolerated.
3. **`main_nf_container` version-match warning.** Also present on `parabricks`.
   Harmless for a single-image module.

How the conda/version/test items were solved (no maintainers needed):

- **Conda:** removed the `conda` directive and `environment.yml`, and added the
  `parabricks`-style guard that errors under `-profile conda/mamba`.
- **Versions:** reported via `eval("cat /usr/local/share/nis/VERSION")`, a single
  authoritative source baked into the container (`ARG NIS_VERSION`).
- **Snapshot (`test_snapshot_exists`):** generated `tests/main.nf.test.snap` from a
  `-stub` run. Content is portable (output basenames + empty-file md5s + version),
  identical to what the published container would produce.

Remaining work happens in a fork of nf-core/modules (not pushable from this repo):
publish the container, add a minimal tile + `.pt` models to
[nf-core/test-datasets](https://github.com/nf-core/test-datasets), then open a PR
linked to issue #11311.

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
