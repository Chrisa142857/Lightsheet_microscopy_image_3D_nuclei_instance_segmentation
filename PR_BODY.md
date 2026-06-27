Closes #11311

## Description

`cellpheno/nis` runs whole-brain 3D nuclei instance segmentation on a single lightsheet
microscopy tile: a 2D U-Net per slice, 2D→3D flow conversion, flow-following to
obtain instances per depth-chunk, and GNN gap-stitching across chunks. It wraps
the C++/LibTorch CellPheno NIS executable from
https://github.com/Chrisa142857/Lightsheet_microscopy_image_3D_nuclei_instance_segmentation.

This is a **GPU, container-only** module (no Conda package), following the
established `parabricks/*` and `numorph/3dunet` pattern: a single vendor container,
no `environment.yml`, an in-script guard that errors under `-profile conda/mamba`,
`label 'process_gpu'`, and versions via the `topic: versions` + `eval` mechanism.

## PR checklist

- [x] This comment contains a description of changes (with reason).
- [x] If you've added a new tool - followed the module conventions in the contribution docs.
- [ ] If necessary, include test data in your PR. <!-- pending nf-core/test-datasets -->
- [x] Remove all TODO statements.
- [x] Broadcast software version numbers to `topic: versions`.
- [x] Follow the naming conventions.
- [x] Follow the input/output options guidelines.
- [x] Add a resource `label` (`process_high` + `process_gpu`).
- [ ] Use BioConda and BioContainers if possible. <!-- N/A: custom CUDA/LibTorch GPU binary, no conda package; ships as a vendor container like parabricks -->
- Ensure the test works with Docker / Singularity:
  - [x] `nf-core modules test cellpheno/nis --profile docker` <!-- stub; real GPU test pending test-data -->
  - [x] `nf-core modules test cellpheno/nis --profile singularity`
  - [ ] `nf-core modules test cellpheno/nis --profile conda` <!-- intentionally unsupported (GPU binary) -->

### Notes for reviewers
- No conda support is intentional and matches `parabricks/*`.
- `process_gpu` emits a non-standard-label warning (as it does for every GPU
  module); paired with `process_high`.
- Versions come from `eval("cat /usr/local/share/cellpheno-nis/VERSION")` because the binary
  has no `--version`; the Dockerfile writes that file from the same `ARG` as the
  image tag to keep them in lockstep.
- **Container hosting:** the image is currently published at
  `ghcr.io/chrisa142857/cellpheno-nis:1.0.0` (I don't yet have `quay.io/nf-core`
  push access). `nf-core modules lint --registry ghcr.io` is clean (42 passed, 0
  failed). I'm happy to have the team re-host it under
  `quay.io/nf-core/cellpheno-nis` during review and will update the `container`
  line to match.
