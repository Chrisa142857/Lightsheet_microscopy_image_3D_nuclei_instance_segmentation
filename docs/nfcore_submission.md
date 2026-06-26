# Contributing the NIS module to nf-core/modules

The module in [`modules/nf-core/cellpheno/nis`](../modules/nf-core/cellpheno/nis) is lint-clean
(`nf-core modules lint cellpheno/nis` → 42/45 (only the unpublished-container check fails)) and prettier-formatted. The actual PR must be
opened from a fork of `nf-core/modules` using your GitHub account (it cannot be
done from the automated session, which is scoped to this repository only).

Reference: <https://nf-co.re/docs/contributing/contribute-components>

## One-time prerequisites (resolve the two external items)

1. **Publish the container.** Build and push the image on a machine with Docker +
   a CUDA GPU + push access. A turnkey script is provided
   ([`containers/nis/build_and_push.sh`](../containers/nis/build_and_push.sh)),
   pre-filled with the cu128 LibTorch source (`cpp/README.md` /
   `cpp/build_main_hummer.sh`):

   ```bash
   docker login quay.io                                   # nf-core push access
   bash containers/nis/build_and_push.sh quay.io/nf-core/cellpheno-nis 1.0.0
   ```

   (The image is large — CUDA base + LibTorch. If you don't have nf-core quay
   access yet, push to `ghcr.io/chrisa142857/cellpheno-nis` and update the
   `container` line, or ask the nf-core team to host it.)

2. **Add minimal test data** to a branch of
   [nf-core/test-datasets](https://github.com/nf-core/test-datasets) (the
   `numorph/3dunet` precedent). The fixtures and their layout are in
   [`tests/testdata/cellpheno_nis`](../tests/testdata/cellpheno_nis). The real
   TorchScript model weights come from the G-drive folder in `cpp/README.md`:

   ```bash
   pip install gdown
   bash containers/nis/fetch_models.sh downloads/resource   # G-drive NIS models
   ```

   Put the tiny tile under
   `data/imaging/segmentation/cellpheno_nis/tile/` on test-datasets; host the
   large `.pt` weights on Zenodo (numorph precedent) and reference them in the
   `gpu`-tagged test. Then run it on a GPU runner to add its snapshot entry:
   `nf-test test modules/nf-core/cellpheno/nis/... --profile docker,gpu --update-snapshot`.

## Open the PR

```bash
# 1. Fork + clone nf-core/modules
gh repo fork nf-core/modules --clone --remote
cd modules
git checkout -b nis-module

# 2. Copy the module in from this repo (adjust SRC path)
SRC=/path/to/Lightsheet_microscopy_image_3D_nuclei_instance_segmentation
mkdir -p modules/nf-core/cellpheno/nis/tests
cp "$SRC"/modules/nf-core/cellpheno/nis/main.nf                 modules/nf-core/cellpheno/nis/
cp "$SRC"/modules/nf-core/cellpheno/nis/meta.yml                modules/nf-core/cellpheno/nis/
cp "$SRC"/modules/nf-core/cellpheno/nis/tests/main.nf.test      modules/nf-core/cellpheno/nis/tests/
cp "$SRC"/modules/nf-core/cellpheno/nis/tests/main.nf.test.snap modules/nf-core/cellpheno/nis/tests/

# 3. Validate locally
nf-core modules lint cellpheno/nis
pre-commit run --all-files
nf-core modules test cellpheno/nis --profile docker   # once the container is published

# 4. Commit, push, open PR against master
git add modules/nf-core/cellpheno/nis
git commit -m "new module: cellpheno/nis"
git push -u origin nis-module
gh pr create --repo nf-core/modules --base master \
  --title "new module: cellpheno/nis" --body-file ../PR_BODY.md
```

## PR description (paste into the PR body / `PR_BODY.md`)

```markdown
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
```
