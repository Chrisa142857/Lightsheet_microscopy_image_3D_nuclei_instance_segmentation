# Finishing the CellPheno NIS submission on a local GPU server

This is the runbook for the three steps the automated cloud session could not do
(no Docker daemon, blocked registry hosts, GitHub scope). Run it on your GPU
server, where Docker + GPU + network + your credentials are available. You can
either run the commands yourself, or launch **Claude Code** in this repo and say:
*"complete the CellPheno NIS nf-core submission per docs/local_gpu_run.md"* — the
agent will execute these steps and verify each one.

## 0. One-time setup

```bash
git clone https://github.com/Chrisa142857/Lightsheet_microscopy_image_3D_nuclei_instance_segmentation
cd Lightsheet_microscopy_image_3D_nuclei_instance_segmentation
git checkout claude/nis-nfcore-conversion-afd7qu

bash scripts/setup_local_gpu.sh     # installs nf-core, nextflow, nf-test, gdown; checks Docker+GPU
docker login ghcr.io                # your GitHub container registry
docker login quay.io                # only if you have nf-core org push access
gh auth login                       # for the fork + PR
```

## 1. Build + push the GPU container

You are not an nf-core org member, so push to **your own GHCR** (the proven
no-membership path — `scvitools/*` ships from `ghcr.io/scverse` the same way).
The module already references `ghcr.io/chrisa142857/cellpheno-nis:1.0.0`.

```bash
bash containers/nis/build_and_push.sh ghcr.io/chrisa142857/cellpheno-nis 1.0.0
```

Then make the package **public** so nf-core CI can pull it:
github.com → your profile → **Packages** → `cellpheno-nis` →
**Package settings → Change visibility → Public**.

## 2. Verify the module is fully green

Lint must run inside a clone of nf-core/modules (it compares against that repo).

```bash
git clone --depth 1 https://github.com/nf-core/modules.git /tmp/nfcore-modules
cp -r modules/nf-core/cellpheno /tmp/nfcore-modules/modules/nf-core/
cd /tmp/nfcore-modules
nf-core modules lint cellpheno/nis        # container check now PASSES (image is reachable)
cd -
```

Expected: the previous `container_links` failure is gone (image now reachable).
The remaining `process_gpu` / container-version messages are non-fatal warnings
that every GPU module carries.

## 3. (Optional, for the real GPU test) prepare test data

The committed test has a `-stub` test (already snapshotted) plus a `gpu`-tagged
real test. The real test needs data; the stub is enough to open the PR. To enable
the real test later:

```bash
bash containers/nis/fetch_models.sh downloads/resource   # G-drive NIS .pt models (cpp/README.md)
# Tiny tile fixture is in tests/testdata/cellpheno_nis/ — replace with a real small tile.
# Then: PR the tile to nf-core/test-datasets, host .pt weights on Zenodo, and run
nf-test test modules/nf-core/cellpheno/nis/tests/main.nf.test --profile docker,gpu --update-snapshot
```

## 4. Fork nf-core/modules and open the PR

```bash
gh repo fork nf-core/modules --clone --remote
cd modules
git checkout -b cellpheno-nis
cp -r ../Lightsheet_microscopy_image_3D_nuclei_instance_segmentation/modules/nf-core/cellpheno \
      modules/nf-core/

# Allowlist your GHCR registry so the container_links prefix check passes
# (exactly like ghcr.io/scverse, ghcr.io/marcelauliano, ... already in the list)
yq -i '.["container-registry"] += ["ghcr.io/chrisa142857"]' .nf-core.yml 2>/dev/null || \
  sed -i 's#^\(container-registry:\)#\1\n  - ghcr.io/chrisa142857#' .nf-core.yml

nf-core modules lint cellpheno/nis      # should be fully green now (image public + prefix allowlisted)
pre-commit run --all-files
git add modules/nf-core/cellpheno .nf-core.yml
git commit -m "new module: cellpheno/nis"
git push -u origin cellpheno-nis
gh pr create --repo nf-core/modules --base master \
    --title "new module: cellpheno/nis" \
    --body-file ../Lightsheet_microscopy_image_3D_nuclei_instance_segmentation/PR_BODY.md
```

The PR body text is in [`docs/nfcore_submission.md`](nfcore_submission.md) — copy
it into `PR_BODY.md` first (or paste it into the PR on GitHub).

## Notes

- **Registry (no nf-core membership needed):** push to your own public
  `ghcr.io/chrisa142857/cellpheno-nis` and add `ghcr.io/chrisa142857` to the
  nf-core/modules `.nf-core.yml` `container-registry` allowlist in your PR — the
  same pattern as the merged `scvitools/*` (`ghcr.io/scverse`) modules. nf-core
  may later mirror it to `quay.io/nf-core`; that's a trivial one-line change they
  handle during review.
- **Conda CI:** the module intentionally errors under `-profile conda/mamba`
  (GPU binary, no conda) — this is expected and matches `parabricks/*`.
