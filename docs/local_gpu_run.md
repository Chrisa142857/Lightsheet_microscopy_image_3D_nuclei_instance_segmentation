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

```bash
# Preferred: nf-core org namespace (matches numorph/3dunet). Needs nf-core quay access.
bash containers/nis/build_and_push.sh quay.io/nf-core/cellpheno-nis 1.0.0

# Fallback you fully control (works without nf-core membership):
bash containers/nis/build_and_push.sh ghcr.io/chrisa142857/cellpheno-nis 1.0.0
```

If you used the **ghcr fallback**, point the module at it so lint can reach the
image, and make the package public (`ghcr.io` → package settings → Public):

```bash
sed -i 's#quay.io/nf-core/cellpheno-nis#ghcr.io/chrisa142857/cellpheno-nis#' \
    modules/nf-core/cellpheno/nis/main.nf
```

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
nf-core modules lint cellpheno/nis
pre-commit run --all-files
git add modules/nf-core/cellpheno
git commit -m "new module: cellpheno/nis"
git push -u origin cellpheno-nis
gh pr create --repo nf-core/modules --base master \
    --title "new module: cellpheno/nis" \
    --body-file ../Lightsheet_microscopy_image_3D_nuclei_instance_segmentation/PR_BODY.md
```

The PR body text is in [`docs/nfcore_submission.md`](nfcore_submission.md) — copy
it into `PR_BODY.md` first (or paste it into the PR on GitHub).

## Notes

- **Registry choice:** `quay.io/nf-core/cellpheno-nis` is the ideal home but needs
  nf-core org access; maintainers commonly mirror a contributor image there during
  review. Using `ghcr.io/chrisa142857/cellpheno-nis` (public) is fine to get a green
  lint and open the PR — note it in the PR and the team can re-host.
- **Conda CI:** the module intentionally errors under `-profile conda/mamba`
  (GPU binary, no conda) — this is expected and matches `parabricks/*`.
