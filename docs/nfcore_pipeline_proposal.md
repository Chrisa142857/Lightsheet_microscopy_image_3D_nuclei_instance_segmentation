# nf-core new-pipeline proposal: CellPheno

*Post this in the nf-core Slack **#new-pipelines** channel (adapt as needed).*

**Proposed name:** `nf-core/cellpheno` (whole-brain 3D nuclei **instance** segmentation & phenotyping for light-sheet microscopy)

**Maintainer:** @Chrisa142857 (Ziquan Wei) · **Preprint:** bioRxiv 10.64898/2026.03.17.712391 · **Data:** BossDB `curtin2026`

---

### What it does
A pipeline that takes raw light-sheet (LSFM) tiles of a whole cleared mouse brain and, in one `nextflow run`, produces a **labelled 3D instance segmentation of every nucleus** (30–50 M per P4 brain) plus whole-brain feature maps:

1. **NIS segmentation** (GPU C++/LibTorch): 2D U-Net per slice → 2D→3D flow (median-filter-pyramid) → flow-following instance extraction → GNN gap-stitch across depth-chunks.
2. **coord → bbox** (per tile)
3. **Tile stitching** — phase correlation (+ optional point-registration refine)
4. **Whole-brain map** — de-duplicate overlaps → 25 µm NIfTI (cell count / avg volume)
5. **Morphometry** (optional) — per-nucleus ellipsoid (SimpleITK)
6. **Co-localization** (optional) — multi-channel ResNet marker classification
7. **QC** — launch bundle for the [cellpheno-viewer](https://cellpheno-viewer.ziquanw.com/) (brain map + on-demand multi-scale zoom)

### Novelty & relationship to existing pipelines
I'm aware of **`nf-core/lsmquant`**, which quantifies LSFM nuclei (3D-U-Net detection + stitching + Allen-atlas registration). CellPheno is complementary rather than duplicative:

| | lsmquant | CellPheno |
|---|---|---|
| Output granularity | nuclei **detection / density** | full **instance** segmentation (every nucleus a labelled 3D object) |
| Core method | 3D U-Net | 2D U-Net + 2D→3D flow + **GNN** gap-stitch (handles anisotropic Z) |
| Enables | counts, atlas registration | per-nucleus **morphometry** + **co-localization** at whole-brain scale |

**Open to either outcome:** a **new pipeline**, or contributing CellPheno's instance-segmentation as **modules/subworkflows into `lsmquant`**. The core `cellpheno/nis` module is already in review at **nf-core/modules#12179** with test data at **nf-core/test-datasets#2136**.

### Practicalities
- **GPU required** (custom CUDA/LibTorch binary; container-only, no conda) — same pattern as `parabricks/*` and `numorph/3dunet`.
- Container hosted on `quay.io/nf-core/cellpheno-nis`; Python post-processing on a `cellpheno-postproc` image.
- Downstream steps (stitch/morphometry/coloc) are wrapped as `cellpheno/*` modules; the pipeline is prototyped and stub-runnable today.

### Ask
Is CellPheno a good fit as a **new nf-core pipeline**, or should the instance-segmentation capability be **added to lsmquant**? Happy to go whichever way the community prefers before I convert to the nf-core template.
