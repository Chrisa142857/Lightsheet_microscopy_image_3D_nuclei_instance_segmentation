# Session handoff — NIS visualization review & on-demand zoom service

Continuation notes so this work can resume on the data server (real brains + NIS
C++ results). Everything below is pushed; nothing depends on the ephemeral
session container.

## Branch (both repos)
`claude/nis-viz-frontend-review-x4abh2`

- **chrisa142857/cellpheno-frontend** — the React/NiiVue viewer + frontend wiring
- **chrisa142857/Lightsheet_microscopy_image_3D_nuclei_instance_segmentation**
  — the NIS C++ pipeline + the new `nis_ondemand_viewer/` backend

## What this session did

### 1. Frontend assessment (cellpheno-frontend)
A working, well-architected **read-only** NIS viewer (React 18 + Vite + NiiVue).
Main view = small 25 µm density map; zoom view = click → pre-cropped 5×5×3 cube
from MinIO. Demo-grade, not production: hardcoded localhost, dead files, stale
heuristics. See commit `803d1f9`.

### 2. Frontend cleanup (commit `803d1f9`)
- Centralized MinIO origin/bucket in `src/configs/minio.ts`
  (`VITE_MINIO_ORIGIN` / `VITE_MINIO_BUCKET`, defaults preserve localhost:8080).
- Removed dead `App.tsx`, `App.css`, `ThreeNii/core.tsx`, and `validateUrls`.
- Fixed `Preview` `brainId` null-type hole + unreachable "Not Found".
- Verified: `pnpm build` clean.

### 3. Cost problem (the core discussion)
`cellpheno-frontend/prepare_nis_to_aws.py` pre-materializes the **whole brain**
into millions of small cube `.nii.gz` files on MinIO (raw image crops + dense
rasterized NIS labels), for every brain — TB-scale, even though users view
<0.1% of cubes. Decided to replace eager materialization with **crop-on-demand**.

### 4. On-demand zoom service (commit `<this branch>`, Lightsheet repo)
New `nis_ondemand_viewer/` FastAPI service crops only the requested cube from the
**native** NIS C++ data — no MinIO precompute, no raw conversion.

### 5. Frontend wiring (cellpheno-frontend commit `316f308`)
`VITE_ZOOM_API_ORIGIN` toggles on-demand mode; unset keeps MinIO behaviour.

## NIS C++ data model (as confirmed this session)
All `.zip` results are **torch tensors** (`*_nis_coordinate.h5` is retired).
Per **stack** (stacks tile the brain, e.g. 2×4×5 in z,x,y):
- `coordinate` `N_voxel × 3` — all cells' voxels concatenated, `(z, x, y)` sort
- `center` `N_cell × 3`, `instance_label` `N_cell × 1`, `nis_volume` `N_cell × 1`

Cell `i` owns `coordinate[start_i:end_i]`, `start_i = cumsum(nis_volume)[i-1]`
(0 for i=0), `end_i = cumsum(nis_volume)[i]`. Whole-brain z =
`within_stack_z + zmin`; original-res z = `global_z * (nis_res_z/orig_res_z)` =
`× 2.5/4`. x/y already at original pixel size. Cubes emitted at **original
resolution** (NIS results remapped to it; viewer reflects true scale).

Resolutions: `map=25 µm`, `orig=(z4, x.75, y.75) µm`, `nis=(z2.5, x.75, y.75) µm`,
grid step `cuben=5, cuben_z=3`. (All overridable via env.)

## Service architecture (`nis_ondemand_viewer/`)
```
geometry.py    dependency-free index math (resolution, cube window, RLE, overlap) — UNIT TESTED
stitch.py      StitchTransform interface + ImarisXmlStitchTransform + GridStitchTransform
nis_cube.py    per-stack torch tensors -> label cube (RLE expand, zmin, z-remap)
raw_cube.py    per-plane TIFF partial reads -> raw cube (NO conversion)
tiff_reader.py pyvips -> tifffile/zarr -> PIL region-read fallback
volume.py      cube -> gzip NIfTI at original resolution (nibabel)
cache.py       thread-safe LRU
config.py      env-driven ServiceConfig
app.py         FastAPI endpoints (+ CORS)
tests/         12 pure-python tests (no numpy/torch needed)
deploy/        Dockerfile, docker-compose.yml, service.env.example, systemd unit
```

### Endpoints
- `GET /zoom/info?brain&x&y&z[&channel]` → `{imgUrl:[...], nisUrl:[...]}`
  (one per tile covering the grid-snapped cube; x,y,z are **map-frame** voxel idx)
- `GET /zoom/nis/{brain}/{tiletok}/{channel}/{x}/{y}/{z}/{filename}` → gzip NIfTI
- `GET /zoom/raw/{brain}/{tiletok}/{channel}/{x}/{y}/{z}/{filename}` → gzip NIfTI
- `GET /healthz` → status + selected TIFF backend

URLs are query-free so NiiVue sniffs `.nii.gz`; the filename carries the
`rawImage`/`nis3d` token the frontend uses to key/colormap volumes.

## How to run on the data server

```bash
# in the Lightsheet repo, on this branch
cd nis_ondemand_viewer/deploy
cp service.env.example service.env      # edit NIS_ROOT / RAW_ROOT / STITCH_ROOT / P_TAG / RAW_CHANNEL
docker compose up -d --build            # -> http://<server>:8090/healthz
# or bare metal: see deploy/nis-ondemand-viewer.service (systemd)

# dev without docker:
pip install -r requirements.txt         # + system libvips for pyvips
export NIS_ROOT=... RAW_ROOT=... STITCH_ROOT=... P_TAG=P14 RAW_CHANNEL=C01
uvicorn nis_ondemand_viewer.app:app --port 8090

# tests (no heavy deps):
python -m unittest discover -s nis_ondemand_viewer/tests
```

Frontend: set `VITE_ZOOM_API_ORIGIN=http://<server>:8090` in
`cellpheno-frontend/.env`, then `pnpm dev`.

## VALIDATION CHECKLIST on real data (what was NOT verifiable here)
numpy/torch/nibabel/pyvips were not installed in the review container, so the
geometry/stitch **core is unit-tested** but the array/I/O layers only compiled.
On the data server, verify in this order:

1. **Tile dir / filename conventions** — `nis_cube.StackTensorStore` assumes
   per-stack files contain `instance_center` and a `_zmin<NNN>` token, with
   sibling `instance_coordinate/volume/label`. Adjust `_CENTER_KEY`/`_FIELD_NAMES`
   /`_ZMIN_RE` in `nis_cube.py` if your filenames differ.
2. **Raw plane files** — `tiff_reader.build_plane_index` parses z from `' Z####'`
   and filters by channel tag; confirm against real filenames.
3. **Stitch params** — replace `stitch.build_stitch_transform` with your real
   accessor (the old `parse_stitch_tform`/`brain_shape_tile_location`, not in
   these repos). Check **offset sign and units** (the Imaris example assumes µm
   stage positions ÷ voxel size, min-normalized to origin).
4. **z-frame alignment** — confirm raw plane Z-numbering shares the cube's global
   z index frame (assumed: each UltraII tile is a full-depth column). If raw z is
   per-tile-relative, add the tile's z offset in `raw_cube`.
5. **Numeric parity** — pick a brain already processed by
   `prepare_nis_to_aws.py`, request the same cube via `/zoom/info`+`/zoom/nis`,
   and diff against the MinIO cube. They should match (axis order, label values,
   pixdim). Then do the same for `/zoom/raw`.
6. **Perf** — confirm a zoom click is sub-second; tune `CACHE_CAPACITY` and the
   `read_region` backend (install libvips for pyvips).

## Open ideas (not done)
- Synthetic-data integration test (fabricated tensors + fake TIFF planes) to
  exercise the numpy/torch paths in CI.
- Longer term: serve the segmentation **sparse** (Neuroglancer precomputed
  meshes / OME-Zarr) instead of rasterized cubes — discussed, not implemented.

## Commit trail
- frontend `803d1f9` — viewer cleanup + configurable MinIO
- frontend `316f308` — VITE_ZOOM_API_ORIGIN wiring
- Lightsheet `<branch head>` — `nis_ondemand_viewer/` service + deploy + this doc
