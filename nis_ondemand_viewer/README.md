# NIS on-demand zoom viewer (backend)

Serves zoom cubes **cropped on request** straight from the native NIS C++
outputs — eliminating the TB-scale offline conversion that
`prepare_nis_to_aws.py` performs (rasterising the whole brain into millions of
small `.nii.gz` cubes and pushing them to MinIO).

Users ever view a tiny fraction of cubes, so we materialise only what is
actually requested and cache it.

## Why this avoids the conversion cost

| | Offline pre-materialise (current) | On-demand (this service) |
|---|---|---|
| NIS labels | dense cubes rasterised for the **whole** 5×5×3 grid, uploaded | `torch.load` the per-stack tensors and rasterise **only the requested window** |
| Raw image | whole-brain tiles cropped + duplicated into cubes | partial-read the ~18 plane TIFFs the cube needs; **no conversion** |
| Storage | whole brain × N brains, up front | small LRU cache of visited cubes |
| New brain | full reprocessing pass | nothing — just point at the data |

The raw data is already chunked along z (one 2D TIFF per plane, channel tag in
the filename), so the raw cube needs no precompute at all: open the plane files
in the cube's z-range and crop the x/y sub-window from each.

## Data model (NIS C++)

Per **stack** (stacks tile the brain, e.g. 2×4×5 in z,x,y), all `.zip` files are
torch tensors:

- `coordinate` `N_voxel × 3` — every cell's voxels concatenated, `(z, x, y)`
- `center` `N_cell × 3`, `instance_label` `N_cell × 1`, `nis_volume` `N_cell × 1`

Cell `i` owns `coordinate[start_i:end_i]` with `start_i = cumsum(nis_volume)[i-1]`
(0 for `i=0`), `end_i = cumsum(nis_volume)[i]` (see `geometry.rle_bounds` /
`nis_cube.extract_cell_voxels`). Whole-brain z is `within_stack_z + zmin`;
original-resolution z is `global_z * (nis_res_z / orig_res_z)` (= `2.5/4`).
x/y already share the NIS and original pixel size. Cubes are emitted at
**original resolution** so the viewer shows true scale.

## Layout

```
geometry.py    dependency-free index math (resolution remap, cube window,
               RLE bounds, tile overlap)            ← unit tested
stitch.py      StitchTransform interface + ImarisXmlStitchTransform (example)
               + GridStitchTransform (runs with no params)
nis_cube.py    per-stack torch tensors → label cube (the spec above)
raw_cube.py    per-plane TIFF partial reads → raw cube
tiff_reader.py pyvips → tifffile/zarr → PIL region-read fallback
volume.py      cube → gzip NIfTI at original resolution (nibabel)
cache.py       thread-safe LRU
config.py      env-driven ServiceConfig
app.py         FastAPI endpoints
tests/         pure-python tests (no numpy/torch needed)
```

## Stitch params are pluggable

The format is intentionally variable. `ImarisXmlStitchTransform` is a worked
example for an Imaris-stitcher-style XML; to use your own format, subclass
`StitchTransform`, implement `placements_at_z`, and return it from
`build_stitch_transform` (replace the factory body with your
`parse_stitch_tform`-equivalent). Everything downstream is unchanged.

## Run

Deploy files in `deploy/` (Docker + systemd); resuming on the data server is
documented in [`SESSION_HANDOFF.md`](./SESSION_HANDOFF.md), including a
validation checklist for real data.

```bash
# Docker:
cd nis_ondemand_viewer/deploy && cp service.env.example service.env  # edit paths
docker compose up -d --build                                          # :8090

# or dev:
pip install -r nis_ondemand_viewer/requirements.txt   # + libvips for pyvips
# point at your data:
export NIS_ROOT=/.../Lightsheet/results RAW_ROOT=/.../image_before_stitch \
       STITCH_ROOT=/.../stitch_params P_TAG=P14 RAW_CHANNEL=C01
uvicorn nis_ondemand_viewer.app:app --host 0.0.0.0 --port 8090
```

Test:
```bash
python -m unittest discover -s nis_ondemand_viewer/tests
```

## Frontend

Set `VITE_ZOOM_API_ORIGIN` (e.g. `http://localhost:8090`) in the
`cellpheno-frontend` `.env`. `fetchZoomVolumes` then calls `GET /zoom/info`
instead of the static MinIO JSON; the returned `{imgUrl, nisUrl}` point at
`/zoom/raw` and `/zoom/nis`, which NiiVue loads directly. Unset → unchanged
MinIO behaviour.

## API

- `GET /zoom/info?brain&x&y&z[&channel]` → `{imgUrl: [...], nisUrl: [...]}`
  (one per tile covering the clicked, grid-snapped cube; `x,y,z` are map-frame
  voxel indices).
- `GET /zoom/nis/{filename}?brain&x&y&z&tile` → gzip NIfTI label cube.
- `GET /zoom/raw/{filename}?brain&x&y&z&tile[&channel]` → gzip NIfTI raw cube.
- `GET /healthz` → status + selected TIFF backend.

## Cost per click

~18 small region reads (a few MB) + array assembly → sub-second, cached by
`(brain, tile, channel, x, y, z)`. Conversion/storage cost of the precompute
step → **zero**.

## Status

The dependency-free geometry/stitch core is unit-tested (`tests/`). The
array/I/O layers are implemented against the spec but were **not executed
against real data here** (numpy/torch/nibabel/TIFF libs not installed in this
environment) — validate against a real brain before production, especially the
stitch offset sign/units for your export and the raw plane↔z indexing.
```
