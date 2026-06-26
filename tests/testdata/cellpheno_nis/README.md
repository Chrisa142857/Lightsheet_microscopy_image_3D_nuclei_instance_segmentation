# cellpheno/nis test data

Minimal fixtures for the `cellpheno/nis` nf-core module test
(`modules/nf-core/cellpheno/nis/tests/main.nf.test`).

## Layout

Mirrors the input layout documented in [`cpp/README.md`](../../../cpp/README.md)
— a single tile is a directory of ordered 2D slices named `*_C1_*.ome.tif`
(`L ... Z####.ome.tif`), plus the TorchScript models directory:

```
cellpheno_nis/
├── tile/                         # one UltraII[xx x yy] tile worth of slices
│   ├── L_TEST_C1_Z0000.ome.tif   # 32×32 uint16 (placeholder; real ≈ 2000×2000)
│   ├── L_TEST_C1_Z0001.ome.tif
│   ├── L_TEST_C1_Z0002.ome.tif
│   └── L_TEST_C1_Z0003.ome.tif
└── models/                       # filenames the binary loads (see cpp/main.cpp)
    ├── nis_unet_cpu.pt           # placeholder; real weights are large
    ├── grad_2Dto3D_cuda:0.pt
    ├── gnn_message_passing_cuda:0.pt
    ├── gnn_classifier_cuda:0.pt
    └── flow_3DtoSeed.pt
```

Production data (`cpp/README.md`) is 2×2 tiles of 100×2000×2000 voxels per tile;
this fixture is shrunk to a few tiny slices so it is suitable as nf-core test
data. The `.pt` files here are **empty placeholders** that let the `-stub` test
stage its inputs — they are not real trained weights.

## Uploading to nf-core/test-datasets

The module test references these via
`params.modules_testdata_base_path + 'imaging/segmentation/cellpheno_nis/...'`.
For the upstream PR:

1. Add `tile/` (real tiny slices) under
   `data/imaging/segmentation/cellpheno_nis/tile/` on the `modules` (or a
   dedicated `cellpheno`) branch of
   [nf-core/test-datasets](https://github.com/nf-core/test-datasets).
2. Host the real TorchScript model weights on **Zenodo** (the `numorph/3dunet`
   precedent for large model files) and reference them in the `gpu`-tagged test,
   or place small stand-ins under `.../cellpheno_nis/models/`.
3. Run the `gpu`-tagged real test on a GPU runner with
   `nf-test test ... --profile docker,gpu --update-snapshot` to add its snapshot
   entry (the committed snapshot currently contains only the `-stub` entry).
