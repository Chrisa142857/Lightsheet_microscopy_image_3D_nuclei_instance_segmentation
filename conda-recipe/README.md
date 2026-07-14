# Bioconda recipe for `cellpheno-nis`

Draft recipe to package the NIS C++/LibTorch/CUDA executable on bioconda, so the
nf-core module can consume a versioned conda package / BioContainer instead of a
build-from-source Dockerfile (per the nf-core/modules#12179 review).

## Files
- `meta.yaml` — package metadata, source (pinned to the `v1.0.0` release tarball + sha256), and requirements.
- `build.sh` — compiles `cpp/` against conda's LibTorch/OpenCV and installs the binary as `cellpheno-nis`.
- `conda_build_config.yaml` — the CUDA version to build against.

## Design notes (validated locally)
- NIS has **no custom `.cu` kernels**, but CUDA-enabled LibTorch's CMake config
  (`Caffe2Config`) still calls `enable_language(CUDA)`, so the build needs `nvcc`
  (`{{ compiler('cuda') }}`) and the CUDA dev libraries.
- `libtorch_cuda.so` references NCCL symbols, so `nccl` is a host dependency.
- The binary is installed as `cellpheno-nis` (upstream names it `main`, which must
  not be placed on the global PATH).
- Local validation against conda-forge `libtorch` 2.12 + `libopencv` 5.0: **all
  sources compile cleanly** (no LibTorch API drift from the 2.5-era source). The
  remaining link step needs conda-build's isolated sandbox (to avoid host
  `/usr/lib` leakage) plus the `nccl` host dep declared above.

## Submitting to bioconda
1. If needed, cut a release that contains the modern `cmake_minimum_required` and
   update `version`/`sha256` here (the `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` flag in
   `build.sh` also lets the current `v1.0.0` tarball configure under CMake 4).
2. Fork `bioconda/bioconda-recipes`, add this as `recipes/cellpheno-nis/`.
3. Align `cuda_version` with a value bioconda's CI supports (their CUDA pinning).
4. Open a PR; iterate with bioconda CI (which provides the CUDA build sandbox).
5. Once merged, switch the nf-core module's `container` to the resulting BioContainer.
