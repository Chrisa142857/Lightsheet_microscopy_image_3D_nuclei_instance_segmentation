# Container image for the NIS (Nuclei Instance Segmentation) executable.
#
# Builds the C++/LibTorch/OpenCV/CUDA binary from `cpp/` and ships it on the
# PATH as `main`, so the nf-core module `modules/local/nis` can call it directly.
#
# Build from the repository root:
#   docker build -f containers/nis/Dockerfile -t ghcr.io/chrisa142857/lightsheet-nis:1.0.0 .
#
# The image is CUDA-enabled and must be run with GPU access, e.g. with
# `docker run --gpus all ...` or, via Nextflow, `docker.runOptions = '--gpus all'`
# / `singularity.runOptions = '--nv'`.

ARG CUDA_VERSION=12.8.0
ARG UBUNTU_VERSION=22.04

# ---------------------------------------------------------------------------
# Stage 1: build the NIS binary
# ---------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION} AS build

ARG LIBTORCH_URL=https://download.pytorch.org/libtorch/cu128/libtorch-cxx11-abi-shared-with-deps-2.7.0%2Bcu128.zip

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        unzip \
        ca-certificates \
        libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Fetch LibTorch (CUDA build) matching the runtime CUDA version.
WORKDIR /opt
RUN wget -q -O libtorch.zip "${LIBTORCH_URL}" \
    && unzip -q libtorch.zip \
    && rm libtorch.zip

# Build the NIS executable.
COPY cpp /src/cpp
WORKDIR /src/cpp/build
RUN cmake -DCMAKE_PREFIX_PATH=/opt/libtorch -DCMAKE_BUILD_TYPE=Release .. \
    && cmake --build . --config Release -j "$(nproc)"

# ---------------------------------------------------------------------------
# Stage 2: slim runtime image
# ---------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu${UBUNTU_VERSION} AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        libopencv-core4.5d \
        libopencv-imgcodecs4.5d \
        libopencv-imgproc4.5d \
        libgomp1 \
        procps \
    && rm -rf /var/lib/apt/lists/*

# LibTorch shared libraries + the compiled binary.
COPY --from=build /opt/libtorch /opt/libtorch
COPY --from=build /src/cpp/build/main /usr/local/bin/main

# Single authoritative version source. The nf-core module reports the NIS
# version via `eval("cat /usr/local/share/nis/VERSION")`, so the container tag
# (ARG) and the reported version stay in lockstep. Bump together on release.
ARG NIS_VERSION=1.0.0
RUN mkdir -p /usr/local/share/nis && echo "${NIS_VERSION}" > /usr/local/share/nis/VERSION

ENV LD_LIBRARY_PATH=/opt/libtorch/lib:${LD_LIBRARY_PATH}

LABEL org.opencontainers.image.title="lightsheet-nis" \
      org.opencontainers.image.description="3D nuclei instance segmentation for whole-brain lightsheet microscopy" \
      org.opencontainers.image.source="https://github.com/Chrisa142857/Lightsheet_microscopy_image_3D_nuclei_instance_segmentation" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.version="1.0.0"

ENTRYPOINT []
CMD ["main", "--help"]
