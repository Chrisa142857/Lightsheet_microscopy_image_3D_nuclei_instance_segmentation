# cellpheno-postproc: Python/torch container for the downstream NIS pipeline steps
# (coord_to_bbox -> stitch -> brain map -> coloc). It bundles this repository so the
# bin/ CLI wrappers can import the research modules (brainmap_nogui, coord_to_bbox,
# image_stitch/*, coloc/*).
#
# Build from the repository root:
#   bash containers/postproc/build_and_push.sh ghcr.io/chrisa142857/cellpheno-postproc 1.0.0
#
# GPU container: run with `--gpus all` (docker) / `--nv` (singularity).

# Pinned to a torch version that has prebuilt pytorch-scatter wheels (see below).
ARG PYTORCH_TAG=2.5.1-cuda12.1-cudnn9-runtime
FROM pytorch/pytorch:${PYTORCH_TAG}

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        tar gzip procps libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python deps actually imported by the downstream scripts (coord_to_bbox,
# image_stitch/phase_correlation_stitch, brainmap_nogui, coloc):
#   coord_to_bbox : numpy torch tqdm nibabel
#   stitch        : numpy scipy scikit-image pillow matplotlib nibabel tqdm
#   brain map     : torch torch_scatter dask numpy nibabel tqdm
#   coloc         : torch torchvision pandas (+ data-root config, see notes)
RUN pip install --no-cache-dir \
        numpy scipy pandas tqdm \
        nibabel scikit-image pillow matplotlib seaborn dask SimpleITK

# brainmap_nogui uses torch_scatter.scatter_max for IoU de-duplication. Install the
# prebuilt wheel matching the base image's torch/CUDA — compiling from source would
# need nvcc, which the -runtime base does not ship.
RUN pip install --no-cache-dir torch-scatter \
        -f https://data.pyg.org/whl/torch-2.5.1+cu121.html

# Bundle the repository (code only; see .dockerignore) so wrappers can import it.
ENV CELLPHENO_HOME=/opt/cellpheno
COPY . ${CELLPHENO_HOME}
ENV PYTHONPATH=${CELLPHENO_HOME}:${CELLPHENO_HOME}/image_stitch:${CELLPHENO_HOME}/coloc

# Put the CLI wrappers on PATH.
RUN cp ${CELLPHENO_HOME}/bin/cellpheno_*.py /usr/local/bin/ \
    && chmod +x /usr/local/bin/cellpheno_*.py

# Single authoritative version source (mirrors the cellpheno-nis container pattern).
ARG POSTPROC_VERSION=1.0.0
RUN mkdir -p /usr/local/share/cellpheno-postproc \
    && echo "${POSTPROC_VERSION}" > /usr/local/share/cellpheno-postproc/VERSION

LABEL org.opencontainers.image.title="cellpheno-postproc" \
      org.opencontainers.image.description="Downstream postprocessing for whole-brain NIS (coord_to_bbox, stitch, brain map, coloc)" \
      org.opencontainers.image.source="https://github.com/Chrisa142857/Lightsheet_microscopy_image_3D_nuclei_instance_segmentation" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.version="1.0.0"

CMD ["cellpheno_coordtobbox.py", "--help"]
