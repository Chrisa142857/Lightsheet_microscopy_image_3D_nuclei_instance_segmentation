#!/usr/bin/env bash
#
# Build and push the CellPheno NIS GPU container referenced by
# modules/nf-core/cellpheno/nis (container "quay.io/nf-core/cellpheno-nis:1.0.0").
#
# Run this on a machine with Docker + a CUDA GPU + push access to the target
# registry. It is the step the automated session cannot do (no Docker daemon,
# blocked registry hosts). Once the image is pushed, the module's only remaining
# lint failure (container-not-reachable) clears.
#
# Usage:
#   bash containers/nis/build_and_push.sh [REGISTRY_IMAGE] [VERSION]
# Examples:
#   bash containers/nis/build_and_push.sh quay.io/nf-core/cellpheno-nis 1.0.0
#   bash containers/nis/build_and_push.sh quay.io/nf-core/cellpheno-nis 1.0.0
set -euo pipefail

IMAGE="${1:-quay.io/nf-core/cellpheno-nis}"
VERSION="${2:-1.0.0}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# LibTorch CUDA build. The Dockerfile targets CUDA 12.1 (cu121) for broad driver
# compatibility (runs on driver >= 530; covers sm_50..sm_90 incl. Ada sm_89) and to
# match the cellpheno-postproc container. Override via build-arg for a different
# CUDA/LibTorch version (e.g. cu128 for Blackwell sm_100/120).
LIBTORCH_URL="${LIBTORCH_URL:-https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu121.zip}"

echo ">> Building ${IMAGE}:${VERSION} (LibTorch: ${LIBTORCH_URL})"
docker build \
    -f "${REPO_ROOT}/containers/nis/Dockerfile" \
    --build-arg "LIBTORCH_URL=${LIBTORCH_URL}" \
    --build-arg "NIS_VERSION=${VERSION}" \
    -t "${IMAGE}:${VERSION}" \
    "${REPO_ROOT}"

echo ">> Pushing ${IMAGE}:${VERSION}"
echo "   (run 'docker login quay.io' first if pushing to quay.io/nf-core)"
docker push "${IMAGE}:${VERSION}"

echo ">> Done. If you used a registry other than quay.io/nf-core/cellpheno-nis,"
echo "   update the container line in modules/nf-core/cellpheno/nis/main.nf to match."
