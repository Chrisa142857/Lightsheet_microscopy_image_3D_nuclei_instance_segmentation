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
#   bash containers/nis/build_and_push.sh ghcr.io/chrisa142857/cellpheno-nis 1.0.0
set -euo pipefail

IMAGE="${1:-quay.io/nf-core/cellpheno-nis}"
VERSION="${2:-1.0.0}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# LibTorch CUDA build. The repo's cpp/README.md downloads LibTorch from
# download.pytorch.org; cpp/build_main_hummer.sh targets CUDA 12.8 (cu128).
# The Dockerfile defaults to the matching cu128 LibTorch; override via build-arg
# if you need a different CUDA/LibTorch version.
LIBTORCH_URL="${LIBTORCH_URL:-https://download.pytorch.org/libtorch/cu128/libtorch-cxx11-abi-shared-with-deps-2.7.0%2Bcu128.zip}"

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
