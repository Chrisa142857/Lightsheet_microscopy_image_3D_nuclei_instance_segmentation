#!/usr/bin/env bash
#
# Build and push the cellpheno-postproc container used by the downstream pipeline
# modules (cellpheno/coordtobbox, cellpheno/stitch, cellpheno/brainmap, cellpheno/coloc).
#
# Run on a machine with Docker (+ a CUDA GPU to validate) and push access.
#
# Usage:
#   bash containers/postproc/build_and_push.sh [REGISTRY_IMAGE] [VERSION]
# Examples:
#   bash containers/postproc/build_and_push.sh ghcr.io/chrisa142857/cellpheno-postproc 1.0.0
#   bash containers/postproc/build_and_push.sh quay.io/nf-core/cellpheno-postproc 1.0.0
set -euo pipefail

IMAGE="${1:-ghcr.io/chrisa142857/cellpheno-postproc}"
VERSION="${2:-1.0.0}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo ">> Building ${IMAGE}:${VERSION}"
docker build \
    -f "${REPO_ROOT}/containers/postproc/Dockerfile" \
    --build-arg "POSTPROC_VERSION=${VERSION}" \
    -t "${IMAGE}:${VERSION}" \
    "${REPO_ROOT}"

echo ">> Pushing ${IMAGE}:${VERSION}"
echo "   (run 'docker login ghcr.io' / 'docker login quay.io' first)"
docker push "${IMAGE}:${VERSION}"

echo ">> Done. If you used a registry other than ghcr.io/chrisa142857/cellpheno-postproc,"
echo "   update the container line in modules/local/cellpheno/*/main.nf to match."
