#!/usr/bin/env bash
#
# Fetch the TorchScript model weights the CellPheno NIS binary loads at runtime.
# These are NOT baked into the container; they are passed to the module as the
# `models` input (and are needed to prepare real test data for nf-core).
#
# Source: cpp/README.md -> G-drive model folder.
#
# Usage:
#   pip install gdown
#   bash containers/nis/fetch_models.sh [DEST_DIR]
set -euo pipefail

DEST="${1:-downloads/resource}"
# G-drive folder with the NIS .pt models (see cpp/README.md).
GDRIVE_MODELS_FOLDER="https://drive.google.com/drive/folders/12YGRtoW4DHftVyhaGoZMl-xdc02Mj9SB"

mkdir -p "${DEST}"
echo ">> Downloading NIS models into ${DEST}"
gdown --folder "${GDRIVE_MODELS_FOLDER}" -O "${DEST}"

echo ">> Expected files (loaded by cpp/main.cpp):"
echo "   nis_unet_cpu.pt  grad_2Dto3D_<device>.pt  gnn_message_passing_<device>.pt"
echo "   gnn_classifier_<device>.pt  flow_3DtoSeed.pt"
echo ">> Note: only cuda:0 and cuda:1 device variants are provided."
