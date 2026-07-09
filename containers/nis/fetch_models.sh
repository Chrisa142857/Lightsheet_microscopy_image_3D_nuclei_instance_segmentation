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
echo "   nis_unet_cpu.pt  grad_2Dto3D.pt  gnn_message_passing.pt"
echo "   gnn_classifier.pt  flow_3DtoSeed.pt"
echo ">> These are device-independent: the binary map_locations them onto whatever"
echo "   --device you pass (cuda:0, cuda:1, cuda:2, ... or cpu)."
echo ">> If your G-drive copy still has device-suffixed names (e.g. grad_2Dto3D_cuda:0.pt),"
echo "   rename them by dropping the _<device> suffix (keep one copy of each)."
