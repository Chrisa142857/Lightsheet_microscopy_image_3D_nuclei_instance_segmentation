#!/usr/bin/env bash
#
# Set up a local GPU server to finish the CellPheno NIS nf-core contribution
# (build/push the container, lint, fork/PR). Installs the toolchain and checks
# prerequisites. Credentials (quay.io / ghcr.io / GitHub) are configured
# interactively at the end — they are never stored by this script.
#
# Usage:  bash scripts/setup_local_gpu.sh
set -euo pipefail

echo "==> Checking Docker + NVIDIA GPU runtime"
docker info >/dev/null 2>&1 || { echo "ERROR: Docker daemon not usable"; exit 1; }
if ! docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
    echo "WARN: 'docker run --gpus all' failed — install the NVIDIA Container Toolkit:"
    echo "      https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
fi

echo "==> Installing nf-core tools, gdown (pip)"
python3 -m pip install --quiet --upgrade nf-core gdown

echo "==> Installing Nextflow (if missing)"
command -v nextflow >/dev/null 2>&1 || { curl -s https://get.nextflow.io | bash && sudo mv nextflow /usr/local/bin/; }

echo "==> Installing nf-test (if missing)"
command -v nf-test >/dev/null 2>&1 || { curl -fsSL https://code.askimed.com/install/nf-test | bash && sudo mv nf-test /usr/local/bin/ 2>/dev/null || true; }

echo "==> Checking GitHub CLI"
command -v gh >/dev/null 2>&1 || echo "WARN: install gh (https://cli.github.com) for the fork/PR step"

cat <<'NEXT'

==> Toolchain ready. Now authenticate (these prompt for / read your stored creds):

    docker login quay.io          # if you have nf-core org push access
    docker login ghcr.io          # your GitHub container registry (always works)
    gh auth login                 # for the fork + PR

Then run the contribution steps (see docs/local_gpu_run.md), or launch Claude
Code in this repo and tell it: "complete the CellPheno NIS nf-core submission
per docs/local_gpu_run.md".
NEXT
