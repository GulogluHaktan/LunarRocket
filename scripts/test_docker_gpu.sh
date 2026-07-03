#!/usr/bin/env bash
set -euo pipefail

docker run --rm --gpus all nvidia/cuda:13.0.2-base-ubuntu24.04 nvidia-smi
