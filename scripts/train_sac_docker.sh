#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTEPS="${SAC_TIMESTEPS:-10000}"
HEADLESS_FLAG="${SAC_HEADLESS_FLAG:---headless}"
DEVICE="${SAC_DEVICE:-cpu}"
BUFFER_SIZE="${SAC_BUFFER_SIZE:-50000}"

export ISAAC_RENDERER="${ISAAC_RENDERER:-RayTracedLighting}"
export ISAAC_CAMERA_FLAG="${ISAAC_CAMERA_FLAG:---disable-camera}"

docker rm -f lunar-rocket-sac-train >/dev/null 2>&1 || true

IMAGE="${ISAAC_SIM_IMAGE:-nvcr.io/nvidia/isaac-sim:6.0.1}"
CACHE_ROOT="${ISAAC_DOCKER_CACHE:-$HOME/docker/isaac-sim}"

mkdir -p \
  "$PROJECT_ROOT/outputs/models" \
  "$CACHE_ROOT/cache/ov" \
  "$CACHE_ROOT/cache/pip" \
  "$CACHE_ROOT/cache/glcache" \
  "$CACHE_ROOT/cache/kit" \
  "$CACHE_ROOT/data" \
  "$CACHE_ROOT/documents" \
  "$CACHE_ROOT/rl_deps_py312"

chmod -R ugo+rwX "$CACHE_ROOT/rl_deps_py312"

docker run --rm \
  --name lunar-rocket-sac-train \
  --entrypoint /bin/bash \
  --gpus all \
  --network=host \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e ACCEPT_EULA=Y \
  -e PRIVACY_CONSENT=Y \
  -e PYTHONPATH=/workspace/LunarRocket \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v "$PROJECT_ROOT:/workspace/LunarRocket" \
  -v "$CACHE_ROOT/cache/ov:/root/.cache/ov" \
  -v "$CACHE_ROOT/cache/pip:/root/.cache/pip" \
  -v "$CACHE_ROOT/cache/glcache:/root/.cache/nvidia" \
  -v "$CACHE_ROOT/cache/kit:/root/.cache/kit" \
  -v "$CACHE_ROOT/data:/root/.local/share/ov" \
  -v "$CACHE_ROOT/documents:/root/Documents" \
  -v "$CACHE_ROOT/rl_deps_py312:/workspace/isaac_rl_deps" \
  -w /workspace/LunarRocket \
  "$IMAGE" \
  -lc "umask 000 && /workspace/LunarRocket/scripts/isaac_train_entrypoint.sh $HEADLESS_FLAG --timesteps '$TIMESTEPS' --device '$DEVICE' --buffer-size '$BUFFER_SIZE'"
