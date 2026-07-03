#!/usr/bin/env bash
set -euo pipefail

IMAGE="${ISAAC_SIM_IMAGE:-nvcr.io/nvidia/isaac-sim:6.0.1}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_ROOT="${ISAAC_DOCKER_CACHE:-$HOME/docker/isaac-sim}"
STEPS="${1:-10}"
RENDERER="${ISAAC_RENDERER:-None}"
CAMERA_FLAG="${ISAAC_CAMERA_FLAG:---disable-camera}"
DOCKER_TTY=()
if [ -t 0 ]; then
  DOCKER_TTY=(-it)
fi

mkdir -p \
  "$PROJECT_ROOT/assets/rocket" \
  "$PROJECT_ROOT/assets/moon_heightmaps" \
  "$CACHE_ROOT/cache/ov" \
  "$CACHE_ROOT/cache/pip" \
  "$CACHE_ROOT/cache/glcache" \
  "$CACHE_ROOT/cache/kit" \
  "$CACHE_ROOT/data" \
  "$CACHE_ROOT/documents"

chmod ugo+rwx "$PROJECT_ROOT/assets/rocket" "$PROJECT_ROOT/assets/moon_heightmaps"

docker run --rm "${DOCKER_TTY[@]}" \
  --name lunar-rocket-isaac \
  --entrypoint /bin/bash \
  --gpus all \
  --network=host \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e ACCEPT_EULA=Y \
  -e PRIVACY_CONSENT=Y \
  -e PYTHONPATH=/workspace/LunarRocket \
  -v "$PROJECT_ROOT:/workspace/LunarRocket" \
  -v "$CACHE_ROOT/cache/ov:/root/.cache/ov" \
  -v "$CACHE_ROOT/cache/pip:/root/.cache/pip" \
  -v "$CACHE_ROOT/cache/glcache:/root/.cache/nvidia" \
  -v "$CACHE_ROOT/cache/kit:/root/.cache/kit" \
  -v "$CACHE_ROOT/data:/root/.local/share/ov" \
  -v "$CACHE_ROOT/documents:/root/Documents" \
  -w /workspace/LunarRocket \
  "$IMAGE" \
  -lc "umask 000 && cd /isaac-sim && ./python.sh /workspace/LunarRocket/app/main.py --headless --renderer '$RENDERER' $CAMERA_FLAG --no-render-step --steps '$STEPS'"
