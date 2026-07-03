#!/usr/bin/env bash
set -euo pipefail

IMAGE="${ISAAC_SIM_IMAGE:-nvcr.io/nvidia/isaac-sim:6.0.1}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_ROOT="${ISAAC_DOCKER_CACHE:-$HOME/docker/isaac-sim}"
WIDTH="${ISAAC_SCREEN_WIDTH:-1280}"
HEIGHT="${ISAAC_SCREEN_HEIGHT:-720}"
FPS="${ISAAC_SCREEN_FPS:-24}"
DURATION="${ISAAC_LIVE_DURATION:-120}"
RENDERER="${ISAAC_RENDERER:-HydraStorm}"
EXPERIENCE="${ISAAC_EXPERIENCE:-/isaac-sim/apps/isaacsim.exp.full.kit}"
TERRAIN_RESOLUTION="${ISAAC_TERRAIN_RESOLUTION:-256}"
LANDING_RESOLUTION="${ISAAC_LANDING_RESOLUTION:-512}"
LANDING_SIZE="${ISAAC_LANDING_SIZE:-24}"
SIMULATE_PHYSICS="${ISAAC_SIMULATE_PHYSICS:-0}"
PHYSICS_ARG=""
if [[ "$SIMULATE_PHYSICS" == "1" || "$SIMULATE_PHYSICS" == "true" ]]; then
  PHYSICS_ARG="--simulate-physics"
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

docker run --rm \
  --name lunar-rocket-isaac-live-demo \
  --entrypoint /bin/bash \
  --gpus all \
  --network=host \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e ACCEPT_EULA=Y \
  -e PRIVACY_CONSENT=Y \
  -e PYTHONPATH=/workspace/LunarRocket \
  -e DISPLAY="${DISPLAY:-:0}" \
  -e WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-}" \
  -e XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}" \
  -e QT_X11_NO_MITSHM=1 \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "${XDG_RUNTIME_DIR:-/run/user/$(id -u)}:${XDG_RUNTIME_DIR:-/run/user/$(id -u)}" \
  -v "$PROJECT_ROOT:/workspace/LunarRocket" \
  -v "$CACHE_ROOT/cache/ov:/root/.cache/ov" \
  -v "$CACHE_ROOT/cache/pip:/root/.cache/pip" \
  -v "$CACHE_ROOT/cache/glcache:/root/.cache/nvidia" \
  -v "$CACHE_ROOT/cache/kit:/root/.cache/kit" \
  -v "$CACHE_ROOT/data:/root/.local/share/ov" \
  -v "$CACHE_ROOT/documents:/root/Documents" \
  -w /workspace/LunarRocket \
  "$IMAGE" \
  -lc "umask 000 && cd /isaac-sim && ./python.sh /workspace/LunarRocket/app/isaac_live_demo.py --renderer '$RENDERER' --experience '$EXPERIENCE' --duration '$DURATION' --fps '$FPS' --width '$WIDTH' --height '$HEIGHT' --terrain-resolution '$TERRAIN_RESOLUTION' --landing-resolution '$LANDING_RESOLUTION' --landing-size '$LANDING_SIZE' $PHYSICS_ARG"
