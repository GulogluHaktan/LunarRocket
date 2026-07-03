#!/usr/bin/env bash
set -euo pipefail

IMAGE="${ISAAC_SIM_IMAGE:-nvcr.io/nvidia/isaac-sim:6.0.1}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_ROOT="${ISAAC_DOCKER_CACHE:-$HOME/docker/isaac-sim}"
OUTPUT="${1:-outputs/videos/isaac_lunar_landing_real.mp4}"
FRAMES="${ISAAC_RENDER_FRAMES:-120}"
FPS="${ISAAC_RENDER_FPS:-24}"
WIDTH="${ISAAC_RENDER_WIDTH:-960}"
HEIGHT="${ISAAC_RENDER_HEIGHT:-540}"
RENDERER="${ISAAC_RENDERER:-RayTracedLighting}"
TERRAIN_RESOLUTION="${ISAAC_TERRAIN_RESOLUTION:-96}"
HEADLESS="${ISAAC_RENDER_HEADLESS:-1}"
DOCKER_TTY=()
if [ -t 0 ]; then
  DOCKER_TTY=(-it)
fi
DOCKER_DISPLAY_ARGS=()
ISAAC_RENDER_ARGS=()
DOCKER_USER_ARGS=()
CONTAINER_RUN_PREFIX=""
if [ "$HEADLESS" = "0" ]; then
  DOCKER_DISPLAY_ARGS=(
    -e DISPLAY="${DISPLAY:-:0}"
    -e WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-}"
    -e XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
    -e QT_X11_NO_MITSHM=1
    -e NVIDIA_DRIVER_CAPABILITIES=all
    -v /tmp/.X11-unix:/tmp/.X11-unix
    -v "${XDG_RUNTIME_DIR:-/run/user/$(id -u)}:${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
  )
  ISAAC_RENDER_ARGS=(--no-headless)
fi

mkdir -p \
  "$PROJECT_ROOT/assets/rocket" \
  "$PROJECT_ROOT/assets/moon_heightmaps" \
  "$PROJECT_ROOT/outputs/videos" \
  "$CACHE_ROOT/cache/ov" \
  "$CACHE_ROOT/cache/pip" \
  "$CACHE_ROOT/cache/glcache" \
  "$CACHE_ROOT/cache/kit" \
  "$CACHE_ROOT/data" \
  "$CACHE_ROOT/documents"

chmod ugo+rwx "$PROJECT_ROOT/assets/rocket" "$PROJECT_ROOT/assets/moon_heightmaps" "$PROJECT_ROOT/outputs/videos"

docker run --rm "${DOCKER_TTY[@]}" \
  --name lunar-rocket-isaac-render \
  --entrypoint /bin/bash \
  "${DOCKER_USER_ARGS[@]}" \
  --gpus all \
  --network=host \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e ACCEPT_EULA=Y \
  -e PRIVACY_CONSENT=Y \
  -e PYTHONPATH=/workspace/LunarRocket \
  "${DOCKER_DISPLAY_ARGS[@]}" \
  -v "$PROJECT_ROOT:/workspace/LunarRocket" \
  -v "$CACHE_ROOT/cache/ov:/root/.cache/ov" \
  -v "$CACHE_ROOT/cache/pip:/root/.cache/pip" \
  -v "$CACHE_ROOT/cache/glcache:/root/.cache/nvidia" \
  -v "$CACHE_ROOT/cache/kit:/root/.cache/kit" \
  -v "$CACHE_ROOT/data:/root/.local/share/ov" \
  -v "$CACHE_ROOT/documents:/root/Documents" \
  -w /workspace/LunarRocket \
  "$IMAGE" \
  -lc "umask 000 && cd /isaac-sim && $CONTAINER_RUN_PREFIX ./python.sh /workspace/LunarRocket/app/isaac_render_video.py --renderer '$RENDERER' --frames '$FRAMES' --fps '$FPS' --width '$WIDTH' --height '$HEIGHT' --terrain-resolution '$TERRAIN_RESOLUTION' --output '/workspace/LunarRocket/$OUTPUT' ${ISAAC_RENDER_ARGS[*]}"

FRAME_DIR="$PROJECT_ROOT/outputs/videos/_frames_isaac_real"
if ! compgen -G "$FRAME_DIR/frame_*.png" >/dev/null; then
  echo "Isaac did not write any viewport frames to: $FRAME_DIR" >&2
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is required on the host to encode the real Isaac frames into MP4." >&2
  exit 1
fi

ffmpeg -y \
  -framerate "$FPS" \
  -i "$FRAME_DIR/frame_%04d.png" \
  -c:v libx264 \
  -pix_fmt yuv420p \
  -movflags +faststart \
  "$PROJECT_ROOT/$OUTPUT"

echo "[LunarRocket] real Isaac Sim MP4 written: $PROJECT_ROOT/$OUTPUT"
