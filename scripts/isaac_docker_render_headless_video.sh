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

echo "[LunarRocket] Running headless Isaac Sim container to render frames..."

docker run --rm \
  --name lunar-rocket-isaac-render-headless \
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
  -lc "umask 000 && cd /isaac-sim && ./python.sh /workspace/LunarRocket/app/isaac_render_headless_video.py --renderer '$RENDERER' --frames '$FRAMES' --fps '$FPS' --width '$WIDTH' --height '$HEIGHT' --terrain-resolution '$TERRAIN_RESOLUTION' --output '/workspace/LunarRocket/$OUTPUT'"

FRAME_DIR="$PROJECT_ROOT/outputs/videos/_frames_isaac_real"
if ! compgen -G "$FRAME_DIR/frame_*.png" >/dev/null; then
  echo "Isaac did not write any viewport frames to: $FRAME_DIR" >&2
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is required on the host to encode the real Isaac frames into MP4." >&2
  exit 1
fi

echo "[LunarRocket] Encoding frames with ffmpeg..."
ffmpeg -y \
  -framerate "$FPS" \
  -i "$FRAME_DIR/frame_%04d.png" \
  -c:v libx264 \
  -pix_fmt yuv420p \
  -movflags +faststart \
  "$PROJECT_ROOT/$OUTPUT"

echo "[LunarRocket] Headless Isaac Sim MP4 successfully written: $PROJECT_ROOT/$OUTPUT"
