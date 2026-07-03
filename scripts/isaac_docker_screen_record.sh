#!/usr/bin/env bash
set -euo pipefail

IMAGE="${ISAAC_SIM_IMAGE:-nvcr.io/nvidia/isaac-sim:6.0.1}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_ROOT="${ISAAC_DOCKER_CACHE:-$HOME/docker/isaac-sim}"
OUTPUT="${1:-outputs/videos/lunar_landing_screen_record.mp4}"
WIDTH="${ISAAC_SCREEN_WIDTH:-1280}"
HEIGHT="${ISAAC_SCREEN_HEIGHT:-720}"
FPS="${ISAAC_SCREEN_FPS:-24}"
DURATION="${ISAAC_SCREEN_DURATION:-5}"
RENDERER="${ISAAC_RENDERER:-HydraStorm}"
TERRAIN_RESOLUTION="${ISAAC_TERRAIN_RESOLUTION:-256}"
CONTAINER_NAME="lunar-rocket-isaac-live-record"
LOG_PATH="$PROJECT_ROOT/outputs/videos/live_record_isaac.log"

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

cleanup() {
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
rm -f "$PROJECT_ROOT/$OUTPUT" "$LOG_PATH"

echo "[LunarRocket] Starting live Isaac Sim window for screen recording..."
docker run --rm \
  --name "$CONTAINER_NAME" \
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
  -lc "umask 000 && cd /isaac-sim && ./python.sh /workspace/LunarRocket/app/isaac_live_demo.py --renderer '$RENDERER' --duration 12 --fps '$FPS' --width '$WIDTH' --height '$HEIGHT' --terrain-resolution '$TERRAIN_RESOLUTION'" \
  >"$LOG_PATH" 2>&1 &

echo "[LunarRocket] Waiting for Isaac scene to become ready..."
sleep 2
for _ in $(seq 1 90); do
  if grep -q "live Isaac scene ready" "$LOG_PATH" 2>/dev/null; then
    break
  fi
  if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    if [ -s "$LOG_PATH" ]; then
      echo "[LunarRocket] Isaac container exited before recording. Log:"
      tail -n 80 "$LOG_PATH" || true
      exit 1
    fi
  fi
  sleep 1
done

if ! grep -q "live Isaac scene ready" "$LOG_PATH" 2>/dev/null; then
  echo "[LunarRocket] Isaac did not report readiness in time. Log:"
  tail -n 120 "$LOG_PATH" || true
  exit 1
fi

echo "[LunarRocket] Recording ${DURATION}s from display ${DISPLAY:-:0} to $PROJECT_ROOT/$OUTPUT"
ffmpeg -y \
  -f x11grab \
  -video_size "${WIDTH}x${HEIGHT}" \
  -framerate "$FPS" \
  -i "${DISPLAY:-:0}+0,0" \
  -t "$DURATION" \
  -c:v libx264 \
  -pix_fmt yuv420p \
  -movflags +faststart \
  "$PROJECT_ROOT/$OUTPUT"

echo "[LunarRocket] Screen recording written: $PROJECT_ROOT/$OUTPUT"
