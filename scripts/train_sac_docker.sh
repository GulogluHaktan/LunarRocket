#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTEPS="${SAC_TIMESTEPS:-10000}"
HEADLESS_FLAG="${SAC_HEADLESS_FLAG:---headless}"
DEVICE="${SAC_DEVICE:-cpu}"
BUFFER_SIZE="${SAC_BUFFER_SIZE:-50000}"
PROGRESS_FREQ="${SAC_PROGRESS_FREQ:-1000}"
CHECKPOINT_FREQ="${SAC_CHECKPOINT_FREQ:-10000}"
CHECKPOINT_DIR="${SAC_CHECKPOINT_DIR:-outputs/models/checkpoints}"
ANALYTIC_ENV="${SAC_ANALYTIC_ENV:-0}"
if [[ "$ANALYTIC_ENV" == "1" || "$ANALYTIC_ENV" == "true" ]]; then
  FAST_TERRAIN="${SAC_FAST_TERRAIN:-0}"
else
  FAST_TERRAIN="${SAC_FAST_TERRAIN:-1}"
fi
VISUAL_DETAIL_FAST_PHYSICS="${SAC_VISUAL_DETAIL_FAST_PHYSICS:-0}"
TERRAIN_QUALITY="${SAC_TERRAIN_QUALITY:-}"
LOCAL_DETAIL_RESOLUTION="${SAC_LOCAL_DETAIL_RESOLUTION:-}"
PHYSICS_DT="${SAC_PHYSICS_DT:-}"
ANALYTIC_CONTROL_DT="${SAC_ANALYTIC_CONTROL_DT:-0.05}"
CLOSE_APP="${SAC_CLOSE_APP:-0}"
SKIP_CLOSE_FLAG="--skip-close"
if [[ "$CLOSE_APP" == "1" || "$CLOSE_APP" == "true" ]]; then
  SKIP_CLOSE_FLAG=""
fi
FAST_TERRAIN_FLAG=""
if [[ "$FAST_TERRAIN" == "1" || "$FAST_TERRAIN" == "true" ]]; then
  FAST_TERRAIN_FLAG="--fast-terrain"
fi
VISUAL_DETAIL_FAST_PHYSICS_FLAG=""
if [[ "$VISUAL_DETAIL_FAST_PHYSICS" == "1" || "$VISUAL_DETAIL_FAST_PHYSICS" == "true" ]]; then
  VISUAL_DETAIL_FAST_PHYSICS_FLAG="--visual-detail-fast-physics"
fi
ANALYTIC_ENV_FLAG=""
if [[ "$ANALYTIC_ENV" == "1" || "$ANALYTIC_ENV" == "true" ]]; then
  ANALYTIC_ENV_FLAG="--analytic-env --analytic-control-dt '$ANALYTIC_CONTROL_DT'"
fi
TERRAIN_QUALITY_FLAG=""
if [[ -n "$TERRAIN_QUALITY" ]]; then
  TERRAIN_QUALITY_FLAG="--terrain-quality '$TERRAIN_QUALITY'"
fi
LOCAL_DETAIL_RESOLUTION_FLAG=""
if [[ -n "$LOCAL_DETAIL_RESOLUTION" ]]; then
  LOCAL_DETAIL_RESOLUTION_FLAG="--local-detail-resolution '$LOCAL_DETAIL_RESOLUTION'"
fi
PHYSICS_DT_FLAG=""
if [[ -n "$PHYSICS_DT" ]]; then
  PHYSICS_DT_FLAG="--physics-dt '$PHYSICS_DT'"
fi

export ISAAC_RENDERER="${ISAAC_RENDERER:-RayTracedLighting}"
export ISAAC_CAMERA_FLAG="${ISAAC_CAMERA_FLAG:---disable-camera}"

docker rm -f lunar-rocket-sac-train >/dev/null 2>&1 || true

IMAGE="${ISAAC_SIM_IMAGE:-nvcr.io/nvidia/isaac-sim:6.0.1}"
CACHE_ROOT="${ISAAC_DOCKER_CACHE:-$HOME/docker/isaac-sim}"

mkdir -p \
  "$PROJECT_ROOT/outputs/models" \
  "$PROJECT_ROOT/$CHECKPOINT_DIR" \
  "$CACHE_ROOT/cache/ov" \
  "$CACHE_ROOT/cache/pip" \
  "$CACHE_ROOT/cache/glcache" \
  "$CACHE_ROOT/cache/kit" \
  "$CACHE_ROOT/data" \
  "$CACHE_ROOT/documents" \
  "$CACHE_ROOT/rl_deps_py312"

chmod 0777 "$PROJECT_ROOT/outputs" "$PROJECT_ROOT/outputs/models" "$PROJECT_ROOT/$CHECKPOINT_DIR"

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
  -e PYTHONUNBUFFERED=1 \
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
  -lc "umask 000 && /workspace/LunarRocket/scripts/isaac_train_entrypoint.sh $HEADLESS_FLAG $SKIP_CLOSE_FLAG $FAST_TERRAIN_FLAG $VISUAL_DETAIL_FAST_PHYSICS_FLAG $ANALYTIC_ENV_FLAG $TERRAIN_QUALITY_FLAG $LOCAL_DETAIL_RESOLUTION_FLAG $PHYSICS_DT_FLAG --timesteps '$TIMESTEPS' --device '$DEVICE' --buffer-size '$BUFFER_SIZE' --progress-freq '$PROGRESS_FREQ' --checkpoint-freq '$CHECKPOINT_FREQ' --checkpoint-dir '$CHECKPOINT_DIR'"
