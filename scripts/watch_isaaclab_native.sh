#!/usr/bin/env bash
# Native (non-Docker) equivalent of watch_isaaclab_docker.sh: opens the
# lightweight interactive watch scene, or plays back a trained checkpoint,
# directly on the host using the venv set up by ./scripts/install.sh --native.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LUNAR_NATIVE_ROOT="${LUNAR_NATIVE_ROOT:-$HOME/isaac-lab-native}"
VENV_DIR="${LUNAR_NATIVE_VENV:-$LUNAR_NATIVE_ROOT/env_isaaclab}"
ISAACLAB_HOST_ROOT="${ISAACLAB_HOST_ROOT:-$LUNAR_NATIVE_ROOT/IsaacLab}"

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "[LunarRocket] native environment not found at $VENV_DIR" >&2
  echo "[LunarRocket] run ./scripts/install.sh --native first" >&2
  exit 1
fi
if [[ ! -x "$ISAACLAB_HOST_ROOT/isaaclab.sh" ]]; then
  echo "[LunarRocket] Isaac Lab not found at $ISAACLAB_HOST_ROOT" >&2
  echo "[LunarRocket] run ./scripts/install.sh --native first" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

TASK="${ISAACLAB_TASK:-LunarRocket-Lander-Direct-v0}"
NUM_ENVS="${ISAACLAB_NUM_ENVS:-1}"
DEVICE="${ISAACLAB_DEVICE:-cuda:0}"
CHECKPOINT="${ISAACLAB_CHECKPOINT:-}"
HEADLESS_FLAG="${ISAACLAB_HEADLESS_FLAG-}"
VIDEO_FLAG="${ISAACLAB_VIDEO_FLAG-}"
VIDEO_LENGTH="${ISAACLAB_VIDEO_LENGTH:-600}"
EVALUATION_STEPS="${ISAACLAB_EVALUATION_STEPS:-}"
PREVIEW_DURATION="${ISAACLAB_PREVIEW_DURATION:-120}"
VIZ_FLAG="${ISAACLAB_VIZ_FLAG---viz kit}"
WATCH_SPAWN_ALTITUDE="${ISAACLAB_WATCH_SPAWN_ALTITUDE:-2.5}"
WATCH_THROTTLE="${ISAACLAB_WATCH_THROTTLE:-0.09}"
WATCH_CONTROLLER="${ISAACLAB_WATCH_CONTROLLER:-constant}"

export ISAACLAB_TERRAIN_QUALITY="${ISAACLAB_TERRAIN_QUALITY:-high_train}"
export ISAACLAB_TERRAIN_LOCAL_RESOLUTION="${ISAACLAB_TERRAIN_LOCAL_RESOLUTION:-256}"
export ISAACLAB_TERRAIN_LOCAL_DETAIL="${ISAACLAB_TERRAIN_LOCAL_DETAIL:-0}"
export ISAACLAB_USE_DEM_TERRAIN="${ISAACLAB_USE_DEM_TERRAIN:-1}"
export ISAACLAB_TERRAIN_POOL_SIZE="${ISAACLAB_TERRAIN_POOL_SIZE:-15}"
export ISAACLAB_TERRAIN_POOL_REFRESH_ASSIGNMENTS="${ISAACLAB_TERRAIN_POOL_REFRESH_ASSIGNMENTS:-60}"
export ISAACLAB_TERRAIN_POOL_USE_DEM="${ISAACLAB_TERRAIN_POOL_USE_DEM:-1}"
export ISAACLAB_TERRAIN_POOL_DEM_QUALITY="${ISAACLAB_TERRAIN_POOL_DEM_QUALITY:-low_debug}"
export LUNAR_ROCKET_ROOT="$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export ACCEPT_EULA="${ACCEPT_EULA:-Y}"
export PRIVACY_CONSENT="${PRIVACY_CONSENT:-Y}"

cd "$PROJECT_ROOT"

if [[ -n "$CHECKPOINT" ]]; then
  PLAY_ARGS=(
    "$ISAACLAB_HOST_ROOT/isaaclab.sh" -p source/lunar_rocket_lab/scripts/play_sac.py
    --task "$TASK"
    --agent sb3_sac_cfg_entry_point
    --num_envs "$NUM_ENVS"
    --device "$DEVICE"
    --video_length "$VIDEO_LENGTH"
    --checkpoint "$CHECKPOINT"
  )
  if [[ -n "$EVALUATION_STEPS" ]]; then
    PLAY_ARGS+=(--evaluation_steps "$EVALUATION_STEPS")
  fi
else
  PLAY_ARGS=(
    "$ISAACLAB_HOST_ROOT/isaaclab.sh" -p source/lunar_rocket_lab/scripts/preview_scene.py
    --task "$TASK"
    --agent sb3_sac_cfg_entry_point
    --num_envs "$NUM_ENVS"
    --device "$DEVICE"
    --duration "$PREVIEW_DURATION"
    --spawn_altitude "$WATCH_SPAWN_ALTITUDE"
    --throttle "$WATCH_THROTTLE"
    --controller "$WATCH_CONTROLLER"
  )
fi
if [[ -n "$HEADLESS_FLAG" ]]; then
  PLAY_ARGS+=($HEADLESS_FLAG)
fi
if [[ -n "$VIZ_FLAG" ]]; then
  PLAY_ARGS+=($VIZ_FLAG)
fi
if [[ -n "$VIDEO_FLAG" ]]; then
  PLAY_ARGS+=($VIDEO_FLAG)
fi

echo "[LunarRocket] watching natively: task=$TASK num_envs=$NUM_ENVS device=$DEVICE"
exec "${PLAY_ARGS[@]}"
