#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB_CACHE="${ISAACLAB_CACHE:-$HOME/docker/isaac-lab}"
ISAACLAB_HOST_ROOT="${ISAACLAB_HOST_ROOT:-$ISAACLAB_CACHE/IsaacLab}"
ISAACLAB_REPO="${ISAACLAB_REPO:-https://github.com/isaac-sim/IsaacLab.git}"
ISAACLAB_REF="${ISAACLAB_REF:-develop}"
IMAGE="${ISAAC_SIM_IMAGE:-nvcr.io/nvidia/isaac-sim:6.0.1}"
LAB_IMAGE="${ISAACLAB_IMAGE:-lunar-rocket-isaaclab:6.0.1}"

TASK="${ISAACLAB_TASK:-LunarRocket-Lander-Direct-v0}"
NUM_ENVS="${ISAACLAB_NUM_ENVS:-1}"
DEVICE="${ISAACLAB_DEVICE:-cuda:0}"
CHECKPOINT="${ISAACLAB_CHECKPOINT:-}"
HEADLESS_FLAG="${ISAACLAB_HEADLESS_FLAG-}"
VIDEO_FLAG="${ISAACLAB_VIDEO_FLAG-}"
VIDEO_LENGTH="${ISAACLAB_VIDEO_LENGTH:-600}"
PREVIEW_DURATION="${ISAACLAB_PREVIEW_DURATION:-120}"
VIZ_FLAG="${ISAACLAB_VIZ_FLAG---viz kit}"
INSTALL_MODE="${ISAACLAB_INSTALL:-0}"
TERRAIN_QUALITY="${ISAACLAB_TERRAIN_QUALITY:-high_detail_landing}"
TERRAIN_LOCAL_RESOLUTION="${ISAACLAB_TERRAIN_LOCAL_RESOLUTION:-512}"
WATCH_SPAWN_ALTITUDE="${ISAACLAB_WATCH_SPAWN_ALTITUDE:-2.5}"
WATCH_THROTTLE="${ISAACLAB_WATCH_THROTTLE:-0.09}"
RL_DEPS_ROOT="${ISAAC_DOCKER_CACHE:-$HOME/docker/isaac-sim}/rl_deps_py312"
XHOST_GRANTED="${ISAACLAB_XHOST_GRANTED:-0}"

cleanup_xhost() {
  if [[ "$XHOST_GRANTED" == "1" ]] && command -v xhost >/dev/null 2>&1; then
    xhost -SI:localuser:root >/dev/null 2>&1 || true
  fi
}
trap cleanup_xhost EXIT

mkdir -p "$ISAACLAB_CACHE"
if [[ ! -d "$ISAACLAB_HOST_ROOT/.git" ]]; then
  git clone --depth 1 --branch "$ISAACLAB_REF" "$ISAACLAB_REPO" "$ISAACLAB_HOST_ROOT"
else
  git -C "$ISAACLAB_HOST_ROOT" fetch --depth 1 origin "$ISAACLAB_REF"
  git -C "$ISAACLAB_HOST_ROOT" checkout FETCH_HEAD
fi

if ! docker image inspect "$LAB_IMAGE" >/dev/null 2>&1; then
  docker build \
    --build-arg ISAAC_SIM_IMAGE="$IMAGE" \
    -t "$LAB_IMAGE" \
    -f "$PROJECT_ROOT/docker/Dockerfile.isaaclab" \
    "$PROJECT_ROOT"
fi

docker rm -f lunar-rocket-isaaclab-watch >/dev/null 2>&1 || true

DOCKER_ARGS=(
  --rm
  --name lunar-rocket-isaaclab-watch
  --entrypoint /bin/bash
  --user root
  --gpus all
  --network=host
  --ipc=host
  --ulimit memlock=-1
  --ulimit stack=67108864
  -e ACCEPT_EULA=Y
  -e PRIVACY_CONSENT=Y
  -e TERM=xterm
  -e PYTHONUNBUFFERED=1
  -e NVIDIA_DRIVER_CAPABILITIES=all
  -e QT_X11_NO_MITSHM=1
  -e ISAACLAB_ROOT=/workspace/IsaacLab
  -e LUNAR_ROCKET_ROOT=/workspace/LunarRocket
  -e INSTALL_MODE="$INSTALL_MODE"
  -e SKIP_TRAIN=1
  -e ISAACLAB_TERRAIN_QUALITY="$TERRAIN_QUALITY"
  -e ISAACLAB_TERRAIN_LOCAL_RESOLUTION="$TERRAIN_LOCAL_RESOLUTION"
  -e ISAACLAB_WATCH_SPAWN_ALTITUDE="$WATCH_SPAWN_ALTITUDE"
  -e PYTHONPATH=/workspace/isaac_rl_deps/site-packages:/workspace/IsaacLab/source/isaaclab:/workspace/IsaacLab/source/isaaclab_assets:/workspace/IsaacLab/source/isaaclab_tasks:/workspace/IsaacLab/source/isaaclab_rl:/workspace/IsaacLab/source/isaaclab_contrib:/workspace/IsaacLab/source/isaaclab_newton:/workspace/IsaacLab/source/isaaclab_physx:/workspace/IsaacLab/source/isaaclab_ov:/workspace/IsaacLab/source/isaaclab_ovphysx:/workspace/IsaacLab/source/isaaclab_visualizers:/workspace/LunarRocket/source/lunar_rocket_lab:/workspace/LunarRocket
  -v "$PROJECT_ROOT:/workspace/LunarRocket"
  -v "$ISAACLAB_HOST_ROOT:/workspace/IsaacLab"
  -v "$RL_DEPS_ROOT:/workspace/isaac_rl_deps"
  -w /workspace/LunarRocket
)

if [[ -n "${DISPLAY:-}" ]]; then
  if [[ -z "${XAUTHORITY:-}" ]] && command -v xhost >/dev/null 2>&1; then
    xhost +SI:localuser:root >/dev/null
    XHOST_GRANTED=1
  fi

  DOCKER_ARGS+=(
    -e DISPLAY="$DISPLAY"
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw
  )
fi

if [[ -n "${XAUTHORITY:-}" && -f "${XAUTHORITY:-}" ]]; then
  DOCKER_ARGS+=(
    -e XAUTHORITY=/tmp/.docker.xauth
    -v "$XAUTHORITY:/tmp/.docker.xauth:ro"
  )
fi

if [[ -n "$CHECKPOINT" ]]; then
  PLAY_ARGS=(
    /workspace/IsaacLab/isaaclab.sh -p source/lunar_rocket_lab/scripts/play_sac.py
    --task "$TASK"
    --agent sb3_sac_cfg_entry_point
    --num_envs "$NUM_ENVS"
    --device "$DEVICE"
    --video_length "$VIDEO_LENGTH"
  )
  PLAY_ARGS+=(--checkpoint "$CHECKPOINT")
else
  PLAY_ARGS=(
    /workspace/IsaacLab/isaaclab.sh -p source/lunar_rocket_lab/scripts/preview_scene.py
    --task "$TASK"
    --agent sb3_sac_cfg_entry_point
    --num_envs "$NUM_ENVS"
    --device "$DEVICE"
    --duration "$PREVIEW_DURATION"
    --spawn_altitude "$WATCH_SPAWN_ALTITUDE"
    --throttle "$WATCH_THROTTLE"
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

docker run "${DOCKER_ARGS[@]}" "$LAB_IMAGE" -lc "
  /workspace/LunarRocket/scripts/isaaclab_train_entrypoint.sh >/tmp/lunar_rocket_watch_deps.log &&
  cd /workspace/LunarRocket &&
  ${PLAY_ARGS[*]}
"
