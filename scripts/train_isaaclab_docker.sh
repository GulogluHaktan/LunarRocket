#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# =======================================================
# Automatic Host Dependencies & CDI Setup (Ubuntu/Debian)
# =======================================================
if [[ -f /etc/debian_version ]]; then
  # 1. Install Docker if missing
  if ! command -v docker &> /dev/null; then
    echo "[Setup] Docker is missing. Installing docker.io (requires sudo)..."
    sudo apt-get update && sudo apt-get install -y docker.io curl gpg
    sudo systemctl enable --now docker
    sudo usermod -aG docker "$USER"
    echo "[Setup] Docker installed. You may need to run 'newgrp docker' or re-login if docker commands fail."
  fi

  # 2. Install NVIDIA Container Toolkit if missing
  if ! command -v nvidia-ctk &> /dev/null; then
    echo "[Setup] nvidia-container-toolkit is missing. Setting up official repository..."
    sudo apt-get update && sudo apt-get install -y curl gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
  fi

  # 3. Generate NVIDIA CDI specification if missing
  if [[ ! -f "/etc/cdi/nvidia.yaml" ]]; then
    echo "[Setup] Generating NVIDIA CDI specification (/etc/cdi/nvidia.yaml)..."
    sudo nvidia-ctk cdi generate --output="/etc/cdi/nvidia.yaml"
    echo "[Setup] Restarting Docker to apply CDI updates..."
    sudo systemctl restart docker
  fi
fi
ISAACLAB_CACHE="${ISAACLAB_CACHE:-$HOME/docker/isaac-lab}"
ISAACLAB_HOST_ROOT="${ISAACLAB_HOST_ROOT:-$ISAACLAB_CACHE/IsaacLab}"
ISAACLAB_REPO="${ISAACLAB_REPO:-https://github.com/isaac-sim/IsaacLab.git}"
# See scripts/install.sh for why this is pinned instead of "develop".
ISAACLAB_REF="${ISAACLAB_REF:-v3.0.0-beta2.patch1}"
ISAACLAB_UPDATE="${ISAACLAB_UPDATE:-0}"
IMAGE="${ISAAC_SIM_IMAGE:-nvcr.io/nvidia/isaac-sim:6.0.1}"
LAB_IMAGE="${ISAACLAB_IMAGE:-lunar-rocket-isaaclab:6.0.1}"

TASK="${ISAACLAB_TASK:-LunarRocket-Lander-Direct-v0}"
ALGO="${ISAACLAB_ALGO:-sac}"
# train_sac.py: total_steps = MAX_ITERATIONS * steps_per_iteration(32) * NUM_ENVS.
# 1200 * 32 * 512 ~= 19.7M steps -- this task's terminal-reward-gated soft
# landing needs far more than the ~1-3M steps prior default runs completed in.
NUM_ENVS="${ISAACLAB_NUM_ENVS:-512}"
MAX_ITERATIONS="${ISAACLAB_MAX_ITERATIONS:-1200}"
DEVICE="${ISAACLAB_DEVICE:-cuda:0}"
HEADLESS_FLAG="${ISAACLAB_HEADLESS_FLAG:---headless}"
LOG_INTERVAL="${ISAACLAB_LOG_INTERVAL:-10000}"
CHECKPOINT="${ISAACLAB_CHECKPOINT:-}"
INSTALL_MODE="${ISAACLAB_INSTALL:-0}"
TERRAIN_QUALITY="${ISAACLAB_TERRAIN_QUALITY:-high_train}"
TERRAIN_LOCAL_RESOLUTION="${ISAACLAB_TERRAIN_LOCAL_RESOLUTION:-256}"
TERRAIN_LOCAL_DETAIL="${ISAACLAB_TERRAIN_LOCAL_DETAIL:-0}"
# The real USD/DEM terrain mesh is only ever spawned up to legacy_terrain_usd_max_envs
# (16) envs; at 512 envs it silently falls back to a ground plane parked at z=-100
# with zero terrain variety (all clones share env_0's static mesh). The GPU
# terrain-pool path (below) is the actual training terrain: it is env-parallel,
# randomizes per reset, and is what the reward/termination/observations use.
# Use ISAACLAB_USE_DEM_TERRAIN=1 with ISAACLAB_NUM_ENVS<=16 only for real-contact
# fine-tuning/eval after pretraining here.
USE_DEM_TERRAIN="${ISAACLAB_USE_DEM_TERRAIN:-0}"
FAST_APPROX_TERRAIN="${ISAACLAB_FAST_APPROX_TERRAIN:-0}"
EPISODE_LENGTH_S="${ISAACLAB_EPISODE_LENGTH_S:-}"
CURRICULUM_INITIAL_DIFFICULTY="${ISAACLAB_CURRICULUM_INITIAL_DIFFICULTY:-}"
TERRAIN_POOL_SIZE="${ISAACLAB_TERRAIN_POOL_SIZE:-15}"
TERRAIN_POOL_REFRESH_ASSIGNMENTS="${ISAACLAB_TERRAIN_POOL_REFRESH_ASSIGNMENTS:-60}"
TERRAIN_POOL_USE_DEM="${ISAACLAB_TERRAIN_POOL_USE_DEM:-1}"
TERRAIN_POOL_DEM_QUALITY="${ISAACLAB_TERRAIN_POOL_DEM_QUALITY:-low_debug}"
# Fine-detail fade shape for the terrain-pool path: smoothstep (shipped
# default) | gaussian | hybrid. See experiments/terrain_transition/.
TERRAIN_BLEND_METHOD="${ISAACLAB_TERRAIN_BLEND_METHOD:-smoothstep}"
TERRAIN_HYBRID_CONTACT_RADIUS_M="${ISAACLAB_TERRAIN_HYBRID_CONTACT_RADIUS_M:-1.5}"
# SAC agent seed override; empty means "use sb3_sac_cfg.yaml's own seed (42)".
SEED="${ISAACLAB_SEED:-}"
RL_DEPS_ROOT="${ISAAC_DOCKER_CACHE:-$HOME/docker/isaac-sim}/rl_deps_py312"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-source/lunar_rocket_lab/scripts/train_sac.py}"
AGENT_ENTRY="sb3_sac_cfg_entry_point"

if [[ "$ALGO" == "ppo" ]]; then
  TRAIN_SCRIPT="source/lunar_rocket_lab/scripts/train_sb3.py"
  AGENT_ENTRY="sb3_cfg_entry_point"
elif [[ "$ALGO" == "td3" ]]; then
  # A2 algorithm-comparison arm: same train_sac.py entrypoint, which selects
  # the SB3 algorithm class from the resolved agent config's "algo" key.
  AGENT_ENTRY="sb3_td3_cfg_entry_point"
elif [[ "$ALGO" == "ddpg" ]]; then
  AGENT_ENTRY="sb3_ddpg_cfg_entry_point"
elif [[ "$ALGO" != "sac" ]]; then
  echo "Unsupported ISAACLAB_ALGO='$ALGO'. Use 'sac', 'ppo', 'td3', or 'ddpg'." >&2
  exit 2
fi

mkdir -p "$ISAACLAB_CACHE"
if [[ ! -d "$ISAACLAB_HOST_ROOT/.git" ]]; then
  git clone --depth 1 --branch "$ISAACLAB_REF" "$ISAACLAB_REPO" "$ISAACLAB_HOST_ROOT"
elif [[ "$ISAACLAB_UPDATE" == "1" || "$ISAACLAB_UPDATE" == "true" ]]; then
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

docker rm -f lunar-rocket-isaaclab-train >/dev/null 2>&1 || true

docker run --rm \
  --name lunar-rocket-isaaclab-train \
  --entrypoint /bin/bash \
  --user root \
  --device nvidia.com/gpu=all \
  --network=host \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e ACCEPT_EULA=Y \
  -e PRIVACY_CONSENT=Y \
  -e TERM=xterm \
  -e PYTHONUNBUFFERED=1 \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e OMNI_KIT_ALLOW_ROOT=1 \
  -e VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
  -e VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json \
  -e __GLX_VENDOR_LIBRARY_NAME=nvidia \
  -e __NV_PRIME_RENDER_OFFLOAD=1 \
  -e ISAACLAB_ROOT=/workspace/IsaacLab \
  -e LUNAR_ROCKET_ROOT=/workspace/LunarRocket \
  -e TRAIN_SCRIPT="$TRAIN_SCRIPT" \
  -e TASK="$TASK" \
  -e AGENT_ENTRY="$AGENT_ENTRY" \
  -e NUM_ENVS="$NUM_ENVS" \
  -e MAX_ITERATIONS="$MAX_ITERATIONS" \
  -e DEVICE="$DEVICE" \
  -e LOG_INTERVAL="$LOG_INTERVAL" \
  -e CHECKPOINT="$CHECKPOINT" \
  -e HEADLESS_FLAG="$HEADLESS_FLAG" \
  -e VIDEO="${ISAACLAB_VIDEO:-0}" \
  -e VIDEO_LENGTH="${ISAACLAB_VIDEO_LENGTH:-100}" \
  -e INSTALL_MODE="$INSTALL_MODE" \
  -e ISAACLAB_TERRAIN_QUALITY="$TERRAIN_QUALITY" \
  -e ISAACLAB_TERRAIN_LOCAL_RESOLUTION="$TERRAIN_LOCAL_RESOLUTION" \
  -e ISAACLAB_TERRAIN_LOCAL_DETAIL="$TERRAIN_LOCAL_DETAIL" \
  -e ISAACLAB_USE_DEM_TERRAIN="$USE_DEM_TERRAIN" \
  -e ISAACLAB_FAST_APPROX_TERRAIN="$FAST_APPROX_TERRAIN" \
  -e ISAACLAB_EPISODE_LENGTH_S="$EPISODE_LENGTH_S" \
  -e ISAACLAB_CURRICULUM_INITIAL_DIFFICULTY="$CURRICULUM_INITIAL_DIFFICULTY" \
  -e ISAACLAB_TERRAIN_POOL_SIZE="$TERRAIN_POOL_SIZE" \
  -e ISAACLAB_TERRAIN_POOL_REFRESH_ASSIGNMENTS="$TERRAIN_POOL_REFRESH_ASSIGNMENTS" \
  -e ISAACLAB_TERRAIN_POOL_USE_DEM="$TERRAIN_POOL_USE_DEM" \
  -e ISAACLAB_TERRAIN_POOL_DEM_QUALITY="$TERRAIN_POOL_DEM_QUALITY" \
  -e ISAACLAB_TERRAIN_BLEND_METHOD="$TERRAIN_BLEND_METHOD" \
  -e ISAACLAB_TERRAIN_HYBRID_CONTACT_RADIUS_M="$TERRAIN_HYBRID_CONTACT_RADIUS_M" \
  -e SEED="$SEED" \
  -e PYTHONPATH=/isaac-sim/extsDeprecated/omni.isaac.ml_archive/pip_prebundle:/workspace/isaac_rl_deps/site-packages:/workspace/IsaacLab/source/isaaclab:/workspace/IsaacLab/source/isaaclab_assets:/workspace/IsaacLab/source/isaaclab_tasks:/workspace/IsaacLab/source/isaaclab_rl:/workspace/IsaacLab/source/isaaclab_contrib:/workspace/IsaacLab/source/isaaclab_newton:/workspace/IsaacLab/source/isaaclab_physx:/workspace/IsaacLab/source/isaaclab_ov:/workspace/IsaacLab/source/isaaclab_ovphysx:/workspace/IsaacLab/source/isaaclab_visualizers:/workspace/LunarRocket/source/lunar_rocket_lab:/workspace/LunarRocket \
  -v "$PROJECT_ROOT:/workspace/LunarRocket" \
  -v "$ISAACLAB_HOST_ROOT:/workspace/IsaacLab" \
  -v "$RL_DEPS_ROOT:/workspace/isaac_rl_deps" \
  -v /usr/share/vulkan/icd.d/nvidia_icd.json:/etc/vulkan/icd.d/nvidia_icd.json:ro \
  -w /workspace/LunarRocket \
  "$LAB_IMAGE" \
  -lc "/workspace/LunarRocket/scripts/isaaclab_train_entrypoint.sh"
