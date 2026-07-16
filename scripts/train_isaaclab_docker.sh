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
ISAACLAB_REF="${ISAACLAB_REF:-develop}"
IMAGE="${ISAAC_SIM_IMAGE:-nvcr.io/nvidia/isaac-sim:6.0.1}"
LAB_IMAGE="${ISAACLAB_IMAGE:-lunar-rocket-isaaclab:6.0.1}"

TASK="${ISAACLAB_TASK:-LunarRocket-Lander-Direct-v0}"
ALGO="${ISAACLAB_ALGO:-sac}"
NUM_ENVS="${ISAACLAB_NUM_ENVS:-16}"
MAX_ITERATIONS="${ISAACLAB_MAX_ITERATIONS:-200}"
DEVICE="${ISAACLAB_DEVICE:-cuda:0}"
HEADLESS_FLAG="${ISAACLAB_HEADLESS_FLAG:---headless}"
LOG_INTERVAL="${ISAACLAB_LOG_INTERVAL:-10000}"
INSTALL_MODE="${ISAACLAB_INSTALL:-0}"
TERRAIN_QUALITY="${ISAACLAB_TERRAIN_QUALITY:-high_detail_landing}"
TERRAIN_LOCAL_RESOLUTION="${ISAACLAB_TERRAIN_LOCAL_RESOLUTION:-512}"
FAST_APPROX_TERRAIN="${ISAACLAB_FAST_APPROX_TERRAIN:-0}"
RL_DEPS_ROOT="${ISAAC_DOCKER_CACHE:-$HOME/docker/isaac-sim}/rl_deps_py312"
TRAIN_SCRIPT="source/lunar_rocket_lab/scripts/train_sac.py"
AGENT_ENTRY="sb3_sac_cfg_entry_point"

if [[ "$ALGO" == "ppo" ]]; then
  TRAIN_SCRIPT="source/lunar_rocket_lab/scripts/train_sb3.py"
  AGENT_ENTRY="sb3_cfg_entry_point"
elif [[ "$ALGO" != "sac" ]]; then
  echo "Unsupported ISAACLAB_ALGO='$ALGO'. Use 'sac' or 'ppo'." >&2
  exit 2
fi

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
  -e HEADLESS_FLAG="$HEADLESS_FLAG" \
  -e INSTALL_MODE="$INSTALL_MODE" \
  -e ISAACLAB_TERRAIN_QUALITY="$TERRAIN_QUALITY" \
  -e ISAACLAB_TERRAIN_LOCAL_RESOLUTION="$TERRAIN_LOCAL_RESOLUTION" \
  -e ISAACLAB_FAST_APPROX_TERRAIN="$FAST_APPROX_TERRAIN" \
  -e PYTHONPATH=/workspace/isaac_rl_deps/site-packages:/workspace/IsaacLab/source/isaaclab:/workspace/IsaacLab/source/isaaclab_assets:/workspace/IsaacLab/source/isaaclab_tasks:/workspace/IsaacLab/source/isaaclab_rl:/workspace/IsaacLab/source/isaaclab_contrib:/workspace/IsaacLab/source/isaaclab_newton:/workspace/IsaacLab/source/isaaclab_physx:/workspace/IsaacLab/source/isaaclab_ov:/workspace/IsaacLab/source/isaaclab_ovphysx:/workspace/IsaacLab/source/isaaclab_visualizers:/workspace/LunarRocket/source/lunar_rocket_lab:/workspace/LunarRocket:/isaac-sim/extsDeprecated/omni.isaac.ml_archive/pip_prebundle \
  -v "$PROJECT_ROOT:/workspace/LunarRocket" \
  -v "$ISAACLAB_HOST_ROOT:/workspace/IsaacLab" \
  -v "$RL_DEPS_ROOT:/workspace/isaac_rl_deps" \
  -v /usr/share/vulkan/icd.d/nvidia_icd.json:/etc/vulkan/icd.d/nvidia_icd.json:ro \
  -w /workspace/LunarRocket \
  "$LAB_IMAGE" \
  -lc "/workspace/LunarRocket/scripts/isaaclab_train_entrypoint.sh"
