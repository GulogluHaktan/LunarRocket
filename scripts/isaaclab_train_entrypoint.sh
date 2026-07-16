#!/usr/bin/env bash
set -euo pipefail

DEPS_DIR="/workspace/isaac_rl_deps/site-packages"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-source/lunar_rocket_lab/scripts/train_sac.py}"
TASK="${TASK:-LunarRocket-Lander-Direct-v0}"
AGENT_ENTRY="${AGENT_ENTRY:-sb3_sac_cfg_entry_point}"
NUM_ENVS="${NUM_ENVS:-16}"
MAX_ITERATIONS="${MAX_ITERATIONS:-200}"
DEVICE="${DEVICE:-cuda:0}"
LOG_INTERVAL="${LOG_INTERVAL:-10000}"
HEADLESS_FLAG="${HEADLESS_FLAG:---headless}"
INSTALL_MODE="${INSTALL_MODE:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

export TERM=xterm
ln -sfn /isaac-sim /workspace/IsaacLab/_isaac_sim
mkdir -p "$DEPS_DIR"

remove_torch_cache() {
  rm -rf \
    /workspace/isaac_rl_deps/torch /workspace/isaac_rl_deps/torch-* \
    /workspace/isaac_rl_deps/torchvision /workspace/isaac_rl_deps/torchvision-* \
    /workspace/isaac_rl_deps/torchaudio /workspace/isaac_rl_deps/torchaudio-* \
    /workspace/isaac_rl_deps/triton /workspace/isaac_rl_deps/triton-* \
    /workspace/isaac_rl_deps/functorch \
    "$DEPS_DIR"/torch "$DEPS_DIR"/torch-* \
    "$DEPS_DIR"/torchvision "$DEPS_DIR"/torchvision-* \
    "$DEPS_DIR"/torchaudio "$DEPS_DIR"/torchaudio-* \
    "$DEPS_DIR"/triton "$DEPS_DIR"/triton-* \
    "$DEPS_DIR"/functorch
}

torch_cuda_ok() {
  /isaac-sim/python.sh - <<'PY'
import sys
sys.path.insert(0, '/isaac-sim/extsDeprecated/omni.isaac.ml_archive/pip_prebundle')
import torch
assert hasattr(torch._C, "_cuda_setDevice")
assert torch.cuda.is_available()
PY
}

deps_ok() {
  /isaac-sim/python.sh - <<'PY'
import hydra, tensorboard, toml, tqdm, rich, flatdict, lazy_loader, warp, prettytable
import isaaclab_visualizers
import PIL, requests, yaml
import gymnasium as gym
from stable_baselines3 import SAC
assert hasattr(gym.vector, "AutoresetMode")
PY
}

if [[ "$INSTALL_MODE" == "1" || "$INSTALL_MODE" == "true" ]]; then
  cd /workspace/IsaacLab
  ./isaaclab.sh --install sb3
else
  if ! torch_cuda_ok >/dev/null 2>&1; then
    echo "[LunarRocket] installing CUDA PyTorch into Isaac Lab dependency cache"
    remove_torch_cache
    /isaac-sim/python.sh -m pip install \
      --target "$DEPS_DIR" \
      --upgrade \
      --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cu128 \
      "torch>=2.8,<3.0"
  fi

  if ! deps_ok >/dev/null 2>&1; then
    echo "[LunarRocket] installing Isaac Lab SB3 dependencies"
    /isaac-sim/python.sh -m pip install \
      --target "$DEPS_DIR" \
      --upgrade \
      --no-cache-dir \
      hydra-core tensorboard toml tqdm rich flatdict lazy_loader warp-lang prettytable \
      requests pillow pyyaml \
      "gymnasium>=1.2.0" "protobuf>=4.25.8,!=5.26.0" "packaging<24"
    /isaac-sim/python.sh -m pip install \
      --target "$DEPS_DIR" \
      --upgrade \
      --no-cache-dir \
      --no-deps \
      stable-baselines3
  else
    echo "[LunarRocket] Isaac Lab smoke deps found"
  fi
fi

if [[ "$SKIP_TRAIN" == "1" || "$SKIP_TRAIN" == "true" ]]; then
  exit 0
fi

cd /workspace/LunarRocket
/workspace/IsaacLab/isaaclab.sh -p "$TRAIN_SCRIPT" \
  --task "$TASK" \
  --agent "$AGENT_ENTRY" \
  --num_envs "$NUM_ENVS" \
  --max_iterations "$MAX_ITERATIONS" \
  --device "$DEVICE" \
  --log_interval "$LOG_INTERVAL" \
  $HEADLESS_FLAG
