#!/usr/bin/env bash
set -euo pipefail

DEPS_ROOT="${LUNAR_RL_DEPS_ROOT:-/workspace/isaac_rl_deps}"
DEPS_DIR="$DEPS_ROOT/site-packages"
PROJECT_DIR="/workspace/LunarRocket"
TMP_ROOT="$DEPS_ROOT/tmp"

cd /isaac-sim
export PYTHONPATH="$DEPS_DIR:$PROJECT_DIR:${PYTHONPATH:-}"
export TMPDIR="$TMP_ROOT"

install_rl_deps() {
  echo "[LunarRocket] installing RL deps once into: $DEPS_DIR"
  rm -rf "$TMP_ROOT"
  mkdir -p "$DEPS_DIR" "$TMP_ROOT"

  if ! ./python.sh -c "import torch" >/dev/null 2>&1; then
    ./python.sh -m pip install \
      --target "$DEPS_DIR" \
      --upgrade \
      --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      "torch>=2.3,<2.8"
  else
    echo "[LunarRocket] torch already available"
  fi

  if ! ./python.sh -c "import gymnasium, pandas, matplotlib" >/dev/null 2>&1; then
    ./python.sh -m pip install \
      --target "$DEPS_DIR" \
      --upgrade \
      --no-cache-dir \
      "gymnasium>=0.29,<1.1" \
      "cloudpickle>=1.2.0" \
      "farama-notifications>=0.0.1" \
      "pandas>=1.0" \
      "matplotlib>=3.5"
  else
    echo "[LunarRocket] gymnasium/pandas/matplotlib already available"
  fi

  if ! ./python.sh -c "from stable_baselines3 import SAC" >/dev/null 2>&1; then
    ./python.sh -m pip install \
      --target "$DEPS_DIR" \
      --upgrade \
      --no-cache-dir \
      --no-deps \
      "stable-baselines3>=2.3,<3.0"
  else
    echo "[LunarRocket] stable-baselines3 already available"
  fi

  ./python.sh -c "import gymnasium, torch; from stable_baselines3 import SAC; print('[LunarRocket] RL deps import check ok')"
  echo "[LunarRocket] RL deps installed"
}

if ./python.sh -c "import gymnasium, torch; from stable_baselines3 import SAC" >/dev/null 2>&1; then
  echo "[LunarRocket] RL deps found: $DEPS_DIR"
else
  install_rl_deps
fi

cd "$PROJECT_DIR"
exec /isaac-sim/python.sh "$PROJECT_DIR/app/train_sac.py" "$@"
