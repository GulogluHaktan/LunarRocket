#!/usr/bin/env bash
set -euo pipefail

DEPS_DIR="${LUNAR_RL_DEPS_DIR:-/workspace/isaac_rl_deps}"
PROJECT_DIR="/workspace/LunarRocket"
TMP_ROOT="$DEPS_DIR/.tmp"

cd /isaac-sim
export PYTHONPATH="$DEPS_DIR:$PROJECT_DIR:${PYTHONPATH:-}"
export TMPDIR="$TMP_ROOT"

install_rl_deps() {
  echo "[LunarRocket] installing RL deps once into: $DEPS_DIR"
  rm -rf "$DEPS_DIR"/*
  mkdir -p "$DEPS_DIR" "$TMP_ROOT"
  chmod -R ugo+rwX "$DEPS_DIR" || true

  ./python.sh -m pip install \
    --target "$DEPS_DIR" \
    --upgrade \
    --no-cache-dir \
    --index-url "https://download.pytorch.org/whl/cpu" \
    --extra-index-url "https://pypi.org/simple" \
    "torch>=2.3,<3.0"

  ./python.sh -m pip install \
    --target "$DEPS_DIR" \
    --upgrade \
    --no-cache-dir \
    "gymnasium>=0.29,<1.1" \
    "cloudpickle>=1.2.0" \
    "pandas>=1.0" \
    "matplotlib>=3.0"

  ./python.sh -m pip install \
    --target "$DEPS_DIR" \
    --upgrade \
    --no-cache-dir \
    --no-deps \
    "stable-baselines3>=2.3,<3.0"

  ./python.sh -c "import gymnasium, stable_baselines3, torch; print('[LunarRocket] RL deps import check ok')"
  echo "[LunarRocket] RL deps installed"
}

if ./python.sh -c "import gymnasium, stable_baselines3" >/dev/null 2>&1; then
  echo "[LunarRocket] RL deps found: $DEPS_DIR"
else
  install_rl_deps
fi

exec ./python.sh "$PROJECT_DIR/app/train_sac.py" "$@"
