#!/usr/bin/env bash
set -euo pipefail

DEPS_DIR="${LUNAR_RL_DEPS_DIR:-/workspace/isaac_rl_deps}"
PROJECT_DIR="/workspace/LunarRocket"
REQ_FILE="$PROJECT_DIR/requirements-rl.txt"

cd /isaac-sim
export PYTHONPATH="$DEPS_DIR:$PROJECT_DIR:${PYTHONPATH:-}"

if ./python.sh -c "import gymnasium, stable_baselines3" >/dev/null 2>&1; then
  echo "[LunarRocket] RL deps found: $DEPS_DIR"
else
  echo "[LunarRocket] installing RL deps once into: $DEPS_DIR"
  mkdir -p "$DEPS_DIR"
  ./python.sh -m pip install --target "$DEPS_DIR" -r "$REQ_FILE"
  echo "[LunarRocket] RL deps installed"
fi

exec ./python.sh "$PROJECT_DIR/app/train_sac.py" "$@"
