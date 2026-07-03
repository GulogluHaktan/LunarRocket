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
  rm -rf "$DEPS_DIR" "$TMP_ROOT"
  mkdir -p "$DEPS_DIR" "$TMP_ROOT"

  ./python.sh -m pip install \
    --target "$DEPS_DIR" \
    --upgrade \
    --no-cache-dir \
    --no-deps \
    "gymnasium>=0.29,<1.1" \
    "stable-baselines3>=2.3,<3.0" \
    "cloudpickle>=1.2.0" \
    "farama-notifications>=0.0.1"

  ./python.sh -c "import importlib.util; assert importlib.util.find_spec('gymnasium'); assert importlib.util.find_spec('stable_baselines3'); print('[LunarRocket] RL deps spec check ok')"
  echo "[LunarRocket] RL deps installed"
}

if [[ -d "$DEPS_DIR/torch" ]] || compgen -G "$DEPS_DIR/torch-*.dist-info" >/dev/null; then
  echo "[LunarRocket] removing cached PyTorch to avoid mixing with Isaac Sim torch"
  install_rl_deps
elif ./python.sh -c "import importlib.util; assert importlib.util.find_spec('gymnasium'); assert importlib.util.find_spec('stable_baselines3')" >/dev/null 2>&1; then
  echo "[LunarRocket] RL deps found: $DEPS_DIR"
else
  install_rl_deps
fi

exec ./python.sh "$PROJECT_DIR/app/train_sac.py" "$@"
