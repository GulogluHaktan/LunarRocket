#!/usr/bin/env bash
# Native (non-Docker) equivalent of train_isaaclab_docker.sh.
#
# Requires that ./scripts/install.sh --native has already been run once on
# this machine (creates the Python venv + Isaac Sim/Isaac Lab install under
# LUNAR_NATIVE_ROOT). This script only activates that environment and runs
# training directly on the host -- no containers involved.
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
ALGO="${ISAACLAB_ALGO:-sac}"
# See train_isaaclab_docker.sh for the reasoning behind these defaults:
# 1200 * 32 * 512 ~= 19.7M steps -- sized for SAC to actually converge on
# this task's terminal-reward-gated soft landing.
NUM_ENVS="${ISAACLAB_NUM_ENVS:-512}"
MAX_ITERATIONS="${ISAACLAB_MAX_ITERATIONS:-1200}"
DEVICE="${ISAACLAB_DEVICE:-cuda:0}"
HEADLESS_FLAG="${ISAACLAB_HEADLESS_FLAG:---headless}"
LOG_INTERVAL="${ISAACLAB_LOG_INTERVAL:-10000}"
CHECKPOINT="${ISAACLAB_CHECKPOINT:-}"

TRAIN_SCRIPT="source/lunar_rocket_lab/scripts/train_sac.py"
AGENT_ENTRY="sb3_sac_cfg_entry_point"
if [[ "$ALGO" == "ppo" ]]; then
  TRAIN_SCRIPT="source/lunar_rocket_lab/scripts/train_sb3.py"
  AGENT_ENTRY="sb3_cfg_entry_point"
elif [[ "$ALGO" == "td3" ]]; then
  AGENT_ENTRY="sb3_td3_cfg_entry_point"
elif [[ "$ALGO" == "ddpg" ]]; then
  AGENT_ENTRY="sb3_ddpg_cfg_entry_point"
elif [[ "$ALGO" != "sac" ]]; then
  echo "Unsupported ISAACLAB_ALGO='$ALGO'. Use 'sac', 'ppo', 'td3', or 'ddpg'." >&2
  exit 2
fi

# All terrain/curriculum knobs pass straight through to the env cfg via
# os.environ, exactly like the Docker path -- no translation needed here.
export ISAACLAB_TERRAIN_QUALITY="${ISAACLAB_TERRAIN_QUALITY:-high_train}"
export ISAACLAB_TERRAIN_LOCAL_RESOLUTION="${ISAACLAB_TERRAIN_LOCAL_RESOLUTION:-256}"
export ISAACLAB_TERRAIN_LOCAL_DETAIL="${ISAACLAB_TERRAIN_LOCAL_DETAIL:-0}"
export ISAACLAB_USE_DEM_TERRAIN="${ISAACLAB_USE_DEM_TERRAIN:-0}"
export ISAACLAB_FAST_APPROX_TERRAIN="${ISAACLAB_FAST_APPROX_TERRAIN:-0}"
export ISAACLAB_EPISODE_LENGTH_S="${ISAACLAB_EPISODE_LENGTH_S:-}"
export ISAACLAB_CURRICULUM_INITIAL_DIFFICULTY="${ISAACLAB_CURRICULUM_INITIAL_DIFFICULTY:-}"
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

TRAIN_ARGS=(
  "$ISAACLAB_HOST_ROOT/isaaclab.sh" -p "$TRAIN_SCRIPT"
  --task "$TASK"
  --agent "$AGENT_ENTRY"
  --num_envs "$NUM_ENVS"
  --max_iterations "$MAX_ITERATIONS"
  --device "$DEVICE"
  --log_interval "$LOG_INTERVAL"
)
if [[ -n "$CHECKPOINT" ]]; then
  TRAIN_ARGS+=(--checkpoint "$CHECKPOINT")
fi
if [[ -n "$HEADLESS_FLAG" ]]; then
  TRAIN_ARGS+=($HEADLESS_FLAG)
fi

echo "[LunarRocket] training natively: task=$TASK algo=$ALGO num_envs=$NUM_ENVS max_iterations=$MAX_ITERATIONS device=$DEVICE"
exec "${TRAIN_ARGS[@]}"
