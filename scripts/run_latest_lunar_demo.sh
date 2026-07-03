#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

docker rm -f lunar-rocket-isaac-live-demo >/dev/null 2>&1 || true

if command -v xhost >/dev/null 2>&1; then
  xhost +local:docker >/dev/null 2>&1 || true
fi

export ISAAC_RENDERER="${ISAAC_RENDERER:-HydraStorm}"
export ISAAC_TERRAIN_RESOLUTION="${ISAAC_TERRAIN_RESOLUTION:-128}"
export ISAAC_LANDING_RESOLUTION="${ISAAC_LANDING_RESOLUTION:-512}"
export ISAAC_LANDING_SIZE="${ISAAC_LANDING_SIZE:-5.12}"
export ISAAC_LIVE_DURATION="${ISAAC_LIVE_DURATION:-120}"
export ISAAC_SCREEN_WIDTH="${ISAAC_SCREEN_WIDTH:-1280}"
export ISAAC_SCREEN_HEIGHT="${ISAAC_SCREEN_HEIGHT:-720}"
export ISAAC_SCREEN_FPS="${ISAAC_SCREEN_FPS:-24}"
export ISAAC_SIMULATE_PHYSICS="${ISAAC_SIMULATE_PHYSICS:-0}"

echo "[LunarRocket] launching latest lunar demo"
echo "[LunarRocket] terrain=${ISAAC_TERRAIN_RESOLUTION} landing=${ISAAC_LANDING_RESOLUTION} landing_size=${ISAAC_LANDING_SIZE}m renderer=${ISAAC_RENDERER}"
echo "[LunarRocket] physics_sim=${ISAAC_SIMULATE_PHYSICS} duration=${ISAAC_LIVE_DURATION}s"

exec "$PROJECT_ROOT/scripts/isaac_docker_live_demo.sh"
