#!/usr/bin/env bash
set -euo pipefail

INSTALL_SYSTEM=0
if [[ "${1:-}" == "--install-system" ]]; then
  INSTALL_SYSTEM=1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_ROOT="${ISAAC_DOCKER_CACHE:-$HOME/docker/isaac-sim}"

echo "[LunarRocket] host setup"
echo "[LunarRocket] project: $PROJECT_ROOT"
echo "[LunarRocket] docker cache: $CACHE_ROOT"

if ! command -v docker >/dev/null 2>&1; then
  echo "[LunarRocket] Docker is missing."
  if [[ "$INSTALL_SYSTEM" == "1" ]]; then
    if command -v pacman >/dev/null 2>&1; then
      sudo pacman -S --needed docker
      sudo systemctl enable --now docker
    elif command -v apt-get >/dev/null 2>&1; then
      sudo apt-get update
      sudo apt-get install -y docker.io
      sudo systemctl enable --now docker
    else
      echo "[LunarRocket] unsupported package manager. Install Docker manually."
      exit 1
    fi
  else
    echo "[LunarRocket] install Docker, or run: ./scripts/setup_host.sh --install-system"
    exit 1
  fi
fi

if ! docker info >/dev/null 2>&1; then
  echo "[LunarRocket] Docker daemon is not reachable."
  echo "[LunarRocket] try: sudo systemctl enable --now docker"
  echo "[LunarRocket] if needed: sudo usermod -aG docker $USER"
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[LunarRocket] nvidia-smi not found. Install NVIDIA proprietary drivers first."
  exit 1
fi

if [[ "$INSTALL_SYSTEM" == "1" ]]; then
  if command -v pacman >/dev/null 2>&1; then
    sudo pacman -S --needed nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
  elif command -v apt-get >/dev/null 2>&1; then
    echo "[LunarRocket] NVIDIA Container Toolkit install differs by distro."
    echo "[LunarRocket] Follow NVIDIA's install guide, then run this script again."
  fi
fi

mkdir -p \
  "$PROJECT_ROOT/assets/moon_heightmaps" \
  "$PROJECT_ROOT/assets/rocket" \
  "$PROJECT_ROOT/outputs/models" \
  "$PROJECT_ROOT/outputs/videos" \
  "$CACHE_ROOT/cache/ov" \
  "$CACHE_ROOT/cache/pip" \
  "$CACHE_ROOT/cache/glcache" \
  "$CACHE_ROOT/cache/kit" \
  "$CACHE_ROOT/data" \
  "$CACHE_ROOT/documents" \
  "$CACHE_ROOT/rl_deps_py312"

chmod +x "$PROJECT_ROOT"/scripts/*.sh "$PROJECT_ROOT/python.sh"

echo "[LunarRocket] checking Docker GPU access"
"$PROJECT_ROOT/scripts/test_docker_gpu.sh"

echo "[LunarRocket] setup complete"
echo "[LunarRocket] smoke test: ./scripts/isaac_docker_run.sh 10"
echo "[LunarRocket] live demo:  ./scripts/run_latest_lunar_demo.sh"
echo "[LunarRocket] SAC test:   SAC_TIMESTEPS=1000 ./scripts/train_sac_docker.sh"
