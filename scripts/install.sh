#!/usr/bin/env bash
# One-shot installer for LunarRocket.
#
# Asks (or is told via flag/env var) whether to set the machine up for the
# Docker training path or the native (no-container) path, then performs every
# install step for that path: system packages, NVIDIA container toolkit or a
# pip-based Isaac Sim + Isaac Lab install, and the project's own Python deps.
#
# Usage:
#   ./scripts/install.sh              # asks interactively
#   ./scripts/install.sh --docker     # force the Docker path, no prompt
#   ./scripts/install.sh --native     # force the native (no Docker) path
#   LUNAR_INSTALL_MODE=native ./scripts/install.sh   # same, via env var
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LUNAR_NATIVE_ROOT="${LUNAR_NATIVE_ROOT:-$HOME/isaac-lab-native}"
VENV_DIR="${LUNAR_NATIVE_VENV:-$LUNAR_NATIVE_ROOT/env_isaaclab}"
ISAACLAB_HOST_ROOT="${ISAACLAB_HOST_ROOT:-$LUNAR_NATIVE_ROOT/IsaacLab}"
ISAACLAB_REPO="${ISAACLAB_REPO:-https://github.com/isaac-sim/IsaacLab.git}"
# Pinned, not "develop": the develop branch's HEAD is a moving target and can
# (and did) ship regressions -- e.g. isaaclab.assets.visual_material eager-
# importing pxr before SimulationApp starts, breaking every task's cfg-class
# import. v3.0.0-beta2.patch1 is the release IsaacLab's own changelog says
# was updated specifically "to support Isaac Sim 6.0.1" -- the exact image
# this project's Docker path uses (nvcr.io/nvidia/isaac-sim:6.0.1).
ISAACLAB_REF="${ISAACLAB_REF:-v3.0.0-beta2.patch1}"
ISAAC_SIM_PIP_SPEC="${ISAAC_SIM_PIP_SPEC:-isaacsim[all,extscache]==6.0.1.0}"
ISAAC_SIM_PY="${ISAAC_SIM_PY:-python3.12}"
TORCH_CUDA_INDEX="${TORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu128}"

log() { echo "[LunarRocket] $*"; }
die() { echo "[LunarRocket] ERROR: $*" >&2; exit 1; }

MODE="${LUNAR_INSTALL_MODE:-}"
for arg in "$@"; do
  case "$arg" in
    --docker) MODE="docker" ;;
    --native) MODE="native" ;;
    -h|--help)
      sed -n '2,16p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *) die "unknown argument: $arg (use --docker or --native)" ;;
  esac
done

# ---------------------------------------------------------------------------
# Shared preflight: OS + NVIDIA driver. Both paths need a real NVIDIA GPU with
# the proprietary driver already installed -- neither Docker nor a pip install
# can safely do that unattended (it usually wants a reboot / a display-manager
# restart), so we only check for it and tell the user how to fix it.
# ---------------------------------------------------------------------------
OS_ID="unknown"
OS_PRETTY="unknown"
if [[ -f /etc/os-release ]]; then
  # shellcheck disable=SC1091
  source /etc/os-release
  OS_ID="${ID:-unknown}"
  OS_PRETTY="${PRETTY_NAME:-unknown}"
fi
log "detected OS: $OS_PRETTY"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  die "nvidia-smi not found. Install the NVIDIA proprietary driver first, then re-run this script."
fi
if ! nvidia-smi >/dev/null 2>&1; then
  die "nvidia-smi is present but failed to run. Check 'nvidia-smi' output and fix the driver before continuing."
fi
log "GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"

# ---------------------------------------------------------------------------
# Pick a path
# ---------------------------------------------------------------------------
if [[ -z "$MODE" ]]; then
  if [[ -t 0 ]]; then
    echo
    echo "How should LunarRocket be installed on this machine?"
    echo "  1) Docker (recommended default: isolated, matches the tested nvcr.io/nvidia/isaac-sim:6.0.1 image)"
    echo "  2) Native, no Docker (installs Isaac Sim + Isaac Lab directly via pip into a venv)"
    read -r -p "Choice [1/2]: " choice
    case "$choice" in
      2) MODE="native" ;;
      *) MODE="docker" ;;
    esac
  else
    die "no TTY to prompt on; pass --docker or --native (or set LUNAR_INSTALL_MODE)."
  fi
fi
log "install mode: $MODE"

# ---------------------------------------------------------------------------
# Docker path: reuse the existing host-setup + GPU-check scripts.
# ---------------------------------------------------------------------------
install_docker() {
  chmod +x "$PROJECT_ROOT"/scripts/*.sh
  "$PROJECT_ROOT/scripts/setup_host.sh" --install-system

  if [[ "$OS_ID" == "ubuntu" ]] && ! command -v nvidia-ctk >/dev/null 2>&1; then
    log "installing NVIDIA Container Toolkit (Ubuntu apt repo)"
    sudo apt-get update
    sudo apt-get install -y curl gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
      | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
      | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
      | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
  fi
  if [[ ! -f /etc/cdi/nvidia.yaml ]]; then
    log "generating NVIDIA CDI spec (/etc/cdi/nvidia.yaml)"
    sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
    sudo systemctl restart docker
  fi

  log "pre-pulling nvcr.io/nvidia/isaac-sim:6.0.1 (several GB, this can take a while)"
  docker pull nvcr.io/nvidia/isaac-sim:6.0.1

  log "docker install complete."
  log "next: ./scripts/train_isaaclab_docker.sh   (or ISAACLAB_MAX_ITERATIONS=1 ISAACLAB_NUM_ENVS=16 ./scripts/train_isaaclab_docker.sh for a smoke test)"
}

# ---------------------------------------------------------------------------
# Native path: pip-install Isaac Sim + Isaac Lab into a dedicated venv, no
# containers. Requires Ubuntu 22.04+ / GLIBC 2.35+ per NVIDIA's pip packages.
# ---------------------------------------------------------------------------
check_glibc() {
  local ver major minor
  ver="$(ldd --version 2>&1 | head -1 | grep -oE '[0-9]+\.[0-9]+$' || true)"
  if [[ -z "$ver" ]]; then
    log "could not determine GLIBC version; continuing anyway"
    return 0
  fi
  major="${ver%%.*}"
  minor="${ver##*.}"
  if (( major < 2 || (major == 2 && minor < 35) )); then
    die "GLIBC $ver detected; Isaac Sim's pip packages require GLIBC 2.35+ (Ubuntu 22.04+). Use the Docker path instead: ./scripts/install.sh --docker"
  fi
  log "GLIBC $ver OK"
}

ensure_python312() {
  if command -v "$ISAAC_SIM_PY" >/dev/null 2>&1; then
    return 0
  fi
  log "$ISAAC_SIM_PY not found; installing"
  if [[ "$OS_ID" == "ubuntu" || "$OS_ID" == "debian" ]]; then
    if ! sudo apt-get install -y python3.12 python3.12-venv 2>/dev/null; then
      log "python3.12 not in the default repos; adding the deadsnakes PPA"
      sudo apt-get install -y software-properties-common
      sudo add-apt-repository -y ppa:deadsnakes/ppa
      sudo apt-get update
      sudo apt-get install -y python3.12 python3.12-venv
    fi
  else
    die "$ISAAC_SIM_PY not found and this OS's package manager isn't handled automatically. Install Python 3.12 yourself, then re-run."
  fi
}

check_disk_space() {
  # isaacsim[all,extscache] alone downloads/installs ~20-30 GB; add torch
  # (~5 GB) and margin. NVIDIA's own docs list 50 GB as the minimum for a
  # full Isaac Sim install, so warn well before hitting that wall mid-download
  # instead of pip failing partway through a multi-GB wheel.
  mkdir -p "$LUNAR_NATIVE_ROOT"
  local avail_gb
  avail_gb="$(df -Pk "$LUNAR_NATIVE_ROOT" | awk 'NR==2 {print int($4/1024/1024)}')"
  log "available disk space at $LUNAR_NATIVE_ROOT: ${avail_gb} GB"
  if (( avail_gb < 40 )); then
    echo
    echo "[LunarRocket] WARNING: only ${avail_gb} GB free at $LUNAR_NATIVE_ROOT."
    echo "  isaacsim[all,extscache] + PyTorch need roughly 30-40 GB; NVIDIA's own"
    echo "  minimum for a full Isaac Sim install is 50 GB. This will likely fail"
    echo "  partway through the isaacsim download."
    echo "  Options: free up space, or point installs at a bigger disk with:"
    echo "    LUNAR_NATIVE_ROOT=/path/with/more/space ./scripts/install.sh --native"
    echo "  (pip's own cache/build temp dirs also need headroom -- check 'df -h /tmp' too)"
    if [[ -t 0 ]]; then
      read -r -p "Continue anyway? [y/N]: " confirm
      [[ "$confirm" =~ ^[Yy]$ ]] || die "aborted: not enough disk space"
    else
      die "not enough disk space (${avail_gb} GB free, ~40 GB needed) and no TTY to confirm past this warning"
    fi
  fi
}

install_native() {
  check_glibc
  check_disk_space
  if [[ "$OS_ID" == "ubuntu" || "$OS_ID" == "debian" ]]; then
    log "installing system packages (build-essential, cmake, git)"
    sudo apt-get update
    sudo apt-get install -y build-essential cmake git curl
  else
    log "non-Debian OS ($OS_PRETTY); skipping apt step, assuming build-essential/cmake/git are already present"
  fi
  ensure_python312

  mkdir -p "$LUNAR_NATIVE_ROOT"
  if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    log "creating venv at $VENV_DIR"
    "$ISAAC_SIM_PY" -m venv "$VENV_DIR"
  fi
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  pip install --upgrade pip

  # --no-cache-dir: these wheels are multi-GB; pip's default cache would keep
  # a second full copy under ~/.cache/pip on top of the installed copy.
  if ! python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1; then
    log "installing CUDA-enabled PyTorch ($TORCH_CUDA_INDEX)"
    pip install --no-cache-dir --upgrade torch --index-url "$TORCH_CUDA_INDEX"
  else
    log "PyTorch with CUDA already present, skipping"
  fi

  # `import isaacsim` alone is not proof the [all,extscache] extras landed --
  # the bare package can import fine without Kit actually being present.
  # IsaacLab's own launcher gates on `omni.kit` being importable
  # (isaaclab_tasks/utils/sim_launcher.py), so check that instead.
  if ! python -c "import omni.kit" >/dev/null 2>&1; then
    log "installing Isaac Sim ($ISAAC_SIM_PIP_SPEC) -- this downloads 20-30 GB and can take a long time"
    pip install --no-cache-dir --force-reinstall "$ISAAC_SIM_PIP_SPEC" --extra-index-url https://pypi.nvidia.com
  else
    log "isaacsim (omni.kit) already importable in $VENV_DIR, skipping"
  fi

  if [[ ! -d "$ISAACLAB_HOST_ROOT/.git" ]]; then
    log "cloning Isaac Lab ($ISAACLAB_REF) into $ISAACLAB_HOST_ROOT"
    git clone --depth 1 --branch "$ISAACLAB_REF" "$ISAACLAB_REPO" "$ISAACLAB_HOST_ROOT"
  fi
  if ! python -c "import isaaclab" >/dev/null 2>&1; then
    log "running isaaclab.sh --install sb3 (installs Isaac Lab extensions + Stable-Baselines3 into the venv)"
    (cd "$ISAACLAB_HOST_ROOT" && ./isaaclab.sh --install sb3)
  else
    log "isaaclab already installed in $VENV_DIR, skipping"
  fi

  log "installing LunarRocket's own Python packages"
  pip install -e "$PROJECT_ROOT/source/lunar_rocket_lab"
  pip install -r "$PROJECT_ROOT/requirements-rl.txt"
  pip install -r "$PROJECT_ROOT/requirements-presentation.txt"

  log "native install complete: venv=$VENV_DIR  isaaclab=$ISAACLAB_HOST_ROOT"
  log "next: ./scripts/train_isaaclab_native.sh   (or ISAACLAB_MAX_ITERATIONS=1 ISAACLAB_NUM_ENVS=16 ./scripts/train_isaaclab_native.sh for a smoke test)"
}

chmod +x "$PROJECT_ROOT"/scripts/*.sh

if [[ "$MODE" == "docker" ]]; then
  install_docker
else
  install_native
fi

log "run 'python3 -m unittest discover -s tests' any time to sanity-check the static test suite (no GPU/Isaac Lab required)."
