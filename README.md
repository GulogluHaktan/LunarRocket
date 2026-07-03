# LunarRocket

Isaac Sim lunar landing environment for a MuJoCo-derived rocket. The project builds a 3D Moon terrain from cached/downloaded NASA lunar DEM data, adds procedural crater and regolith detail, spawns the rocket with lunar gravity, and exposes a Gymnasium/SAC-ready RL scaffold.

This repo is designed to run on Linux through the official Isaac Sim Docker image. Native Isaac Sim is optional.

## Quick Start

```bash
git clone https://github.com/GulogluHaktan/LunarRocket.git
cd LunarRocket
./scripts/setup_host.sh
./scripts/isaac_docker_run.sh 10
```

If you prefer `make`:

```bash
make setup
make smoke
```

The first Isaac run may take time because Docker pulls the Isaac Sim image and the app downloads/caches the lunar DEM.

## Requirements

- Linux with NVIDIA proprietary driver
- Docker
- NVIDIA Container Toolkit
- NVIDIA GPU with RTX support
- 32 GB RAM recommended
- Isaac Sim Docker image: `nvcr.io/nvidia/isaac-sim:6.0.1`

On Arch/CachyOS, the usual host setup is:

```bash
sudo pacman -S --needed docker nvidia-container-toolkit
sudo systemctl enable --now docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Then verify GPU access:

```bash
./scripts/test_docker_gpu.sh
```

`./scripts/setup_host.sh --install-system` can attempt the Docker/toolkit setup for Arch-like hosts, but driver installation is still your responsibility.

## Main Commands

Headless Isaac smoke test:

```bash
./scripts/isaac_docker_run.sh 10
```

Live GUI demo for screen recording:

```bash
./scripts/run_latest_lunar_demo.sh
```

Low-memory SAC training smoke test:

```bash
SAC_TIMESTEPS=1000 SAC_DEVICE=cpu SAC_BUFFER_SIZE=50000 ./scripts/train_sac_docker.sh
```

Static tests without Isaac Sim:

```bash
python3 -m unittest discover -s tests
```

Local non-Isaac preview image:

```bash
python3 app/preview.py --seed 42 --size 1400
```

## What Runs Automatically

- `assets/moon_heightmaps/` is used as the NASA DEM cache.
- `assets/rocket/hopper_lunar.xml` is the tracked MuJoCo rocket source.
- Generated USD rocket files are cache/output files and are ignored by git.
- SAC dependencies are installed once into `~/docker/isaac-sim/rl_deps_py312` and reused by later containers.
- Isaac/Omniverse Docker caches live under `~/docker/isaac-sim` by default.

You can override the cache/image:

```bash
ISAAC_SIM_IMAGE=nvcr.io/nvidia/isaac-sim:6.0.1 \
ISAAC_DOCKER_CACHE=$HOME/docker/isaac-sim \
./scripts/isaac_docker_run.sh 10
```

## Project Layout

```text
app/
  terrain_generator.py       Moon DEM, craters, roughness, two-tier landing patch
  world_builder.py           Isaac World, gravity, physics, lighting, terrain, rocket
  rocket_asset.py            MJCF/XML to USD import and rocket rigid body setup
  sensors.py                 RGB camera setup
  env_lunar_landing.py       Gymnasium environment wrapper
  world_adapter.py           Isaac-to-RL state/sensor adapter
  reward_function.py         Reward terms and terminal landing classification hooks
  train_sac.py               SAC training entrypoint

assets/
  rocket/hopper_lunar.xml    Tracked MuJoCo rocket source
  moon_heightmaps/           Runtime DEM cache, ignored except .gitkeep

configs/
  sim_config.yaml            Physics, gravity, renderer, episode defaults
  terrain_config.yaml        DEM, crater, roughness, local landing detail
  rocket_config.yaml         Rocket spawn/mass/thrust parameters
  rl_config.yaml             Observation, reward, SAC defaults

scripts/
  setup_host.sh              Host/Docker/GPU setup check
  isaac_docker_run.sh        Headless smoke test
  run_latest_lunar_demo.sh   GUI demo
  train_sac_docker.sh        Docker SAC training
```

## RL Status

The RL code is intentionally a scaffold, not a finished controller. Current Stage 1 training uses `state_lidar` observations:

- state vector: 27 normalized values
- LiDAR: 32 synthetic/adapter range values
- action: `[main_thrust, gimbal_x, gimbal_y]`
- gravity: `1.62 m/s^2`
- rocket mass config: `configs/rocket_config.yaml`

Detailed sensor, reward, and observation notes are in:

```text
docs/RL_REWARD_OBSERVATION_REPORT.md
```

## Terrain Notes

The terrain is a layered lunar regolith height field:

```text
height(x, y) =
    NASA DEM macro elevation
  + randomized slope/undulation
  + crater bowl/rim deformation
  + dense local landing-zone regolith detail
  + small footprint-scale bumps and pits
```

The default demo uses a light global mesh plus a denser local landing patch so the rocket feet see visible surface detail without exploding VRAM.

Useful overrides:

```bash
ISAAC_TERRAIN_RESOLUTION=128 \
ISAAC_LANDING_RESOLUTION=512 \
ISAAC_LANDING_SIZE=24 \
./scripts/run_latest_lunar_demo.sh
```

## Common Problems

Docker cannot see the GPU:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
./scripts/test_docker_gpu.sh
```

SAC says dependencies are missing:

```bash
./scripts/train_sac_docker.sh
```

The script now installs RL dependencies once into the persistent Docker cache. If the cache is corrupted, delete:

```bash
rm -rf ~/docker/isaac-sim/rl_deps_py312
```

Then run training again.

GUI demo does not open:

```bash
xhost +local:docker
./scripts/run_latest_lunar_demo.sh
```

On Wayland, XWayland/Xorg support may be needed for the Isaac window.
