# LunarRocket

Isaac Lab direct-workflow lunar rocket landing task.

The repository is now centered on one training path: `LunarRocket-Lander-Direct-v0` under `source/lunar_rocket_lab`. The old standalone Isaac Sim/Gym scaffold has been removed.

## Requirements

- Linux with NVIDIA proprietary driver
- Docker
- NVIDIA Container Toolkit
- NVIDIA GPU with RTX support
- Isaac Sim Docker image: `nvcr.io/nvidia/isaac-sim:6.0.1`

## Quick Start

```bash
./scripts/setup_host.sh
./scripts/test_docker_gpu.sh
./scripts/train_isaaclab_docker.sh
```

Useful overrides:

```bash
ISAACLAB_ALGO=sac \
ISAACLAB_NUM_ENVS=512 \
ISAACLAB_MAX_ITERATIONS=200 \
ISAACLAB_DEVICE=cuda:0 \
./scripts/train_isaaclab_docker.sh
```

For PPO:

```bash
ISAACLAB_ALGO=ppo ./scripts/train_isaaclab_docker.sh
```

## Project Layout

```text
source/lunar_rocket_lab/
  lunar_rocket_lab/tasks/direct/lunar_lander/lunar_lander_env.py
  lunar_rocket_lab/tasks/direct/lunar_lander/agents/
  scripts/train_sac.py
  scripts/train_sb3.py

assets/rocket/
  lunar_rocket_single_body.usda

docker/
  Dockerfile.isaaclab

scripts/
  setup_host.sh
  test_docker_gpu.sh
  train_isaaclab_docker.sh

tests/
  test_isaaclab_static.py
```

## Observation

The policy observation is a flat Isaac Lab/SB3-friendly vector of 123 values:

- 27 proprioceptive/navigation values
- 64 terrain-relative LiDAR range samples
- 24 local terrain scan samples
- 8 sensor features: body-frame accelerometer, gyro, altimeter, and nearest LiDAR return

The task also randomizes procedural slope, ripples, craters, and rocks per environment reset, and computes reward/done conditions relative to that terrain.

## Validation

Static checks:

```bash
python3 -m unittest discover -s tests
```

Isaac Lab training smoke:

```bash
ISAACLAB_MAX_ITERATIONS=1 ISAACLAB_NUM_ENVS=16 ./scripts/train_isaaclab_docker.sh
```
