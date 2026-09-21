# LunarRocket

Isaac Lab direct-workflow lunar rocket landing task. The repository is
centered on one training path: `LunarRocket-Lander-Direct-v0` under
`source/lunar_rocket_lab`. A SAC (Soft Actor-Critic) policy learns to land a
4-legged hopper rocket, under a two-axis thrust-vectored (gimballed) engine,
on real cratered/sloped lunar terrain at a randomized target.

This README is written for someone setting the project up on a machine they
did not develop it on (no assumed familiarity with the code). If you only
need the short version:

```bash
git clone <this-repo> LunarRocket && cd LunarRocket
./scripts/install.sh          # asks Docker vs. native, installs everything
./scripts/train_isaaclab_docker.sh   # or train_isaaclab_native.sh
```

## 1. Requirements

- Linux (Ubuntu 22.04+ recommended) with the **NVIDIA proprietary driver**
  already installed and working (`nvidia-smi` must succeed).
- An NVIDIA GPU with RTX support (RTX 50-series/Blackwell included).
- Either:
  - **Docker** + NVIDIA Container Toolkit (the tested, reproducible path;
    everything else about the container is installed automatically), or
  - **~25 GB free disk** and Python 3.12 for a **native, no-Docker** install
    (pip-installed Isaac Sim + Isaac Lab directly on the host).
- `./scripts/install.sh` cannot install the NVIDIA driver itself (it usually
  needs a reboot / display-manager restart) -- install that first with your
  distro's standard method (e.g. `ubuntu-drivers install` on Ubuntu), reboot,
  confirm `nvidia-smi` works, then run the installer.

## 2. Installation

Run the installer from the project root:

```bash
./scripts/install.sh
```

It checks for `nvidia-smi`, then asks:

1. **Docker** (recommended default) -- installs Docker + the NVIDIA
   Container Toolkit if missing, generates the NVIDIA CDI spec, and
   pre-pulls the `nvcr.io/nvidia/isaac-sim:6.0.1` image. Nothing else needs
   to be installed on the host; Isaac Lab itself is cloned and built into a
   Docker image automatically the first time you train.
2. **Native / no Docker** -- installs `build-essential`/`cmake`/`git`,
   Python 3.12 (via the `deadsnakes` PPA if your distro doesn't ship it),
   creates a venv at `~/isaac-lab-native/env_isaaclab`, installs CUDA
   PyTorch, `pip install isaacsim[all,extscache]==6.0.1.0`, clones and
   builds Isaac Lab (`isaaclab.sh --install sb3`) into
   `~/isaac-lab-native/IsaacLab`, and installs this project's own Python
   packages (`source/lunar_rocket_lab`, `requirements-rl.txt`,
   `requirements-presentation.txt`) into that same venv.

To skip the prompt (e.g. running this from another script), pass a flag or
env var:

```bash
./scripts/install.sh --docker
./scripts/install.sh --native
LUNAR_INSTALL_MODE=native ./scripts/install.sh
```

**Docker vs. native --  which one to pick.** Docker is the path this project
was actually developed and tested against (pinned Isaac Sim image, no
host Python/CUDA version drift), so prefer it unless there's a specific
reason not to run containers on the target machine. The native path exists
for machines where Docker is unavailable or undesired; it follows NVIDIA's
official pip-install instructions for Isaac Sim 6.0.1 / Isaac Lab, but has
**not** been run end-to-end in this repository's CI -- treat first use of
`--native` on a new machine as a smoke test (`ISAACLAB_MAX_ITERATIONS=1
ISAACLAB_NUM_ENVS=16`, see below) before trusting a long training run to it.
The native path also requires GLIBC 2.35+ (Ubuntu 22.04 or newer); the
installer checks this and tells you to fall back to `--docker` if it fails.

### Re-running the installer

Every step in `install.sh` is idempotent (it checks whether Docker/the venv/
Isaac Sim/Isaac Lab are already present before doing anything), so it's safe
to run again after an interrupted install, or to pick up a newer Isaac Lab
checkout with `ISAACLAB_UPDATE=1`.

## 3. Quick start

Docker path:

```bash
./scripts/test_docker_gpu.sh              # sanity check: GPU visible inside a container
./scripts/train_isaaclab_docker.sh
```

Native path:

```bash
./scripts/train_isaaclab_native.sh
```

Both default to `ISAACLAB_NUM_ENVS=512` and `ISAACLAB_MAX_ITERATIONS=1200`
(`MAX_ITERATIONS * 32 * NUM_ENVS` ~= 19.7M env steps), sized for SAC to
actually converge on this task rather than stopping after a quick smoke run.
A full run at these defaults takes many hours on a single GPU. Useful
overrides (same env vars on both paths):

```bash
ISAACLAB_ALGO=sac \
ISAACLAB_NUM_ENVS=512 \
ISAACLAB_MAX_ITERATIONS=1200 \
ISAACLAB_DEVICE=cuda:0 \
./scripts/train_isaaclab_docker.sh
```

For PPO instead of SAC:

```bash
ISAACLAB_ALGO=ppo ./scripts/train_isaaclab_docker.sh
```

TD3 or DDPG instead of SAC (algorithm-comparison arms; same tuned net_arch
and buffer/batch sizing as `sb3_sac_cfg.yaml`, see
`source/lunar_rocket_lab/lunar_rocket_lab/tasks/direct/lunar_lander/agents/sb3_td3_cfg.yaml`
/ `sb3_ddpg_cfg.yaml`):

```bash
ISAACLAB_ALGO=td3 ./scripts/train_isaaclab_docker.sh
ISAACLAB_ALGO=ddpg ./scripts/train_isaaclab_docker.sh
```

Run SAC/TD3/DDPG across several seeds back-to-back (resumable, same
JSONL-manifest pattern as `train_terrain_blend_comparison.sh`; PPO uses a
separate entrypoint script, `train_sb3.py`, and is not part of this
comparison):

```bash
./scripts/train_algo_comparison.sh
```

Smoke test before committing to a long run (~a minute or two):

```bash
ISAACLAB_MAX_ITERATIONS=1 ISAACLAB_NUM_ENVS=16 ./scripts/train_isaaclab_docker.sh
# or: ISAACLAB_MAX_ITERATIONS=1 ISAACLAB_NUM_ENVS=16 ./scripts/train_isaaclab_native.sh
```

Open the lightweight interactive watch scene (visual sanity check, one env,
no training):

```bash
./scripts/watch_isaaclab_docker.sh
# or: ./scripts/watch_isaaclab_native.sh
```

Play back a trained checkpoint and render/inspect it:

```bash
ISAACLAB_CHECKPOINT=logs/sb3_sac/LunarRocket-Lander-Direct-v0/<run>/model.zip \
  ./scripts/watch_isaaclab_docker.sh
```

Watch mode uses one environment, a 128x128 visible USD terrain, and no local
detail mesh by default. Override these only when a higher-quality still/demo
is needed.

## 4. Where things end up

- **Checkpoints & TensorBoard logs**: `logs/sb3_sac/LunarRocket-Lander-Direct-v0/<timestamp>/`
  - `model.zip` -- final SAC policy (and `model_vecnormalize.pkl` if
    observation normalization is enabled).
  - `<step>_steps.zip` -- periodic checkpoints from `CheckpointCallback`.
  - `events.out.tfevents.*` -- TensorBoard scalars, including
    `Metrics/success_rate`, `Metrics/timeout_rate`, `Metrics/harsh_landing_rate`,
    `Curriculum/difficulty`, and a full per-episode reward-term breakdown
    (`Episode_Reward/*`). View with:
    ```bash
    tensorboard --logdir logs/sb3_sac
    ```
- **Resuming a run**: pass `ISAACLAB_CHECKPOINT=<path to model.zip>` to the
  training script.
- To validate an environment-mechanics change without a GPU at all:
  ```bash
  python3 -m unittest discover -s tests
  ```
  These are static/logic tests (string/AST checks on the env source, terrain
  math, terrain pool). They do not need Isaac Lab or a GPU and run in under a
  second -- use them as a first check after editing `lunar_lander_env.py`.

## 5. Project layout

```text
source/lunar_rocket_lab/
  lunar_rocket_lab/tasks/direct/lunar_lander/lunar_lander_env.py  # env, reward, curriculum
  lunar_rocket_lab/tasks/direct/lunar_lander/agents/              # SB3 SAC/PPO hyperparameters
  scripts/train_sac.py, scripts/train_sb3.py                      # training entrypoints
  scripts/play_sac.py, scripts/preview_scene.py                   # checkpoint playback / watch scene

assets/rocket/
  hopper_lunar.usd            # Isaac conversion of assets/rocket/hopper_lunar.xml

app/
  terrain_pool.py             # CPU-side terrain-pool batch generator (background thread)
  terrain_generator.py, randomization.py, config.py, ...          # legacy DEM/terrain tooling
                                                                    # reused by the terrain pool
                                                                    # and by ISAACLAB_USE_DEM_TERRAIN=1

docker/
  Dockerfile.isaaclab         # thin build-tools layer on top of nvcr.io/nvidia/isaac-sim:6.0.1

scripts/
  install.sh                  # <- start here
  setup_host.sh, test_docker_gpu.sh
  train_isaaclab_docker.sh, watch_isaaclab_docker.sh
  train_isaaclab_native.sh, watch_isaaclab_native.sh

tests/
  test_isaaclab_static.py, test_terrain_landability.py, test_terrain_pool.py
```

## 6. Observation, reward, and termination

The policy observation is a flat 137-value vector:

- 41 proprioceptive/navigation values (position/velocity/orientation/angular
  velocity/body-frame acceleration relative to the target, current + previous
  9-dim action [throttle + 8 RCS thruster duties], and the target's local
  terrain slope magnitude *and* direction, roughness, and safe-zone score).
- 64 terrain-relative LiDAR range samples.
- 24 local terrain scan samples.
- 8 sensor features: body-frame accelerometer, gyro, altimeter, and nearest
  LiDAR return.

The training task randomizes GPU-resident slope, ripples, craters, and rock
height profiles independently on every environment reset via a continuously
refreshed **terrain pool**:

- 15 NASA DEM macro-surface slots are sampled and uploaded before the first
  reset.
- Each slot also has randomized slope, craters, rocks, and an 8 m
  high-detail regolith patch with centimeter-scale relief.
- A reset only copies one ready slot into the resetting environment; active
  episodes never change underneath the rocket.
- A single background CPU worker prepares the next generation of slots;
  the staged generation is only copied to the GPU between reset assignments,
  never during an active episode.

Tunable via env vars (same names work for both training paths):

```bash
ISAACLAB_TERRAIN_POOL_SIZE=15 \
ISAACLAB_TERRAIN_POOL_REFRESH_ASSIGNMENTS=60 \
ISAACLAB_TERRAIN_POOL_USE_DEM=1 \
ISAACLAB_TERRAIN_POOL_DEM_QUALITY=low_debug \
./scripts/train_isaaclab_docker.sh
```

Reward is dense and terminal:

```text
reward = xy_progress + altitude_progress + stability + velocity
       + guidance + control + terrain + time + terminal
```

`terminal` pays out once per episode based on landing classification
(`soft` / `harsh` / `failed` (out of bounds or escaped) / `timeout`), gated
on foot contact, tilt against the *local terrain normal* (not world-vertical
-- a real slope requires banking to match it), horizontal/vertical/angular
speed, and footprint fit to the terrain. See the per-term weights and the
rationale behind each (`rew_*` fields, with comments on why several of them
are set to counter-intuitive-looking values) directly in
`lunar_lander_env.py`; that file is the single source of truth, kept
up to date as the reward has been iterated on.

A **curriculum** ramps spawn distance, target distance, and the policy's
RCS thruster authority together as success rate improves (advances when a
rolling window's success rate clears `curriculum_success_threshold`, backs
off on sustained underperformance). The window size scales with `num_envs`
so it can't be satisfied by a handful of resets at high env counts, and the
policy-commandable RCS authority ramps smoothly across the *entire*
curriculum range rather than saturating early. Spawn also carries a
randomized horizontal drift velocity (an "orbital approach" cue -- the
rocket starts mid-descent with lateral velocity to null out, not
stationary), ramped by the same curriculum.

**TVC -> RCS conversion.** The rocket used to steer via a two-axis
thrust-vectored (gimballed) main engine; it now has a fixed main engine
(throttle only, no gimbal) plus an 8-thruster reaction-control (RCS) ring
near the top of the body for all attitude control (`rcs_thruster_layout` in
`lunar_lander_env_cfg.py`, and the `rcs_0`..`rcs_7` sites in
`assets/rocket/hopper_lunar.xml`). This was a direct response to the
gimbal-authority ceiling documented below: TVC coupled lateral control
authority to `g * tan(gimbal_angle)` at hover throttle, and every attempt to
raise that angle past ~4-8 deg collapsed training outright (see the archived
history in `lunar_lander_env_cfg.py`'s comments). RCS decouples attitude
torque from throttle entirely. The RCS authority curriculum
(`policy_min/max_rcs_thrust_n`, `curriculum_full_rcs_difficulty`) and the
`rew_wb_rcs_*` control-effort reward term are fresh designs, not retuned
from real training data yet -- treat their defaults as untrained starting
points.

```
        rcs_1 (CW)   rcs_2 (CCW)   rcs_3 (CW)
              \        |        /
  rcs_0 (CCW)--+------[top ring]------+--rcs_4 (CCW)
              /        |        \
        rcs_7 (CW)   rcs_6 (CCW)   rcs_5 (CW)

                 |    main body   |
                 |  (cylinder)    |
                 |________________|
                    |   engine   |     <- fixed, straight down (-Z),
                    |  (thrust)  |        no gimbal
                     \    |    /
                4 legs + feet (unchanged)
```

**A previous training collapse and its fix, for reference.** The most recent
run in this repo's logs (`logs/sb3_sac/.../2026-08-06_13-01-55`) shows
`Metrics/success_rate` collapsing from ~48% to single digits while
`Metrics/timeout_rate` climbs to 80%+ and `Curriculum/difficulty` gets stuck
around 0.3 and never recovers over 3M+ steps -- the policy learns to hover
to the episode timeout instead of attempting a landing. Two compounding
causes, both already fixed in this codebase (compare against `git show
HEAD` if you want to see the before/after):

1. **Reward scale collapsing SAC's entropy.** Terminal rewards span
   +300/-220 and dense per-step terms accumulate to +/-1000+ per episode.
   Left unnormalized (`normalize_value: false`), this collapsed `ent_coef`
   from 0.85 to 0.025 within ~50k steps -- the actor went deterministic
   before it had learned anything, because the value/critic targets
   dominated the SAC update scale. Fixed in
   `agents/sb3_sac_cfg.yaml` via `normalize_value: true`, a wider
   `target_entropy: -1.5` (default is `-action_dim`, `-3` at the time of
   this incident under the old 3-dim TVC action space -- action_dim is now
   9 after the TVC->RCS conversion, so SB3's default heuristic would compute
   differently today; `target_entropy: -1.5` stays an explicit override
   either way, which pushed entropy down even harder combined with the
   scale problem), and
   `gradient_steps: 32` (one update per vector step under-trains SAC when
   a vector step already contributes `num_envs` transitions).
2. **Gimbal-authority cliff at the curriculum level this run happened to
   plateau at.** (Historical -- this run predates the TVC->RCS conversion
   above; the mechanism and fix shape now apply to RCS authority instead,
   see `curriculum_full_rcs_difficulty`.) The policy-commandable gimbal
   range used to ramp to full authority by a fixed, low curriculum level;
   once curriculum difficulty (which the window logic above computed
   independently) reached that level, the hardest episodes in the batch
   suddenly got much larger torque authority than the still-undertrained
   policy could handle. Fixed via `curriculum_full_gimbal_difficulty = 1.0`
   (spreads the ramp across the *entire* curriculum range instead of
   saturating early) and widening the curriculum window so difficulty
   can't outrace what the policy has actually demonstrated.

If you see the same signature again in a new run's TensorBoard logs
(`Curriculum/difficulty` stuck, `Metrics/success_rate` collapsing,
`Metrics/timeout_rate` climbing toward 80%+, `Episode_Reward/xy_progress`
near zero while `Episode_Reward/guidance` and `.../velocity` dominate the
total), start by checking whether `ent_coef` (in `train/ent_coef` on
TensorBoard) collapsed early -- that's the fastest tell for cause 1 above.

## 7. Validation

Static checks (no GPU, no Isaac Lab):

```bash
python3 -m unittest discover -s tests
```

Isaac Lab training smoke test:

```bash
ISAACLAB_MAX_ITERATIONS=1 ISAACLAB_NUM_ENVS=16 ./scripts/train_isaaclab_docker.sh
```

Reset-path smoke test (forces several short episodes, exercises the
terrain-pool reset path quickly):

```bash
ISAACLAB_MAX_ITERATIONS=2 \
ISAACLAB_NUM_ENVS=4 \
ISAACLAB_EPISODE_LENGTH_S=0.2 \
./scripts/train_isaaclab_docker.sh
```

## 8. Troubleshooting

- **`nvidia-smi` fails / not found**: install/repair the NVIDIA proprietary
  driver and reboot before running `install.sh`.
- **`./scripts/install.sh --native` fails the GLIBC check**: your distro's
  GLIBC is older than 2.35 (e.g. Ubuntu 20.04). Use `--docker` instead, or
  upgrade the OS.
- **Docker can't see the GPU**: re-run `./scripts/test_docker_gpu.sh`; if it
  fails, re-run `./scripts/install.sh --docker` (it regenerates the NVIDIA
  CDI spec and restarts Docker), and confirm your user is in the `docker`
  group (`sudo usermod -aG docker $USER`, then log out/in).
- **512 envs run out of VRAM**: lower `ISAACLAB_NUM_ENVS` (halves roughly
  linearly with VRAM use) or `ISAACLAB_TERRAIN_POOL_DEM_QUALITY`.
- **Training success rate collapses and stays down for millions of steps**:
  see the curriculum/gimbal note in section 6 -- check
  `Curriculum/difficulty` and `Metrics/timeout_rate` in TensorBoard first.
