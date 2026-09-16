# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Isaac Lab direct-workflow RL task: a SAC (Soft Actor-Critic) policy learns to land a 4-legged
hopper rocket with a two-axis thrust-vectored (gimballed) engine on real cratered/sloped lunar
terrain at a randomized target. The one task that matters is
`LunarRocket-Lander-Direct-v0`, registered and implemented under `source/lunar_rocket_lab/`.

Training requires Isaac Lab/Isaac Sim + a CUDA GPU and normally runs in Docker or a native
Isaac Sim venv, not in the environment Claude Code itself runs in — Claude Code cannot execute
a real training run directly. What Claude Code *can* run directly is the static test suite
(no GPU, no Isaac Lab import needed).

## Commands

```bash
# Static/logic tests — the only thing Claude Code should run to validate changes.
# No GPU, no Isaac Lab, no torch required. ~23 tests, runs in under a second.
python3 -m unittest discover -s tests

# Run a single test file / case
python3 -m unittest tests.test_isaaclab_static -v
python3 -m unittest tests.test_isaaclab_static.IsaacLabStaticTests.test_reward_is_progress_based_and_logs_completed_episodes
```

Everything below needs a real GPU + Isaac Lab and is normally run by the user, not Claude:

```bash
./scripts/install.sh                     # first-time setup (Docker or native, asks which)
./scripts/train_isaaclab_docker.sh       # train (Docker path)
./scripts/train_isaaclab_native.sh       # train (native path)
./scripts/watch_isaaclab_docker.sh       # interactive one-env visual scene, no training
./scripts/train_terrain_blend_comparison.sh   # sequential multi-seed (blend method x seed) SAC runs,
                                              # resumable via a JSONL manifest, then auto-analyzes the logs
make setup / make train / make smoke / make test / make gpu   # thin wrappers, see Makefile
                                              # NOTE: `make train` defaults to ISAACLAB_MAX_ITERATIONS=200,
                                              # NOT the 1200 the training scripts use directly.

# Smoke test after an env change (~1-2 min, still needs GPU/Isaac Lab):
ISAACLAB_MAX_ITERATIONS=1 ISAACLAB_NUM_ENVS=16 ./scripts/train_isaaclab_docker.sh
```

`ISAACLAB_ALGO` (`sac`|`ppo`|`td3`|`ddpg`), `ISAACLAB_NUM_ENVS`, `ISAACLAB_MAX_ITERATIONS`, `ISAACLAB_DEVICE`,
and the `ISAACLAB_TERRAIN_POOL_*` family are the env vars used to configure a run; they're read
by `scripts/isaaclab_train_entrypoint.sh` and the training scripts under
`source/lunar_rocket_lab/scripts/`. See `README.md` sections 3 and 6 for the full list.

## Architecture

There are **two parallel codebases** in this repo (plus a third, standalone research tree) —
know which one you're editing:

1. **`source/lunar_rocket_lab/`** — the current, actively-trained Isaac Lab extension. This is
   an installable Python package (`pyproject.toml`, name `lunar-rocket-lab`) containing:
   - `lunar_rocket_lab/tasks/direct/lunar_lander/lunar_lander_env.py` — the entire env: physics
     setup, observation/reward/termination logic, curriculum. This is the single source of
     truth for reward weights and their rationale (read the inline comments before changing any
     `rew_*` field or curriculum constant — several values look counter-intuitive on purpose,
     e.g. `curriculum_full_gimbal_difficulty = 1.0`, `target_entropy: -1.5` in the SAC config,
     documented as fixes for a specific training collapse in README.md section 6).
   - `lunar_rocket_lab/tasks/direct/lunar_lander/agents/sb3_sac_cfg.yaml` /
     `sb3_ppo_cfg.yaml` — SB3 hyperparameters, registered as `sb3_sac_cfg_entry_point` /
     `sb3_cfg_entry_point` via `gym.register` in the task's `__init__.py`.
   - `scripts/train_sac.py`, `scripts/train_sb3.py` — training entrypoints invoked by
     `isaaclab_train_entrypoint.sh` inside the Isaac Lab container/venv.
   - `scripts/play_sac.py`, `scripts/preview_scene.py` — checkpoint playback / visual watch
     scene, invoked by `watch_isaaclab_docker.sh` / `watch_isaaclab_native.sh`.

2. **`app/`** — a legacy, pre-Isaac-Lab standalone environment (`env_lunar_landing.py`,
   `reward_function.py`, `observation_builder.py`, `landing_classifier.py`, `world_builder.py`,
   `world_adapter.py`/`analytic_world_adapter.py`, video/demo recording scripts, etc.), still
   covered by `tests/test_rl_scaffold.py` and `tests/test_static_smoke.py`. Most of `app/` is
   not on the training path anymore, with two exceptions that *are* live:
   - `app/terrain_pool.py` — CPU-side batch generator (`generate_terrain_pool_batch`) imported
     directly by `lunar_lander_env.py`, used by the background terrain-refresh worker.
   - `app/terrain_generator.py` / `app/randomization.py` / `app/config.py` — legacy DEM/terrain
     tooling, reused by the terrain pool and by the (opt-in) `ISAACLAB_USE_DEM_TERRAIN=1` path.
   Treat the rest of `app/` as reference material for the legacy design, not a place new
   Isaac Lab features belong.

3. **`experiments/terrain_transition/`** — a self-contained CPU/NumPy research study (no GPU,
   no Isaac Sim) backing a paper comparing local-detail terrain blend shapes (smoothstep /
   Gaussian / a proposed hybrid). It has its *own* `.venv` and is not imported by the training
   path. Note the two shipped blend implementations it treats as baselines: the **live RL env**
   fades local detail in with a **smoothstep** analytic height field
   (`LunarLanderEnv._local_detail_height`, `terrain_detail_radius_m = 8.0`); the **legacy
   `ISAACLAB_USE_DEM_TERRAIN=1` mesh pipeline** uses a **Gaussian** falloff
   (`MoonTerrainGenerator._blend_local_patch_multiscale`). `scripts/train_terrain_blend_comparison.sh`
   is the GPU follow-up that runs the same comparison as real multi-seed SAC training.

**Terrain pool.** The training env doesn't procedurally generate terrain synchronously on
reset. A background CPU thread (`ThreadPoolExecutor`) continuously stages a pool of terrain
"slots" (NASA DEM macro-surface + randomized slope/craters/rocks/high-detail regolith patch);
a reset only ever copies one already-ready slot onto the GPU for that environment. Active
episodes are never mutated mid-episode — new generations are only swapped in between reset
assignments. This is why terrain-related changes usually touch both `lunar_lander_env.py` (GPU
upload/assignment logic) and `app/terrain_pool.py` / `app/terrain_generator.py` (CPU-side
generation), and why `tests/test_terrain_pool.py` and `tests/test_terrain_landability.py` exist
as fast CPU-only checks of that generation logic.

**Curriculum.** Spawn distance, target distance, and the policy's commandable gimbal authority
ramp together as a rolling-window success rate improves (`curriculum_success_threshold`,
`curriculum_window_episodes`, `curriculum_increment` in `LunarLanderEnvCfg`). The window size
scales with `num_envs` so it can't be satisfied by a handful of resets at high env counts —
this was the direct fix for a documented training collapse (README.md section 6); don't
reintroduce a fixed/small window without accounting for that.

**Reward.** Dense + terminal:
`reward = xy_progress + altitude_progress + stability + velocity + guidance + control + terrain + time + terminal`.
`terminal` is a one-time-per-episode payout based on landing classification (`soft`/`harsh`/
`failed`/`timeout`), gated on foot contact, tilt against the *local terrain normal* (not
world-vertical), and speed/footprint-fit thresholds. Weight/threshold changes should be
validated against `tests/test_isaaclab_static.py`, which asserts on literal source strings in
`lunar_lander_env.py` (string/AST-based checks, not behavioral) — update both together.

**Test suite is static-only.** All five files under `tests/` are logic/string/AST checks on
source files or pure-Python terrain math — none import Isaac Lab or need a GPU. This is the
one part of the repo Claude Code should actually execute; anything that needs simulation
(training runs, checkpoint playback, visual scenes) needs to be described/reviewed, not run,
unless the user is driving it on their own GPU machine.

## Notes

- `LunarRocket.zip` and `recovered/` in the repo root are not part of the source tree — ignore
  them unless the user specifically asks about them.
- `test_two_tier_terrain.py` in the repo root is a standalone legacy `app/`-terrain check, run
  directly (`python3 test_two_tier_terrain.py`); it is *not* picked up by
  `python3 -m unittest discover -s tests`.
- Several top-level `*.md` files (`IMPLEMENTATION_SUMMARY.md`, `TWO_TIER_LANDING.md`,
  `TERRAIN_QUALITY.md`, `TERRAIN_QUICK_REFERENCE.md`, `VERIFICATION_CHECKLIST.md`) document the
  legacy `app/`-based terrain system, not the current Isaac Lab terrain pool — useful for
  understanding the DEM/two-tier terrain *concepts* (still reused), but not authoritative for
  current behavior; `lunar_lander_env.py` and `app/terrain_pool.py` are.
- `README.md` is written for a fresh machine setup and is kept detailed and current — check it
  first for anything about installation, env vars, or the documented training-collapse
  postmortem before re-deriving that information from source.
