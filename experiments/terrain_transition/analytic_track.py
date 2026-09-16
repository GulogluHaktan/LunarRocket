"""Analytic track: the live RL environment's height-field representation.

No mesh at all -- ``LunarLanderEnv._terrain_height`` (lunar_lander_env.py:904)
is a closed-form function evaluated per query point (LiDAR rays, terrain
scan samples, foot-clearance checks), batched across all parallel envs every
physics step. What matters for training throughput here is per-step query
evaluation time, not "generation time" in the mesh sense -- there's nothing
to regenerate; the terrain pool (app/terrain_pool.py) only resamples the
random *parameters*, independent of which blend shape is used to fade the
fine layer in.
"""

from __future__ import annotations

import time

import numpy as np

from blend_functions import make_fade_fn
from terrain_layers import TerrainSlot, fine_height, macro_height

# Matches LunarLanderEnvCfg defaults: lidar_ray_count=64, terrain_scan_count=24,
# 4 foot-clearance queries per env, at the ISAACLAB_NUM_ENVS=512 default.
DEFAULT_NUM_ENVS = 512
DEFAULT_QUERIES_PER_ENV = 64 + 24 + 4


def make_combined_height_fn(
    slot: TerrainSlot,
    method: str,
    radius_m: float,
    contact_radius_m: float | None = None,
    detail_only: bool = False,
):
    """Deployable height field: the fine-detail contribution is hard-truncated
    to exactly 0 at r >= radius_m.

    This matters specifically for the Gaussian shapes: exp(-(r/R)**p) never
    reaches exactly 0 analytically (unbounded, C-infinity smooth), so left
    untruncated it never shows a seam at any finite radius -- which would
    hide the actual engineering problem. Any deployable representation
    (mesh, heightmap, or "no detail beyond the target's local patch")
    necessarily truncates the fine layer somewhere; forcing that truncation
    to happen at the *same* nominal radius_m for every method is what makes
    this a fair, same-transition-budget comparison. Smoothstep and the
    calibrated Gaussian are engineered so this truncation is (near-)lossless;
    the as-shipped Gaussian (k=1) is not -- see README "Finding".

    ``detail_only=True`` returns just the truncated fade*fine contribution,
    isolating the blend method's own seam from ordinary macro-terrain
    roughness (used for the headline continuity metric).
    """
    fade_fn = make_fade_fn(method, radius_m, contact_radius_m)

    def height_fn(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        r = np.sqrt(x**2 + y**2)
        fade = np.where(r < radius_m, fade_fn(r), 0.0)
        detail = fade * fine_height(x, y, slot)
        if detail_only:
            return detail
        return macro_height(x, y, slot) + detail

    return height_fn


def benchmark_query_throughput(
    height_fn,
    n_queries: int = DEFAULT_NUM_ENVS * DEFAULT_QUERIES_PER_ENV,
    repeats: int = 50,
    seed: int = 0,
) -> dict[str, float]:
    """Wall-clock cost of one "RL step's worth" of terrain-height queries."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-20.0, 20.0, size=n_queries)
    y = rng.uniform(-20.0, 20.0, size=n_queries)

    height_fn(x, y)  # warm-up (page faults, etc.)

    times = np.empty(repeats)
    for i in range(repeats):
        t0 = time.perf_counter()
        height_fn(x, y)
        times[i] = time.perf_counter() - t0

    return {
        "n_queries": n_queries,
        "mean_s": float(np.mean(times)),
        "std_s": float(np.std(times)),
        "min_s": float(np.min(times)),
        "queries_per_second": float(n_queries / np.mean(times)),
    }


def benchmark_fade_only(fade_fn, n_queries: int = DEFAULT_NUM_ENVS * DEFAULT_QUERIES_PER_ENV, repeats: int = 200, seed: int = 0) -> dict[str, float]:
    """Isolated cost of the fade formula itself, decoupled from macro/fine terrain evaluation."""
    rng = np.random.default_rng(seed)
    r = rng.uniform(0.0, 16.0, size=n_queries)

    fade_fn(r)
    times = np.empty(repeats)
    for i in range(repeats):
        t0 = time.perf_counter()
        fade_fn(r)
        times[i] = time.perf_counter() - t0
    return {"mean_s": float(np.mean(times)), "std_s": float(np.std(times))}
