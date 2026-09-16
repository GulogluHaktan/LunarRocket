"""Macro/fine terrain height layers, ported 1:1 from the production formulas.

Macro layer mirrors ``LunarLanderEnv._procedural_height_raw`` (minus the
optional NASA-DEM ``base_height`` term, so this module has zero external
asset/GPU dependencies) in
``source/lunar_rocket_lab/lunar_rocket_lab/tasks/direct/lunar_lander/lunar_lander_env.py``.

Fine layer mirrors ``LunarLanderEnv._local_detail_height`` (same file), minus
the "subtract detail-at-target" trick (that just re-zeroes the target point;
irrelevant to a blend-shape study and would be a needless deviation from a
generic radial analysis).

Random terrain *content* (slope, ripple, craters, rocks, fine-detail
amplitude/phase) is sampled with the actual production generator,
``app.terrain_pool.generate_terrain_pool_batch``, so trial-to-trial content
is drawn from the exact distribution used by real training runs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.terrain_pool import generate_terrain_pool_batch

# Defaults copied from LunarLanderEnvCfg (lunar_lander_env.py lines ~157-162).
TERRAIN_HEIGHT_SCALE_M = 0.45
TERRAIN_SLOPE_SCALE = 0.035
TERRAIN_CRATER_COUNT = 3
TERRAIN_ROCK_COUNT = 5
TERRAIN_DETAIL_AMPLITUDE_RANGE_M = (0.006, 0.020)
TERRAIN_DETAIL_RADIUS_M = 8.0  # == legacy blend_radius_m default; shared transition budget


@dataclass(frozen=True)
class TerrainSlot:
    slope: np.ndarray  # (2,)
    phase: np.ndarray  # (2,)
    amplitude: float
    detail_phase: np.ndarray  # (4,)
    detail_amplitude: float
    crater_xy: np.ndarray  # (crater_count, 2)
    crater_radius: np.ndarray  # (crater_count,)
    crater_depth: np.ndarray  # (crater_count,)
    rock_xy: np.ndarray  # (rock_count, 2)
    rock_radius: np.ndarray  # (rock_count,)
    rock_height: np.ndarray  # (rock_count,)


def sample_terrain_slot(seed: int) -> TerrainSlot:
    """Draw one terrain-content sample using the real production generator."""
    batch = generate_terrain_pool_batch(
        seed=seed,
        pool_size=1,
        crater_count=TERRAIN_CRATER_COUNT,
        rock_count=TERRAIN_ROCK_COUNT,
        slope_scale=TERRAIN_SLOPE_SCALE,
        height_scale=TERRAIN_HEIGHT_SCALE_M,
        safe_feature_radius=0.0,
        detail_amplitude_range=TERRAIN_DETAIL_AMPLITUDE_RANGE_M,
    )
    return TerrainSlot(
        slope=batch["slope"][0],
        phase=batch["phase"][0],
        amplitude=float(batch["amplitude"][0]),
        detail_phase=batch["detail_phase"][0],
        detail_amplitude=float(batch["detail_amplitude"][0]),
        crater_xy=batch["crater_xy"][0],
        crater_radius=batch["crater_radius"][0],
        crater_depth=batch["crater_depth"][0],
        rock_xy=batch["rock_xy"][0],
        rock_radius=batch["rock_radius"][0],
        rock_height=batch["rock_height"][0],
    )


def macro_height(x: np.ndarray, y: np.ndarray, slot: TerrainSlot) -> np.ndarray:
    """Coarse global terrain: slope + ripple + craters + rocks. Matches
    ``_procedural_height_raw`` (lunar_lander_env.py:982-1015) minus the DEM term."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    slope = slot.slope[0] * x + slot.slope[1] * y
    ripple = slot.amplitude * (
        0.55 * np.sin(0.55 * x + slot.phase[0]) + 0.45 * np.sin(0.45 * y + slot.phase[1])
    )

    dxc = x[..., None] - slot.crater_xy[:, 0]
    dyc = y[..., None] - slot.crater_xy[:, 1]
    cdist = np.sqrt(dxc**2 + dyc**2)
    csigma = np.clip(slot.crater_radius * 0.55, 1e-3, None)
    bowl = -slot.crater_depth * np.exp(-0.5 * (cdist / csigma) ** 2)
    rsigma = np.clip(slot.crater_radius * 0.18, 1e-3, None)
    rim = 0.35 * slot.crater_depth * np.exp(-0.5 * ((cdist - slot.crater_radius) / rsigma) ** 2)
    crater_total = np.sum(bowl + rim, axis=-1)

    dxr = x[..., None] - slot.rock_xy[:, 0]
    dyr = y[..., None] - slot.rock_xy[:, 1]
    rdist = np.sqrt(dxr**2 + dyr**2)
    rradius = np.clip(slot.rock_radius, 1e-3, None)
    rock_profile = slot.rock_height * np.clip(1.0 - rdist / rradius, 0.0, None) ** 2
    rock_total = np.sum(rock_profile, axis=-1)

    return slope + ripple + crater_total + rock_total


def fine_height(x: np.ndarray, y: np.ndarray, slot: TerrainSlot) -> np.ndarray:
    """Fine "regolith" detail layer. Matches ``_local_detail_height``
    (lunar_lander_env.py:954-972), sine-product terms only."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ph = slot.detail_phase
    amp = slot.detail_amplitude
    return amp * (
        0.55 * np.sin(3.2 * x + ph[0]) * np.sin(2.7 * y + ph[1])
        + 0.30 * np.sin(6.4 * x + ph[2]) * np.sin(5.8 * y + ph[3])
        + 0.15 * np.sin(9.5 * (x + y) + ph[0])
    )
