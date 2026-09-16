from __future__ import annotations

import numpy as np


def generate_terrain_pool_batch(
    *,
    seed: int,
    pool_size: int,
    crater_count: int,
    rock_count: int,
    slope_scale: float,
    height_scale: float,
    safe_feature_radius: float,
    detail_amplitude_range: tuple[float, float],
) -> dict[str, np.ndarray]:
    """Generate a deterministic terrain-parameter batch for asynchronous staging."""
    if pool_size < 1:
        raise ValueError("pool_size must be positive")
    if not 0.0 <= detail_amplitude_range[0] <= detail_amplitude_range[1]:
        raise ValueError("detail_amplitude_range must be ordered and non-negative")

    rng = np.random.default_rng(seed)

    def sample_annulus(feature_count: int, max_radius: float) -> np.ndarray:
        angle = rng.uniform(0.0, 2.0 * np.pi, size=(pool_size, feature_count))
        radius = np.sqrt(
            rng.uniform(
                safe_feature_radius * safe_feature_radius,
                max_radius * max_radius,
                size=(pool_size, feature_count),
            )
        )
        return np.stack((radius * np.cos(angle), radius * np.sin(angle)), axis=-1).astype(np.float32)

    return {
        "slope": rng.uniform(-slope_scale, slope_scale, size=(pool_size, 2)).astype(np.float32),
        "phase": rng.uniform(0.0, 2.0 * np.pi, size=(pool_size, 2)).astype(np.float32),
        "amplitude": rng.uniform(0.05, height_scale, size=pool_size).astype(np.float32),
        "detail_phase": rng.uniform(0.0, 2.0 * np.pi, size=(pool_size, 4)).astype(np.float32),
        "detail_amplitude": rng.uniform(*detail_amplitude_range, size=pool_size).astype(np.float32),
        "crater_xy": sample_annulus(crater_count, 8.0),
        "crater_radius": rng.uniform(1.2, 4.5, size=(pool_size, crater_count)).astype(np.float32),
        "crater_depth": rng.uniform(0.08, 0.35, size=(pool_size, crater_count)).astype(np.float32),
        "rock_xy": sample_annulus(rock_count, 7.0),
        "rock_radius": rng.uniform(0.25, 0.65, size=(pool_size, rock_count)).astype(np.float32),
        "rock_height": rng.uniform(0.05, 0.22, size=(pool_size, rock_count)).astype(np.float32),
    }
