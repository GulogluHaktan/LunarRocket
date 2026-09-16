"""The three detail-transition blend shapes under comparison.

- ``fade_smoothstep``  -- the live RL path's fade, exact formula from
  ``LunarLanderEnv._local_detail_height`` (lunar_lander_env.py:978-979):
  ``t = clamp(1 - r/R, 0, 1); fade = t*t*(3-2*t)``. Exactly 0 for r >= R by
  construction (compact support).

- ``fade_gaussian``    -- the legacy mesh pipeline's fade, exact shape from
  ``MoonTerrainGenerator._blend_local_patch_multiscale``
  (app/terrain_generator.py:956-963): ``exp(-(r/R)**power)``. NOTE: with the
  shipped ``blend_scale=2.0`` and no extra scale constant (k=1 here), this
  does *not* reach 0 at r=R -- exp(-1) ~= 0.368 residual. See
  GAUSSIAN_K_SHIPPED vs GAUSSIAN_K_CALIBRATED below.

- ``fade_hybrid``      -- proposed method: Gaussian near the landing-gear
  contact zone (r <= Rc), smoothstep beyond it out to R, joined by a smooth
  (itself-smoothstep) switch so no new seam is introduced at r=Rc.
"""

from __future__ import annotations

import math

import numpy as np

# exp(-(r/R)^power) evaluated at r=R equals exp(-k) for our reparameterized
# fade_gaussian(r, R, k, power) = exp(-k*(r/R)**power). The shipped legacy
# code is k=1, power=blend_scale=2.0, leaving exp(-1) ~= 36.8% residual at
# the nominal blend radius -- a real, measurable seam (see README "Finding").
GAUSSIAN_K_SHIPPED = 1.0
# Calibrated so the Gaussian's practical support matches R at the 0.1% level
# (exp(-k) = 1e-3), the same "same transition budget" the smoothstep gets
# for free from its compact support. Used as the primary Method-B baseline
# in the fair 3-way comparison; GAUSSIAN_K_SHIPPED is reported separately as
# a diagnostic of the as-shipped legacy behavior.
GAUSSIAN_K_CALIBRATED = math.log(1000.0)


def _smoothstep01(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def fade_smoothstep(r: np.ndarray, radius: float) -> np.ndarray:
    t = np.clip(1.0 - r / radius, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def fade_gaussian(
    r: np.ndarray, radius: float, k: float = GAUSSIAN_K_CALIBRATED, power: float = 2.0
) -> np.ndarray:
    return np.exp(-k * (r / radius) ** power)


def switch_weight(r: np.ndarray, contact_radius: float, width: float) -> np.ndarray:
    """1 for r <= contact_radius - width/2, 0 for r >= contact_radius + width/2."""
    lo = contact_radius - width / 2.0
    hi = contact_radius + width / 2.0
    t = (hi - r) / max(hi - lo, 1e-9)
    return _smoothstep01(t)


def fade_hybrid(
    r: np.ndarray,
    radius: float,
    contact_radius: float,
    switch_width: float = 0.3,
    gaussian_k: float = GAUSSIAN_K_CALIBRATED,
) -> np.ndarray:
    w = switch_weight(r, contact_radius, switch_width)
    return w * fade_gaussian(r, radius, k=gaussian_k) + (1.0 - w) * fade_smoothstep(r, radius)


BLEND_METHODS = ("smoothstep", "gaussian_shipped", "gaussian_calibrated", "hybrid")


def make_fade_fn(method: str, radius: float, contact_radius: float | None = None):
    if method == "smoothstep":
        return lambda r: fade_smoothstep(r, radius)
    if method == "gaussian_shipped":
        return lambda r: fade_gaussian(r, radius, k=GAUSSIAN_K_SHIPPED)
    if method == "gaussian_calibrated":
        return lambda r: fade_gaussian(r, radius, k=GAUSSIAN_K_CALIBRATED)
    if method == "hybrid":
        assert contact_radius is not None, "hybrid requires contact_radius"
        return lambda r: fade_hybrid(r, radius, contact_radius, gaussian_k=GAUSSIAN_K_CALIBRATED)
    raise ValueError(f"unknown blend method: {method}")
