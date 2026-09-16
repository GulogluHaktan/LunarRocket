from __future__ import annotations

import re
import unittest
from pathlib import Path

import numpy as np

from app.terrain_pool import generate_terrain_pool_batch

ROOT = Path(__file__).resolve().parents[1]
_LUNAR_LANDER_DIR = ROOT / "source/lunar_rocket_lab/lunar_rocket_lab/tasks/direct/lunar_lander"
ENV_SOURCE = _LUNAR_LANDER_DIR / "lunar_lander_env.py"
# LunarLanderEnvCfg lives in its own module (see lunar_lander_env.py's import
# comment) so gym.register's env_cfg_entry_point can resolve without pulling
# in DirectRLEnv/pxr. Reward/cfg literals this file greps for now live there.
CFG_SOURCE = _LUNAR_LANDER_DIR / "lunar_lander_env_cfg.py"


def _extract_float(source: str, name: str) -> float:
    match = re.search(rf"^\s*{name}\s*=\s*([0-9.eE+-]+)", source, re.MULTILINE)
    if match is None:
        raise AssertionError(f"could not find `{name} = <value>` in env source")
    return float(match.group(1))


def _foot_offsets_xy(source: str) -> np.ndarray:
    match = re.search(r"rocket_foot_offsets_m = \((.*?)\n    \)", source, re.DOTALL)
    if match is None:
        raise AssertionError("could not find rocket_foot_offsets_m in env source")
    tuples = re.findall(r"\(([^()]+)\)", match.group(1))
    offsets = np.array([[float(v) for v in t.split(",")] for t in tuples])
    return offsets[:, :2]


class TerrainLandabilityTests(unittest.TestCase):
    """The soft-landing gate must stay reachable on the REAL, unflattened terrain.

    A perfectly upright, motionless rocket only satisfied the old
    world-vertical foot_clearance_spread gate 30-65% of the time on this
    terrain (median local slope ~9-13 deg vs. a 10 deg tilt budget) because
    landing on a slope geometrically requires banking to match it -- judging
    tilt against world vertical made the gate nearly unreachable regardless of
    policy skill.

    The fix (_tilt_against_terrain) judges tilt against the real terrain's own
    surface normal at the target instead of world vertical, so the gate
    becomes a check on residual attitude error after banking to the slope, not
    on the slope's magnitude. These tests prove that residual error stays
    small using the same procedural terrain math _procedural_height_raw uses
    (craters, rocks, slope, ripple) -- without touching the real cached DEM
    asset, which is intentionally kept out of git (see test_static_smoke.py's
    synthetic-DEM convention for the same reason).
    """

    def setUp(self) -> None:
        self.source = ENV_SOURCE.read_text() + "\n" + CFG_SOURCE.read_text()

    def test_no_terrain_flattening_around_the_target(self) -> None:
        # An earlier version of this fix flattened terrain into an invisible
        # flat pad near the target to make the gate reachable. That was
        # reverted: the rocket must land on the real cratered/sloped surface.
        self.assertNotIn("_blend_safe_zone", self.source)
        self.assertNotIn("terrain_safe_radius_m", self.source)
        self.assertIn(
            "return raw_height + self._local_detail_height(xy_w, env_ids, origin_xy, radius)",
            self.source,
        )

    def test_tilt_is_judged_against_the_local_terrain_normal_near_touchdown(self) -> None:
        self.assertIn("def _terrain_normal(", self.source)
        self.assertIn("def _tilt_against_terrain(", self.source)
        # Reward side (whiteboard's e^(IMU theta)) takes the angle; the
        # termination gate takes the penalty form. Both must come from the
        # terrain-normal-aware helper, never from raw world-vertical tilt.
        self.assertIn("_, tilt_angle = self._tilt_against_terrain(quat, near_ground)", self.source)
        self.assertIn("tilt_penalty, tilt_angle = self._tilt_against_terrain(quat, near_ground)", self.source)

    def test_policy_observes_target_slope_direction_not_just_magnitude(self) -> None:
        # Without direction (which way is downhill), the policy has no signal
        # for which way to bank to align with _tilt_against_terrain's target.
        self.assertIn("grad_x_norm = torch.clamp(grad_x / 0.5", self.source)
        self.assertIn("grad_y_norm = torch.clamp(grad_y / 0.5", self.source)
        self.assertIn("observation_space = 125", self.source)

    def test_gimbal_authority_ramps_with_curriculum_instead_of_fixed_full_range(self) -> None:
        self.assertIn("curriculum_full_gimbal_difficulty", self.source)
        self.assertIn("gimbal_frac = torch.clamp(", self.source)

    def test_timeout_penalty_is_worse_than_the_best_possible_harsh_landing(self) -> None:
        # Otherwise SAC learns hovering to timeout is competitive with (or
        # better than) actually attempting a landing.
        rew_harsh_landing = _extract_float(self.source, "rew_harsh_landing")
        rew_landing_quality = _extract_float(self.source, "rew_landing_quality")
        rew_timeout = _extract_float(self.source, "rew_timeout")
        self.assertLess(rew_timeout, rew_harsh_landing + rew_landing_quality)

    def test_matching_local_slope_leaves_only_a_small_curvature_residual(self) -> None:
        """If the vehicle perfectly banks to the target's local terrain
        gradient, the only remaining foot-clearance spread comes from terrain
        curvature within the ~0.8 m footprint (craters/rocks aren't exactly
        planar there). Prove that residual comfortably clears the landing
        gate on real procedural terrain: the redesigned gate is geometrically
        reachable by a policy that learns to track the slope, not just in the
        idealized flat-ground case the original gate assumed.
        """
        gate_m = _extract_float(self.source, "landing_max_foot_clearance_m")
        foot_offsets_xy = _foot_offsets_xy(self.source)

        pool = generate_terrain_pool_batch(
            seed=123,
            pool_size=8,
            crater_count=3,
            rock_count=5,
            slope_scale=0.035,
            height_scale=0.45,
            safe_feature_radius=0.0,
            detail_amplitude_range=(0.006, 0.020),
        )

        rng = np.random.default_rng(0)
        n_targets = 2000
        eps = 0.35
        worst_fraction_over_gate = 0.0
        for slot in range(pool["slope"].shape[0]):
            slope_xy = pool["slope"][slot]
            phase = pool["phase"][slot]
            amp = float(pool["amplitude"][slot])
            crater_xy = pool["crater_xy"][slot]
            crater_radius = pool["crater_radius"][slot]
            crater_depth = pool["crater_depth"][slot]
            rock_xy = pool["rock_xy"][slot]
            rock_radius = pool["rock_radius"][slot]
            rock_height = pool["rock_height"][slot]

            def height(xy: np.ndarray) -> np.ndarray:
                slope_term = slope_xy[0] * xy[:, 0] + slope_xy[1] * xy[:, 1]
                ripple = amp * (
                    0.55 * np.sin(0.55 * xy[:, 0] + phase[0])
                    + 0.45 * np.sin(0.45 * xy[:, 1] + phase[1])
                )
                crater_dist = np.linalg.norm(xy[:, None, :] - crater_xy[None, :, :], axis=-1)
                crater_sigma = np.clip(crater_radius * 0.55, 1e-3, None)
                bowl = -crater_depth * np.exp(-0.5 * (crater_dist / crater_sigma) ** 2)
                rim_sigma = np.clip(crater_radius * 0.18, 1e-3, None)
                rim = 0.35 * crater_depth * np.exp(-0.5 * ((crater_dist - crater_radius) / rim_sigma) ** 2)
                rock_dist = np.linalg.norm(xy[:, None, :] - rock_xy[None, :, :], axis=-1)
                rock_r = np.clip(rock_radius, 1e-3, None)
                rocks = rock_height * np.clip(1.0 - rock_dist / rock_r, 0.0, None) ** 2
                return slope_term + ripple + np.sum(bowl + rim, axis=-1) + np.sum(rocks, axis=-1)

            targets = rng.uniform(-14.0, 14.0, size=(n_targets, 2))
            h0 = height(targets)
            gx = (height(targets + np.array([eps, 0.0])) - h0) / eps
            gy = (height(targets + np.array([0.0, eps])) - h0) / eps

            foot_xy = targets[:, None, :] + foot_offsets_xy[None, :, :]
            h_actual = height(foot_xy.reshape(-1, 2)).reshape(n_targets, -1)
            h_plane = (
                h0[:, None]
                + gx[:, None] * foot_offsets_xy[None, :, 0]
                + gy[:, None] * foot_offsets_xy[None, :, 1]
            )
            residual = h_actual - h_plane
            residual_spread = residual.max(axis=1) - residual.min(axis=1)
            worst_fraction_over_gate = max(
                worst_fraction_over_gate, float((residual_spread > gate_m).mean())
            )

        self.assertLess(
            worst_fraction_over_gate,
            0.05,
            "more than 5% of random targets leave a curvature-only foot-clearance "
            "spread above the landing gate even when perfectly banked to the local "
            "slope -- the gate may be geometrically unreachable again",
        )


if __name__ == "__main__":
    unittest.main()
