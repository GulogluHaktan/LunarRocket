from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from app.landing_classifier import classify_landing
from app.observation_builder import build_observation
from app.reward_function import compute_reward
from app.rl_types import ContactState, EpisodeTimeState, RewardMemory, RocketKinematicState, TerrainSafetyState


def _state() -> RocketKinematicState:
    return RocketKinematicState(
        position=np.array([0.2, 0.1, 0.05], dtype=np.float32),
        target_position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        linear_velocity=np.array([0.05, 0.02, -0.2], dtype=np.float32),
        quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        angular_velocity=np.array([0.01, 0.01, 0.01], dtype=np.float32),
        linear_acceleration=np.zeros(3, dtype=np.float32),
        altitude_to_ground=0.05,
        main_thrust=0.4,
        gimbal_x=0.0,
        gimbal_y=0.0,
    )


def _terrain() -> TerrainSafetyState:
    return TerrainSafetyState(
        local_slope_deg=4.0,
        local_roughness_m=0.04,
        safe_zone_score=0.9,
        terrain_height_at_target=0.0,
        slope_norm=4.0 / 45.0,
        roughness_norm=0.04,
    )


class RlScaffoldTests(unittest.TestCase):
    def test_observation_shapes(self) -> None:
        obs = build_observation(
            _state(),
            {
                "lidar": np.ones(32, dtype=np.float32) * 10.0,
                "depth": np.ones((64, 64), dtype=np.float32) * 5.0,
                "rgb": np.zeros((84, 84, 3), dtype=np.uint8),
            },
            _terrain(),
            np.zeros(3, dtype=np.float32),
        )
        self.assertEqual(obs["state"].shape, (27,))
        self.assertEqual(obs["lidar"].shape, (32,))
        self.assertEqual(obs["depth"].shape, (1, 64, 64))
        self.assertEqual(obs["rgb"].shape, (3, 84, 84))

    def test_state_lidar_observation_excludes_images(self) -> None:
        obs = build_observation(
            _state(),
            {"lidar": np.ones(32, dtype=np.float32) * 10.0},
            _terrain(),
            np.zeros(3, dtype=np.float32),
            {"include_lidar": True, "include_depth": False, "include_rgb": False},
        )
        self.assertEqual(set(obs), {"state", "lidar"})
        self.assertEqual(obs["state"].shape, (27,))
        self.assertEqual(obs["lidar"].shape, (32,))

    def test_soft_landing_classification(self) -> None:
        contact = ContactState(
            leg_contact=np.array([True, True, True, True]),
            leg_contact_force=np.array([10.0, 11.0, 9.0, 10.0], dtype=np.float32),
            body_contact=False,
        )
        landing = classify_landing(
            _state(),
            contact,
            _terrain(),
            EpisodeTimeState(step_count=20, elapsed_s=1.0, max_steps=1000, max_episode_s=50.0),
        )
        self.assertEqual(landing, "soft_landing")

    def test_reward_breakdown_contains_terms(self) -> None:
        contact = ContactState(
            leg_contact=np.array([False, False, False, False]),
            leg_contact_force=np.zeros(4, dtype=np.float32),
            body_contact=False,
        )
        flying_state = replace(_state(), position=np.array([0.2, 0.1, 5.0], dtype=np.float32), altitude_to_ground=5.0)
        reward, terminated, info = compute_reward(
            flying_state,
            np.array([0.4, 0.1, -0.1], dtype=np.float32),
            np.array([0.3, 0.0, 0.0], dtype=np.float32),
            contact,
            _terrain(),
            RewardMemory(target_distance_xy=1.0),
            EpisodeTimeState(step_count=10, elapsed_s=0.5, max_steps=1000, max_episode_s=50.0),
        )
        self.assertFalse(terminated)
        self.assertIsInstance(reward, float)
        self.assertIn("reward_terms", info)
        self.assertIn("metrics", info)
        self.assertIn("progress", info["reward_terms"])
        self.assertEqual(info["metrics"]["landing_type"], "flying")


if __name__ == "__main__":
    unittest.main()
