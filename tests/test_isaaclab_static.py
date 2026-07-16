from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENV_SOURCE = ROOT / "source/lunar_rocket_lab/lunar_rocket_lab/tasks/direct/lunar_lander/lunar_lander_env.py"


class IsaacLabStaticTests(unittest.TestCase):
    def test_isaaclab_env_uses_terrain_relative_observations_and_dones(self) -> None:
        source = ENV_SOURCE.read_text()

        self.assertIn("observation_space = 123", source)
        self.assertIn("terrain_height = self._terrain_height(pos[:, :2])", source)
        self.assertIn("altitude = pos[:, 2] - terrain_height", source)
        self.assertIn("terrain_metrics = self._terrain_metrics", source)
        self.assertIn("lidar = self._lidar_ranges", source)
        self.assertIn("terrain_scan = self._terrain_scan", source)
        self.assertIn("sensor_features = self._sensor_features", source)
        self.assertIn("lidar_ray_count = 64", source)

    def test_isaaclab_thrust_is_body_frame_and_generates_torque(self) -> None:
        source = ENV_SOURCE.read_text()

        self.assertIn("thrust_dir_b", source)
        self.assertIn("thrust_dir_w = _quat_rotate(quat, thrust_dir_b)", source)
        self.assertIn("torch.cross(lever_w, force_w", source)

    def test_isaaclab_manual_rocket_spawn_uses_concrete_source_prim(self) -> None:
        source = ENV_SOURCE.read_text()

        self.assertIn('rocket_cfg.spawn.spawn_path = "/World/envs/env_0/Rocket"', source)
        self.assertIn("self._rocket = RigidObject(rocket_cfg)", source)
