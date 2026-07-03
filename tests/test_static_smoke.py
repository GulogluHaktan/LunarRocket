from __future__ import annotations

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from app.config import load_configs
from app.randomization import sample_reset
from app.terrain_generator import MoonTerrainGenerator


class StaticSmokeTests(unittest.TestCase):
    def test_configs_parse(self) -> None:
        configs = load_configs("configs")
        self.assertEqual(configs["sim"]["gravity_mps2"], 1.62)
        self.assertIn("dem", configs["terrain"])
        self.assertEqual(configs["rocket"]["mjcf_path"], "assets/rocket/hopper_lunar.xml")

    def test_hopper_asset_is_terrain_free(self) -> None:
        root = ET.parse("assets/rocket/hopper_lunar.xml").getroot()
        geom_names = {geom.attrib.get("name") for geom in root.findall(".//geom")}
        body_names = {body.attrib.get("name") for body in root.findall(".//body")}
        site_names = {site.attrib.get("name") for site in root.findall(".//site")}

        self.assertNotIn("ground", geom_names)
        self.assertNotIn("landing_pad", body_names)
        self.assertIn("hopper", body_names)
        self.assertIn("thrust_site", site_names)

    def test_terrain_generation_from_synthetic_dem_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dem_path = Path(tmp) / "dem.npy"
            dem = np.linspace(-2.0, 2.0, 64 * 64, dtype=np.float32).reshape(64, 64)
            np.save(dem_path, dem)

            terrain_cfg = {
                "seed": 1,
                "dem": {
                    "cache_dir": tmp,
                    "filename": "dem.npy",
                    "units": "meters",
                    "remove_patch_mean": True,
                },
                "terrain_size_m": [20.0, 20.0],
                "patch_shape_pixels": [32, 32],
                "mesh_resolution": 16,
                "crater_count_range": [3, 3],
                "crater_radius_m_range": [1.0, 2.0],
                "crater_depth_m_range": [0.2, 0.3],
                "slope_x_range": [0.01, 0.01],
                "slope_y_range": [0.02, 0.02],
                "roughness_scale_range": [0.1, 0.1],
                "roughness_smoothing_passes": 1,
            }
            rocket_cfg = {"spawn": {"altitude_m": 10.0, "xy_range_m": [0.0, 0.0], "attitude_deg_range": [0.0, 0.0]}}
            reset = sample_reset({**terrain_cfg, "dem_height_pixels": 64, "dem_width_pixels": 64}, rocket_cfg, np.random.default_rng(5))

            first = MoonTerrainGenerator(terrain_cfg).generate(reset)
            second = MoonTerrainGenerator(terrain_cfg).generate(reset)

            np.testing.assert_allclose(first.heights, second.heights)
            self.assertEqual(len(first.vertices), 16 * 16)
            self.assertEqual(len(first.face_counts), 15 * 15)
            self.assertGreater(float(np.std(first.heights)), 0.0)

    def test_reset_randomization_bounds(self) -> None:
        terrain_cfg = {
            "patch_shape_pixels": [16, 16],
            "dem_height_pixels": 64,
            "dem_width_pixels": 80,
            "crater_count_range": [2, 4],
            "slope_x_range": [-0.1, 0.1],
            "slope_y_range": [-0.2, 0.2],
            "roughness_scale_range": [0.5, 1.0],
        }
        rocket_cfg = {"spawn": {"altitude_m": 25.0, "xy_range_m": [-3.0, 3.0], "attitude_deg_range": [-5.0, 5.0]}}
        sample = sample_reset(terrain_cfg, rocket_cfg, np.random.default_rng(12))
        self.assertGreaterEqual(sample.terrain_origin[0], 0)
        self.assertLessEqual(sample.terrain_origin[0], 48)
        self.assertGreaterEqual(sample.terrain_origin[1], 0)
        self.assertLessEqual(sample.terrain_origin[1], 64)
        self.assertGreaterEqual(sample.crater_count, 2)
        self.assertLessEqual(sample.crater_count, 4)
        self.assertEqual(sample.rocket_position[2], 25.0)


if __name__ == "__main__":
    unittest.main()
