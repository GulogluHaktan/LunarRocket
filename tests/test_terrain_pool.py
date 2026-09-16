from __future__ import annotations

import unittest

import numpy as np

from app.terrain_pool import generate_terrain_pool_batch


class TerrainPoolTests(unittest.TestCase):
    def _generate(self, seed: int) -> dict[str, np.ndarray]:
        return generate_terrain_pool_batch(
            seed=seed,
            pool_size=15,
            crater_count=3,
            rock_count=5,
            slope_scale=0.035,
            height_scale=0.45,
            safe_feature_radius=3.0,
            detail_amplitude_range=(0.006, 0.020),
        )

    def test_pool_is_deterministic_and_has_expected_shapes(self) -> None:
        first = self._generate(7)
        second = self._generate(7)

        self.assertEqual(first["slope"].shape, (15, 2))
        self.assertEqual(first["crater_xy"].shape, (15, 3, 2))
        self.assertEqual(first["rock_xy"].shape, (15, 5, 2))
        self.assertEqual(first["detail_phase"].shape, (15, 4))
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])

    def test_features_stay_outside_safe_landing_zone(self) -> None:
        pool = self._generate(11)
        crater_distance = np.linalg.norm(pool["crater_xy"], axis=-1)
        rock_distance = np.linalg.norm(pool["rock_xy"], axis=-1)

        self.assertTrue(np.all(crater_distance >= 3.0))
        self.assertTrue(np.all(rock_distance >= 3.0))

    def test_refill_seed_produces_a_different_pool(self) -> None:
        first = self._generate(7)
        refill = self._generate(8)

        self.assertFalse(np.array_equal(first["crater_xy"], refill["crater_xy"]))
        self.assertFalse(np.array_equal(first["detail_phase"], refill["detail_phase"]))


if __name__ == "__main__":
    unittest.main()
