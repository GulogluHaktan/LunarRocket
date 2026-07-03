#!/usr/bin/env python3
"""Quick test to verify two-tier terrain system works correctly."""

import sys
import numpy as np
from pathlib import Path

# Add app to path
app_dir = Path(__file__).parent / "app"
sys.path.insert(0, str(app_dir))

from terrain_generator import MoonTerrainGenerator
from config import load_yaml
from randomization import sample_reset


def test_two_tier_terrain():
    """Test that high_detail_landing preset produces expected stats."""
    print("[Test] Loading lunar terrain system...")
    
    # Load config
    config = load_yaml(Path(__file__).parent / "configs" / "terrain_config.yaml")
    
    # Test all presets
    presets_to_test = [
        ("low_debug", False),
        ("medium_train", False),
        ("high_train", False),
        ("high_detail_landing", True),  # Should show two-tier
        ("demo_high", False),
    ]
    
    print("\n[Test] Terrain Quality Preset Test")
    print("=" * 80)
    
    for preset_name, expect_two_tier in presets_to_test:
        print(f"\n[Test] Testing preset: {preset_name}")
        
        # Set quality
        config["terrain_quality"] = preset_name
        
        # Create generator
        gen = MoonTerrainGenerator(config)
        
        # Generate terrain
        rng = np.random.default_rng(42)
        reset_sample = sample_reset(config, {"spawn": {}}, rng)
        terrain_data = gen.generate(reset_sample)
        
        # Check stats
        is_two_tier = terrain_data.collision_resolution > terrain_data.heightmap_resolution
        
        print(f"  heightmap_resolution: {terrain_data.heightmap_resolution}")
        print(f"  collision_resolution: {terrain_data.collision_resolution}")
        print(f"  triangle_count: {terrain_data.triangle_count}")
        print(f"  is_two_tier: {is_two_tier} (expected: {expect_two_tier})")
        
        if is_two_tier:
            local_triangles = int((terrain_data.collision_resolution - 1) ** 2 * 2 * 0.15)
            effective = terrain_data.triangle_count + local_triangles
            print(f"  effective_triangles: ~{effective}")
        
        # Verify expectations
        if expect_two_tier and not is_two_tier:
            print(f"  ❌ ERROR: Expected two-tier system for {preset_name}!")
            return False
        if not expect_two_tier and is_two_tier:
            print(f"  ❌ ERROR: Did not expect two-tier system for {preset_name}!")
            return False
        
        print(f"  ✅ OK")
    
    print("\n" + "=" * 80)
    print("[Test] All tests passed! ✅")
    return True


if __name__ == "__main__":
    success = test_two_tier_terrain()
    sys.exit(0 if success else 1)
