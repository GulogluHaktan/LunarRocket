#!/usr/bin/env python3
"""
Terrain Quality Benchmark Script

Runs terrain generation across different quality presets (128, 192, 256 resolution)
and saves detailed statistics for VRAM usage and rendering performance.

Usage:
    python app/benchmark_terrain_quality.py --config-dir configs --output-dir outputs/terrain_quality_benchmark

The benchmark will:
1. Create terrain at 128x128, 192x192, and 256x256 resolutions
2. For each resolution, generate multiple random samples
3. Track peak VRAM usage and generation time
4. Save detailed stats to JSON files
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import load_configs
from app.terrain_generator import MoonTerrainGenerator
from app.randomization import sample_reset


def get_vram_usage_mb() -> float:
    """Get current GPU VRAM usage in MB."""
    try:
        import subprocess
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
            encoding="utf-8",
            stderr=subprocess.DEVNULL
        )
        return float(result.strip().split("\n")[0].strip())
    except Exception:
        return 0.0


def benchmark_quality_preset(
    terrain_gen: MoonTerrainGenerator,
    quality_name: str,
    sample_count: int = 3,
) -> dict[str, Any]:
    """Benchmark a single terrain quality preset."""
    print(f"\n[Benchmark] Testing quality preset: {quality_name}", flush=True)
    
    # Update generator config to use this quality preset
    terrain_gen.config["terrain_quality"] = quality_name
    terrain_gen._quality_preset = terrain_gen._get_quality_preset()
    
    resolution = terrain_gen.get_mesh_resolution()
    
    stats = {
        "quality_preset": quality_name,
        "heightmap_resolution": resolution,
        "sample_count": sample_count,
        "samples": [],
        "avg_generation_time_s": 0.0,
        "avg_vertex_count": 0,
        "avg_triangle_count": 0,
        "avg_grid_spacing_m": 0.0,
        "avg_min_elevation_m": 0.0,
        "avg_max_elevation_m": 0.0,
        "avg_mean_slope_deg": 0.0,
        "avg_max_slope_deg": 0.0,
    }
    
    vram_before = get_vram_usage_mb()
    generation_times = []
    
    for i in range(sample_count):
        # Generate a random reset sample
        rng = np.random.default_rng(int(42 + i))
        reset_sample = sample_reset(terrain_gen.config, {"spawn": {}}, rng)
        
        # Benchmark terrain generation
        gen_start = time.time()
        terrain_data = terrain_gen.generate(reset_sample)
        gen_time = time.time() - gen_start
        generation_times.append(gen_time)
        
        vram_after = get_vram_usage_mb()
        
        sample_stat = {
            "sample_id": i,
            "generation_time_s": float(gen_time),
            "vram_used_mb": float(vram_after - vram_before),
            "vertex_count": terrain_data.vertex_count,
            "triangle_count": terrain_data.triangle_count,
            "collision_resolution": terrain_data.collision_resolution,
            "grid_spacing_m": float(terrain_data.grid_spacing_m),
            "min_elevation_m": float(terrain_data.min_height_m),
            "max_elevation_m": float(terrain_data.max_height_m),
            "mean_slope_deg": float(terrain_data.mean_slope_deg),
            "max_slope_deg": float(terrain_data.max_slope_deg),
        }
        stats["samples"].append(sample_stat)
        
        # Check for two-tier system
        is_two_tier = terrain_data.collision_resolution > terrain_data.heightmap_resolution
        if is_two_tier:
            local_triangles = int((terrain_data.collision_resolution - 1) ** 2 * 2 * 0.15)
            effective_triangles = terrain_data.triangle_count + local_triangles
            print(
                f"  Sample {i+1}/{sample_count}: {gen_time*1000:.1f}ms, "
                f"base_triangles={terrain_data.triangle_count}, "
                f"landing_zone_detail={terrain_data.collision_resolution}×{terrain_data.collision_resolution}, "
                f"effective_triangles~{effective_triangles}",
                flush=True,
            )
        else:
            print(
                f"  Sample {i+1}/{sample_count}: {gen_time*1000:.1f}ms, "
                f"vertices={terrain_data.vertex_count}, "
                f"triangles={terrain_data.triangle_count}",
                flush=True,
            )
    
    vram_peak = get_vram_usage_mb()
    
    # Compute averages
    if stats["samples"]:
        stats["avg_generation_time_s"] = float(np.mean(generation_times))
        stats["avg_vertex_count"] = int(np.mean([s["vertex_count"] for s in stats["samples"]]))
        stats["avg_triangle_count"] = int(np.mean([s["triangle_count"] for s in stats["samples"]]))
        stats["avg_grid_spacing_m"] = float(np.mean([s["grid_spacing_m"] for s in stats["samples"]]))
        stats["avg_min_elevation_m"] = float(np.mean([s["min_elevation_m"] for s in stats["samples"]]))
        stats["avg_max_elevation_m"] = float(np.mean([s["max_elevation_m"] for s in stats["samples"]]))
        stats["avg_mean_slope_deg"] = float(np.mean([s["mean_slope_deg"] for s in stats["samples"]]))
        stats["avg_max_slope_deg"] = float(np.mean([s["max_slope_deg"] for s in stats["samples"]]))
    
    stats["vram_before_mb"] = float(vram_before)
    stats["vram_peak_mb"] = float(vram_peak)
    
    print(
        f"  Average generation time: {stats['avg_generation_time_s']*1000:.1f}ms\n"
        f"  Average vertices: {stats['avg_vertex_count']}\n"
        f"  Average triangles: {stats['avg_triangle_count']}\n"
        f"  Peak VRAM: {vram_peak:.0f} MB",
        flush=True,
    )
    
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark lunar terrain quality presets"
    )
    parser.add_argument("--config-dir", default="configs", help="Directory with config files")
    parser.add_argument("--output-dir", default="outputs/terrain_quality_benchmark", help="Output directory")
    parser.add_argument("--samples-per-preset", type=int, default=3, help="Samples per quality preset")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("[Benchmark] Loading configs...", flush=True)
    configs = load_configs(Path(args.config_dir))
    
    print("[Benchmark] Creating terrain generator...", flush=True)
    terrain_gen = MoonTerrainGenerator(configs["terrain"])
    
    print("[Benchmark] Pre-loading DEM...", flush=True)
    terrain_gen.load_dem()
    
    # Test all quality presets (dynamically from config)
    presets_config = configs["terrain"].get("terrain_quality_presets", {})
    quality_presets = list(presets_config.keys())
    
    if not quality_presets:
        print("[Benchmark] ERROR: No quality presets found in config!", flush=True)
        return
    
    benchmark_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config_dir": str(args.config_dir),
        "presets": {},
        "summary": {},
    }
    
    for preset_name in quality_presets:
        try:
            preset_stats = benchmark_quality_preset(
                terrain_gen,
                preset_name,
                sample_count=args.samples_per_preset,
            )
            benchmark_results["presets"][preset_name] = preset_stats
        except Exception as e:
            print(f"[Benchmark] ERROR with preset {preset_name}: {e}", flush=True)
            benchmark_results["presets"][preset_name] = {"error": str(e)}
    
    # Save benchmark results
    results_file = output_dir / "benchmark_results.json"
    with open(results_file, "w") as f:
        json.dump(benchmark_results, f, indent=2)
    
    print(f"\n[Benchmark] Results saved to: {results_file}", flush=True)
    
    # Print summary
    print("\n[Benchmark] SUMMARY", flush=True)
    print("=" * 100, flush=True)
    for preset_name in quality_presets:
        if preset_name in benchmark_results["presets"]:
            stats = benchmark_results["presets"][preset_name]
            if "error" not in stats:
                # Check if samples contain collision_resolution for two-tier info
                is_two_tier = False
                collision_res = 0
                if stats.get("samples") and len(stats["samples"]) > 0:
                    collision_res = stats["samples"][0].get("collision_resolution", 0)
                    is_two_tier = collision_res > stats['heightmap_resolution']
                
                if is_two_tier:
                    # Two-tier display
                    local_triangles = int((collision_res - 1) ** 2 * 2 * 0.15)
                    effective_triangles = stats['avg_triangle_count'] + local_triangles
                    print(
                        f"{preset_name:20} | Base: {stats['heightmap_resolution']:3d}×{stats['heightmap_resolution']:3d} "
                        f"({stats['avg_triangle_count']:6d}tri) + Landing: {collision_res:3d}×{collision_res:3d} "
                        f"({local_triangles:6d}tri) ≈ {effective_triangles:7d} total | "
                        f"Gen: {stats['avg_generation_time_s']*1000:6.1f}ms",
                        flush=True,
                    )
                else:
                    # Regular single-tier display
                    print(
                        f"{preset_name:20} | Res: {stats['heightmap_resolution']:3d}×{stats['heightmap_resolution']:3d} | "
                        f"Tri: {stats['avg_triangle_count']:7d} | "
                        f"Gen: {stats['avg_generation_time_s']*1000:6.1f}ms | "
                        f"VRAM: {stats['vram_peak_mb']:7.0f}MB",
                        flush=True,
                    )
            else:
                print(f"{preset_name:20} | ERROR: {stats['error']}", flush=True)
    print("=" * 100, flush=True)
    
    print("[Benchmark] Done!", flush=True)


if __name__ == "__main__":
    main()
