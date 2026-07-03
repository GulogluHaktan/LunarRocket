# Terrain Quality Quick Reference

## One-Minute Summary

### TL;DR
- **Training**: Use `medium_train` (192×192) — balanced detail & VRAM
- **Debugging**: Use `low_debug` (128×128) — minimal VRAM
- **Demo**: Uses `demo_high` (256×256) — high quality, same physics as training
- **VRAM**: 128→820MB, 192→1.8GB, 256→2.7GB

## Quick Start

### Use Default Training (192×192)
```bash
python app/main.py --config-dir configs
```
Uses `medium_train` preset automatically.

### Use High-Quality Training (256×256)
Edit `configs/terrain_config.yaml`:
```yaml
terrain_quality: high_train
```
Then run:
```bash
python app/main.py --config-dir configs
```

### Run Benchmark
```bash
python app/benchmark_terrain_quality.py --config-dir configs --output-dir outputs/terrain_quality_benchmark
```
Compares all presets: low_debug, medium_train, high_train, demo_high.

### Demo Recording (High Quality)
```bash
python app/record_demo_light.py
```
Automatically uses `demo_high` preset (256×256, same geometry as training).

### Debug with Minimal VRAM
```bash
python app/debug_min_vram.py
```
Uses `low_debug` preset (128×128).

## Preset Comparison

| Preset | Resolution | Vertices | Triangles | VRAM | Use Case |
|--------|-----------|----------|-----------|------|----------|
| `low_debug` | 128×128 | ~16k | ~32k | ~820MB | Debugging only |
| `medium_train` | 192×192 | ~36k | ~72k | ~1.8GB | **Training (default)** |
| `high_train` | 256×256 | ~64k | ~128k | ~2.7GB | High-detail training |
| `demo_high` | 256×256 | ~64k | ~128k | ~2.7GB | Video rendering |

## Important Constraints

✅ **Training ≠ Demo Geometry**: Both share identical terrain (DEM, craters, slopes)
✅ **Minimum Training Resolution**: Never use lower than 192×192 for training
✅ **64×64 Not Allowed for Training**: Only in `debug_min_vram` for smoke tests
✅ **Reproducibility**: Same seed = same terrain across all runs

## Config Files by Use Case

| Use Case | Config File | Quality Preset |
|----------|------------|-----------------|
| Standard training | `configs/sim_config.yaml` | `medium_train` |
| High-quality training | `configs/train_high_quality.yaml` | `high_train` |
| Demo recording | `configs/record_demo_light.yaml` | `demo_high` |
| Debug/VRAM test | `configs/debug_min_vram.yaml` | `low_debug` |

## Terrain Statistics Logged

Every terrain generation prints:
```
heightmap_resolution: 192
vertex_count: 36864
triangle_count: 72576
grid_spacing_m: 0.4167
min_elevation_m: -0.4521
max_elevation_m: 0.3892
mean_slope_deg: 2.47
max_slope_deg: 18.32
```

## VRAM Allocation

```
Total available GPU VRAM: X GB
Terrain creation: 128×128=0.8GB, 192×192=1.8GB, 256×256=2.7GB
Rocket creation: +0.2-0.5GB
Camera/sensors: +0.5-1GB
Physics/rendering: +1-2GB
────────────────────────
Total needed: roughly 2-5 GB depending on preset
```

## Decision Tree

```
Do you have > 8GB VRAM?
  ├─ YES: Can use high_train (256×256) for training
  │       │
  │       └─ Need to debug quickly?
  │           ├─ YES: Use low_debug temporarily
  │           └─ NO: Use medium_train or high_train
  │
  └─ NO (4-8GB): Use medium_train (192×192)
                 └─ If VRAM errors, switch to low_debug

Record demo video?
  └─ Always use demo_high (same physics as training)

Publish/present?
  └─ Use demo_high with better materials/lighting
```

## Troubleshooting

### "Out of VRAM" Error
```
Solution: Switch to lower preset
1. Edit configs/terrain_config.yaml
2. Change terrain_quality: medium_train → low_debug
3. Run benchmark to verify: python app/benchmark_terrain_quality.py
```

### Terrain Looks Different Between Runs
```
Solution: Check reproducibility settings
1. Verify seed is same: grep seed configs/*.yaml
2. Check terrain_quality hasn't changed
3. Verify DEM file exists: assets/moon_heightmaps/ldem_4.tif
```

### Need Reproducible Terrain
```
Solution: Use ResetSample with fixed seed
from app.randomization import ResetSample
reset_sample = ResetSample(
    seed=12345,
    terrain_origin=(1000, 1000),
    rocket_position=(2.5, 1.2, 30.0),
    rocket_euler_deg=(0.5, 0.2, 45.0),
    crater_count=5,
    slope=(0.005, -0.008),
    roughness_scale=0.012,
)
terrain = terrain_gen.generate(reset_sample)
```

## Performance Tips

1. **Benchmark first**: Run `benchmark_terrain_quality.py` to establish baseline
2. **Start lower**: Begin with `medium_train`, increase if needed
3. **Monitor VRAM**: Use `nvidia-smi` to track usage during training
4. **Profile terrain gen**: Check generation times in logs
5. **Use local patches**: Enable `local_detail_patch` for landing zone detail without global resolution increase

## For More Details

See [TERRAIN_QUALITY.md](TERRAIN_QUALITY.md) for comprehensive documentation including:
- Quality preset specifications
- Multi-resolution landing zone system
- Train/demo consistency guarantees
- Full statistics and metrics
- Benchmarking results
- Migration guide
