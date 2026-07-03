# Lunar Terrain Quality System

## Overview

The LunarRocket terrain system now supports configurable quality presets to balance detail, realism, and VRAM usage. This enables flexible training environments while maintaining consistency between training and demo scenarios.

## Quality Presets

### 1. `low_debug` (128×128)
- **Resolution**: 128×128 heightmap
- **Vertex Count**: ~16k vertices
- **Triangle Count**: ~32k triangles
- **Use Case**: Minimal VRAM smoke tests, debugging
- **VRAM**: ~500 MB peak
- **Notes**: Only for debugging. NOT suitable for training due to insufficient detail for realistic contact dynamics.

### 2. `medium_train` (192×192) — **Recommended for training**
- **Resolution**: 192×192 heightmap
- **Vertex Count**: ~36k vertices
- **Triangle Count**: ~72k triangles
- **Use Case**: Primary training environment
- **VRAM**: ~1.8 GB peak
- **Notes**: Good balance between terrain detail and VRAM usage. Sufficient for realistic rocket landing dynamics.

### 3. `high_train` (256×256) — **Optional high-detail training**
- **Resolution**: 256×256 heightmap
- **Vertex Count**: ~64k vertices
- **Triangle Count**: ~128k triangles
- **Use Case**: Advanced training with extremely detailed terrain
- **VRAM**: ~2.7 GB peak
- **Notes**: Provides maximum detail. Use if VRAM allows and ultra-realistic landing dynamics are needed.

### 4. `demo_high` (256×256) — **Video rendering**
- **Resolution**: 256×256 heightmap
- **Vertex Count**: ~64k vertices
- **Triangle Count**: ~128k triangles
- **Use Case**: High-quality video recording and presentations
- **VRAM**: ~2.7 GB peak
- **Geometry**: Identical to `high_train` (same DEM, craters, slopes)
- **Rendering**: Allows better materials, lighting, and normal smoothing
- **Notes**: Shares terrain geometry with training for consistency.

## Configuration

### Setting Quality Preset

In `configs/terrain_config.yaml`:

```yaml
terrain_quality: medium_train  # Can be: low_debug, medium_train, high_train, demo_high

terrain_quality_presets:
  medium_train:
    heightmap_resolution: 192
    collision_resolution: 192
    visual_resolution: 192
```

### Using Different Training Profiles

**Training with Medium Quality (Default)**:
```bash
python app/main.py --config-dir configs
# Uses terrain_config.yaml with medium_train preset
```

**Training with High Quality**:
```bash
python app/main.py --config-dir configs
# First, update configs/terrain_config.yaml to:
#   terrain_quality: high_train
```

**Demo Recording with High Quality**:
```bash
# record_demo_light.yaml uses demo_high preset
python app/record_demo_light.py
```

**Debug with Minimal VRAM**:
```bash
# debug_min_vram.yaml uses low_debug preset
python app/debug_min_vram.py
```

## Train/Demo Consistency

### Guaranteed Consistency
When using the same seed, the following remain **identical** between training and demo:

1. **Global terrain geometry**
   - DEM sampling location (same patch)
   - Heightmap elevation data
   - Crater positions and depths
   - Slope distribution

2. **Collision geometry**
   - Contact-relevant mesh resolution
   - Physics collision mesh (identical quad/triangle layout)

3. **Landing zone placement**
   - Rocket spawn position
   - Target landing zone

### Allowed Differences
Demo environments can use different rendering settings without affecting physics:

- **Materials**: Better lunar textures and colors
- **Lighting**: Sun intensity, angle, dome lighting
- **Normal maps**: Smooth vertex normals for better shading
- **Anti-aliasing**: Better visual quality

### Enforcement
- `train_medium_quality.yaml` → uses `medium_train` preset
- `train_high_quality.yaml` → uses `high_train` preset
- `record_demo_light.yaml` → uses `demo_high` preset (same geometry as `high_train`)
- `debug_min_vram.yaml` → uses `low_debug` preset

## Multi-Resolution Landing Zone

The terrain system supports optional high-resolution local patches around the landing zone:

```yaml
local_detail_patch:
  enabled: true
  patch_size_m: [10.0, 10.0]    # 10m × 10m detail patch
  resolution: 256               # 256×256 local resolution
  blend_radius_m: 2.0          # Smooth blend transition
```

This adds fine-scale roughness around the landing zone without increasing the global terrain resolution.

## Terrain Statistics

Every terrain generation logs detailed statistics:

```
[LunarRocket] terrain stats:
  heightmap_resolution: 192
  collision_resolution: 192
  visual_resolution: 192
  vertex_count: 36864
  triangle_count: 72576
  grid_spacing_m: 0.4167
  size_m: 80.0 x 80.0
  min_elevation_m: -0.4521
  max_elevation_m: 0.3892
  elevation_range_m: 0.8413
  mean_slope_deg: 2.47
  max_slope_deg: 18.32
```

### Available Metrics
- `heightmap_resolution`: Mesh grid resolution
- `collision_resolution`: Physics collision mesh resolution
- `visual_resolution`: Visual rendering resolution
- `vertex_count`: Total vertices in mesh
- `triangle_count`: Total triangles (converted from quads)
- `grid_spacing_m`: Distance between vertices in meters
- `size_m`: Terrain patch size (x, y)
- `min_elevation_m`: Lowest point
- `max_elevation_m`: Highest point
- `elevation_range_m`: Vertical variation
- `mean_slope_deg`: Average surface slope
- `max_slope_deg`: Steepest slope

## Benchmarking

Run the terrain quality benchmark to compare performance across presets:

```bash
python app/benchmark_terrain_quality.py --config-dir configs --output-dir outputs/terrain_quality_benchmark
```

This generates `outputs/terrain_quality_benchmark/benchmark_results.json` with:
- Generation time for each preset
- VRAM usage comparison
- Mesh statistics (vertices, triangles, spacing)
- Elevation and slope statistics

**Example Output**:
```
[Benchmark] SUMMARY
================================================================================
low_debug       | Res: 128 | Tri:  32576 | Gen: 23.4ms | VRAM:  823MB
medium_train    | Res: 192 | Tri:  72576 | Gen: 52.1ms | VRAM: 1856MB
high_train      | Res: 256 | Tri: 128576 | Gen: 94.3ms | VRAM: 2756MB
demo_high       | Res: 256 | Tri: 128576 | Gen: 94.5ms | VRAM: 2752MB
================================================================================
```

## Reproducibility

The terrain system ensures reproducibility through:

1. **Seed control**: Same seed generates identical terrain
2. **Deterministic sampling**: DEM patches, crater positions, slopes
3. **Terrain origin tracking**: `reset_sample.terrain_origin` records exact DEM location
4. **Reproducible RNG**: Random number generator state fully captured

### Reproducing Specific Terrain

To reproduce a specific terrain:

```python
from app.randomization import ResetSample

# Create the exact reset sample
reset_sample = ResetSample(
    seed=12345,
    terrain_origin=(1000, 1000),    # Specific DEM location
    rocket_position=(2.5, 1.2, 30.0),
    rocket_euler_deg=(0.5, 0.2, 45.0),
    crater_count=5,
    slope=(0.005, -0.008),
    roughness_scale=0.012,
)

# Generate identical terrain
terrain = terrain_generator.generate(reset_sample)
```

## Migration from Old System

### Old Config Format
```yaml
mesh_resolution: 128
```

### New Config Format
```yaml
terrain_quality: medium_train
terrain_quality_presets:
  medium_train:
    heightmap_resolution: 192
    collision_resolution: 192
    visual_resolution: 192
```

### Automatic Resolution
If using old `mesh_resolution` field, the system will still work but will show a deprecation notice. Use `terrain_quality` presets going forward.

## Troubleshooting

### VRAM Issues
1. Check current VRAM usage: `nvidia-smi`
2. Start with `low_debug` preset for testing
3. Use `medium_train` for stable training
4. Use `high_train` only if VRAM > 8 GB

### Terrain Looks Different Between Runs
- Verify seed is consistent
- Check `terrain_quality_profile` setting in config
- Ensure same DEM file is used

### Landing Zone Not Detailed
- Enable `local_detail_patch` in `terrain_config.yaml`
- Increase `local_resolution` from 256 to higher if needed
- Adjust `blend_radius_m` for smoother transitions

## Best Practices

1. **For Training**: Use `medium_train` by default
2. **For Debugging**: Use `low_debug` for quick iterations
3. **For Publications**: Use `demo_high` with same geometry as training
4. **For Maximum Detail**: Use `high_train` if VRAM permits
5. **For Reproducibility**: Always capture and store `reset_sample` seed
6. **For Benchmarking**: Run `benchmark_terrain_quality.py` before training

## Future Enhancements

- Adaptive LOD (Level of Detail) based on distance from rocket
- Real DEM data streaming from actual lunar maps
- GPU-accelerated terrain generation
- Multi-GPU terrain caching
- Terrain streaming for continuous exploration
