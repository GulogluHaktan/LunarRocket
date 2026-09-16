# Terrain detail-transition method comparison

Experiment code backing the paper "Isaac Lab'da Ay İnişi için Arazi
Detay-Geçiş Yöntemlerinin Karşılaştırılması ve Hibrit Bir Yaklaşım."
Compares three ways to blend a fine local "landing-zone detail" layer into
a coarse global terrain layer: **smoothstep**, **Gaussian**, and a proposed
**hybrid** (Gaussian near the landing-gear contact zone, smoothstep beyond
it). Runs entirely on CPU/NumPy -- no GPU or Isaac Sim required.

## Important correction vs. the original framing

The initial framing was "the current system is two-mesh + smoothstep;
compare it against single-grid Gaussian." Grepping the actual codebase
shows the opposite pairing:

- The **live RL training path**
  (`LunarLanderEnv._local_detail_height`,
  `source/lunar_rocket_lab/lunar_rocket_lab/tasks/direct/lunar_lander/lunar_lander_env.py:954-980`)
  fades its fine detail layer in with a **smoothstep** (`t*t*(3-2*t)`),
  radius `terrain_detail_radius_m = 8.0`. It is a pure analytic height
  field, not a mesh at all.
- The **legacy visual/demo mesh pipeline**
  (`MoonTerrainGenerator._blend_local_patch_multiscale`,
  `app/terrain_generator.py:924-991`, reachable only via
  `ISAACLAB_USE_DEM_TERRAIN=1`) blends its local detail patch into the
  global grid with a **Gaussian** falloff (`exp(-(d/r)**blend_scale)`),
  radius `blend_radius_m = 8.0`.

So this experiment treats both as real, shipped baselines rather than
picking one as "current" -- it runs **two parallel tracks**:

- `mesh_track.py` -- the literal global-grid + local-patch mesh
  representation (the "two-mesh" framing), generalized so any of the three
  blend shapes can be dropped into the same pipeline the legacy Gaussian
  code was hard-coded into.
- `analytic_track.py` -- the closed-form height field the live RL env
  actually queries every physics step (LiDAR rays, terrain-scan samples,
  foot-clearance checks), generalized the same way around the shipped
  smoothstep fade.

Terrain *content* (slope, ripple, craters, rocks, fine-detail
amplitude/phase) is sampled every trial with the real production generator,
`app.terrain_pool.generate_terrain_pool_batch`, so results reflect the
actual training-time content distribution, not synthetic filler.

## Methods compared

| method | formula | source |
|---|---|---|
| `smoothstep` | `t=clamp(1-r/R,0,1); fade=t²(3-2t)` | shipped, `_local_detail_height` |
| `gaussian_shipped` | `exp(-(r/R)²)` (k=1) | shipped, `_blend_local_patch_multiscale`, `blend_scale=2.0` |
| `gaussian_calibrated` | `exp(-6.908·(r/R)²)` | k tuned so `fade(R) ≈ 1e-3`, i.e. the same practical R-radius support smoothstep gets for free; primary "fair" Method-B baseline |
| `hybrid` (proposed) | Gaussian for `r ≲ Rc`, smoothstep for `r ≳ Rc`, joined by a smoothstep-weighted switch over a 0.3 m band | this work |

`R = 8.0 m` (shared by both shipped formulas). `Rc = 1.5 m`, sized from the
rocket's actual foot layout (`rocket_foot_offsets_m`, footprint
half-diagonal ≈ 0.57 m) plus a touchdown-drift margin — the "near the
landing gear" zone the hybrid method should special-case.

## Metrics

1. **Production/generation time** — mesh track: wall-clock to build the
   global grid, local patch, and blend step (`GridSpec(128, 80m)` global /
   `GridSpec(256, 30m)` local, matching `low_debug` / `local_detail_patch`
   defaults). Analytic track: per-"RL step" query throughput at the actual
   default batch shape (512 envs × (64 LiDAR + 24 terrain-scan + 4 foot)
   queries ≈ 47k queries), plus an isolated fade-formula-only microbenchmark
   decoupled from macro/fine terrain evaluation.
2. **VRAM proxy** — vertex/triangle counts × 32 bytes/vertex (pos+normal+uv)
   × 2 (render + PhysX collision mesh), reported for both a literal
   `two_mesh_literal` deployment (global mesh + local mesh as separate
   prims) and the `single_grid_baked` deployment the shipped code actually
   uses (local detail resampled/added into the global grid, no extra
   geometry). **This is an analytic estimate, not a measured GPU
   allocation** — no GPU/Isaac Sim is available in this environment.
3. **Edge/seam continuity** — C0 (height) and C1 (surface-normal angle)
   discontinuity sampled around a 720-point ring at `r = R`, finite-difference
   based. Reported two ways:
   - **headline / seam-only**: continuity of `fade(r)·fine_height(x,y)`
     alone, with **all methods hard-truncated to exactly 0 at r ≥ R** (a
     deployable mesh or a "no detail beyond the patch" analytic rule can't
     carry an unbounded Gaussian tail; truncating every method at the same
     nominal radius is what makes this a fair, same-transition-budget
     comparison — see "Finding" below).
   - **secondary / combined**: continuity of the full macro+detail surface,
     i.e. what a foot or LiDAR ray actually perceives (dominated by ambient
     macro-terrain roughness, not the seam itself — included for realism).

30 random seeds per method (`--seeds`, default 30); all summary numbers are
seed-mean ± seed-stdev.

## Reproduce

```bash
cd experiments/terrain_transition
python3 run_experiment.py --seeds 30
```

Outputs land in `results/`: `raw_results.json` (all per-seed rows),
`summary_table.txt`, and three plots (`radial_profile.png`,
`continuity_comparison.png`, `cost_comparison.png`).

## Findings (30-seed run, defaults above)

1. **The as-shipped Gaussian formula leaves a real seam; smoothstep,
   calibrated-Gaussian, and the hybrid do not.** Headline seam-only C0/C1
   (analytic track, mean over 30 seeds):

   | method | seam C0 (m) | seam C1 (deg) |
   |---|---|---|
   | smoothstep | 1e-6 | 0.001 |
   | **gaussian_shipped** | **1.7e-3** | **0.67** |
   | gaussian_calibrated | 5e-6 | 0.002 |
   | hybrid | 1e-6 | 0.001 |

   Root cause: `exp(-(r/R)²)` is C∞ smooth but never reaches exactly 0 —
   at `r=R` it still carries `exp(-1) ≈ 36.8%` of its center value. Left
   *unbounded* it therefore shows no seam at any finite radius (verified);
   the seam appears the moment any deployable representation truncates it
   at a finite radius, which every real mesh/heightmap must do. The legacy
   code's own truncation (`blend_range = 1.5×blend_idx` in
   `app/terrain_generator.py:947`) pushes this out to `1.5R = 12m`, where
   the residual is still `exp(-2.25) ≈ 10.5%` — i.e. the as-shipped
   implementation is carrying a real, measurable discontinuity today,
   independent of this study's methods. Re-radius-matching the Gaussian
   (`gaussian_calibrated`, k so `fade(R)=1e-3`) fixes this entirely at
   negligible extra cost — cheapest actionable fix if the legacy pipeline
   is kept.

2. **VRAM is not a real differentiator for the shipped (single-grid-baked)
   deployment, in either track.** All three methods land on the same
   1.05 MB single-grid-baked estimate (mesh track) because they all reuse
   the same global-resolution grid — only the *blend shape*, not the
   *geometry*, differs. In the analytic track all three read the same
   42 floats/env (168 bytes/env) of terrain-content parameters; the fade
   function adds none of its own. A literal `two_mesh_literal` deployment
   (real separate local-patch mesh) would cost ~5.24 MB regardless of
   which blend shape is used — so mesh-vs-analytic and
   baked-vs-literal are the memory-relevant design axes here, not the
   choice of blend function.
3. **Generation time and end-to-end query throughput are essentially
   blend-shape-independent** (mesh: 5.4–5.7 ms across methods; analytic:
   3.1–3.3M queries/s) — both are dominated by the shared macro/fine
   terrain evaluation, not the fade formula. The fade formula's own cost
   *is* measurably different in isolation (fade-only microbenchmark):
   smoothstep ≈ 89 µs, Gaussian ≈ 118–121 µs (+~35%, from `exp`), **hybrid
   ≈ 388 µs (+~4.3× vs. smoothstep)** — the hybrid evaluates both branches
   and the switch weight every call. This cost is invisible at the current
   batch size/terrain complexity but would matter if the terrain evaluation
   itself were made cheaper or the fade were called at a much higher rate.
4. **The hybrid's internal switch (Gaussian↔smoothstep at r=Rc) introduces
   no measurable new seam**: `|fade(Rc+ε)−fade(Rc−ε)| = 8.7e-5`, slope
   discontinuity `1.3e-4` (pure 1D check on the fade weight itself, both
   consistent with the finite-difference step size used, i.e.
   indistinguishable from the two component functions' own smoothness).
   The hybrid inherits calibrated-Gaussian-quality smoothness in the
   landing-gear contact zone and smoothstep's cheaper compact support
   everywhere else, at the cost noted in (3).

## Limitations

- No GPU/Isaac Sim available in this environment: VRAM figures are an
  analytic proxy (vertex/triangle count × assumed bytes/vertex), not a
  measured `nvidia-smi` allocation. Cross-checking order of magnitude
  against the empirical figures in `TERRAIN_QUALITY.md` (which *do*
  include full Isaac Sim/PhysX scene overhead, not just this mesh) is a
  useful sanity check but not an apples-to-apples validation.
- The fine/detail layer formula intentionally omits the shipped
  `_local_detail_height`'s "subtract detail-at-target" term (a
  target-height-matching quirk orthogonal to blend-shape comparison).
- `Rc = 1.5 m` and the 0.3 m switch band are single fixed choices, not
  swept — a follow-up ablation over `Rc` and switch width would strengthen
  the hybrid-method section of the paper.
