"""Mesh/heightmap track: the literal "two-mesh" (global grid + local detail
patch) representation, matching the legacy visual/demo pipeline
(``app/terrain_generator.py: generate_local_detail_patch`` +
``_blend_local_patch_multiscale``, only reachable via
``ISAACLAB_USE_DEM_TERRAIN=1``), generalized so any of the three blend
shapes from ``blend_functions.py`` can be dropped into the same pipeline the
legacy Gaussian formula was hard-coded into.

Two ways a local detail patch can be turned into a deployable mesh:

1. ``two_mesh_literal``  -- the global grid and the local patch stay two
   separate meshes/prims (their own vertex/triangle budgets). This is what
   TWO_TIER_LANDING.md describes and what "iki-mesh" literally means.
2. ``single_grid_baked`` -- (what the shipped code actually does) the local
   patch is bilinearly resampled and additively blended into the *same*
   global-resolution grid, so the final asset is one mesh at global
   resolution; no extra vertex/triangle cost, at the price of losing
   resolution wherever local_spacing < global_spacing (aliasing).

Both accountings are reported; continuity/timing use the single-grid-baked
height field, since that's the deployable, physics-queryable surface.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from blend_functions import make_fade_fn
from terrain_layers import TerrainSlot, fine_height, macro_height


@dataclass(frozen=True)
class GridSpec:
    resolution: int
    size_m: float


def build_grid(spec: GridSpec) -> tuple[np.ndarray, np.ndarray]:
    coords = np.linspace(-spec.size_m / 2.0, spec.size_m / 2.0, spec.resolution)
    xx, yy = np.meshgrid(coords, coords)
    return xx, yy


def vertex_triangle_counts(resolution: int) -> tuple[int, int]:
    vertices = resolution * resolution
    triangles = (resolution - 1) * (resolution - 1) * 2
    return vertices, triangles


def _bilinear_sample(values: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    res_i, res_j = values.shape
    i = np.clip(i, 0.0, res_i - 1.0 - 1e-6)
    j = np.clip(j, 0.0, res_j - 1.0 - 1e-6)
    i0 = np.floor(i).astype(int)
    j0 = np.floor(j).astype(int)
    i1 = np.clip(i0 + 1, 0, res_i - 1)
    j1 = np.clip(j0 + 1, 0, res_j - 1)
    fi = i - i0
    fj = j - j0
    return (
        values[i0, j0] * (1 - fi) * (1 - fj)
        + values[i1, j0] * fi * (1 - fj)
        + values[i0, j1] * (1 - fi) * fj
        + values[i1, j1] * fi * fj
    )


def build_mesh_track_heightfield(
    slot: TerrainSlot,
    method: str,
    global_spec: GridSpec,
    local_spec: GridSpec,
    blend_radius_m: float,
    contact_radius_m: float | None = None,
) -> dict:
    fade_fn = make_fade_fn(method, blend_radius_m, contact_radius_m)

    t0 = time.perf_counter()
    gxx, gyy = build_grid(global_spec)
    global_h = macro_height(gxx, gyy, slot)
    t_global_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    lxx, lyy = build_grid(local_spec)
    local_fine = fine_height(lxx, lyy, slot)
    t_local_s = time.perf_counter() - t1

    t2 = time.perf_counter()
    li = (gyy + local_spec.size_m / 2.0) / local_spec.size_m * (local_spec.resolution - 1)
    lj = (gxx + local_spec.size_m / 2.0) / local_spec.size_m * (local_spec.resolution - 1)
    in_patch = (li >= 0) & (li <= local_spec.resolution - 1) & (lj >= 0) & (lj <= local_spec.resolution - 1)
    local_val = _bilinear_sample(local_fine, li, lj)
    r = np.sqrt(gxx**2 + gyy**2)
    # Hard-truncate the fine-detail contribution at the nominal blend radius
    # (same rationale as analytic_track.make_combined_height_fn: a deployable
    # mesh can't carry an unbounded Gaussian tail, and truncating every
    # method at the same radius is what makes this a fair comparison).
    fade = np.where((r < blend_radius_m) & in_patch, fade_fn(r), 0.0)
    detail_grid = fade * local_val
    combined = global_h + detail_grid
    t_blend_s = time.perf_counter() - t2

    global_vertices, global_triangles = vertex_triangle_counts(global_spec.resolution)
    local_vertices, local_triangles = vertex_triangle_counts(local_spec.resolution)

    return {
        "method": method,
        "combined": combined,
        "detail_grid": detail_grid,
        "global_grid": (gxx, gyy),
        "global_size_m": global_spec.size_m,
        "global_resolution": global_spec.resolution,
        "t_global_s": t_global_s,
        "t_local_s": t_local_s,
        "t_blend_s": t_blend_s,
        "t_total_s": t_global_s + t_local_s + t_blend_s,
        "global_vertices": global_vertices,
        "global_triangles": global_triangles,
        "local_vertices": local_vertices,
        "local_triangles": local_triangles,
        "two_mesh_literal_vertices": global_vertices + local_vertices,
        "two_mesh_literal_triangles": global_triangles + local_triangles,
        "single_grid_baked_vertices": global_vertices,
        "single_grid_baked_triangles": global_triangles,
        "global_grid_spacing_m": global_spec.size_m / (global_spec.resolution - 1),
        "local_grid_spacing_m": local_spec.size_m / (local_spec.resolution - 1),
    }


def make_grid_query_fn(result: dict, detail_only: bool = False):
    """Bilinear-interpolated height_fn(x, y) over the single-grid-baked surface."""
    combined = result["detail_grid"] if detail_only else result["combined"]
    size_m = result["global_size_m"]
    resolution = result["global_resolution"]

    def query(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        i = (np.asarray(y) + size_m / 2.0) / size_m * (resolution - 1)
        j = (np.asarray(x) + size_m / 2.0) / size_m * (resolution - 1)
        return _bilinear_sample(combined, i, j)

    return query
