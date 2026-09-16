"""Shared metrics: boundary continuity (C0/C1) and mesh-memory estimation."""

from __future__ import annotations

from typing import Callable

import numpy as np

HeightFn = Callable[[np.ndarray, np.ndarray], np.ndarray]

# Bytes per vertex assumed for the VRAM proxy: float32 position(3) + normal(3)
# + uv(2) = 32 bytes, a standard Isaac Sim/USD render-mesh attribute layout.
# Isaac Lab direct-workflow tasks also keep a separate PhysX collision mesh
# for terrain, so we count render + collision buffers unless told otherwise.
# This is an ANALYTIC ESTIMATE, not a measured GPU allocation: no GPU/Isaac
# Sim is available in this environment. It is offered only as a consistent,
# reproducible relative proxy across the three methods, not an absolute
# VRAM figure -- see README "Limitations".
BYTES_PER_VERTEX = 32
COLLISION_MESH_MULTIPLIER = 2.0


def estimate_mesh_memory_bytes(vertex_count: int, with_collision: bool = True) -> float:
    mult = COLLISION_MESH_MULTIPLIER if with_collision else 1.0
    return vertex_count * BYTES_PER_VERTEX * mult


def boundary_continuity(
    height_fn: HeightFn,
    radius: float,
    n_angles: int = 720,
    eps: float = 0.05,
    grad_eps: float = 0.02,
) -> dict[str, float]:
    """C0/C1 discontinuity of ``height_fn`` across the circle r == radius.

    C0: |h(r-eps) - h(r+eps)| sampled at n_angles points around the ring.
    C1: angle (deg) between the finite-difference surface normals evaluated
    just inside vs. just outside the ring.
    """
    angles = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    cos_a, sin_a = np.cos(angles), np.sin(angles)

    x_in, y_in = (radius - eps) * cos_a, (radius - eps) * sin_a
    x_out, y_out = (radius + eps) * cos_a, (radius + eps) * sin_a

    h_in = height_fn(x_in, y_in)
    h_out = height_fn(x_out, y_out)
    c0_err = np.abs(h_in - h_out)

    def normals(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        hx1 = height_fn(x + grad_eps, y)
        hx0 = height_fn(x - grad_eps, y)
        hy1 = height_fn(x, y + grad_eps)
        hy0 = height_fn(x, y - grad_eps)
        gx = (hx1 - hx0) / (2.0 * grad_eps)
        gy = (hy1 - hy0) / (2.0 * grad_eps)
        normal = np.stack([-gx, -gy, np.ones_like(gx)], axis=-1)
        return normal / np.clip(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-9, None)

    n_in = normals(x_in, y_in)
    n_out = normals(x_out, y_out)
    cos_ang = np.clip(np.sum(n_in * n_out, axis=-1), -1.0, 1.0)
    c1_err_deg = np.degrees(np.arccos(cos_ang))

    return {
        "c0_max_m": float(np.max(c0_err)),
        "c0_rms_m": float(np.sqrt(np.mean(c0_err**2))),
        "c1_max_deg": float(np.max(c1_err_deg)),
        "c1_rms_deg": float(np.sqrt(np.mean(c1_err_deg**2))),
    }


def radial_profile(
    height_fn: HeightFn, r_max: float, n_samples: int = 400, angle: float = 0.37
) -> tuple[np.ndarray, np.ndarray]:
    """1D height cross-section along a fixed ray, for visualization."""
    r = np.linspace(0.0, r_max, n_samples)
    x = r * np.cos(angle)
    y = r * np.sin(angle)
    return r, height_fn(x, y)
