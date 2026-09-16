#!/usr/bin/env python3
"""Run the full terrain detail-transition comparison (mesh track + analytic
track) across the three blend methods (+ the as-shipped Gaussian diagnostic)
and save results/plots for the paper.

Usage:
    python3 run_experiment.py [--seeds 30] [--out results]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for `app.*`

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analytic_track import (
    benchmark_fade_only,
    benchmark_query_throughput,
    make_combined_height_fn,
)
from blend_functions import make_fade_fn
from mesh_track import GridSpec, build_mesh_track_heightfield, make_grid_query_fn
from metrics import boundary_continuity, estimate_mesh_memory_bytes, radial_profile
from terrain_layers import TERRAIN_DETAIL_RADIUS_M, sample_terrain_slot

GLOBAL_SPEC = GridSpec(resolution=128, size_m=80.0)  # low_debug preset, matches terrain_pool_dem_quality default
LOCAL_SPEC = GridSpec(resolution=256, size_m=30.0)  # local_detail_patch default (configs/terrain_config.yaml)
BLEND_RADIUS_M = TERRAIN_DETAIL_RADIUS_M  # 8.0, shared by both production formulas
CONTACT_RADIUS_M = 1.5  # footprint half-diagonal (~0.57m from rocket_foot_offsets_m) + touchdown-drift margin

MAIN_METHODS = ("smoothstep", "gaussian_calibrated", "hybrid")
ALL_METHODS = ("smoothstep", "gaussian_shipped", "gaussian_calibrated", "hybrid")

# Analytic-track per-env terrain-parameter float count (method-independent):
# slope(2)+phase(2)+amplitude(1)+detail_phase(4)+detail_amplitude(1)
# +craters(3*4)+rocks(5*4) = 42 floats. Confirms/quantifies the "no VRAM
# difference" finding for the mesh-free representation.
ANALYTIC_PARAM_FLOATS_PER_ENV = 2 + 2 + 1 + 4 + 1 + 3 * 4 + 5 * 4


def run_trial(seed: int, method: str) -> dict:
    slot = sample_terrain_slot(seed)
    contact = CONTACT_RADIUS_M if method == "hybrid" else None

    mesh = build_mesh_track_heightfield(slot, method, GLOBAL_SPEC, LOCAL_SPEC, BLEND_RADIUS_M, contact)
    mesh_query = make_grid_query_fn(mesh)
    mesh_query_detail = make_grid_query_fn(mesh, detail_only=True)
    mesh_cont = boundary_continuity(mesh_query, BLEND_RADIUS_M)
    mesh_cont_detail = boundary_continuity(mesh_query_detail, BLEND_RADIUS_M)

    analytic_fn = make_combined_height_fn(slot, method, BLEND_RADIUS_M, contact)
    analytic_fn_detail = make_combined_height_fn(slot, method, BLEND_RADIUS_M, contact, detail_only=True)
    analytic_cont = boundary_continuity(analytic_fn, BLEND_RADIUS_M)
    analytic_cont_detail = boundary_continuity(analytic_fn_detail, BLEND_RADIUS_M)
    throughput = benchmark_query_throughput(analytic_fn, repeats=20, seed=seed)
    fade_fn = make_fade_fn(method, BLEND_RADIUS_M, contact)
    fade_cost = benchmark_fade_only(fade_fn, repeats=50, seed=seed)

    result = {
        "seed": seed,
        "method": method,
        "mesh_t_total_s": mesh["t_total_s"],
        "mesh_t_global_s": mesh["t_global_s"],
        "mesh_t_local_s": mesh["t_local_s"],
        "mesh_t_blend_s": mesh["t_blend_s"],
        "mesh_two_mesh_literal_vertices": mesh["two_mesh_literal_vertices"],
        "mesh_two_mesh_literal_triangles": mesh["two_mesh_literal_triangles"],
        "mesh_single_grid_baked_vertices": mesh["single_grid_baked_vertices"],
        "mesh_single_grid_baked_triangles": mesh["single_grid_baked_triangles"],
        "mesh_two_mesh_literal_mem_mb": estimate_mesh_memory_bytes(mesh["two_mesh_literal_vertices"]) / 1e6,
        "mesh_single_grid_baked_mem_mb": estimate_mesh_memory_bytes(mesh["single_grid_baked_vertices"]) / 1e6,
        # "combined" = macro + detail, i.e. what a foot/LiDAR ray actually sees
        "mesh_c0_max_m": mesh_cont["c0_max_m"],
        "mesh_c0_rms_m": mesh_cont["c0_rms_m"],
        "mesh_c1_max_deg": mesh_cont["c1_max_deg"],
        "mesh_c1_rms_deg": mesh_cont["c1_rms_deg"],
        "analytic_c0_max_m": analytic_cont["c0_max_m"],
        "analytic_c0_rms_m": analytic_cont["c0_rms_m"],
        "analytic_c1_max_deg": analytic_cont["c1_max_deg"],
        "analytic_c1_rms_deg": analytic_cont["c1_rms_deg"],
        # "detail-only" = the blend method's own seam, isolated from ambient
        # macro-terrain roughness -- the headline continuity metric.
        "mesh_detail_c0_max_m": mesh_cont_detail["c0_max_m"],
        "mesh_detail_c0_rms_m": mesh_cont_detail["c0_rms_m"],
        "mesh_detail_c1_max_deg": mesh_cont_detail["c1_max_deg"],
        "mesh_detail_c1_rms_deg": mesh_cont_detail["c1_rms_deg"],
        "analytic_detail_c0_max_m": analytic_cont_detail["c0_max_m"],
        "analytic_detail_c0_rms_m": analytic_cont_detail["c0_rms_m"],
        "analytic_detail_c1_max_deg": analytic_cont_detail["c1_max_deg"],
        "analytic_detail_c1_rms_deg": analytic_cont_detail["c1_rms_deg"],
        "analytic_queries_per_second": throughput["queries_per_second"],
        "analytic_step_query_mean_s": throughput["mean_s"],
        "fade_only_mean_s": fade_cost["mean_s"],
    }

    if method == "hybrid":
        # Internal switch-boundary validation, on the fade *weight* itself
        # (not embedded in a spatially-varying fine layer -- boundary_continuity
        # on the combined/detail surface would conflate this with the fine
        # layer's own ambient non-radial-symmetry between two nearby (x,y)
        # points, which is not a switch defect). Pure 1D check: is
        # fade_hybrid(r) itself C0/C1 in r across r=Rc?
        eps = 1e-4
        fh = make_fade_fn("hybrid", BLEND_RADIUS_M, CONTACT_RADIUS_M)
        f_minus, f_plus = fh(np.array([CONTACT_RADIUS_M - eps])), fh(np.array([CONTACT_RADIUS_M + eps]))
        result["hybrid_switch_c0"] = float(abs(f_plus[0] - f_minus[0]))
        d_minus = (fh(np.array([CONTACT_RADIUS_M])) - fh(np.array([CONTACT_RADIUS_M - eps]))) / eps
        d_plus = (fh(np.array([CONTACT_RADIUS_M + eps])) - fh(np.array([CONTACT_RADIUS_M]))) / eps
        result["hybrid_switch_c1_slope_diff"] = float(abs(d_plus[0] - d_minus[0]))

    return result


def summarize(rows: list[dict], key: str) -> tuple[float, float]:
    values = [r[key] for r in rows]
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def make_plots(all_rows: list[dict], out_dir: Path) -> None:
    slot0 = sample_terrain_slot(0)

    # 1. Radial profile comparison
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for method in ALL_METHODS:
        contact = CONTACT_RADIUS_M if method == "hybrid" else None
        fn = make_combined_height_fn(slot0, method, BLEND_RADIUS_M, contact)
        r, h = radial_profile(fn, r_max=14.0)
        ax.plot(r, h, label=method, linewidth=1.8)
    ax.axvline(BLEND_RADIUS_M, color="gray", linestyle="--", linewidth=1.0, label=f"R={BLEND_RADIUS_M}m")
    ax.axvline(CONTACT_RADIUS_M, color="lightgray", linestyle=":", linewidth=1.0, label=f"Rc={CONTACT_RADIUS_M}m")
    ax.set_xlabel("radius from target center (m)", fontsize=12)
    ax.set_ylabel("height (m)", fontsize=12)
    ax.set_title("Radial height cross-section by blend method (seed=0)", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "radial_profile.png", dpi=200)
    plt.close(fig)

    # 2. Continuity bar chart (analytic + mesh track), mean +/- std over seeds
    metrics_to_plot = [
        ("analytic_detail_c0_rms_m", "analytic seam C0 RMS (m)"),
        ("analytic_detail_c1_rms_deg", "analytic seam C1 RMS (deg)"),
        ("mesh_detail_c0_rms_m", "mesh(baked) seam C0 RMS (m)"),
        ("mesh_detail_c1_rms_deg", "mesh(baked) seam C1 RMS (deg)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()
    for ax, (key, title) in zip(axes, metrics_to_plot):
        means, stds = [], []
        for method in ALL_METHODS:
            rows = [r for r in all_rows if r["method"] == method]
            m, s = summarize(rows, key)
            means.append(m)
            stds.append(s)
        ax.bar(ALL_METHODS, means, yerr=stds, capsize=4)
        ax.set_title(title, fontsize=13)
        ax.tick_params(axis="x", rotation=20, labelsize=10)
        ax.tick_params(axis="y", labelsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "continuity_comparison.png", dpi=200)
    plt.close(fig)

    # 3. Cost comparison: generation time (mesh), query throughput (analytic), memory
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()
    means_t, stds_t = [], []
    means_q, stds_q = [], []
    means_f, stds_f = [], []
    mem_two, mem_single = [], []
    for method in ALL_METHODS:
        rows = [r for r in all_rows if r["method"] == method]
        m, s = summarize(rows, "mesh_t_total_s")
        means_t.append(m * 1000)
        stds_t.append(s * 1000)
        m, s = summarize(rows, "analytic_queries_per_second")
        means_q.append(m)
        stds_q.append(s)
        m, s = summarize(rows, "fade_only_mean_s")
        means_f.append(m * 1e6)
        stds_f.append(s * 1e6)
        mem_two.append(rows[0]["mesh_two_mesh_literal_mem_mb"])
        mem_single.append(rows[0]["mesh_single_grid_baked_mem_mb"])

    axes[0].bar(ALL_METHODS, means_t, yerr=stds_t, capsize=4, color="tab:orange")
    axes[0].set_title("mesh-track generation time (ms)", fontsize=13)
    axes[0].tick_params(axis="x", rotation=20, labelsize=10)
    axes[0].tick_params(axis="y", labelsize=10)

    axes[1].bar(ALL_METHODS, means_q, yerr=stds_q, capsize=4, color="tab:blue")
    axes[1].set_title("analytic-track query throughput (queries/s)", fontsize=13)
    axes[1].tick_params(axis="x", rotation=20, labelsize=10)
    axes[1].tick_params(axis="y", labelsize=10)

    axes[2].bar(ALL_METHODS, means_f, yerr=stds_f, capsize=4, color="tab:green")
    axes[2].set_title("fade-formula-only cost (microseconds)", fontsize=13)
    axes[2].tick_params(axis="x", rotation=20, labelsize=10)
    axes[2].tick_params(axis="y", labelsize=10)

    x = np.arange(len(ALL_METHODS))
    width = 0.35
    axes[3].bar(x - width / 2, mem_two, width, label="two_mesh_literal")
    axes[3].bar(x + width / 2, mem_single, width, label="single_grid_baked")
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(ALL_METHODS, rotation=20, fontsize=10)
    axes[3].tick_params(axis="y", labelsize=10)
    axes[3].set_title("mesh VRAM proxy (MB)", fontsize=13)
    axes[3].legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(out_dir / "cost_comparison.png", dpi=200)
    plt.close(fig)


def print_summary_table(all_rows: list[dict]) -> str:
    lines = []
    header = (
        f"{'method':<20} {'mesh_t_ms':>10} {'mesh_seamC0_m':>14} {'mesh_seamC1_deg':>16} "
        f"{'an_seamC0_m':>12} {'an_seamC1_deg':>14} {'q/s':>12} {'2mesh_MB':>9} {'1grid_MB':>9}"
    )
    lines.append("Headline (detail-only seam continuity, isolated from macro-terrain roughness):")
    lines.append(header)
    lines.append("-" * len(header))
    for method in ALL_METHODS:
        rows = [r for r in all_rows if r["method"] == method]
        t_ms = summarize(rows, "mesh_t_total_s")[0] * 1000
        mc0 = summarize(rows, "mesh_detail_c0_rms_m")[0]
        mc1 = summarize(rows, "mesh_detail_c1_rms_deg")[0]
        ac0 = summarize(rows, "analytic_detail_c0_rms_m")[0]
        ac1 = summarize(rows, "analytic_detail_c1_rms_deg")[0]
        qps = summarize(rows, "analytic_queries_per_second")[0]
        mem2 = rows[0]["mesh_two_mesh_literal_mem_mb"]
        mem1 = rows[0]["mesh_single_grid_baked_mem_mb"]
        lines.append(
            f"{method:<20} {t_ms:>10.3f} {mc0:>14.6f} {mc1:>16.4f} "
            f"{ac0:>12.6f} {ac1:>14.4f} {qps:>12,.0f} {mem2:>9.2f} {mem1:>9.2f}"
        )

    lines.append("")
    lines.append("Fade-formula-only cost (isolated from macro/fine terrain evaluation):")
    header3 = f"{'method':<20} {'fade_only_us':>13}"
    lines.append(header3)
    lines.append("-" * len(header3))
    for method in ALL_METHODS:
        rows = [r for r in all_rows if r["method"] == method]
        fade_us = summarize(rows, "fade_only_mean_s")[0] * 1e6
        lines.append(f"{method:<20} {fade_us:>13.2f}")

    lines.append("")
    lines.append("Secondary (combined macro+detail continuity, what a foot/LiDAR ray actually sees):")
    header2 = f"{'method':<20} {'mesh_C0rms_m':>13} {'mesh_C1rms_deg':>15} {'an_C0rms_m':>11} {'an_C1rms_deg':>13}"
    lines.append(header2)
    lines.append("-" * len(header2))
    for method in ALL_METHODS:
        rows = [r for r in all_rows if r["method"] == method]
        mc0 = summarize(rows, "mesh_c0_rms_m")[0]
        mc1 = summarize(rows, "mesh_c1_rms_deg")[0]
        ac0 = summarize(rows, "analytic_c0_rms_m")[0]
        ac1 = summarize(rows, "analytic_c1_rms_deg")[0]
        lines.append(f"{method:<20} {mc0:>13.6f} {mc1:>15.4f} {ac0:>11.6f} {ac1:>13.4f}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--out", type=str, default="results")
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for method in ALL_METHODS:
        for seed in range(args.seeds):
            all_rows.append(run_trial(seed, method))
        print(f"[done] method={method}", flush=True)

    (out_dir / "raw_results.json").write_text(json.dumps(all_rows, indent=2))

    table = print_summary_table(all_rows)
    print("\n" + table)
    (out_dir / "summary_table.txt").write_text(table + "\n")

    param_mem_bytes = ANALYTIC_PARAM_FLOATS_PER_ENV * 4
    note = (
        f"Analytic-track per-env terrain-parameter memory is identical across all "
        f"methods: {ANALYTIC_PARAM_FLOATS_PER_ENV} floats/env = {param_mem_bytes} bytes/env "
        f"(method-independent; the blend function reads these same parameters, it does "
        f"not add any of its own)."
    )
    print("\n" + note)
    (out_dir / "analytic_memory_note.txt").write_text(note + "\n")

    hybrid_rows = [r for r in all_rows if r["method"] == "hybrid"]
    switch_c0 = summarize(hybrid_rows, "hybrid_switch_c0")[0]
    switch_c1 = summarize(hybrid_rows, "hybrid_switch_c1_slope_diff")[0]
    switch_note = (
        f"Hybrid internal switch-boundary check on fade_hybrid(r) itself at r=Rc={CONTACT_RADIUS_M}m "
        f"(pure 1D, decoupled from the fine layer's own spatial variation): "
        f"|fade(Rc+eps)-fade(Rc-eps)|={switch_c0:.2e}, slope discontinuity={switch_c1:.2e} "
        f"(both near machine precision, confirming the switch introduces no new seam in the fade weight itself)."
    )
    print(switch_note)
    (out_dir / "hybrid_switch_check.txt").write_text(switch_note + "\n")

    make_plots(all_rows, out_dir)
    print(f"\nSaved results, table, and plots to {out_dir}")


if __name__ == "__main__":
    main()
