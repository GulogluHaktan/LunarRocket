#!/usr/bin/env python3
"""Ablation sweeps + paired significance tests, extending run_experiment.py.

Produces CSV + Markdown tables and PNG plots under results/ablations/:

  A) significance_tests   -- paired tests (30 seeds) on the main comparison
  B) rc_sweep              -- hybrid contact-radius Rc sensitivity
  C) switch_width_sweep    -- hybrid switch-band width vs. curvature (C2) cost
  D) radius_sweep          -- blend radius R sensitivity (is gaussian_shipped's
                               ~37% residual structural, i.e. R-independent?)
  E) resolution_sweep      -- mesh-track discretization error vs. grid
                               resolution, with an empirical convergence-rate fit

Requires scipy (see .venv/ in this directory: `python3 -m venv .venv &&
.venv/bin/pip install numpy matplotlib scipy`).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sstats

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Larger default fonts so 2-panel figures stay legible once embedded at
# ~6in width in the paper (the matplotlib defaults were sized for on-screen
# viewing, not for a printed page column).
plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
})

from analytic_track import benchmark_fade_only, make_combined_height_fn
from blend_functions import GAUSSIAN_K_CALIBRATED, fade_gaussian, fade_hybrid, fade_smoothstep, make_fade_fn
from mesh_track import GridSpec, build_mesh_track_heightfield, make_grid_query_fn
from metrics import boundary_continuity
from terrain_layers import TERRAIN_DETAIL_RADIUS_M, sample_terrain_slot

SEEDS = list(range(30))
BASE_R = TERRAIN_DETAIL_RADIUS_M  # 8.0
BASE_RC = 1.5
BASE_WIDTH = 0.3
GLOBAL_SPEC = GridSpec(resolution=128, size_m=80.0)
LOCAL_SPEC = GridSpec(resolution=256, size_m=30.0)

OUT = Path(__file__).resolve().parent / "results" / "ablations"
OUT.mkdir(parents=True, exist_ok=True)


def write_csv(rows: list[dict], name: str) -> None:
    if not rows:
        return
    with open(OUT / f"{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_markdown_table(headers: list[str], rows: list[list], name: str) -> None:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(v) for v in r) + " |")
    (OUT / f"{name}.md").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# A) Significance tests on the main 30-seed comparison
# ---------------------------------------------------------------------------


def significance_tests() -> None:
    raw = json.loads((Path(__file__).resolve().parent / "results" / "raw_results.json").read_text())
    by_method = {m: [r for r in raw if r["method"] == m] for m in ("smoothstep", "gaussian_shipped", "gaussian_calibrated", "hybrid")}
    baseline = "smoothstep"
    rows = []
    for key in ("analytic_detail_c0_rms_m", "analytic_detail_c1_rms_deg"):
        for method in ("gaussian_shipped", "gaussian_calibrated", "hybrid"):
            a = np.array([r[key] for r in by_method[baseline]])
            b = np.array([r[key] for r in by_method[method]])
            log_a, log_b = np.log10(a + 1e-15), np.log10(b + 1e-15)
            diff = log_b - log_a

            if np.allclose(diff, 0.0, atol=1e-9):
                rows.append(
                    {
                        "metric": key,
                        "comparison": f"{method} vs {baseline}",
                        "mean_log10_ratio": 0.0,
                        "log10_ratio_min": 0.0,
                        "log10_ratio_max": 0.0,
                        "wilcoxon_pvalue": "n/a (identical, 0/30 nonzero diffs)",
                        "n": len(diff),
                        "note": "analytically identical at r=R by construction",
                    }
                )
                continue

            # Near-deterministic paired differences (very small cross-seed
            # variance is expected here -- the *shape* of the discontinuity
            # is a fixed property of the two formulas, largely independent
            # of which random terrain realization it's evaluated on) make
            # classical Cohen's d / t-CI blow up or degenerate. We report
            # the robust, assumption-free signal instead: Wilcoxon
            # signed-rank p-value plus the actual min/max range observed
            # across all 30 independent terrain samples (a tight range is
            # itself the finding: the effect is not a fluke of one seed).
            try:
                w_stat, p_w = sstats.wilcoxon(diff)
            except ValueError:
                w_stat, p_w = float("nan"), float("nan")
            rows.append(
                {
                    "metric": key,
                    "comparison": f"{method} vs {baseline}",
                    "mean_log10_ratio": round(float(diff.mean()), 4),
                    "log10_ratio_min": round(float(diff.min()), 4),
                    "log10_ratio_max": round(float(diff.max()), 4),
                    "wilcoxon_pvalue": f"{p_w:.3e}",
                    "n": len(diff),
                    "note": "",
                }
            )
    write_csv(rows, "significance_tests")
    write_markdown_table(list(rows[0].keys()), [list(r.values()) for r in rows], "significance_tests")
    print("\n[A] Significance tests (paired, n=30, log10-ratio of seam error vs. smoothstep baseline):")
    for r in rows:
        print(
            f"  [{r['metric']}] {r['comparison']:<32} mean_log10_ratio={r['mean_log10_ratio']:>7} "
            f"range=[{r['log10_ratio_min']},{r['log10_ratio_max']}] wilcoxon_p={r['wilcoxon_pvalue']:>28} {r['note']}"
        )


# ---------------------------------------------------------------------------
# B) Rc sweep (hybrid)
# ---------------------------------------------------------------------------


def rc_sweep() -> list[dict]:
    rows = []
    for rc in (1.0, 1.5, 2.0, 3.0, 4.0):
        for seed in SEEDS:
            def fh(r, rc=rc):
                return fade_hybrid(r, BASE_R, rc, BASE_WIDTH)

            eps = 1e-4
            f_minus = fh(np.array([rc - eps]))[0]
            f0 = fh(np.array([rc]))[0]
            f_plus = fh(np.array([rc + eps]))[0]
            c0 = abs(f_plus - f_minus)
            d_minus = (f0 - f_minus) / eps
            d_plus = (f_plus - f0) / eps
            c1_slope = abs(d_plus - d_minus)
            fade_cost_us = benchmark_fade_only(fh, n_queries=47104, repeats=10, seed=seed)["mean_s"] * 1e6
            rows.append({"rc_m": rc, "seed": seed, "switch_c0": c0, "switch_c1_slope": c1_slope, "fade_cost_us": fade_cost_us})
    write_csv(rows, "rc_sweep")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    rcs = sorted(set(r["rc_m"] for r in rows))
    c0_means = [np.mean([r["switch_c0"] for r in rows if r["rc_m"] == rc]) for rc in rcs]
    cost_means = [np.mean([r["fade_cost_us"] for r in rows if r["rc_m"] == rc]) for rc in rcs]
    cost_stds = [np.std([r["fade_cost_us"] for r in rows if r["rc_m"] == rc]) for rc in rcs]
    axes[0].plot(rcs, c0_means, "o-")
    axes[0].set_xlabel("Rc (m)")
    axes[0].set_ylabel("internal switch C0 error")
    axes[0].set_title("Switch-boundary continuity vs. Rc\n(stays ~machine precision)")
    axes[1].errorbar(rcs, cost_means, yerr=cost_stds, fmt="o-", color="tab:green")
    axes[1].set_xlabel("Rc (m)")
    axes[1].set_ylabel("fade-only cost (us)")
    axes[1].set_title("Hybrid fade cost vs. Rc")
    fig.tight_layout()
    fig.savefig(OUT / "rc_sweep.png", dpi=200)
    plt.close(fig)
    print(f"[B] Rc sweep: switch_c0 range [{min(c0_means):.2e}, {max(c0_means):.2e}]; "
          f"fade cost range [{min(cost_means):.1f}, {max(cost_means):.1f}] us")
    return rows


# ---------------------------------------------------------------------------
# C) Switch-width sweep -- C0/C1 stay ~0, but curvature (C2) blows up as
#    width -> 0 (a sharper switch is "less smooth" in a 2nd-derivative sense
#    even though it's still formally C1)
# ---------------------------------------------------------------------------


def switch_width_sweep() -> list[dict]:
    rows = []
    for width in (0.05, 0.1, 0.3, 0.6, 1.0, 2.0):
        for seed in SEEDS:
            def fh(r, width=width):
                return fade_hybrid(r, BASE_R, BASE_RC, width)

            eps = 1e-4
            f_minus, f0, f_plus = (fh(np.array([BASE_RC + d]))[0] for d in (-eps, 0.0, eps))
            c0 = abs(f_plus - f_minus)
            curvature_eps = max(width * 0.1, 1e-3)
            fm, f0c, fp = (fh(np.array([BASE_RC + d]))[0] for d in (-curvature_eps, 0.0, curvature_eps))
            c2 = abs((fp - 2 * f0c + fm) / curvature_eps**2)
            rows.append({"switch_width_m": width, "seed": seed, "switch_c0": c0, "curvature_c2": c2})
    write_csv(rows, "switch_width_sweep")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    widths = sorted(set(r["switch_width_m"] for r in rows))
    c0_means = [np.mean([r["switch_c0"] for r in rows if r["switch_width_m"] == w]) for w in widths]
    c2_means = [np.mean([r["curvature_c2"] for r in rows if r["switch_width_m"] == w]) for w in widths]
    axes[0].semilogy(widths, np.array(c0_means) + 1e-18, "o-")
    axes[0].set_xlabel("switch width (m)")
    axes[0].set_ylabel("switch C0 (log scale)")
    axes[0].set_title("C0 stays ~0 regardless of width")
    axes[1].loglog(widths, c2_means, "o-", color="tab:red")
    axes[1].set_xlabel("switch width (m)")
    axes[1].set_ylabel("|2nd derivative| at Rc")
    axes[1].set_title("Curvature cost grows as width -> 0\n(formally still C1, but less smooth)")
    fig.tight_layout()
    fig.savefig(OUT / "switch_width_sweep.png", dpi=200)
    plt.close(fig)
    print(f"[C] Switch-width sweep: C2 at width={widths[0]}m -> {c2_means[0]:.3f}, at width={widths[-1]}m -> {c2_means[-1]:.3f}")
    return rows


# ---------------------------------------------------------------------------
# D) Blend radius R sweep -- is the shipped Gaussian's residual structural
#    (scale-invariant, since exp(-(r/R)^2) depends only on r/R) or an
#    artifact of the specific R=8m default?
# ---------------------------------------------------------------------------


def radius_sweep() -> list[dict]:
    rows = []
    for R in (4.0, 6.0, 8.0, 12.0, 16.0):
        for method in ("smoothstep", "gaussian_shipped", "gaussian_calibrated"):
            for seed in SEEDS:
                slot = sample_terrain_slot(seed)
                fn = make_combined_height_fn(slot, method, R, detail_only=True)
                cont = boundary_continuity(fn, R, n_angles=180)
                rows.append({"R_m": R, "method": method, "seed": seed, "seam_c0_rms_m": cont["c0_rms_m"], "seam_c1_rms_deg": cont["c1_rms_deg"]})
    write_csv(rows, "radius_sweep")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    Rs = sorted(set(r["R_m"] for r in rows))
    for method, color in zip(("smoothstep", "gaussian_shipped", "gaussian_calibrated"), ("tab:blue", "tab:orange", "tab:green")):
        c0 = [np.mean([r["seam_c0_rms_m"] for r in rows if r["R_m"] == R and r["method"] == method]) for R in Rs]
        c1 = [np.mean([r["seam_c1_rms_deg"] for r in rows if r["R_m"] == R and r["method"] == method]) for R in Rs]
        axes[0].semilogy(Rs, np.array(c0) + 1e-12, "o-", label=method, color=color)
        axes[1].semilogy(Rs, np.array(c1) + 1e-12, "o-", label=method, color=color)
    axes[0].set_xlabel("blend radius R (m)")
    axes[0].set_ylabel("seam C0 RMS (m, log scale)")
    axes[0].set_title("Seam error vs. R")
    axes[0].legend()
    axes[1].set_xlabel("blend radius R (m)")
    axes[1].set_ylabel("seam C1 RMS (deg, log scale)")
    axes[1].set_title("Normal-angle error vs. R")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "radius_sweep.png", dpi=200)
    plt.close(fig)
    gs = [np.mean([r["seam_c1_rms_deg"] for r in rows if r["R_m"] == R and r["method"] == "gaussian_shipped"]) for R in Rs]
    print(f"[D] Radius sweep: gaussian_shipped seam C1 RMS across R={Rs} -> {[round(v,3) for v in gs]} (structural if roughly constant)")
    return rows


# ---------------------------------------------------------------------------
# E) Mesh resolution sweep -- does the mesh-track's extra (discretization)
#    seam error over the pure-analytic seam shrink with grid resolution,
#    and at what empirical convergence order?
# ---------------------------------------------------------------------------


def resolution_sweep() -> list[dict]:
    rows = []
    global_resolutions = (64, 96, 128, 192, 256)
    n_seeds = 10
    for res in global_resolutions:
        spec = GridSpec(resolution=res, size_m=80.0)
        for method in ("smoothstep", "gaussian_calibrated", "hybrid"):
            for seed in range(n_seeds):
                slot = sample_terrain_slot(seed)
                contact = BASE_RC if method == "hybrid" else None
                mesh = build_mesh_track_heightfield(slot, method, spec, LOCAL_SPEC, BASE_R, contact)
                query_detail = make_grid_query_fn(mesh, detail_only=True)
                cont = boundary_continuity(query_detail, BASE_R, n_angles=180)
                rows.append(
                    {
                        "global_resolution": res,
                        "grid_spacing_m": mesh["global_grid_spacing_m"],
                        "method": method,
                        "seed": seed,
                        "seam_c0_rms_m": cont["c0_rms_m"],
                        "t_total_ms": mesh["t_total_s"] * 1000,
                        "single_grid_baked_vertices": mesh["single_grid_baked_vertices"],
                    }
                )
    write_csv(rows, "resolution_sweep")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for method, color in zip(("smoothstep", "gaussian_calibrated", "hybrid"), ("tab:blue", "tab:green", "tab:red")):
        spacings = sorted(set(r["grid_spacing_m"] for r in rows), reverse=True)
        c0 = [np.mean([r["seam_c0_rms_m"] for r in rows if r["grid_spacing_m"] == s and r["method"] == method]) for s in spacings]
        axes[0].loglog(spacings, np.array(c0) + 1e-9, "o-", label=method, color=color)
        # empirical convergence order: slope of log(error) vs log(spacing)
        log_s, log_e = np.log(spacings), np.log(np.array(c0) + 1e-9)
        slope, intercept = np.polyfit(log_s, log_e, 1)
        print(f"[E] {method}: empirical convergence order p ~= {slope:.2f} (error ~ spacing^p)")

    t_res = sorted(set(r["global_resolution"] for r in rows))
    t_means = [np.mean([r["t_total_ms"] for r in rows if r["global_resolution"] == res]) for res in t_res]
    axes[1].plot(t_res, t_means, "o-", color="tab:purple")
    axes[1].set_xlabel("global grid resolution")
    axes[1].set_ylabel("mesh generation time (ms)")
    axes[1].set_title("Generation time vs. resolution")

    axes[0].set_xlabel("grid spacing (m, log scale)")
    axes[0].set_ylabel("discretized seam C0 RMS (m, log scale)")
    axes[0].set_title("Discretization error vs. grid spacing")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(OUT / "resolution_sweep.png", dpi=200)
    plt.close(fig)
    return rows


def main() -> None:
    significance_tests()
    rc_sweep()
    switch_width_sweep()
    radius_sweep()
    resolution_sweep()
    print(f"\nAblation tables/plots saved to {OUT}")


if __name__ == "__main__":
    main()
