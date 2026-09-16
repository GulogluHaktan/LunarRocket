#!/usr/bin/env python3
"""Compare the real Isaac Lab SAC training runs launched by
scripts/train_terrain_blend_comparison.sh (smoothstep / gaussian / hybrid
terrain fine-detail blend method, now with multiple seeds per method) using
their TensorBoard logs.

This is the GPU-backed follow-up to the CPU-only study in this directory:
same three blend methods, but now measured as actual training outcomes
(success rate, timeout rate, harsh-landing rate, curriculum progression)
on the real Isaac Lab / PhysX terrain-pool path, not a NumPy replica.

Multi-seed statistics (v2): each (method, seed) run contributes ONE
final-window summary value per metric (mean of the last TAIL_FRACTION of
logged points within that run). Comparisons across methods are then done on
these per-seed values -- genuinely independent samples -- rather than on
autocorrelated points within a single run's moving-window metric (the v1
approach, which is a pseudoreplication problem: consecutive points in one
run's rolling-window statistic are not independent observations). With a
handful of seeds per method this remains a modest-power comparison, but it
is the statistically correct unit of replication for RL training outcomes
(Henderson et al., 2018).

Usage (from repo root, after scripts/train_terrain_blend_comparison.sh
finishes -- or partway through, to check progress on completed runs):
    experiments/terrain_transition/.venv/bin/python \\
        experiments/terrain_transition/analyze_training_comparison.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Larger default fonts so multi-panel figures stay legible once embedded at
# ~6in width in the paper.
plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})

from scipy import stats as sstats
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_ROOT = REPO_ROOT / "logs" / "sb3_sac" / "LunarRocket-Lander-Direct-v0"
OUT = Path(__file__).resolve().parent / "results" / "training_comparison"
# logs/sb3_sac/.../ is root-owned (Docker writes it as root through the bind
# mount) -- scripts/train_terrain_blend_comparison.sh therefore writes the
# manifest under OUT (host-user-owned) instead of alongside the run dirs.
MANIFEST_JSONL = OUT / "comparison_manifest.jsonl"
MANIFEST_LEGACY = OUT / "comparison_manifest.json"  # v1 single-seed format, read-only fallback

TAGS = (
    "Metrics/success_rate",
    "Metrics/timeout_rate",
    "Metrics/harsh_landing_rate",
    "Curriculum/difficulty",
    "train/ent_coef",
    "Episode_Reward/xy_progress",
)

COLORS = {"smoothstep": "tab:blue", "gaussian": "tab:orange", "hybrid": "tab:red"}
STAT_METRICS = ("Metrics/success_rate", "Metrics/timeout_rate", "Metrics/harsh_landing_rate")
TAIL_FRACTION = 0.10  # last 10% of logged points within one run, treated as its "converged" summary

STAT_CAVEAT = (
    "Each (method, seed) run contributes ONE final-window summary value per metric "
    "(mean of the run's last {:.0%} of logged points). Cross-method comparisons below "
    "use these per-seed values as independent samples (Mann-Whitney U, unpaired), which "
    "is the statistically appropriate unit for RL training outcomes -- unlike treating "
    "within-run autocorrelated points as independent (pseudoreplication). With only a "
    "handful of seeds per method, this remains LOW-POWER: a non-significant p-value here "
    "does not establish equivalence, and even a significant one should be read as an "
    "indicative effect size, not a definitive claim, per Henderson et al. (2018) 'Deep "
    "Reinforcement Learning That Matters' (AAAI)."
).format(TAIL_FRACTION)


def load_manifest_rows() -> list[dict]:
    rows = []
    if MANIFEST_JSONL.exists():
        for line in MANIFEST_JSONL.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    elif MANIFEST_LEGACY.exists():
        print(f"[note] no {MANIFEST_JSONL.name}; falling back to legacy {MANIFEST_LEGACY.name} (seed=42 only)")
        data = json.loads(MANIFEST_LEGACY.read_text())
        for method, entry in data.items():
            if isinstance(entry, dict):
                rows.append({"method": method, "seed": 42, **entry})
            else:
                rows.append({"method": method, "seed": 42, "run_dir": entry, "wall_seconds": None, "status": "ok"})
    else:
        print(f"No manifest found ({MANIFEST_JSONL} or {MANIFEST_LEGACY}) -- "
              f"run scripts/train_terrain_blend_comparison.sh first.", file=sys.stderr)
        sys.exit(1)
    return rows


def load_scalars(run_dir: Path) -> dict[str, tuple[list[float], list[float]]]:
    sac_dirs = sorted(p for p in run_dir.glob("SAC_*") if p.is_dir())
    if not sac_dirs:
        raise FileNotFoundError(f"no SAC_*/ subdir under {run_dir}")
    tb_dir = sac_dirs[-1]
    ea = EventAccumulator(str(tb_dir), size_guidance={"scalars": 0})
    ea.Reload()
    available = set(ea.Tags().get("scalars", []))
    out = {}
    for tag in TAGS:
        if tag not in available:
            continue
        events = ea.Scalars(tag)
        out[tag] = ([e.step for e in events], [e.value for e in events])
    return out


def tail_mean(values: list[float], frac: float = TAIL_FRACTION) -> float:
    n = max(1, int(len(values) * frac))
    return float(np.mean(values[-n:]))


def bootstrap_ci_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 5000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        sa = rng.choice(a, size=len(a), replace=True)
        sb = rng.choice(b, size=len(b), replace=True)
        diffs[i] = sa.mean() - sb.mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(diffs.mean()), float(lo), float(hi)


def auc_sample_efficiency(steps: list[float], values: list[float]) -> float:
    if len(steps) < 2:
        return float(values[0]) if values else float("nan")
    return float(np.trapezoid(values, steps) / (steps[-1] - steps[0]))


def steps_to_threshold(steps: list[float], values: list[float], threshold: float, smooth_window: int = 5) -> float | None:
    steps_arr, values_arr = np.asarray(steps, dtype=float), np.asarray(values, dtype=float)
    if len(values_arr) >= smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        values_arr = np.convolve(values_arr, kernel, mode="valid")
        steps_arr = steps_arr[smooth_window - 1:]
    hits = np.nonzero(values_arr >= threshold)[0]
    return float(steps_arr[hits[0]]) if len(hits) else None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_manifest_rows()

    # method -> seed -> {tag: (steps, values)}
    per_seed_scalars: dict[str, dict[int, dict]] = defaultdict(dict)
    wall_seconds: dict[str, dict[int, float | None]] = defaultdict(dict)
    for row in rows:
        method, seed = row.get("method"), int(row.get("seed", -1))
        if row.get("status") != "ok" or not row.get("run_dir"):
            print(f"[skip] {method} seed={seed}: status={row.get('status')!r}, run_dir={row.get('run_dir')!r}")
            continue
        run_dir = LOG_ROOT / row["run_dir"]
        if not run_dir.exists():
            print(f"[skip] {method} seed={seed}: {run_dir} does not exist")
            continue
        try:
            scalars = load_scalars(run_dir)
        except FileNotFoundError as e:
            print(f"[skip] {method} seed={seed}: {e}")
            continue
        print(f"[load] {method} seed={seed}: {run_dir}")
        per_seed_scalars[method][seed] = scalars
        wall_seconds[method][seed] = row.get("wall_seconds")

    if not per_seed_scalars:
        print("No completed runs found in the manifest yet.", file=sys.stderr)
        sys.exit(1)

    methods = list(per_seed_scalars.keys())
    n_seeds = {m: len(per_seed_scalars[m]) for m in methods}
    print("\nSeeds loaded per method:", n_seeds)

    # --- per-seed final-window summary table -----------------------------
    summary_rows = []
    for method in methods:
        for seed, scalars in per_seed_scalars[method].items():
            row = {"method": method, "seed": seed, "wall_seconds": wall_seconds[method][seed]}
            for tag in ("Metrics/success_rate", "Metrics/timeout_rate", "Metrics/harsh_landing_rate", "Curriculum/difficulty"):
                if tag in scalars and scalars[tag][1]:
                    row[f"final_{tag.split('/')[-1]}"] = round(tail_mean(scalars[tag][1]), 4)
                    row[f"total_steps"] = scalars[tag][0][-1]
            summary_rows.append(row)
    (OUT / "summary.json").write_text(json.dumps(summary_rows, indent=2))
    print("\nPer-(method,seed) final-window summary:")
    for row in summary_rows:
        print(f"  {row}")

    # --- per-method aggregate (mean +/- std across seeds) ------------------
    agg_rows = []
    for method in methods:
        row = {"method": method, "n_seeds": n_seeds[method], "seeds": sorted(per_seed_scalars[method].keys())}
        for tag in STAT_METRICS + ("Curriculum/difficulty",):
            short = tag.split("/")[-1]
            vals = [tail_mean(per_seed_scalars[method][s][tag][1]) for s in per_seed_scalars[method] if tag in per_seed_scalars[method][s]]
            if vals:
                row[f"{short}_mean"] = round(float(np.mean(vals)), 4)
                row[f"{short}_std"] = round(float(np.std(vals, ddof=1)), 4) if len(vals) > 1 else None
                row[f"{short}_values"] = [round(v, 4) for v in vals]
        agg_rows.append(row)
    (OUT / "aggregate_by_method.json").write_text(json.dumps(agg_rows, indent=2))
    print("\nPer-method aggregate (mean +/- std across seeds):")
    for row in agg_rows:
        print(f"  {row}")

    # --- multi-seed line plots: every seed's curve, per method -------------
    # Split into two 1x3 figures (rather than one 2x3 figure) so each panel
    # keeps a wider share of the page width once embedded -- a 2x3 figure
    # wide enough to stay legible at its own native size shrinks far more
    # than a 1x3 one does when both are scaled down to the same ~6in page
    # width.
    TAG_GROUPS = (
        (TAGS[:3], "training_comparison_metrics.png",
         "Isaac Lab SAC training: outcome metrics (all seeds)"),
        (TAGS[3:], "training_comparison_diagnostics.png",
         "Isaac Lab SAC training: curriculum/optimizer diagnostics (all seeds)"),
    )
    for group_tags, filename, suptitle in TAG_GROUPS:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
        for ax, tag in zip(axes, group_tags):
            for method in methods:
                for i, (seed, scalars) in enumerate(sorted(per_seed_scalars[method].items())):
                    if tag not in scalars:
                        continue
                    steps, values = scalars[tag]
                    ax.plot(steps, values, color=COLORS.get(method), alpha=0.55,
                            label=method if i == 0 else None)
            ax.set_title(tag, fontsize=15)
            ax.set_xlabel("env steps", fontsize=13)
            ax.tick_params(labelsize=11)
            ax.legend(fontsize=12)
        fig.suptitle(suptitle, fontsize=15)
        fig.tight_layout()
        fig.savefig(OUT / filename, dpi=200)
        plt.close(fig)

    # --- statistical analysis (across-seed, independent samples) ----------
    print("\n" + "=" * 78)
    print(STAT_CAVEAT)
    print("=" * 78)

    efficiency_rows = []
    for method in methods:
        for seed, scalars in per_seed_scalars[method].items():
            row = {"method": method, "seed": seed}
            for tag in STAT_METRICS:
                if tag not in scalars:
                    continue
                steps, values = scalars[tag]
                short = tag.split("/")[-1]
                row[f"auc_{short}"] = round(auc_sample_efficiency(steps, values), 4)
                if short == "success_rate":
                    row["steps_to_20pct_success"] = steps_to_threshold(steps, values, 0.20)
                    row["steps_to_30pct_success"] = steps_to_threshold(steps, values, 0.30)
            efficiency_rows.append(row)
    (OUT / "sample_efficiency.json").write_text(json.dumps(efficiency_rows, indent=2))
    print("\nSample-efficiency summary (AUC = mean value over the whole run; one row per method/seed):")
    for row in efficiency_rows:
        print(f"  {row}")

    stat_rows = []
    for tag in STAT_METRICS:
        short = tag.split("/")[-1]
        for m1, m2 in combinations(methods, 2):
            a = np.array([tail_mean(per_seed_scalars[m1][s][tag][1]) for s in per_seed_scalars[m1] if tag in per_seed_scalars[m1][s]])
            b = np.array([tail_mean(per_seed_scalars[m2][s][tag][1]) for s in per_seed_scalars[m2] if tag in per_seed_scalars[m2][s]])
            if len(a) < 2 or len(b) < 2:
                note = f"n={len(a)} vs n={len(b)} -- too few seeds for a test (need >=2 per side)"
                stat_rows.append({"metric": tag, "comparison": f"{m1} vs {m2}", "n": [len(a), len(b)], "note": note})
                continue
            if np.allclose(a.mean(), b.mean()) and np.std(np.concatenate([a, b])) == 0:
                stat_rows.append({"metric": tag, "comparison": f"{m1} vs {m2}", "n": [len(a), len(b)],
                                   "note": "zero variance across seeds -- degenerate, no test run"})
                continue
            try:
                u_stat, p_mw = sstats.mannwhitneyu(a, b, alternative="two-sided")
                p_mw_str = f"{p_mw:.3e}"
            except ValueError as e:
                p_mw_str = f"n/a ({e})"
            mean_diff, ci_lo, ci_hi = bootstrap_ci_diff(a, b)
            stat_rows.append({
                "metric": tag,
                "comparison": f"{m1} vs {m2}",
                "n": [len(a), len(b)],
                "mean_a": round(float(a.mean()), 4),
                "mean_b": round(float(b.mean()), 4),
                "values_a": [round(v, 4) for v in a],
                "values_b": [round(v, 4) for v in b],
                "mean_diff_a_minus_b": round(mean_diff, 4),
                "bootstrap_95pct_CI": [round(ci_lo, 4), round(ci_hi, 4)],
                "mannwhitney_pvalue": p_mw_str,
            })
    (OUT / "statistics.json").write_text(json.dumps({"caveat": STAT_CAVEAT, "tests": stat_rows}, indent=2))
    print(f"\nAcross-seed pairwise comparison (independent per-seed final-window means):")
    for row in stat_rows:
        if "mean_a" not in row:
            print(f"  [{row['metric']}] {row['comparison']:<28} {row['note']}")
        else:
            print(
                f"  [{row['metric']}] {row['comparison']:<28} n={row['n']} "
                f"mean_diff={row['mean_diff_a_minus_b']:>8} 95%CI={row['bootstrap_95pct_CI']} "
                f"Mann-Whitney p={row['mannwhitney_pvalue']}"
            )

    # --- across-seed dot plot with mean +/- std, one panel per metric -----
    fig, axes = plt.subplots(1, len(STAT_METRICS), figsize=(4.3 * len(STAT_METRICS), 5.0))
    if len(STAT_METRICS) == 1:
        axes = [axes]
    for ax, tag in zip(axes, STAT_METRICS):
        for x, method in enumerate(methods):
            vals = [tail_mean(per_seed_scalars[method][s][tag][1]) for s in per_seed_scalars[method] if tag in per_seed_scalars[method][s]]
            if not vals:
                continue
            xs = np.full(len(vals), x, dtype=float) + np.random.default_rng(0).uniform(-0.06, 0.06, len(vals))
            ax.scatter(xs, vals, color=COLORS.get(method, "tab:gray"), alpha=0.8, zorder=3, s=60)
            mean, std = np.mean(vals), (np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
            ax.errorbar([x], [mean], yerr=[std] if len(vals) > 1 else None, fmt="_", color="black",
                        markersize=20, capsize=6, zorder=4)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, fontsize=12)
        ax.tick_params(axis="y", labelsize=11)
        ax.set_title(f"{tag}\n(per-seed final-window mean;\nbar = mean +/- std)", fontsize=13)
    fig.suptitle("Across-seed final-window values\n(independent samples -- see statistics.json caveat)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.80])
    fig.savefig(OUT / "final_window_boxplot.png", dpi=200)
    plt.close(fig)

    print(f"\nSaved plots + summary + aggregate + statistics to {OUT}")


if __name__ == "__main__":
    main()
