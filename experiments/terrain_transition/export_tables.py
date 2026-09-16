#!/usr/bin/env python3
"""Render every results table as CSV + Markdown + a PNG table image, so the
paper can pull in whichever format is convenient."""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path(__file__).resolve().parent / "results"
ABLATIONS = RESULTS / "ablations"
TABLES = RESULTS / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

ALL_METHODS = ("smoothstep", "gaussian_shipped", "gaussian_calibrated", "hybrid")


def mean_std(rows: list[dict], key: str) -> tuple[float, float]:
    vals = [float(r[key]) for r in rows]
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


def render_table_png(headers: list[str], rows: list[list[str]], path: Path, title: str) -> None:
    n_cols = len(headers)
    col_chars = [max(len(str(headers[j])), max((len(str(r[j])) for r in rows), default=0)) for j in range(n_cols)]
    total_chars = sum(col_chars)
    col_widths = [c / total_chars for c in col_chars]

    fig_h = 0.6 + 0.4 * len(rows)
    fig_w = max(12, 0.16 * total_chars)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center", colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    for j in range(n_cols):
        table[0, j].set_facecolor("#dbe5f1")
        table[0, j].set_text_props(weight="bold")
    ax.set_title(title, fontsize=12, pad=16)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_csv_md(headers: list[str], rows: list[list], name: str) -> None:
    with open(TABLES / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(v) for v in r) + " |")
    (TABLES / f"{name}.md").write_text("\n".join(lines) + "\n")


def main_comparison_table() -> None:
    raw = json.loads((RESULTS / "raw_results.json").read_text())
    by_method = {m: [r for r in raw if r["method"] == m] for m in ALL_METHODS}
    headers = [
        "method", "mesh gen (ms)", "mesh seam C0 (m)", "mesh seam C1 (deg)",
        "analytic seam C0 (m)", "analytic seam C1 (deg)", "query throughput (q/s)",
        "fade-only (us)", "two-mesh VRAM (MB)", "single-grid VRAM (MB)",
    ]
    rows = []
    for m in ALL_METHODS:
        rs = by_method[m]
        t_ms = mean_std(rs, "mesh_t_total_s")[0] * 1000
        mc0 = mean_std(rs, "mesh_detail_c0_rms_m")[0]
        mc1 = mean_std(rs, "mesh_detail_c1_rms_deg")[0]
        ac0 = mean_std(rs, "analytic_detail_c0_rms_m")[0]
        ac1 = mean_std(rs, "analytic_detail_c1_rms_deg")[0]
        qps = mean_std(rs, "analytic_queries_per_second")[0]
        fade_us = mean_std(rs, "fade_only_mean_s")[0] * 1e6
        mem2 = rs[0]["mesh_two_mesh_literal_mem_mb"]
        mem1 = rs[0]["mesh_single_grid_baked_mem_mb"]
        rows.append([
            m, f"{t_ms:.3f}", f"{mc0:.2e}", f"{mc1:.4f}", f"{ac0:.2e}", f"{ac1:.4f}",
            f"{qps:,.0f}", f"{fade_us:.1f}", f"{mem2:.2f}", f"{mem1:.2f}",
        ])
    write_csv_md(headers, rows, "main_comparison")
    render_table_png(headers, rows, TABLES / "main_comparison.png", "Main comparison (30 seeds, mean)")


def csv_to_table_png(csv_path: Path, out_path: Path, title: str, round_map: dict[str, int] | None = None) -> None:
    with open(csv_path) as f:
        r = list(csv.reader(f))
    headers, data = r[0], r[1:]
    render_table_png(headers, data, out_path, title)


def ablation_summary_tables() -> None:
    # aggregate rc_sweep
    with open(ABLATIONS / "rc_sweep.csv") as f:
        rows = list(csv.DictReader(f))
    rcs = sorted(set(row["rc_m"] for row in rows), key=float)
    headers = ["Rc (m)", "switch C0 (mean)", "switch C1 slope (mean)", "fade cost (us, mean)"]
    out_rows = []
    for rc in rcs:
        sub = [row for row in rows if row["rc_m"] == rc]
        c0 = statistics.mean(float(row["switch_c0"]) for row in sub)
        c1 = statistics.mean(float(row["switch_c1_slope"]) for row in sub)
        cost = statistics.mean(float(row["fade_cost_us"]) for row in sub)
        out_rows.append([rc, f"{c0:.2e}", f"{c1:.2e}", f"{cost:.1f}"])
    write_csv_md(headers, out_rows, "rc_sweep_summary")
    render_table_png(headers, out_rows, TABLES / "rc_sweep_summary.png", "Hybrid: Rc sensitivity (30 seeds, mean)")

    with open(ABLATIONS / "switch_width_sweep.csv") as f:
        rows = list(csv.DictReader(f))
    widths = sorted(set(row["switch_width_m"] for row in rows), key=float)
    headers = ["switch width (m)", "switch C0 (mean)", "curvature |f''| at Rc (mean)"]
    out_rows = []
    for w in widths:
        sub = [row for row in rows if row["switch_width_m"] == w]
        c0 = statistics.mean(float(row["switch_c0"]) for row in sub)
        c2 = statistics.mean(float(row["curvature_c2"]) for row in sub)
        out_rows.append([w, f"{c0:.2e}", f"{c2:.3f}"])
    write_csv_md(headers, out_rows, "switch_width_summary")
    render_table_png(headers, out_rows, TABLES / "switch_width_summary.png", "Hybrid: switch-width sensitivity (30 seeds, mean)")

    with open(ABLATIONS / "radius_sweep.csv") as f:
        rows = list(csv.DictReader(f))
    Rs = sorted(set(row["R_m"] for row in rows), key=float)
    methods = ["smoothstep", "gaussian_shipped", "gaussian_calibrated"]
    headers = ["R (m)"] + [f"{m} seam C1 (deg)" for m in methods]
    out_rows = []
    for R in Rs:
        row_vals = [R]
        for m in methods:
            sub = [row for row in rows if row["R_m"] == R and row["method"] == m]
            c1 = statistics.mean(float(row["seam_c1_rms_deg"]) for row in sub)
            row_vals.append(f"{c1:.4f}")
        out_rows.append(row_vals)
    write_csv_md(headers, out_rows, "radius_sweep_summary")
    render_table_png(headers, out_rows, TABLES / "radius_sweep_summary.png", "Blend-radius R sensitivity: seam C1 by method (30 seeds, mean)")

    csv_to_table_png(ABLATIONS / "significance_tests.csv", TABLES / "significance_tests.png", "Paired significance tests vs. smoothstep baseline (n=30, Wilcoxon signed-rank)")


if __name__ == "__main__":
    main_comparison_table()
    ablation_summary_tables()
    print(f"Tables (csv/md/png) saved to {TABLES}")
