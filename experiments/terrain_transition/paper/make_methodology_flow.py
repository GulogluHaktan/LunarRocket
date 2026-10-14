"""Regenerates methodology_flow.png with axis-aligned arrows that only ever
touch box edges (no edge crosses another box)."""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

STY = {
    "src":  ("#f5efe3", "#8a6a35"),
    "core": ("#fdeede", "#c0561c"),
    "sim":  ("#edf1fa", "#3a5a8c"),
    "out":  ("#e9f3e9", "#2e7d32"),
}
W, H = 3.0, 1.3  # default box size


def box(ax, cx, cy, text, kind, w=W, h=H, bold=False):
    fc, ec = STY[kind]
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0,rounding_size=0.08", fc=fc, ec=ec, lw=1.8))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=10.5,
            fontweight="bold" if bold else "normal", linespacing=1.35)
    return dict(l=cx - w / 2, r=cx + w / 2, t=cy + h / 2, b=cy - h / 2, x=cx, y=cy)


def arrow(ax, p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=16,
                 lw=1.8, color="#3a5a8c", shrinkA=0, shrinkB=0))


def main(out):
    fig, ax = plt.subplots(figsize=(12, 6.6))
    ax.set_xlim(-0.2, 14.9); ax.set_ylim(-4.6, 3.4); ax.axis("off")
    ax.text(7.35, 3.15, "Overall methodology pipeline", ha="center",
            fontsize=15, fontweight="bold")

    dem  = box(ax, 1.5, 1.7, "NASA DEM\n(macro terrain)", "src", h=1.2)
    proc = box(ax, 1.5, -0.2, "Procedural fine\ndetail (regolith,\ncraters, rocks)", "src", h=1.4)
    fade = box(ax, 5.6, 0.75, "Fine-detail fade\nweight fade(r), r < R\n\n"
               "smoothstep | gaussian |\nhybrid  (this study)", "core", w=3.6, h=3.0, bold=True)
    comb = box(ax, 9.7, 0.75, "Combined height\nfield (analytic +\nmesh track)", "sim", w=3.0, h=1.6)
    # analytic eval sits to the right of the combined field on the same row
    ana  = box(ax, 13.3, 0.75, "Analytic / CPU eval.:\ncontinuity (C0/C1),\ncost, VRAM, ablations",
               "out", w=2.7, h=1.6)
    isaac = box(ax, 9.7, -1.85, "Isaac Lab direct-\nworkflow env (PhysX,\nGPU-parallel, terrain pool)",
                "sim", w=3.5, h=1.4)
    sb3  = box(ax, 9.7, -3.8, "SB3 SAC training\n(3 methods x 3 seeds,\n5.73M steps each)",
               "sim", w=3.0, h=1.4)
    tb   = box(ax, 13.3, -3.8, "TensorBoard logs ->\nmulti-seed statistics\n(Mann-Whitney, CI)",
               "out", w=2.7, h=1.4)

    arrow(ax, (dem["r"], 1.7), (fade["l"], 1.7))
    arrow(ax, (proc["r"], -0.2), (fade["l"], -0.2))
    arrow(ax, (fade["r"], comb["y"]), (comb["l"], comb["y"]))
    arrow(ax, (comb["r"], comb["y"]), (ana["l"], ana["y"]))
    arrow(ax, (comb["x"], comb["b"]), (isaac["x"], isaac["t"]))
    arrow(ax, (isaac["x"], isaac["b"]), (sb3["x"], sb3["t"]))
    arrow(ax, (sb3["r"], sb3["y"]), (tb["l"], tb["y"]))
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
