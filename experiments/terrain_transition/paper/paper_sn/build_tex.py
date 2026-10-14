#!/usr/bin/env python3
"""Convert the plain-text English sections + CSV tables into class-independent
LaTeX under sections/ and tables/.  Content is taken verbatim from
../sections_en and ../tables_en; only markup is added.

Run:  python3 build_tex.py
"""
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT.parent / "sections_en"
TBL_SRC = ROOT.parent / "tables_en"
SEC_OUT = ROOT / "sections"
TBL_OUT = ROOT / "tables"

# --- code-ish identifiers that must survive as \texttt (they carry _ and -) ---
VERBATIM = [
    "ISAACLAB_TERRAIN_BLEND_METHOD", "ISAACLAB_CHECKPOINT",
    "LunarRocket-Lander-Direct-v0",
    "terrain_detail_radius_m", "terrain_hybrid_contact_radius_m",
    "terrain_hybrid_switch_width_m", "cfg.terrain_gaussian_k",
    "gaussian_shipped", "gaussian_calibrated",
    "replicate_physics", "clone_in_fabric", "env_spacing",
    "learning_starts", "train_freq", "gradient_steps",
    "normalize_input", "normalize_value", "clip_obs", "target_entropy",
    "up_z", "smoothstep", "gaussian", "hybrid",
]
# longest first so substrings do not shadow longer identifiers
VERBATIM.sort(key=len, reverse=True)

# --- citation map: surface form -> bibtex key ---------------------------------
CITE = {
    "Barker et al., 2016": "barker2016",
    "de Boer, 2000": "deboer2000",
    "Duchaineau et al., 1997": "duchaineau1997",
    "Ebert et al., 2002": "ebert2002",
    "Haarnoja et al., 2018": "haarnoja2018",
    "Hardy, 1971": "hardy1971",
    "Henderson et al., 2018": "henderson2018",
    "Ishida et al., 2025": "ishida2025",
    "Johnson & Montgomery, 2008": "johnson2008",
    "Losasso & Hoppe, 2004": "losasso2004",
    "Mittal et al., 2023": "mittal2023",
    "NASA JPL, n.d.": "nasajpl",
    "NVIDIA, 2025": "nvidia2025",
    "Ohtake et al., 2003": "ohtake2003",
    "Raffin et al., 2021": "raffin2021",
    "Rudin et al., 2022": "rudin2022",
    "Tobin et al., 2017": "tobin2017",
    "Wendland, 1995": "wendland1995",
    "Pajarola & Gobbetti, 2007": "pajarola2007",
    "Zhang et al., 2019": "zhang2019",
    "Jain et al., 2024": "jain2024",
    "Ohtake, Belyaev, Alexa et al., 2003": "ohtake2003mpu",
    "Agarwal et al., 2021": "agarwal2021",
    "Colas et al., 2018": "colas2018",
    "Colas et al., 2019": "colas2019",
    "Muratore et al., 2022": "muratore2022",
    "Collins et al., 2021": "collins2021",
    "Nakashima et al., 2010": "nakashima2010",
    "Richard et al., 2024": "richard2024",
    "Scorsoglio et al., 2021": "scorsoglio2021",
}
# textual (\citet) forms: "Name (Year)" possibly with trailing ", Note"
CITET = {
    "Hardy (1971)": "hardy1971",
    "Wendland (1995)": "wendland1995",
    "Ohtake et al. (2003)": "ohtake2003",
    "Rudin et al. (2022)": "rudin2022",
    "Tobin et al. (2017)": "tobin2017",
    "Henderson et al. (2018)": "henderson2018",
    "Haarnoja et al. (2018)": "haarnoja2018",
    "Raffin et al. (2021)": "raffin2021",
    "Mittal et al. (2023, Orbit)": "mittal2023",
    "NVIDIA (2025, Isaac Lab)": "nvidia2025",
    "Mittal et al. (2023)": "mittal2023",
    "NVIDIA (2025)": "nvidia2025",
    "Barker et al. (2016)": "barker2016",
    "Ishida et al. (2025)": "ishida2025",
    "Pajarola & Gobbetti (2007)": "pajarola2007",
    "Zhang et al. (2019)": "zhang2019",
    "Jain et al. (2024)": "jain2024",
    "Agarwal et al. (2021)": "agarwal2021",
    "Colas et al. (2018)": "colas2018",
    "Colas et al. (2019)": "colas2019",
    "Muratore et al. (2022)": "muratore2022",
    "Collins et al. (2021)": "collins2021",
    "Nakashima et al. (2010)": "nakashima2010",
    "Richard et al. (2024)": "richard2024",
    "Scorsoglio et al. (2021)": "scorsoglio2021",
}

SENTINEL = "\x00V%d\x00"


def protect(text, store):
    """Swap verbatim identifiers and citations out for sentinels."""
    # citet first (they contain parens that the citep pass would mangle)
    for surface, key in sorted(CITET.items(), key=lambda kv: -len(kv[0])):
        if surface in text:
            store.append(r"\citet{%s}" % key)
            text = text.replace(surface, SENTINEL % (len(store) - 1))

    # parenthesised groups: (A, 1999; B et al., 2000)
    def paren(m):
        inner = m.group(1)
        parts = [p.strip() for p in inner.split(";")]
        rendered, keys, all_pure = [], [], True
        for part in parts:
            if part in CITE:                       # pure citation
                keys.append(CITE[part])
                rendered.append(r"\citealp{%s}" % CITE[part])
                continue
            # citation embedded in a label, e.g. "geometry clipmaps: Losasso & Hoppe, 2004"
            hit = None
            for surface, key in sorted(CITE.items(), key=lambda kv: -len(kv[0])):
                if part.endswith(surface):
                    hit = (surface, key)
                    break
            all_pure = False
            if hit:
                surface, key = hit
                keys.append(key)
                rendered.append(part[: -len(surface)] + r"\citealp{%s}" % key)
            else:
                rendered.append(part)
        if not keys:
            return m.group(0)
        if all_pure:
            cmd = r"\citep{%s}" % ",".join(keys)
        else:
            cmd = "(" + "; ".join(rendered) + ")"
        store.append(cmd)
        return SENTINEL % (len(store) - 1)

    text = re.sub(r"\(([^()]*?(?:19|20)\d\d[^()]*?|[^()]*?n\.d\.[^()]*?)\)", paren, text)

    for v in VERBATIM:
        if v in text:
            store.append(r"\texttt{%s}" % v.replace("_", r"\_"))
            text = text.replace(v, SENTINEL % (len(store) - 1))
    return text


SPECIALS = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
}


def escape(text):
    return "".join(SPECIALS.get(c, c) for c in text)


def restore(text, store):
    for i, v in enumerate(store):
        text = text.replace(SENTINEL % i, v)
    return text


# --- cross-references: the .txt sources carry hard-coded numbers (written for
# the .docx build).  Map them to \ref so the numbers can never drift from the
# real float/section counters.  Figure numbers are the .docx numbering (no
# geometry schematic); the TikZ schematic fig:geometry is referenced by label.
SEC_REF = {
    "2": "sec:related", "3": "sec:method", "3.1": "sec:env", "3.2": "sec:setup",
    "3.3": "sec:blend", "3.4": "sec:analytic", "3.5": "sec:rl",
    "3.6": "sec:rlprotocol", "4": "sec:results", "4.1": "sec:continuity",
    "4.2": "sec:ablation", "4.3": "sec:rlresults", "5": "sec:discussion",
    "6": "sec:conclusion",
}
FIG_REF = {
    "1": "fig:flow", "2": "fig:scene", "3": "fig:closeup", "4": "fig:radial",
    "5": "fig:contcmp", "6": "fig:costcmp", "7": "fig:rcsweep",
    "8": "fig:wsweep", "9": "fig:rsweep", "10": "fig:ressweep",
    "11": "fig:trainmetrics", "12": "fig:traindiag", "13": "fig:boxplot",
}
TAB_REF = {
    "1": "tab:simconfig", "2": "tab:continuity", "3": "tab:fadecost",
    "4": "tab:vram", "5": "tab:rlsummary", "6": "tab:rlstats",
}


def xref(text, store):
    def sub(table, word):
        def f(m):
            key = m.group(1)
            if key not in table:
                raise SystemExit(f"unmapped {word} {key}")
            store.append(r"%s~\ref{%s}" % (word, table[key]))
            return SENTINEL % (len(store) - 1)
        return f
    text = re.sub(r"\bSection (\d+(?:\.\d+)?)\b", sub(SEC_REF, "Section"), text)
    text = re.sub(r"\bFigure (\d+)\b", sub(FIG_REF, "Figure"), text)
    text = re.sub(r"\bTable (\d+)\b", sub(TAB_REF, "Table"), text)
    return text


def convert_inline(text):
    store = []
    text = protect(text, store)
    text = xref(text, store)
    text = escape(text)
    text = restore(text, store)
    return text.replace(" -- ", "~--- ")


def convert(raw):
    out = []
    for block in raw.strip().split("\n\n"):
        lines = [l.strip() for l in block.strip().splitlines() if l.strip()]
        if not lines:
            continue
        if all(l.startswith("- ") for l in lines):      # bullet list
            items = "\n".join(r"  \item " + convert_inline(l[2:]) for l in lines)
            out.append("\\begin{itemize}\n" + items + "\n\\end{itemize}")
            continue
        out.append(convert_inline(" ".join(lines)))
    return "\n\n".join(out)


def csv_to_tabular(path, caption, label, colspec=None, small=True, headers=None):
    """headers: CSV basliklarini baskida kisaltmak icin liste (veri degismez)."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if headers is not None:
        assert len(headers) == len(rows[0]), f"{path}: baslik sayisi uyusmuyor"
        rows[0] = headers
    ncol = len(rows[0])
    if colspec is None:
        colspec = "l" * ncol
    lines = [r"\begin{table}[htb]", r"\centering",
             r"\caption{%s}" % caption, r"\label{%s}" % label]
    if small:
        lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{%dpt}" % (3 if ncol >= 7 else 4))
    lines.append(r"\begin{tabular}{@{}%s@{}}" % colspec)
    lines.append(r"\toprule")
    for i, row in enumerate(rows):
        store = []
        cells = []
        for c in row:
            c = protect(c, store)
            c = escape(c)
            c = restore(c, store)
            cells.append(c)
        lines.append(" & ".join(cells) + r" \\")
        if i == 0:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def main():
    SEC_OUT.mkdir(exist_ok=True)
    TBL_OUT.mkdir(exist_ok=True)

    for src in sorted(SRC.glob("*.txt")):
        if src.stem in {"08_references", "01_abstract_tr", "04_2_terrain_blending"}:
            continue
        tex = convert(src.read_text(encoding="utf-8"))
        (SEC_OUT / (src.stem + ".tex")).write_text(tex + "\n", encoding="utf-8")

    tables = [
        ("table0_sim_config.csv",
         "Simulation, vehicle, RL training, and terrain-blending parameters shared by "
         "the analytic protocol and the real Isaac Lab/Isaac Sim training runs. Only "
         "the last group (blending parameters) varies from method to method.",
         "tab:simconfig", "llp{0.40\\linewidth}", None),
        ("table1_continuity.csv",
         "Boundary continuity ratios relative to smoothstep, aggregated over thirty "
         "seeds, isolating only the fine-detail component. C0 is the RMS boundary "
         "height error and C1 the RMS slope error; both columns are ratios to the "
         "smoothstep reference. p is the Wilcoxon signed-rank p value.",
         "tab:continuity", "lrrrr",
         ["Method", "C0 ratio", "C1 ratio", "Wilcoxon p", "n"]),
        ("table2_fadecost.csv",
         "Per-query computational cost of the fade formula.",
         "tab:fadecost", "lr", ["Method", "Cost (us/query)"]),
        ("table3_vram.csv",
         "Estimated VRAM usage by mesh accounting scheme.",
         "tab:vram", "lr", None),
        ("table4_rl_summary.csv",
         "Final ten-percent window summary in the real Isaac Sim training, three "
         "seeds per method. All rate columns are mean $\\pm$ standard deviation over "
         "the three seeds; \\emph{Curric.} is the curriculum difficulty reached.",
         "tab:rlsummary", "lrrrrrr",
         ["Method", "n", "Wall time (s)", "Success", "Timeout",
          "Harsh", "Curric."]),
        ("table5_rl_stats.csv",
         "Pairwise comparisons over the final ten-percent window (Mann--Whitney U, "
         "descriptive; see the caveat in Section~\\ref{sec:rlprotocol}). All "
         "comparisons are 3 seeds versus 3 seeds.",
         "tab:rlstats", "llrrlr",
         ["Metric", "Comparison", "n", "Mean diff", "95\\% CI", "p"]),
    ]
    for fname, cap, lab, spec, hdr in tables:
        tex = csv_to_tabular(TBL_SRC / fname, cap, lab, spec, headers=hdr)
        (TBL_OUT / fname.replace(".csv", ".tex")).write_text(tex, encoding="utf-8")

    # leftover-citation audit
    leftovers = []
    for f in sorted(SEC_OUT.glob("*.tex")):
        body = f.read_text(encoding="utf-8")
        body_wo = re.sub(r"\\cite[tp]?\{[^}]*\}", "", body)
        for m in re.finditer(r"[A-Z][A-Za-z]+(?: et al\.)?,? \(?(?:19|20)\d\d\)?", body_wo):
            leftovers.append((f.name, m.group(0)))
    print("sections:", len(list(SEC_OUT.glob('*.tex'))), " tables:", len(list(TBL_OUT.glob('*.tex'))))
    if leftovers:
        print("DONUSMEYEN ATIF ADAYLARI:")
        for n, t in leftovers:
            print("  ", n, "->", t)
    else:
        print("donusmeyen atif yok")


if __name__ == "__main__":
    main()
