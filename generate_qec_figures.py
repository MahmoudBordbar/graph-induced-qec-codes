"""
======================================================================
generate_qec_figures.py
======================================================================
Generate all 5 figures for the manuscript:

    "Graph-Induced Quantum Error-Correcting Codes:
     Exhaustive Search, Temporal Decoding, and
     Leakage-Controlled Machine Learning"

Output directory: ./figures/
    fig01_c5_construction.png  + .pdf
    fig02_code_landscape.png   + .pdf
    fig03_pauli_weights.png    + .pdf
    fig04_temporal_scaling.png + .pdf
    fig05_ml_results.png       + .pdf

Usage:
    python generate_qec_figures.py            # generate all five
    python generate_qec_figures.py --only 4   # generate only figure 4

Requirements:
    matplotlib, numpy, pandas, networkx, qiskit, pylatexenc
======================================================================
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
from qiskit import QuantumCircuit


# ======================================================================
# GLOBAL STYLE
# ======================================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": False,
})

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

# ----------------------------------------------------------------------
# Colour palette (consistent across all figures)
# ----------------------------------------------------------------------
C_TOPOLOGY    = "#4C72B0"   # blue   – topology-derived
C_STABILIZER  = "#DD8452"   # orange – stabilizer-derived
C_CORRECTABLE = "#55A868"   # green  – correctable
C_FAILURE     = "#C44E52"   # red    – logical failure
C_TOTAL       = "#8172B2"   # purple – total


# ======================================================================
# FIGURE 1 — C5 graph + Qiskit circuit + rank-4 subgroup
# ======================================================================
def make_figure_1():
    """
    Three-panel figure:
      (a) five-cycle graph C5
      (b) graph-state preparation circuit for C5
      (c) rank-4 subgroup selection yielding the [[5,1,3]] code
    """
    fig = plt.figure(figsize=(15.5, 4.8))

    # ---------- Panel (a): C5 graph ----------
    ax_a = fig.add_axes([0.02, 0.05, 0.16, 0.85])
    ax_a.set_title("(a) Five-cycle graph $C_5$", fontsize=12)
    ax_a.axis("off")

    n = 5
    angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi,
                         n, endpoint=False)
    pos = {i: (np.cos(a), np.sin(a)) for i, a in enumerate(angles)}
    edges = [(0, 1), (0, 4), (1, 2), (2, 3), (3, 4)]

    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)

    nx.draw_networkx_edges(G, pos, ax=ax_a, width=1.8, edge_color="black")
    nx.draw_networkx_nodes(G, pos, ax=ax_a, node_size=550,
                           node_color="#F2F2F2", edgecolors="black",
                           linewidths=1.5)
    nx.draw_networkx_labels(G, pos, ax=ax_a, font_size=11,
                            font_weight="bold")
    ax_a.set_xlim(-1.4, 1.4)
    ax_a.set_ylim(-1.4, 1.4)

    # ---------- Panel (b): circuit ----------
    # NOTE: no barrier — the grey band has been removed.
    qc = QuantumCircuit(5, name="")
    qc.h(range(5))
    for (u, v) in edges:
        qc.cz(u, v)

    qc_path = OUTPUT_DIR / "_tmp_qc_c5.png"
    fig_qc = qc.draw(output="mpl", style="iqp", scale=0.8,
                     fold=0, justify="left")
    fig_qc.savefig(qc_path, dpi=250, bbox_inches="tight",
                   facecolor="white")
    plt.close(fig_qc)

    ax_b = fig.add_axes([0.20, 0.05, 0.46, 0.85])
    ax_b.set_title("(b) Graph-state preparation for $C_5$", fontsize=12)
    ax_b.axis("off")
    img = mpimg.imread(str(qc_path))
    ax_b.imshow(img)
    ax_b.set_xticks([])
    ax_b.set_yticks([])

    # ---------- Panel (c): rank-4 subgroup ----------
    ax_c = fig.add_axes([0.68, 0.05, 0.31, 0.85])
    ax_c.set_title("(c) Rank-4 subgroup of $S_{C_5}$", fontsize=12)
    ax_c.axis("off")
    ax_c.set_xlim(0, 10)
    ax_c.set_ylim(0, 10)

    ax_c.text(5.0, 9.4, r"$S_{C_5} = \langle K_0, \dots, K_4 \rangle$",
              ha="center", va="center", fontsize=11)
    ax_c.text(5.0, 8.75, r"(rank 5)", ha="center", va="center",
              fontsize=9, color="#666666")

    ax_c.annotate("", xy=(5.0, 7.95), xytext=(5.0, 8.45),
                  arrowprops=dict(arrowstyle="->", lw=1.2))

    ax_c.text(5.0, 7.45, "select rank-4 subgroup", ha="center",
              va="center", fontsize=9, style="italic")

    stabilizers = [
        r"$S_0 = XZZXI$",
        r"$S_1 = IXZZX$",
        r"$S_2 = XIXZZ$",
        r"$S_3 = ZXIXZ$",
    ]
    for i, s in enumerate(stabilizers):
        ax_c.text(5.0, 6.4 - 0.55 * i, s, ha="center", va="center",
                  fontsize=10.5)

    ax_c.annotate("", xy=(5.0, 2.9), xytext=(5.0, 3.4),
                  arrowprops=dict(arrowstyle="->", lw=1.2))

    ax_c.text(5.0, 2.2, r"$[[5,1,3]]$", ha="center", va="center",
              fontsize=16, fontweight="bold")
    ax_c.text(5.0, 1.4, r"$X_L = IIXYX$", ha="center", va="center",
              fontsize=10.5)
    ax_c.text(5.0, 0.75, r"$Z_L = IIZXZ$", ha="center", va="center",
              fontsize=10.5)

    # ---------- save ----------
    out = OUTPUT_DIR / "fig01_c5_construction"
    fig.savefig(out.with_suffix(".png"))
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)

    try:
        qc_path.unlink()
    except OSError:
        pass

    print(f"✅ Saved {out}.png / .pdf")


# ======================================================================
# FIGURE 2 — Exhaustive five-qubit code landscape
# ======================================================================
def make_figure_2():
    """
    Stacked bar chart of the 22,568 graph/subgroup constructions,
    split by code distance d = 1, 2, 3.

    NOTE for caption: the annotated numbers 12, 60, 60 above the d=3
    bars are counts within each of the three graph-isomorphism classes,
    NOT the total height of the d=3 column (which is 132).
    """
    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    csv_path = SCRIPT_DIR / "ml_full_dataset.csv"
    if not csv_path.exists():
        csv_path = SCRIPT_DIR.parent / "ml_full_dataset.csv"

    if not csv_path.exists():
        warnings.warn(
            "ml_full_dataset.csv not found. Using placeholder data.",
            RuntimeWarning, stacklevel=2)
        edge_vals = [4, 5, 6, 7, 8, 9, 10]
        d1 = [1500, 2800, 2700, 1600, 550, 200, 30]
        d2 = [2300, 4100, 3500, 2000, 800, 100, 20]
        d3 = [0, 12, 60, 60, 0, 0, 0]
    else:
        df = pd.read_csv(csv_path)
        grouped = (df.groupby(["edges", "distance"])
                     .size().unstack(fill_value=0))
        edge_vals = sorted(grouped.index.tolist())
        d1, d2, d3 = [], [], []
        for e in edge_vals:
            row = grouped.loc[e]
            d1.append(int(row.get(1.0, 0)))
            d2.append(int(row.get(2.0, 0)))
            d3.append(int(row.get(3.0, 0)))

    x = np.arange(len(edge_vals))
    width = 0.6

    ax.bar(x, d1, width, label="$d = 1$",
           color=C_TOTAL, alpha=0.85)
    ax.bar(x, d2, width, bottom=d1, label="$d = 2$",
           color=C_STABILIZER, alpha=0.85)
    ax.bar(x, d3, width,
           bottom=np.array(d1) + np.array(d2),
           label="$d = 3$", color=C_FAILURE)

    for i, v3 in enumerate(d3):
        if v3 > 0:
            total_height = d1[i] + d2[i] + v3
            ax.text(i, total_height + 150, f"{v3}",
                    ha="center", va="bottom",
                    fontsize=11, fontweight="bold",
                    color=C_FAILURE,
                    bbox=dict(boxstyle="round,pad=0.25",
                              facecolor="white",
                              edgecolor=C_FAILURE,
                              linewidth=0.7,
                              alpha=0.95))

    ax.set_xlabel("Number of graph edges")
    ax.set_ylabel("Number of graph/subgroup constructions")
    ax.set_title("Exhaustive five-qubit code landscape")
    ax.set_xticks(x)
    ax.set_xticklabels(edge_vals)
    ax.legend(loc="upper left", frameon=True)
    ax.set_ylim(0, 7800)

    total = int(np.sum(d1) + np.sum(d2) + np.sum(d3))
    ax.text(0.98, 0.95, f"Total constructions: {total:,}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, style="italic",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8F8F8",
                      edgecolor="#CCCCCC"))

    ax.text(0.98, 0.86, f"$d \\geq 3$: {int(np.sum(d3))}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, fontweight="bold", color=C_FAILURE,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF5F5",
                      edgecolor=C_FAILURE))

    fig.tight_layout()
    out = OUTPUT_DIR / "fig02_code_landscape"
    fig.savefig(out.with_suffix(".png"))
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"✅ Saved {out}.png / .pdf")


# ======================================================================
# FIGURE 3 — Complete Pauli weight distribution
# ======================================================================
def make_figure_3():
    """
    Grouped bars showing correctable vs failing Pauli operators
    for each weight w = 0, ..., 5 out of 4^5 = 1024 total.
    """
    weights     = [0, 1, 2, 3, 4, 5]
    total       = [1, 15, 90, 270, 405, 243]
    correctable = [1, 15,  0,  60, 135,  45]
    failure     = [0,  0, 90, 210, 270, 198]

    x = np.arange(len(weights))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    ax.bar(x - width/2, correctable, width,
           label="Correctable", color=C_CORRECTABLE,
           edgecolor="black", linewidth=0.7)
    ax.bar(x + width/2, failure, width,
           label="Logical failure", color=C_FAILURE,
           edgecolor="black", linewidth=0.7)

    for i, v in enumerate(correctable):
        if v > 0:
            ax.text(i - width/2, v + 8, f"{v}",
                    ha="center", va="bottom",
                    fontsize=9, fontweight="bold",
                    color="#2F6B3F")

    for i, v in enumerate(failure):
        if v > 0:
            ax.text(i + width/2, v + 8, f"{v}",
                    ha="center", va="bottom",
                    fontsize=9, fontweight="bold",
                    color="#8B2E2E")

    ax.text(2.0, 440, "90 weight-2 failures",
            ha="center", va="center",
            fontsize=9.5, color=C_FAILURE,
            bbox=dict(boxstyle="round,pad=0.35",
                      facecolor="#FFF5F5",
                      edgecolor=C_FAILURE,
                      linewidth=0.9),
            zorder=10)

    ax.text(3.4, 440, "60 degenerate\nweight-3 errors",
            ha="center", va="center",
            fontsize=9.5, color=C_CORRECTABLE,
            bbox=dict(boxstyle="round,pad=0.35",
                      facecolor="#F0F8F2",
                      edgecolor=C_CORRECTABLE,
                      linewidth=0.9),
            zorder=10)

    for i, t in enumerate(total):
        max_val = max(correctable[i], failure[i])
        ax.text(i, max_val + 32, f"total {t}",
                ha="center", va="bottom", fontsize=8,
                color="#333333", style="italic")

    ax.set_xlabel("Pauli operator weight")
    ax.set_ylabel("Number of operators (out of $4^5 = 1024$)")
    ax.set_title("Complete Pauli weight distribution "
                 "for the $[[5,1,3]]$ code")
    ax.set_xticks(x)
    ax.set_xticklabels(weights)
    ax.legend(loc="upper left", frameon=True)
    ax.set_ylim(0, 510)

    fig.tight_layout()
    out = OUTPUT_DIR / "fig03_pauli_weights"
    fig.savefig(out.with_suffix(".png"))
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"✅ Saved {out}.png / .pdf")


# ======================================================================
# FIGURE 4 — Temporal two-fault scaling
# ======================================================================
def make_figure_4():
    """
    Same-round, cross-round and total two-fault pattern counts
    versus the number of syndrome-extraction rounds R.

    Markers are offset horizontally by ±0.045 so that the three
    series never visually overlap (the collision occurs at R = 1).
    Numerical values are unchanged.
    """
    R = np.array([1, 2, 3])
    same  = 90 * R
    cross = 90 * R * (R - 1)
    total = 90 * R ** 2

    R_curve     = np.linspace(1, 3, 100)
    same_curve  = 90 * R_curve
    cross_curve = 90 * R_curve * (R_curve - 1)
    total_curve = 90 * R_curve ** 2

    fig, ax = plt.subplots(figsize=(10.0, 5.8))

    # ---------- analytical curves ----------
    ax.plot(R_curve, same_curve, "--", color=C_TOPOLOGY, lw=1.8,
            label=r"$N_2^{\mathrm{same}} = 90R$")
    ax.plot(R_curve, cross_curve, "--", color=C_STABILIZER, lw=1.8,
            label=r"$N_2^{\mathrm{cross}} = 90R(R{-}1)$")
    ax.plot(R_curve, total_curve, "-", color=C_FAILURE, lw=2.2,
            label=r"$N_2^{\mathrm{total}} = 90R^2$")

    # ---------- exhaustive-enumeration markers ----------
    DX = 0.045
    ax.plot(R - DX, same, "o", color=C_TOPOLOGY, markersize=10,
            markeredgecolor="black", markeredgewidth=1.0, zorder=5)
    ax.plot(R,      cross, "s", color=C_STABILIZER, markersize=10,
            markeredgecolor="black", markeredgewidth=1.0, zorder=5)
    ax.plot(R + DX, total, "^", color=C_FAILURE, markersize=12,
            markeredgecolor="black", markeredgewidth=1.0, zorder=5)

    # ---------- annotations ----------
    ax.annotate("90",  xy=(3.0, 270), xytext=(3.30, 270),
                fontsize=13, fontweight="bold", color=C_TOPOLOGY,
                ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.4",
                          facecolor="white",
                          edgecolor=C_TOPOLOGY, linewidth=1.4),
                arrowprops=dict(arrowstyle="-", color=C_TOPOLOGY,
                                lw=1.0, alpha=0.85))

    ax.annotate("360", xy=(3.0, 540), xytext=(3.30, 540),
                fontsize=13, fontweight="bold", color=C_STABILIZER,
                ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.4",
                          facecolor="white",
                          edgecolor=C_STABILIZER, linewidth=1.4),
                arrowprops=dict(arrowstyle="-", color=C_STABILIZER,
                                lw=1.0, alpha=0.85))

    ax.annotate("810", xy=(3.0, 810), xytext=(3.30, 810),
                fontsize=13, fontweight="bold", color=C_FAILURE,
                ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.4",
                          facecolor="white",
                          edgecolor=C_FAILURE, linewidth=1.4),
                arrowprops=dict(arrowstyle="-", color=C_FAILURE,
                                lw=1.0, alpha=0.85))

    ax.set_xlabel("Number of syndrome-extraction rounds $R$")
    ax.set_ylabel(r"Number of logical-failing two-fault patterns $N_2$")
    ax.set_title("Temporal two-fault scaling of the $[[5,1,3]]$ code")
    ax.set_xticks([1, 2, 3])
    ax.set_xlim(0.55, 4.05)
    ax.set_ylim(-120, 1000)
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, alpha=0.25, linestyle=":")

    ax.text(0.98, 0.04,
            "Points: exhaustive enumeration for $R=1,2,3$\n"
            "Curves: corresponding analytical relations",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, style="italic",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8F8F8",
                      edgecolor="#CCCCCC"))

    fig.tight_layout()
    out = OUTPUT_DIR / "fig04_temporal_scaling"
    fig.savefig(out.with_suffix(".png"))
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"✅ Saved {out}.png / .pdf")


# ======================================================================
# FIGURE 5 — ML performance + feature importance
# ======================================================================
def make_figure_5():
    """
    Two-panel figure:
      (a) graph-disjoint cross-validated R² for four feature sets
      (b) top-10 Random Forest feature importances
    """
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    # ---------- Panel (a): predictive performance ----------
    ax = axes[0]

    models = [
        "Topology\nonly (6)",
        "Stabilizer\nonly (8)",
        "Topology +\nStabilizer (14)",
        "Combined +\nEngineered (20)",
    ]
    r2_means = [0.104, 0.338, 0.504, 0.503]
    r2_stds  = [0.020, 0.026, 0.022, 0.023]

    x = np.arange(len(models))
    colors = [C_TOPOLOGY, C_STABILIZER, C_FAILURE, C_TOTAL]

    ax.bar(x, r2_means, yerr=r2_stds, capsize=5,
           color=colors, edgecolor="black", linewidth=0.7)

    for i, (v, s) in enumerate(zip(r2_means, r2_stds)):
        ax.text(i, v + s + 0.012, f"{v:.3f}\n$\\pm$ {s:.3f}",
                ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel(r"Graph-disjoint cross-validated $R^2$")
    ax.set_title("(a) Predictive performance", fontsize=11)
    ax.set_ylim(0, 0.62)
    ax.axhline(0.11, color="#999999", linestyle="--", lw=1.0,
               label="topology ceiling ($R^2 \\approx 0.11$)")
    ax.legend(loc="upper left", fontsize=8)

    # ---------- Panel (b): feature importances ----------
    ax = axes[1]

    features = [
        "mean_stab_weight",
        "stab_weight_4",
        "min_stab_weight",
        "degree_variance",
        "triangles",
        "var_stab_weight",
        "edges",
        "diameter",
        "stab_weight_5",
        "stab_weight_3",
    ]
    importance = [0.323, 0.138, 0.126, 0.104, 0.065,
                  0.052, 0.048, 0.030, 0.027, 0.026]

    features   = features[::-1]
    importance = importance[::-1]

    stab_features = {"mean_stab_weight", "stab_weight_4", "min_stab_weight",
                     "var_stab_weight", "stab_weight_5", "stab_weight_3"}
    colors_b = [C_STABILIZER if f in stab_features else C_TOPOLOGY
                for f in features]

    y = np.arange(len(features))
    ax.barh(y, importance, color=colors_b,
            edgecolor="black", linewidth=0.7)

    for i, v in enumerate(importance):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(features, fontsize=9)
    ax.set_xlabel("Feature importance (Random Forest)")
    ax.set_title("(b) Top-10 feature importances", fontsize=11)
    ax.set_xlim(0, 0.37)

    legend_elements = [
        mpatches.Patch(facecolor=C_STABILIZER, edgecolor="black",
                       label="Stabilizer-derived"),
        mpatches.Patch(facecolor=C_TOPOLOGY, edgecolor="black",
                       label="Topology-derived"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    fig.tight_layout()
    out = OUTPUT_DIR / "fig05_ml_results"
    fig.savefig(out.with_suffix(".png"))
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"✅ Saved {out}.png / .pdf")


# ======================================================================
# MAIN
# ======================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Generate manuscript figures for the "
                    "graph-induced QEC paper.")
    parser.add_argument("--only", type=int, choices=[1, 2, 3, 4, 5],
                        default=None,
                        help="Generate only one figure by number.")
    args = parser.parse_args()

    generators = {
        1: make_figure_1,
        2: make_figure_2,
        3: make_figure_3,
        4: make_figure_4,
        5: make_figure_5,
    }

    print("=" * 65)
    print("Generating manuscript figures ...")
    print("=" * 65)
    print(f"Script directory: {SCRIPT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 65)

    if args.only is None:
        for k in sorted(generators):
            generators[k]()
    else:
        generators[args.only]()

    print("=" * 65)
    print(f"🎉 Figures saved to: {OUTPUT_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()