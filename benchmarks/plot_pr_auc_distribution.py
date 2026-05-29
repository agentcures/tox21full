"""Plot per-assay PR-AUC distributions for the benchmark baseline figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODEL_LABELS = {
    "ecfp_logreg": "ECFP4\nlogistic",
    "ecfp_rf": "ECFP4\nRF",
    "rdkit_hgb": "RDKit\nHGB",
    "chemprop_mpnn": "Chemprop\nD-MPNN",
}

MODEL_ORDER = ["ecfp_logreg", "ecfp_rf", "rdkit_hgb", "chemprop_mpnn"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        default="benchmarks/results/per_assay_metrics_all.csv",
        help="Combined per-assay metrics CSV.",
    )
    parser.add_argument(
        "--out",
        default="benchmarks/results/figures/per_assay_pr_auc_distribution.pdf",
        help="Output figure path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_path = Path(args.metrics)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(metrics_path)
    grouped = [
        metrics.loc[metrics["model"] == model, "pr_auc"].dropna().to_numpy()
        for model in MODEL_ORDER
    ]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(3.55, 2.15))
    box = ax.boxplot(
        grouped,
        patch_artist=True,
        widths=0.55,
        showfliers=False,
        medianprops={"color": "#17201d", "linewidth": 1.25},
        boxprops={"linewidth": 0.8, "color": "#17201d"},
        whiskerprops={"linewidth": 0.8, "color": "#17201d"},
        capprops={"linewidth": 0.8, "color": "#17201d"},
    )

    colors = ["#6b8e23", "#0f766e", "#4f7cac", "#9f6b38"]
    rng = np.random.default_rng(20260506)
    for i, (values, color) in enumerate(zip(grouped, colors, strict=True), start=1):
        box["boxes"][i - 1].set_facecolor(color)
        box["boxes"][i - 1].set_alpha(0.22)
        jitter = rng.normal(0, 0.035, size=len(values))
        ax.scatter(
            np.full_like(values, i, dtype=float) + jitter,
            values,
            s=9,
            color=color,
            alpha=0.55,
            edgecolors="none",
            zorder=3,
        )
        ax.text(
            i,
            min(0.92, values.max() + 0.045),
            f"median {np.median(values):.2f}",
            ha="center",
            va="bottom",
            fontsize=6.5,
            color="#3f4743",
        )

    ax.set_ylabel("Per-assay PR-AUC")
    ax.set_xticks(range(1, len(MODEL_ORDER) + 1))
    ax.set_xticklabels([MODEL_LABELS[model] for model in MODEL_ORDER])
    ax.set_ylim(0, 0.92)
    ax.set_yticks(np.arange(0, 1.0, 0.2))
    ax.grid(axis="y", color="#d9ddd4", linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout(pad=0.4)
    fig.savefig(out_path, bbox_inches="tight")


if __name__ == "__main__":
    main()
