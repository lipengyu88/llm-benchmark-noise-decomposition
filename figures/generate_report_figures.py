"""
Generate the three main-text figures for the conference report.

Outputs (saved to the same directory as this script):
  fig1_accuracy_distribution.png  - per-variant accuracy boxplot + strip
  fig2_bt_sample_size.png         - BT top-1 / top-2 stability vs N
  fig3_three_way_variance.png     - three-way variance decomposition stacked bars

All figures are exported at 600 DPI, suitable for camera-ready submission.
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "bold",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent

MODELS = ["llama", "qwen7b", "qwen32b", "qwen72b"]
MODEL_LABELS = {
    "llama":   "LLaMA-3.1-8B",
    "qwen7b":  "Qwen2.5-7B",
    "qwen32b": "Qwen3-32B",
    "qwen72b": "Qwen2.5-72B",
}
COLORS = {
    "llama":   "#4C72B0",
    "qwen7b":  "#C44E52",
    "qwen32b": "#DD8452",
    "qwen72b": "#55A868",
}
DATASETS = {"arc": "ARC-Challenge", "mmlu": "MMLU-Pro"}



def fig1_accuracy_distribution():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    for ax_idx, ds_key in enumerate(["arc", "mmlu"]):
        ax = axes[ax_idx]
        data = json.loads(
            (ROOT / "exp1" / "analysis_exp1" / f"analysis_{ds_key}.json").read_text()
        )

        positions = np.arange(len(MODELS))
        box_data = []
        box_colors = []
        for model in MODELS:
            accs = data[model]["accuracy_stats"]["per_variant"]
            box_data.append(accs)
            box_colors.append(COLORS[model])

        bp = ax.boxplot(
            box_data,
            positions=positions,
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.5},
            whiskerprops={"color": "black", "linewidth": 1.0},
            capprops={"color": "black", "linewidth": 1.0},
            boxprops={"linewidth": 1.0},
        )
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.4)
            patch.set_edgecolor(color)

        rng = np.random.RandomState(42)
        for i, (accs, color) in enumerate(zip(box_data, box_colors)):
            jitter = rng.uniform(-0.14, 0.14, len(accs))
            ax.scatter(
                positions[i] + jitter,
                accs,
                color=color,
                s=10,
                alpha=0.55,
                edgecolors="white",
                linewidths=0.3,
                zorder=3,
            )

        for i, (accs, model) in enumerate(zip(box_data, MODELS)):
            mean_acc = float(np.mean(accs))
            std_acc = float(np.std(accs))
            ax.text(
                positions[i],
                max(accs) + 0.018,
                f"$\\mu$={mean_acc:.3f}\n$\\sigma$={std_acc:.3f}",
                ha="center", va="bottom", fontsize=8,
                color=COLORS[model], fontweight="bold",
            )

        ax.set_xticks(positions)
        ax.set_xticklabels(
            [MODEL_LABELS[m] for m in MODELS],
            fontsize=9,
            rotation=12,
            ha="right",
        )
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_title(DATASETS[ds_key], fontsize=12)
        ax.set_ylim(top=ax.get_ylim()[1] + 0.07)
        ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle(
        "Accuracy distribution across 100 prompt variants",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig1_accuracy_distribution.png")
    plt.close(fig)
    print("  saved fig1_accuracy_distribution.png")



def fig2_bt_sample_size():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    for ax_idx, ds_key in enumerate(["arc", "mmlu"]):
        ax = axes[ax_idx]
        data = json.loads(
            (ROOT / "exp4" / f"bt_results_{ds_key}.json").read_text()
        )
        ss = data["sample_size_simulation"]
        curves = ss["sample_curves"]
        ns = [c["n_comparisons"] for c in curves]
        top1 = [c["pr_correct_top1"] for c in curves]
        top2 = [c["pr_correct_top2_set"] for c in curves]
        n_top1 = ss["n_needed_top1_95"]
        n_top2 = ss["n_needed_top2_95"]

        ax.plot(
            ns, top2, marker="s", linestyle="--",
            color="#4C72B0", linewidth=2.0, markersize=8,
            label="Top-2 set",
        )
        ax.plot(
            ns, top1, marker="o", linestyle="-",
            color="#C44E52", linewidth=2.0, markersize=8,
            label="Top-1 model",
        )

        ax.axhline(
            0.95, color="gray", linestyle=":", linewidth=1.2, alpha=0.7,
        )
        ax.text(
            ns[-1] * 0.95, 0.955,
            "95% stability",
            color="gray", fontsize=8, ha="right", va="bottom",
        )

        if n_top1 is not None:
            ax.axvline(
                n_top1, color="#C44E52", linestyle=":", linewidth=1.2, alpha=0.6,
            )
            ax.annotate(
                f"top-1: $N$ = {n_top1}",
                xy=(n_top1, 0.5),
                xytext=(n_top1 * 1.4, 0.42),
                fontsize=9, color="#C44E52", fontweight="bold",
                arrowprops=dict(
                    arrowstyle="->", color="#C44E52", lw=1.0, alpha=0.7,
                ),
            )
        if n_top2 is not None:
            ax.axvline(
                n_top2, color="#4C72B0", linestyle=":", linewidth=1.2, alpha=0.6,
            )
            ax.annotate(
                f"top-2: $N$ = {n_top2}",
                xy=(n_top2, 0.95),
                xytext=(n_top2 * 1.3, 0.78),
                fontsize=9, color="#4C72B0", fontweight="bold",
                arrowprops=dict(
                    arrowstyle="->", color="#4C72B0", lw=1.0, alpha=0.7,
                ),
            )

        ax.set_xscale("log")
        ax.set_xlabel("Number of pairwise comparisons $N$", fontsize=11)
        ax.set_ylabel("Pr(recovered set = full-data set)", fontsize=11)
        ax.set_title(DATASETS[ds_key], fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower right", framealpha=0.95, fontsize=9)
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        "Bradley Terry ranking stability vs. sample size",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig2_bt_sample_size.png")
    plt.close(fig)
    print("  saved fig2_bt_sample_size.png")



def fig3_three_way_variance():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    var_colors = {
        "prompt":   "#C44E52",
        "sampling": "#DD8452",
        "testset":  "#4C72B0",
    }
    var_labels = {
        "prompt":   "Var(prompt)",
        "sampling": "Var(sampling)",
        "testset":  "Var(test-set)",
    }

    for ax_idx, ds_key in enumerate(["arc", "mmlu"]):
        ax = axes[ax_idx]
        data = json.loads(
            (ROOT / "exp3" / "analysis_exp3" / f"analysis_{ds_key}_shared150.json").read_text()
        )
        vd_baseline = data["variance_decomposition_3way"]["baseline"]

        e2_to_e1 = {
            "llama-3.1-8b": "llama",
            "qwen2.5-7b":   "qwen7b",
            "qwen3-32b":    "qwen32b",
            "qwen2.5-72b":  "qwen72b",
        }
        order = {e2_to_e1[v["model"]]: v for v in vd_baseline}
        rows = [order[m] for m in MODELS]

        prompt = np.array([r["pct_prompt"]   for r in rows])
        samp   = np.array([r["pct_sampling"] for r in rows])
        test   = np.array([r["pct_testset"]  for r in rows])

        positions = np.arange(len(MODELS))
        width = 0.6

        b1 = ax.bar(
            positions, prompt, width=width,
            color=var_colors["prompt"], alpha=0.85, edgecolor="white", linewidth=1.0,
            label=var_labels["prompt"],
        )
        b2 = ax.bar(
            positions, samp, width=width, bottom=prompt,
            color=var_colors["sampling"], alpha=0.85, edgecolor="white", linewidth=1.0,
            label=var_labels["sampling"],
        )
        b3 = ax.bar(
            positions, test, width=width, bottom=prompt + samp,
            color=var_colors["testset"], alpha=0.85, edgecolor="white", linewidth=1.0,
            label=var_labels["testset"],
        )

        for i, p in enumerate(prompt):
            ax.text(
                positions[i], 102, f"{p:.0f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
                color=var_colors["prompt"],
            )

        ax.set_xticks(positions)
        ax.set_xticklabels(
            [MODEL_LABELS[m] for m in MODELS],
            fontsize=9, rotation=12, ha="right",
        )
        ax.set_ylabel("% of total variance", fontsize=11)
        ax.set_title(DATASETS[ds_key], fontsize=12)
        ax.set_ylim(0, 115)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(True, axis="y", alpha=0.25)

        if ax_idx == 0:
            ax.legend(
                loc="upper right",
                framealpha=0.95, fontsize=9,
                ncol=1, bbox_to_anchor=(1.0, 0.95),
            )

    fig.suptitle(
        "Three-way variance decomposition (top label = prompt share)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig3_three_way_variance.png")
    plt.close(fig)
    print("  saved fig3_three_way_variance.png")



if __name__ == "__main__":
    print("Generating report figures...")
    fig1_accuracy_distribution()
    fig2_bt_sample_size()
    fig3_three_way_variance()
    n = len(list(OUT.glob("*.png")))
    print(f"\nDone. {n} figures saved to {OUT}/")
