from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).parent
FIGDIR = ROOT / "figures_bt"
FIGDIR.mkdir(exist_ok=True)

MODELS = ["llama-3.1-8b", "qwen2.5-7b", "qwen3-32b", "qwen2.5-72b"]
M_DISP = {"llama-3.1-8b": "LLaMA-3.1-8B", "qwen2.5-7b": "Qwen2.5-7B",
          "qwen3-32b": "Qwen3-32B", "qwen2.5-72b": "Qwen2.5-72B"}
MC = {"llama-3.1-8b": "#E8565C", "qwen2.5-7b": "#F5A623",
      "qwen3-32b": "#4A90D9", "qwen2.5-72b": "#45B87F"}
B_DISP = {"arc": "ARC-Challenge", "mmlu": "MMLU-Pro"}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 250,
    "font.family": "sans-serif", "font.size": 13,
    "axes.titlesize": 15, "axes.titleweight": "bold",
    "axes.facecolor": "#FAFAFA", "axes.edgecolor": "#CCC",
    "axes.grid": True, "grid.alpha": .3, "figure.facecolor": "white",
})
sns.set_style("whitegrid")


def save(fig, name):
    fig.savefig(FIGDIR / name, bbox_inches="tight", pad_inches=.25)
    plt.close(fig)
    print(f"  saved {name}")


def load_results(ds: str) -> dict:
    return json.loads((ROOT / f"bt_results_{ds}.json").read_text())



def fig1_bt_ratings():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, ds in zip(axes, ["arc", "mmlu"]):
        data = load_results(ds)
        log_r = np.array(data["log_ratings"])
        lo = np.array(data["bootstrap_ci_low"])
        hi = np.array(data["bootstrap_ci_high"])

        order = np.argsort(-log_r)
        y = np.arange(len(order))
        for i, idx in enumerate(order):
            m = MODELS[idx]
            xerr_low = log_r[idx] - lo[idx]
            xerr_high = hi[idx] - log_r[idx]
            ax.errorbar(log_r[idx], i,
                        xerr=[[xerr_low], [xerr_high]],
                        fmt="D", color=MC[m], markersize=14, linewidth=3,
                        capsize=8, capthick=2.5, zorder=4,
                        markeredgecolor="white", markeredgewidth=1.5)
            ax.text(hi[idx] + 0.03, i, M_DISP[m],
                    va="center", fontsize=11, fontweight="bold")

        ax.set_yticks(y)
        ax.set_yticklabels([f"rank {i+1}" for i in range(len(order))])
        ax.set_xlabel("BT log-strength  (higher = stronger)")
        ax.set_title(B_DISP[ds] + f"  (n={data['n_conditions_total']} conditions)")
        ax.invert_yaxis()
        ax.set_xlim(ax.get_xlim()[0] - 0.1, ax.get_xlim()[1] + 0.6)

    fig.suptitle("Bradley-Terry Ratings with 95% Bootstrap CIs",
                 fontsize=17, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "fig1_bt_ratings.png")



def fig2_rank_posterior():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, ds in zip(axes, ["arc", "mmlu"]):
        data = load_results(ds)
        post = np.array(data["rank_posterior"])
        ranking = data["ranking"]

        order_idx = [MODELS.index(m) for m in ranking]
        post_ordered = post[order_idx]

        im = ax.imshow(post_ordered, cmap="Blues", aspect="auto",
                       vmin=0, vmax=1)
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels([f"Rank {k+1}" for k in range(len(MODELS))])
        ax.set_yticks(range(len(MODELS)))
        ax.set_yticklabels([M_DISP[m] for m in ranking], fontsize=10)

        for i in range(len(MODELS)):
            for k in range(len(MODELS)):
                val = post_ordered[i, k]
                color = "white" if val > 0.5 else "black"
                if val > 0.001:
                    ax.text(k, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=11, fontweight="bold", color=color)
                else:
                    ax.text(k, i, "0", ha="center", va="center",
                            fontsize=10, color="#888")

        ax.set_title(B_DISP[ds])

    fig.colorbar(im, ax=axes, label="Posterior probability", shrink=0.8)
    fig.suptitle("Rank Posterior (Pr(model holds rank k)) from 10,000 Bootstraps",
                 fontsize=17, fontweight="bold", y=1.01)
    save(fig, "fig2_rank_posterior.png")



def fig3_sample_size_curves():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, ds in zip(axes, ["arc", "mmlu"]):
        data = load_results(ds)
        sim = data["sample_size_simulation"]
        curves = sim["sample_curves"]

        ns = [c["n_comparisons"] for c in curves]
        p_top1 = [c["pr_correct_top1"] for c in curves]
        p_top2 = [c["pr_correct_top2_set"] for c in curves]

        ax.plot(ns, p_top1, "o-", color="#4A90D9", linewidth=2.5, markersize=10,
                label="Pr(correct top-1)", zorder=4)
        ax.plot(ns, p_top2, "s-", color="#E8565C", linewidth=2.5, markersize=10,
                label="Pr(correct top-2 set)", zorder=4)

        ax.axhline(0.95, color="#888", linestyle="--", linewidth=1.5,
                   label="95% threshold")

        n1 = sim.get("n_needed_top1_95")
        n2 = sim.get("n_needed_top2_95")
        txt = f"N to reach 95%:\n  top-1: {n1 if n1 else 'not reached'}\n  top-2: {n2 if n2 else 'not reached'}"
        ax.text(0.98, 0.05, txt, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="#888", alpha=0.95))

        ax.set_xscale("log")
        ax.set_xlabel("Number of pairwise comparisons (per dataset)")
        ax.set_ylabel("Pr(reconstructs true ranking)")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{B_DISP[ds]}  (total={sim['n_total_conditions']} conditions)")
        ax.legend(fontsize=10, loc="lower right")

    fig.suptitle("Sample-Size Requirements for Stable Top-k Rankings",
                 fontsize=17, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "fig3_sample_size_curves.png")



def main():
    print(f"Generating BT figures to {FIGDIR}/")
    fig1_bt_ratings()
    fig2_rank_posterior()
    fig3_sample_size_curves()
    print("Done.")


if __name__ == "__main__":
    main()
