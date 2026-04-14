"""
Experiment III: High-Noise Item Analysis — Publication-quality Visualizations

Reads:   analysis_exp3/analysis_*.json, noise_data/noise_*.json
Writes:  figures_exp3/*.png

Figures:
  1. Noise score distribution (histogram + cumulative)
  2. Accuracy stability improvement across removal thresholds (Exp I)
  3. Accuracy stability improvement across removal thresholds (Exp II)
  4. Variance decomposition shift under noise removal
  5. Ranking reversal reduction under noise removal
  6. Three-way variance decomposition (prompt vs sampling vs test-set)
  7. Noise correlation heatmap across models
  8. Noise vs difficulty scatter
  9. Category-level noise analysis (MMLU-Pro)
  10. Summary dashboard
"""
from __future__ import annotations
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from itertools import combinations

ROOT = Path(__file__).parent
NOISE_DIR = ROOT / "noise_data"
ANALYSIS_DIR = ROOT / "analysis_exp3"
FIGDIR = ROOT / "figures_exp3"
FIGDIR.mkdir(exist_ok=True)

MODELS_E2 = ["llama-3.1-8b", "qwen2.5-7b", "qwen3-32b", "qwen2.5-72b"]
MODELS_E1 = ["llama", "qwen7b", "qwen32b", "qwen72b"]
MODEL_MAP = {
    "llama": "llama-3.1-8b", "qwen7b": "qwen2.5-7b",
    "qwen32b": "qwen3-32b", "qwen72b": "qwen2.5-72b",
}
M_DISP = {
    "llama-3.1-8b": "LLaMA-3.1-8B", "qwen2.5-7b": "Qwen2.5-7B",
    "qwen3-32b": "Qwen3-32B", "qwen2.5-72b": "Qwen2.5-72B",
}
MC = {
    "llama-3.1-8b": "#E8565C", "qwen2.5-7b": "#F5A623",
    "qwen3-32b": "#4A90D9", "qwen2.5-72b": "#45B87F",
}
MC_E1 = {
    "llama": "#E8565C", "qwen7b": "#F5A623",
    "qwen32b": "#4A90D9", "qwen72b": "#45B87F",
}
M_DISP_E1 = {
    "llama": "LLaMA-3.1-8B", "qwen7b": "Qwen2.5-7B",
    "qwen32b": "Qwen3-32B", "qwen72b": "Qwen2.5-72B",
}
B_DISP = {"arc": "ARC-Challenge", "mmlu": "MMLU-Pro"}
DATASETS = ["arc", "mmlu"]
THRESHOLDS = ["baseline", "remove_10pct", "remove_20pct", "remove_30pct"]
THRESHOLD_LABELS = {"baseline": "0%", "remove_10pct": "10%", "remove_20pct": "20%", "remove_30pct": "30%"}
THRESHOLD_PCTS = [0, 10, 20, 30]

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 250,
    "font.family": "sans-serif", "font.size": 13,
    "axes.titlesize": 15, "axes.titleweight": "bold",
    "axes.facecolor": "#FAFAFA", "axes.edgecolor": "#CCC",
    "axes.grid": True, "grid.alpha": .3, "figure.facecolor": "white",
})
sns.set_style("whitegrid")


NOISE_TAG = ""


def save(fig, name):
    fig.savefig(FIGDIR / name, bbox_inches="tight", pad_inches=.25)
    plt.close(fig)
    print(f"  saved {name}")


def load_noise(ds):
    return json.loads((NOISE_DIR / f"noise_{ds}{NOISE_TAG}.json").read_text())


def load_analysis(ds):
    return json.loads((ANALYSIS_DIR / f"analysis_{ds}{NOISE_TAG}.json").read_text())



def fig1_noise_distribution():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, ds in zip(axes, DATASETS):
        noise = load_noise(ds)
        scores = [nd["noise_score"] for nd in noise["noise_scores"].values()]

        bins = np.linspace(0, 1, 21)
        ax.hist(scores, bins=bins, color="#4A90D9", alpha=0.7, edgecolor="white",
                linewidth=1.2, zorder=3)

        sorted_scores = sorted(scores, reverse=True)
        n = len(sorted_scores)
        for pct, color, ls in [(10, "#E74C3C", "--"), (20, "#F39C12", "-."), (30, "#8E44AD", ":")]:
            idx = int(n * pct / 100)
            if idx < n:
                cutoff = sorted_scores[idx]
                ax.axvline(cutoff, color=color, linestyle=ls, linewidth=2,
                           label=f"Top {pct}% cutoff ({cutoff:.2f})", zorder=4)

        ax.set_xlabel("Noise Score")
        ax.set_ylabel("Number of Questions")
        ax.set_title(B_DISP[ds])
        ax.legend(fontsize=9, loc="upper left")

        mean_ns = np.mean(scores)
        median_ns = np.median(scores)
        ax.text(0.98, 0.95, f"N={n}\nMean={mean_ns:.3f}\nMedian={median_ns:.3f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    fig.suptitle("Noise Score Distribution (Combined Exp I + Exp II)",
                 fontsize=17, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "fig1_noise_distribution.png")



def fig2_exp1_stability_improvement():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = [
        ("std", "Accuracy Std"),
        ("range", "Accuracy Range"),
    ]

    for row, (metric, ylabel) in enumerate(metrics):
        for col, ds in enumerate(DATASETS):
            ax = axes[row][col]
            analysis = load_analysis(ds)
            tr = analysis["threshold_results"]

            for m_e1 in MODELS_E1:
                vals = []
                for tk in THRESHOLDS:
                    if tk in tr and m_e1 in tr[tk].get("exp1", {}):
                        vals.append(tr[tk]["exp1"][m_e1]["accuracy"][metric])
                    else:
                        vals.append(np.nan)

                ax.plot(THRESHOLD_PCTS, vals, "o-", color=MC_E1[m_e1],
                        label=M_DISP_E1[m_e1], linewidth=2.5, markersize=9, zorder=4)

            ax.set_xlabel("% Noisiest Items Removed")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{B_DISP[ds]} — {ylabel}")
            ax.set_xticks(THRESHOLD_PCTS)
            ax.set_xticklabels(["0%", "10%", "20%", "30%"])
            if row == 0 and col == 0:
                ax.legend(fontsize=9, loc="best")

    fig.suptitle("Exp I: Accuracy Stability Under Noise Removal",
                 fontsize=17, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "fig2_exp1_stability.png")



def fig3_flip_rate_reduction():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, ds in zip(axes, DATASETS):
        analysis = load_analysis(ds)
        tr = analysis["threshold_results"]

        for m_e1 in MODELS_E1:
            vals = []
            for tk in THRESHOLDS:
                if tk in tr and m_e1 in tr[tk].get("exp1", {}):
                    vals.append(tr[tk]["exp1"][m_e1]["flip_rate"] * 100)
                else:
                    vals.append(np.nan)

            ax.plot(THRESHOLD_PCTS, vals, "o-", color=MC_E1[m_e1],
                    label=M_DISP_E1[m_e1], linewidth=2.5, markersize=9, zorder=4)

        ax.set_xlabel("% Noisiest Items Removed")
        ax.set_ylabel("Item Flip Rate (%)")
        ax.set_title(B_DISP[ds])
        ax.set_xticks(THRESHOLD_PCTS)
        ax.set_xticklabels(["0%", "10%", "20%", "30%"])
        ax.legend(fontsize=9, loc="best")

    fig.suptitle("Item Flip Rate Reduction Under Noise Removal",
                 fontsize=17, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "fig3_flip_rate_reduction.png")



def fig4_variance_ratio():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, ds in zip(axes, DATASETS):
        analysis = load_analysis(ds)
        tr = analysis["threshold_results"]

        for m_e1 in MODELS_E1:
            vals = []
            for tk in THRESHOLDS:
                if tk in tr and m_e1 in tr[tk].get("exp1", {}):
                    vd = tr[tk]["exp1"][m_e1]["variance_decomposition"]
                    vals.append(vd["ratio"])
                else:
                    vals.append(np.nan)

            ax.plot(THRESHOLD_PCTS, vals, "o-", color=MC_E1[m_e1],
                    label=M_DISP_E1[m_e1], linewidth=2.5, markersize=9, zorder=4)

        ax.axhline(1.0, color="#888", ls="--", lw=1.2, label="Var_prompt = Var_sampling")
        ax.set_xlabel("% Noisiest Items Removed")
        ax.set_ylabel("Var(prompt) / Var(sampling)")
        ax.set_title(B_DISP[ds])
        ax.set_xticks(THRESHOLD_PCTS)
        ax.set_xticklabels(["0%", "10%", "20%", "30%"])
        ax.set_yscale("log")
        ax.legend(fontsize=8, loc="best")

    fig.suptitle("Variance Ratio Change Under Noise Removal",
                 fontsize=17, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "fig4_variance_ratio.png")



def fig5_reversal_reduction():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    pair_colors = ["#E8565C", "#F5A623", "#4A90D9", "#45B87F", "#8E44AD", "#C0392B"]

    for ax, ds in zip(axes, DATASETS):
        analysis = load_analysis(ds)
        tr = analysis["threshold_results"]

        baseline_rev = tr.get("baseline", {}).get("exp1_ranking", {}).get("reversals", {})
        pairs = list(baseline_rev.keys())

        for pi, pair in enumerate(pairs):
            rates = []
            for tk in THRESHOLDS:
                rev = tr.get(tk, {}).get("exp1_ranking", {}).get("reversals", {})
                if pair in rev:
                    rates.append(rev[pair]["reversal_rate"] * 100)
                else:
                    rates.append(np.nan)

            m_a, m_b = pair.split("_vs_")
            label = f"{M_DISP_E1.get(m_a, m_a)[:10]} vs {M_DISP_E1.get(m_b, m_b)[:10]}"
            ax.plot(THRESHOLD_PCTS, rates, "o-", color=pair_colors[pi % len(pair_colors)],
                    label=label, linewidth=2, markersize=8, zorder=4)

        ax.axhline(50, color="#888", ls=":", lw=1, alpha=0.5)
        ax.set_xlabel("% Noisiest Items Removed")
        ax.set_ylabel("Reversal Rate (%)")
        ax.set_title(B_DISP[ds])
        ax.set_xticks(THRESHOLD_PCTS)
        ax.set_xticklabels(["0%", "10%", "20%", "30%"])
        ax.set_ylim(-5, 105)
        ax.legend(fontsize=7, loc="best", ncol=2)

    fig.suptitle("Ranking Reversal Rate Under Noise Removal (Exp I)",
                 fontsize=17, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "fig5_reversal_reduction.png")



def fig6_three_way_variance():
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, ds in zip(axes, DATASETS):
        analysis = load_analysis(ds)
        vd_all = analysis.get("variance_decomposition_3way", {})

        labels = []
        pp, ps, pt = [], [], []

        for tk in THRESHOLDS:
            vd = vd_all.get(tk, [])
            for v in vd:
                labels.append(f"{M_DISP.get(v['model'], v['model'])[:8]}\n{THRESHOLD_LABELS[tk]}")
                pp.append(v["pct_prompt"])
                ps.append(v["pct_sampling"])
                pt.append(v["pct_testset"])

        if not labels:
            continue

        x = np.arange(len(labels))
        w = 0.65
        ax.bar(x, pp, w, label="Var(prompt)", color="#E8565C", alpha=.85, edgecolor="white")
        ax.bar(x, ps, w, bottom=pp, label="Var(sampling)", color="#95A5A6", alpha=.85, edgecolor="white")
        bottom2 = [a + b for a, b in zip(pp, ps)]
        ax.bar(x, pt, w, bottom=bottom2, label="Var(test-set)", color="#4A90D9", alpha=.85, edgecolor="white")

        for i in range(len(labels)):
            cum = 0
            for pct in [pp[i], ps[i], pt[i]]:
                if pct > 8:
                    ax.text(i, cum + pct / 2, f"{pct:.0f}%", ha="center", va="center",
                            fontsize=7, fontweight="bold", color="white")
                cum += pct

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
        ax.set_ylabel("Variance Share (%)")
        ax.set_ylim(0, 110)
        ax.set_title(B_DISP[ds])
        ax.legend(fontsize=9)

    fig.suptitle("Three-Way Variance Decomposition Under Noise Removal",
                 fontsize=17, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "fig6_three_way_variance.png")



def fig7_noise_correlation():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, ds in zip(axes, DATASETS):
        analysis = load_analysis(ds)
        corrs = analysis.get("noise_correlation", {})

        n = len(MODELS_E2)
        mat = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                key = f"{MODELS_E2[i]}_vs_{MODELS_E2[j]}"
                if key in corrs:
                    mat[i, j] = corrs[key]
                    mat[j, i] = corrs[key]

        labels = [M_DISP[m][:10] for m in MODELS_E2]
        im = ax.imshow(mat, cmap="RdYlGn", vmin=-1, vmax=1, aspect="equal")

        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, fontsize=9, rotation=30, ha="right")
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels, fontsize=9)

        for i in range(n):
            for j in range(n):
                color = "white" if abs(mat[i, j]) > 0.5 else "black"
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                        fontsize=10, color=color, fontweight="bold")

        ax.set_title(B_DISP[ds])

    fig.colorbar(im, ax=axes, shrink=0.8, label="Noise Score Correlation")
    fig.suptitle("Per-Model Noise Score Correlation",
                 fontsize=17, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "fig7_noise_correlation.png")



def fig8_noise_vs_difficulty():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, ds in zip(axes, DATASETS):
        noise = load_noise(ds)
        analysis = load_analysis(ds)
        nvd = analysis.get("noise_vs_difficulty", {})

        difficulties = []
        noises = []
        for qid, nd in noise["noise_scores"].items():
            if nd["total"] > 0:
                difficulties.append(1.0 - nd["correct"] / nd["total"])
                noises.append(nd["noise_score"])

        ax.scatter(difficulties, noises, alpha=0.4, s=30, color="#4A90D9",
                   edgecolors="white", linewidth=0.3, zorder=3)

        z = np.polyfit(difficulties, noises, 2)
        x_fit = np.linspace(0, 1, 100)
        y_fit = np.polyval(z, x_fit)
        ax.plot(x_fit, y_fit, "r-", linewidth=2, alpha=0.7, label="Quadratic fit", zorder=4)

        corr = nvd.get("correlation", 0)
        ax.text(0.02, 0.95, f"r = {corr:.3f}", transform=ax.transAxes,
                fontsize=11, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax.set_xlabel("Item Difficulty (1 - accuracy)")
        ax.set_ylabel("Noise Score")
        ax.set_title(B_DISP[ds])
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=9)

    fig.suptitle("Noise Score vs Item Difficulty",
                 fontsize=17, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "fig8_noise_vs_difficulty.png")



def fig9_category_noise():
    noise = load_noise("mmlu")
    qual = noise.get("qualitative_analysis", {})
    cat_noise = qual.get("category_noise", {})

    if not cat_noise:
        print("  Skipping fig9: no category noise data")
        return

    cats = list(cat_noise.keys())
    means = [cat_noise[c]["mean_noise"] for c in cats]
    n_qs = [cat_noise[c]["n_questions"] for c in cats]
    n_noisy = [cat_noise[c].get("n_noisy_above_0.5", 0) for c in cats]

    fig, ax = plt.subplots(figsize=(12, max(6, len(cats) * 0.4)))

    colors = ["#E74C3C" if m > 0.4 else "#F39C12" if m > 0.25 else "#45B87F" for m in means]
    bars = ax.barh(range(len(cats)), means, color=colors, alpha=0.8,
                   edgecolor="white", height=0.7, zorder=3)

    for i, (bar, nq, nn) in enumerate(zip(bars, n_qs, n_noisy)):
        ax.text(bar.get_width() + 0.01, i,
                f" n={nq}, noisy={nn}", va="center", fontsize=9)

    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels(cats, fontsize=9)
    ax.set_xlabel("Mean Noise Score")
    ax.set_title("MMLU-Pro: Mean Noise Score by Subject Category",
                 fontsize=15, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0, max(means) * 1.3)

    fig.tight_layout()
    save(fig, "fig9_category_noise.png")



def fig10_exp2_stability():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, ds in zip(axes, DATASETS):
        analysis = load_analysis(ds)
        tr = analysis["threshold_results"]

        for m_e2 in MODELS_E2:
            stds = []
            for tk in THRESHOLDS:
                exp2 = tr.get(tk, {}).get("exp2", {})
                acc_summary = exp2.get("accuracy_summary", [])
                match = [a for a in acc_summary if a.get("model_short") == m_e2]
                if match:
                    stds.append(match[0]["std"])
                else:
                    stds.append(np.nan)

            ax.plot(THRESHOLD_PCTS, stds, "o-", color=MC[m_e2],
                    label=M_DISP[m_e2], linewidth=2.5, markersize=9, zorder=4)

        ax.set_xlabel("% Noisiest Items Removed")
        ax.set_ylabel("Accuracy Std Across Paraphrases")
        ax.set_title(B_DISP[ds])
        ax.set_xticks(THRESHOLD_PCTS)
        ax.set_xticklabels(["0%", "10%", "20%", "30%"])
        ax.legend(fontsize=9, loc="best")

    fig.suptitle("Exp II: Paraphrase Stability Under Noise Removal",
                 fontsize=17, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "fig10_exp2_stability.png")



def fig11_summary_dashboard():
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    for ds, color in [("arc", "#4A90D9"), ("mmlu", "#E8565C")]:
        noise = load_noise(ds)
        scores = [nd["noise_score"] for nd in noise["noise_scores"].values()]
        ax1.hist(scores, bins=20, alpha=0.5, color=color, label=B_DISP[ds],
                 edgecolor="white", linewidth=0.5)
    ax1.set_xlabel("Noise Score")
    ax1.set_ylabel("Count")
    ax1.set_title("(A) Noise Distribution", fontsize=11)
    ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(gs[0, 1])
    for ds, ls in [("arc", "-"), ("mmlu", "--")]:
        analysis = load_analysis(ds)
        tr = analysis["threshold_results"]
        for m_e1 in MODELS_E1:
            vals = []
            for tk in THRESHOLDS:
                if tk in tr and m_e1 in tr[tk].get("exp1", {}):
                    vals.append(tr[tk]["exp1"][m_e1]["accuracy"]["std"])
                else:
                    vals.append(np.nan)
            label = f"{M_DISP_E1[m_e1][:8]} ({ds[:3]})" if ls == "-" else None
            ax2.plot(THRESHOLD_PCTS, vals, f"o{ls}", color=MC_E1[m_e1],
                     linewidth=1.5, markersize=6, label=label)
    ax2.set_xlabel("% Removed")
    ax2.set_ylabel("Acc Std")
    ax2.set_title("(B) Prompt Stability", fontsize=11)
    ax2.set_xticks(THRESHOLD_PCTS)
    ax2.legend(fontsize=6, ncol=2)

    ax3 = fig.add_subplot(gs[0, 2])
    for ds, ls in [("arc", "-"), ("mmlu", "--")]:
        analysis = load_analysis(ds)
        tr = analysis["threshold_results"]
        for m_e1 in MODELS_E1:
            vals = []
            for tk in THRESHOLDS:
                if tk in tr and m_e1 in tr[tk].get("exp1", {}):
                    vals.append(tr[tk]["exp1"][m_e1]["flip_rate"] * 100)
                else:
                    vals.append(np.nan)
            label = f"{M_DISP_E1[m_e1][:8]} ({ds[:3]})" if ls == "-" else None
            ax3.plot(THRESHOLD_PCTS, vals, f"o{ls}", color=MC_E1[m_e1],
                     linewidth=1.5, markersize=6, label=label)
    ax3.set_xlabel("% Removed")
    ax3.set_ylabel("Flip Rate (%)")
    ax3.set_title("(C) Flip Rate Reduction", fontsize=11)
    ax3.set_xticks(THRESHOLD_PCTS)
    ax3.legend(fontsize=6, ncol=2)

    ax4 = fig.add_subplot(gs[1, 0])
    for ds, ls in [("arc", "-"), ("mmlu", "--")]:
        analysis = load_analysis(ds)
        tr = analysis["threshold_results"]
        for m_e1 in MODELS_E1:
            vals = []
            for tk in THRESHOLDS:
                if tk in tr and m_e1 in tr[tk].get("exp1", {}):
                    vals.append(tr[tk]["exp1"][m_e1]["variance_decomposition"]["ratio"])
                else:
                    vals.append(np.nan)
            ax4.plot(THRESHOLD_PCTS, vals, f"o{ls}", color=MC_E1[m_e1],
                     linewidth=1.5, markersize=6)
    ax4.axhline(1, color="#888", ls="--", lw=1)
    ax4.set_xlabel("% Removed")
    ax4.set_ylabel("Var Ratio")
    ax4.set_title("(D) Variance Ratio", fontsize=11)
    ax4.set_yscale("log")
    ax4.set_xticks(THRESHOLD_PCTS)

    ax5 = fig.add_subplot(gs[1, 1])
    for ds, color in [("arc", "#4A90D9"), ("mmlu", "#E8565C")]:
        noise = load_noise(ds)
        difficulties = []
        noises = []
        for nd in noise["noise_scores"].values():
            if nd["total"] > 0:
                difficulties.append(1.0 - nd["correct"] / nd["total"])
                noises.append(nd["noise_score"])
        ax5.scatter(difficulties, noises, alpha=0.3, s=15, color=color,
                    label=B_DISP[ds], edgecolors="none")
    ax5.set_xlabel("Difficulty")
    ax5.set_ylabel("Noise Score")
    ax5.set_title("(E) Noise vs Difficulty", fontsize=11)
    ax5.legend(fontsize=8)

    ax6 = fig.add_subplot(gs[1, 2])
    pair_labels = []
    corr_vals_arc = []
    corr_vals_mmlu = []
    for m1, m2 in combinations(MODELS_E2, 2):
        key = f"{m1}_vs_{m2}"
        pair_labels.append(f"{M_DISP[m1][:5]}v{M_DISP[m2][:5]}")
        for ds, vals in [("arc", corr_vals_arc), ("mmlu", corr_vals_mmlu)]:
            analysis = load_analysis(ds)
            corrs = analysis.get("noise_correlation", {})
            vals.append(corrs.get(key, 0))

    x = np.arange(len(pair_labels))
    w = 0.35
    ax6.bar(x - w/2, corr_vals_arc, w, label="ARC", color="#4A90D9", alpha=0.8)
    ax6.bar(x + w/2, corr_vals_mmlu, w, label="MMLU", color="#E8565C", alpha=0.8)
    ax6.set_xticks(x)
    ax6.set_xticklabels(pair_labels, fontsize=6, rotation=45, ha="right")
    ax6.set_ylabel("Noise Correlation")
    ax6.set_title("(F) Model Agreement on Noise", fontsize=11)
    ax6.legend(fontsize=8)

    fig.suptitle("Experiment III: High-Noise Item Analysis — Overview",
                 fontsize=17, fontweight="bold", y=1.01)
    fig.savefig(FIGDIR / "fig11_summary_dashboard.png", bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print("  saved fig11_summary_dashboard.png")



def main():
    import argparse
    parser = argparse.ArgumentParser(description="Experiment III: Visualizations")
    parser.add_argument("--noise-tag", type=str, default="",
                        help="Suffix of noise/analysis files, e.g. '_shared150'")
    args = parser.parse_args()

    global NOISE_TAG, FIGDIR
    NOISE_TAG = args.noise_tag

    if NOISE_TAG:
        FIGDIR = ROOT / f"figures_exp3{NOISE_TAG}"
        FIGDIR.mkdir(exist_ok=True)

    tag_label = NOISE_TAG if NOISE_TAG else "(default)"
    print(f"Generating Experiment III figures [{tag_label}]...")

    fig1_noise_distribution()
    fig2_exp1_stability_improvement()
    fig3_flip_rate_reduction()
    fig4_variance_ratio()
    fig5_reversal_reduction()
    fig6_three_way_variance()
    fig7_noise_correlation()
    fig8_noise_vs_difficulty()
    fig9_category_noise()
    fig10_exp2_stability()
    fig11_summary_dashboard()

    n = len(list(FIGDIR.glob("*.png")))
    print(f"\nDone — {n} figures saved to {FIGDIR}/")


if __name__ == "__main__":
    main()
