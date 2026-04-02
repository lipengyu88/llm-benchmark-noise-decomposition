"""
Experiment II: Visualization — PPT-quality PNG figures.

Supports dual paraphrase sources (GPT-4o and Qwen) with cross-source comparison.

Reads:   analysis_exp2/analysis_*.json, exp2_*.json
Writes:  figures_exp2/*.png
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from itertools import combinations

ROOT = Path(__file__).parent
ANALYSIS = ROOT / "analysis_exp2"
FIGDIR = ROOT / "figures_exp2"
FIGDIR.mkdir(exist_ok=True)

MODELS = ["llama-3.1-8b", "qwen2.5-7b", "qwen3-32b", "qwen2.5-72b"]
DATASETS = {"arc": "arc_challenge", "mmlu": "mmlu_pro"}
SOURCES = ["gpt4o", "qwen"]
M_DISP = {"llama-3.1-8b": "LLaMA-3.1-8B", "qwen2.5-7b": "Qwen2.5-7B",
           "qwen3-32b": "Qwen3-32B", "qwen2.5-72b": "Qwen2.5-72B"}
MC = {"llama-3.1-8b": "#E8565C", "qwen2.5-7b": "#F5A623",
      "qwen3-32b": "#4A90D9", "qwen2.5-72b": "#45B87F"}
B_DISP = {"arc": "ARC-Challenge", "mmlu": "MMLU-Pro"}
S_DISP = {"gpt4o": "GPT-4o", "qwen": "Qwen2.5-72B"}
VL = {0: "Original", 1: "Para-1", 2: "Para-2", 3: "Para-3"}

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


def load_raw(ds_key, source):
    bench = DATASETS[ds_key]
    frames = []
    for m in MODELS:
        p = ROOT / f"exp2_{bench}_{m}_{source}.json"
        if p.exists():
            frames.append(pd.DataFrame(json.loads(p.read_text())))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if "parse_failure" not in df.columns:
        df["parse_failure"] = False
    return df


def get_clean_qids(df):
    keep = set(df.question_id.unique())
    for m in MODELS:
        mdf = df[df.model_short == m]
        keep -= set(mdf.loc[mdf.parse_failure == True, "question_id"])
    return keep


def load_analyses():
    analyses = {}
    for ds in ["arc", "mmlu"]:
        p = ANALYSIS / f"analysis_{ds}.json"
        if p.exists():
            analyses[ds] = json.loads(p.read_text())
    return analyses


# ============================================================
# Figure 1: Accuracy by version (side-by-side GPT-4o and Qwen)
# ============================================================

def fig1_accuracy_by_version(data):
    n_sources = sum(1 for s in SOURCES if any(s in data.get(ds, {}) for ds in ["arc", "mmlu"]))
    if n_sources == 0:
        return

    fig, axes = plt.subplots(n_sources, 2, figsize=(14, 5.5 * n_sources), squeeze=False)
    row = 0

    for source in SOURCES:
        has_data = False
        for col, ds in enumerate(["arc", "mmlu"]):
            if source not in data.get(ds, {}):
                continue
            has_data = True
            ax = axes[row][col]
            df = data[ds][source]
            acc = df.groupby(["model_short", "version"])["is_correct"].mean().reset_index()
            for m in MODELS:
                md = acc[acc.model_short == m].sort_values("version")
                if len(md) == 0:
                    continue
                ax.plot(md.version, md.is_correct, "o-", ms=10, lw=2.5,
                        color=MC[m], label=M_DISP[m], zorder=4)
                ax.fill_between(md.version, md.is_correct.min(), md.is_correct.max(),
                                alpha=.08, color=MC[m])
            ax.set_xticks([0, 1, 2, 3])
            ax.set_xticklabels(list(VL.values()))
            ax.set_ylabel("Accuracy")
            ax.set_title(f"{B_DISP[ds]} ({S_DISP[source]} paraphrases)")
            ax.legend(loc="best", fontsize=9)
        if has_data:
            row += 1

    fig.suptitle("Accuracy Stability Across Paraphrased Versions",
                 fontsize=17, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "fig1_accuracy_by_version.png")


# ============================================================
# Figure 2: Flip Rate comparison (GPT-4o vs Qwen)
# ============================================================

def fig2_flip_rate(analyses):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, ds in zip(axes, ["arc", "mmlu"]):
        a = analyses.get(ds, {})
        x = np.arange(len(MODELS))
        w = 0.35
        has_bars = False

        for si, source in enumerate(SOURCES):
            if source not in a:
                continue
            fr = a[source].get("item_flip_rate", {})
            nf = [fr.get(m, {}).get("n_flipped", 0) for m in MODELS]
            nt = [fr.get(m, {}).get("n_total", 1) for m in MODELS]
            rates = [100 * f / t if t > 0 else 0 for f, t in zip(nf, nt)]
            offset = -w / 2 + si * w
            bars = ax.bar(x + offset, rates, w * 0.9,
                          color=[MC[m] for m in MODELS],
                          alpha=0.85 if si == 0 else 0.5,
                          hatch="" if si == 0 else "///",
                          edgecolor="white", zorder=3,
                          label=S_DISP[source] if not has_bars else None)
            for bar, f, t in zip(bars, nf, nt):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{f}/{t}", ha="center", va="bottom", fontsize=8, fontweight="bold")
            has_bars = True

        ax.set_xticks(x)
        ax.set_xticklabels([M_DISP[m] for m in MODELS], fontsize=10)
        ax.set_ylabel("Item Flip Rate (%)")
        ax.set_title(B_DISP[ds])
        ax.legend(fontsize=10)

    fig.suptitle("Item Flip Rate Under Paraphrasing (GPT-4o vs Qwen)",
                 fontsize=17, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "fig2_flip_rate.png")


# ============================================================
# Figure 3: Pairwise CI (with BH significance)
# ============================================================

def fig3_pairwise_ci(analyses):
    n_sources = sum(1 for s in SOURCES if any(s in analyses.get(ds, {}) for ds in ["arc", "mmlu"]))
    fig, axes = plt.subplots(n_sources, 2, figsize=(15, 5.5 * n_sources), squeeze=False)
    row = 0

    for source in SOURCES:
        has_data = False
        for col, ds in enumerate(["arc", "mmlu"]):
            if source not in analyses.get(ds, {}):
                continue
            has_data = True
            ax = axes[row][col]
            gaps = analyses[ds][source].get("pairwise_gaps", [])
            for i, g in enumerate(gaps):
                sig_bh = g.get("significant_bh", g["significant"])
                c = "#27AE60" if sig_bh else "#C0392B"
                ax.errorbar(g["mean_gap"], i,
                            xerr=[[g["mean_gap"] - g["ci_lower"]],
                                  [g["ci_upper"] - g["mean_gap"]]],
                            fmt="D", color=c, ms=10, lw=2.5, capsize=7, capthick=2, zorder=4)
                p_bh = g.get("p_value_bh", g.get("p_value", 0))
                label = f"p_bh={p_bh:.3f}" if sig_bh else f"N.S. (p_bh={p_bh:.3f})"
                ax.text(g["ci_upper"] + 0.005, i, label,
                        va="center", fontsize=9, color=c, fontweight="bold")
            ax.axvline(0, color="#888", ls="--", lw=1.2)
            ax.set_yticks(range(len(gaps)))
            ax.set_yticklabels([f"{M_DISP[g['model_1']]}\nvs {M_DISP[g['model_2']]}"
                                for g in gaps], fontsize=9)
            ax.set_xlabel("Accuracy Gap (95% Bootstrap CI)")
            ax.set_title(f"{B_DISP[ds]} ({S_DISP[source]})")
        if has_data:
            row += 1

    fig.suptitle("Pairwise Accuracy Gap with Bootstrap CIs (BH-corrected)",
                 fontsize=17, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "fig3_pairwise_ci.png")


# ============================================================
# Figure 4: Rank Distribution (primary source)
# ============================================================

def fig4_rank_distribution(analyses):
    RC = ["#F1C40F", "#BDC3C7", "#CD7F32", "#8E44AD"]
    # Use first available source
    source = next((s for s in SOURCES if any(s in analyses.get(ds, {}) for ds in ["arc", "mmlu"])), None)
    if not source:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, ds in zip(axes, ["arc", "mmlu"]):
        if source not in analyses.get(ds, {}):
            continue
        rd = analyses[ds][source].get("rank_distribution", [])
        if not rd:
            continue
        labels = [M_DISP[r["model"]] for r in rd]
        n = len(rd)
        x, w = np.arange(n), .52
        bottom = np.zeros(n)
        for r_idx in range(n):
            vals = [r.get(f"rank_{r_idx + 1}_prob", 0) for r in rd]
            ax.bar(x, vals, w, bottom=bottom, label=f"Rank {r_idx + 1}",
                   color=RC[r_idx], edgecolor="white", zorder=3)
            for i, v in enumerate(vals):
                if v > .04:
                    ax.text(i, bottom[i] + v / 2, f"{v:.1%}", ha="center", va="center",
                            fontsize=10, fontweight="bold")
            bottom += vals
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1.08)
        ax.set_title(f"{B_DISP[ds]} ({S_DISP[source]})")
        ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("Bootstrap Rank Distribution (10,000 iterations)",
                 fontsize=17, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "fig4_rank_distribution.png")


# ============================================================
# Figure 5: Two-layer accuracy (MMLU parse failures)
# ============================================================

def fig5_two_layer(analyses):
    source = next((s for s in SOURCES if "mmlu" in analyses and s in analyses["mmlu"]
                    and "two_layer_accuracy" in analyses["mmlu"][s]), None)
    if not source:
        return

    fig, ax = plt.subplots(figsize=(10, 5.5))
    tl = analyses["mmlu"][source]["two_layer_accuracy"]
    n_eval = analyses["mmlu"][source]["n_evaluable"]
    x, w = np.arange(len(tl)), .35
    ax.bar(x - w / 2, [r["acc_all_150"] * 100 for r in tl], w,
           label="All 150 (PF=wrong)", color="#4A90D9", alpha=.75, edgecolor="white")
    ax.bar(x + w / 2, [r["acc_clean"] * 100 for r in tl], w,
           label=f"Clean {n_eval}", color="#45B87F", alpha=.85, edgecolor="white")
    for i, r in enumerate(tl):
        if r["n_pf"] > 0:
            ax.text(i - w / 2, r["acc_all_150"] * 100 - 3, f"PF:{r['n_pf']}",
                    ha="center", fontsize=9, color="#C0392B", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([M_DISP.get(r["model"], r["model"]) for r in tl])
    ax.set_ylabel("Accuracy (%)")
    ax.legend(fontsize=10)
    ax.set_title(f"MMLU-Pro: All-150 vs Clean Subset ({S_DISP[source]})",
                 fontsize=17, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig5_two_layer_mmlu.png")


# ============================================================
# Figure 6: Cross-source accuracy comparison
# ============================================================

def fig6_cross_source_accuracy(analyses):
    """Side-by-side accuracy under GPT-4o vs Qwen paraphrases."""
    has_cross = any("cross_source" in analyses.get(ds, {}) for ds in ["arc", "mmlu"])
    if not has_cross:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, ds in zip(axes, ["arc", "mmlu"]):
        cs = analyses.get(ds, {}).get("cross_source", {})
        acc = cs.get("accuracy", [])
        if not acc:
            continue

        x = np.arange(len(acc))
        w = 0.35
        ax.bar(x - w / 2, [a["gpt4o_mean"] * 100 for a in acc], w,
               label="GPT-4o paraphrases", color="#4A90D9", alpha=.85, edgecolor="white")
        ax.bar(x + w / 2, [a["qwen_mean"] * 100 for a in acc], w,
               label="Qwen paraphrases", color="#E8565C", alpha=.85, edgecolor="white")

        # Error bars for std
        ax.errorbar(x - w / 2, [a["gpt4o_mean"] * 100 for a in acc],
                    yerr=[a["gpt4o_std"] * 100 for a in acc],
                    fmt="none", ecolor="black", capsize=4, zorder=5)
        ax.errorbar(x + w / 2, [a["qwen_mean"] * 100 for a in acc],
                    yerr=[a["qwen_std"] * 100 for a in acc],
                    fmt="none", ecolor="black", capsize=4, zorder=5)

        ax.set_xticks(x)
        ax.set_xticklabels([M_DISP.get(a["model"], a["model"]) for a in acc], fontsize=10)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(B_DISP[ds])
        ax.legend(fontsize=10)

    fig.suptitle("Accuracy Under Different Paraphrase Sources",
                 fontsize=17, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "fig6_cross_source_accuracy.png")


# ============================================================
# Figure 7: Cross-source significance agreement
# ============================================================

def fig7_cross_source_significance(analyses):
    """Show which pairwise comparisons agree/disagree between sources."""
    has_cross = any("cross_source" in analyses.get(ds, {}) for ds in ["arc", "mmlu"])
    if not has_cross:
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, ds in zip(axes, ["arc", "mmlu"]):
        cs = analyses.get(ds, {}).get("cross_source", {})
        sig_comp = cs.get("significance_agreement", [])
        if not sig_comp:
            continue

        for i, s in enumerate(sig_comp):
            # Plot both gaps
            ax.scatter(s["gpt4o_gap"] * 100, i - 0.1, marker="D", s=100,
                       color="#4A90D9", zorder=4, label="GPT-4o" if i == 0 else None)
            ax.scatter(s["qwen_gap"] * 100, i + 0.1, marker="s", s=100,
                       color="#E8565C", zorder=4, label="Qwen" if i == 0 else None)

            # Agreement indicator
            bg = "#90EE90" if s["agree"] else "#FFB6C1"
            ax.axhspan(i - 0.4, i + 0.4, alpha=0.15, color=bg, zorder=1)

        ax.axvline(0, color="#888", ls="--", lw=1.2)
        ax.set_yticks(range(len(sig_comp)))
        ax.set_yticklabels([s["pair"].replace(" vs ", "\nvs ") for s in sig_comp], fontsize=9)
        ax.set_xlabel("Accuracy Gap (pp)")
        ax.set_title(B_DISP[ds])
        ax.legend(fontsize=9)

        # Agreement rate
        agree_rate = cs.get("agreement_rate", 0)
        ax.text(0.98, 0.02, f"Agreement: {agree_rate:.0%}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    fig.suptitle("Cross-Source Significance Agreement",
                 fontsize=17, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "fig7_cross_source_significance.png")


# ============================================================
# Figure 8: Cross-experiment flip rate
# ============================================================

def fig8_cross_experiment_flip(analyses):
    """Three-bar cross-experiment flip rate comparison."""
    source = next((s for s in SOURCES if any(
        s in analyses.get(ds, {}) and "cross_experiment" in analyses[ds][s]
        for ds in ["arc", "mmlu"]
    )), None)
    if not source:
        return

    cross_data = {}
    for ds in ["arc", "mmlu"]:
        if source in analyses.get(ds, {}) and "cross_experiment" in analyses[ds][source]:
            cross_data[ds] = analyses[ds][source]["cross_experiment"]
    if not cross_data:
        return

    fig, ax = plt.subplots(figsize=(16, 6))
    labels, e1, e2 = [], [], []
    for ds_label, ds in [("ARC", "arc"), ("MMLU", "mmlu")]:
        cross = cross_data.get(ds, {})
        for c in cross.get("flip_comparison", []):
            labels.append(f"{M_DISP.get(c['model'], c['model'])}\n{ds_label}")
            e1.append(c["exp1_all"])
            e2.append(c["exp2"])
    if not labels:
        return

    x, w = np.arange(len(labels)), .35
    bars1 = ax.bar(x - w / 2, e1, w, label="Exp I (18 prompts)",
                   color="#E8565C", alpha=.85, edgecolor="white")
    bars2 = ax.bar(x + w / 2, e2, w, label=f"Exp II ({S_DISP[source]} paraphrases)",
                   color="#4A90D9", alpha=.85, edgecolor="white")
    for bars, vals in [(bars1, e1), (bars2, e2)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + .6,
                    f"{v:.1f}%", ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Item Flip Rate (%)")
    ax.legend(fontsize=10)
    ax.set_title("Cross-Experiment Flip Rate: Prompt vs Paraphrase",
                 fontsize=17, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig8_cross_experiment_flip.png")


# ============================================================
# Figure 9: Three-way variance decomposition
# ============================================================

def fig9_cross_variance(analyses):
    """Variance decomposition ALL vs NO-EXPL."""
    vd_all = []
    source = next((s for s in SOURCES if any(
        s in analyses.get(ds, {}) and "cross_experiment" in analyses[ds][s]
        for ds in ["arc", "mmlu"]
    )), None)
    if not source:
        return

    for ds in ["arc", "mmlu"]:
        if source not in analyses.get(ds, {}):
            continue
        vd = analyses[ds][source].get("cross_experiment", {}).get("variance_decomposition", [])
        for v in vd:
            v["bench"] = B_DISP[ds]
        vd_all.extend(vd)

    if not vd_all:
        return

    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    for ax, tag, title in zip(axes, ["ALL", "NO-EXPL"],
                               ["All 18 Variants", "Excluding with_explanation (14)"]):
        sub = [v for v in vd_all if v["version"] == tag]
        if not sub:
            continue
        labels = [f"{M_DISP.get(v['model'], v['model'])[:8]}\n{v['bench']}" for v in sub]
        x, w = np.arange(len(sub)), .55
        pp = [v["pct_prompt"] for v in sub]
        ps = [v["pct_sampling"] for v in sub]
        pt = [v["pct_testset"] for v in sub]
        ax.bar(x, pp, w, label="Var(prompt)", color="#E8565C", alpha=.85, edgecolor="white")
        ax.bar(x, ps, w, bottom=pp, label="Var(sampling)", color="#95A5A6", alpha=.85, edgecolor="white")
        ax.bar(x, pt, w, bottom=[a + b for a, b in zip(pp, ps)], label="Var(test-set)",
               color="#4A90D9", alpha=.85, edgecolor="white")
        for i in range(len(sub)):
            cum = 0
            for pct in [pp[i], ps[i], pt[i]]:
                if pct > 6:
                    ax.text(i, cum + pct / 2, f"{pct:.0f}%", ha="center", va="center",
                            fontsize=9, fontweight="bold", color="white")
                cum += pct
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Variance Share (%)")
        ax.set_ylim(0, 108)
        ax.legend(fontsize=9)
        ax.set_title(title)

    fig.suptitle(f"Three-Way Variance Decomposition ({S_DISP[source]} paraphrases)",
                 fontsize=17, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "fig9_variance_decomposition.png")


# ============================================================
# Main
# ============================================================

def main():
    print("Loading data...")
    analyses = load_analyses()
    if not analyses:
        print("Run analyze_experiment2.py first!")
        return

    # Load raw data for accuracy plots
    data = {}
    for ds in ["arc", "mmlu"]:
        data[ds] = {}
        for source in SOURCES:
            df = load_raw(ds, source)
            if len(df) > 0:
                clean = get_clean_qids(df)
                data[ds][source] = df[df.question_id.isin(clean)].copy()

    print(f"Generating figures to {FIGDIR}/")
    fig1_accuracy_by_version(data)
    fig2_flip_rate(analyses)
    fig3_pairwise_ci(analyses)
    fig4_rank_distribution(analyses)
    fig5_two_layer(analyses)
    fig6_cross_source_accuracy(analyses)
    fig7_cross_source_significance(analyses)
    fig8_cross_experiment_flip(analyses)
    fig9_cross_variance(analyses)

    n = len(list(FIGDIR.glob("*.png")))
    print(f"\nDone -- {n} figures saved to {FIGDIR}/")


if __name__ == "__main__":
    main()
