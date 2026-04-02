"""
Experiment II: Visualization — PPT-quality PNG figures.

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
import seaborn as sns
from pathlib import Path
from itertools import combinations

ROOT = Path(__file__).parent
ANALYSIS = ROOT / "analysis_exp2"
FIGDIR = ROOT / "figures_exp2"
FIGDIR.mkdir(exist_ok=True)

MODELS = ["llama-3.1-8b", "qwen2.5-7b", "qwen3-32b", "qwen2.5-72b"]
DATASETS = {"arc": "arc_challenge", "mmlu": "mmlu_pro"}
M_DISP = {"llama-3.1-8b": "LLaMA-3.1-8B", "qwen2.5-7b": "Qwen2.5-7B",
           "qwen3-32b": "Qwen3-32B", "qwen2.5-72b": "Qwen2.5-72B"}
MC = {"llama-3.1-8b": "#E8565C", "qwen2.5-7b": "#F5A623",
      "qwen3-32b": "#4A90D9", "qwen2.5-72b": "#45B87F"}
B_DISP = {"arc": "ARC-Challenge", "mmlu": "MMLU-Pro"}
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


def load_raw(ds_key):
    bench = DATASETS[ds_key]
    frames = []
    for m in MODELS:
        p = ROOT / f"exp2_{bench}_{m}.json"
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


# ============================================================
# Figures
# ============================================================

def fig1_accuracy_by_version(data):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, ds in zip(axes, ["arc", "mmlu"]):
        df = data[ds]
        acc = df.groupby(["model_short", "version"])["is_correct"].mean().reset_index()
        for m in MODELS:
            md = acc[acc.model_short == m].sort_values("version")
            ax.plot(md.version, md.is_correct, "o-", ms=10, lw=2.5,
                    color=MC[m], label=M_DISP[m], zorder=4)
            ax.fill_between(md.version, md.is_correct.min(), md.is_correct.max(),
                            alpha=.08, color=MC[m])
        ax.set_xticks([0,1,2,3]); ax.set_xticklabels(list(VL.values()))
        ax.set_ylabel("Accuracy"); ax.set_title(B_DISP[ds])
        ax.legend(loc="best", fontsize=9)
    fig.suptitle("Accuracy Stability Across Paraphrased Versions",
                 fontsize=17, fontweight="bold", y=1.01)
    fig.tight_layout(); save(fig, "fig1_accuracy_by_version.png")


def fig2_flip_rate(analyses):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, ds in zip(axes, ["arc", "mmlu"]):
        a = analyses[ds]
        fr = a["item_flip_rate"]
        nms = [M_DISP[m] for m in MODELS]
        nf = [fr[m]["n_flipped"] for m in MODELS]
        nt = [fr[m]["n_total"] for m in MODELS]
        bars = ax.barh(nms, nf, color=[MC[m] for m in MODELS],
                       height=.55, edgecolor="white", zorder=3)
        for bar, f, t in zip(bars, nf, nt):
            ax.text(bar.get_width() + .5, bar.get_y() + bar.get_height()/2,
                    f" {f}/{t} ({100*f/t:.1f}%)", va="center", fontsize=11, fontweight="bold")
        ax.set_xlabel("Flipped Items"); ax.set_title(B_DISP[ds])
        ax.set_xlim(0, max(nf) * 1.6); ax.invert_yaxis()
    fig.suptitle("Item Flip Rate Under Paraphrasing",
                 fontsize=17, fontweight="bold", y=1.01)
    fig.tight_layout(); save(fig, "fig2_flip_rate.png")


def fig3_pairwise_ci(analyses):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, ds in zip(axes, ["arc", "mmlu"]):
        gaps = analyses[ds]["pairwise_gaps"]
        for i, g in enumerate(gaps):
            c = "#27AE60" if g["significant"] else "#C0392B"
            ax.errorbar(g["mean_gap"], i,
                        xerr=[[g["mean_gap"]-g["ci_lower"]], [g["ci_upper"]-g["mean_gap"]]],
                        fmt="D", color=c, ms=10, lw=2.5, capsize=7, capthick=2, zorder=4)
            ax.text(g["ci_upper"]+.005, i,
                    "Sig." if g["significant"] else "N.S.",
                    va="center", fontsize=10, color=c, fontweight="bold")
        ax.axvline(0, color="#888", ls="--", lw=1.2)
        ax.set_yticks(range(len(gaps)))
        ax.set_yticklabels([f"{M_DISP[g['model_1']]}\nvs {M_DISP[g['model_2']]}" for g in gaps], fontsize=9)
        ax.set_xlabel("Accuracy Gap (95% Bootstrap CI)"); ax.set_title(B_DISP[ds])
    fig.suptitle("Pairwise Accuracy Gap with Bootstrap CIs",
                 fontsize=17, fontweight="bold", y=1.01)
    fig.tight_layout(); save(fig, "fig3_pairwise_ci.png")


def fig4_rank_distribution(analyses):
    RC = ["#F1C40F", "#BDC3C7", "#CD7F32", "#8E44AD"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, ds in zip(axes, ["arc", "mmlu"]):
        rd = analyses[ds]["rank_distribution"]
        labels = [M_DISP[r["model"]] for r in rd]
        n = len(rd)
        x, w = np.arange(n), .52
        bottom = np.zeros(n)
        for r_idx in range(n):
            vals = [r[f"rank_{r_idx+1}_prob"] for r in rd]
            ax.bar(x, vals, w, bottom=bottom, label=f"Rank {r_idx+1}",
                   color=RC[r_idx], edgecolor="white", zorder=3)
            for i, v in enumerate(vals):
                if v > .04:
                    ax.text(i, bottom[i]+v/2, f"{v:.1%}", ha="center", va="center",
                            fontsize=10, fontweight="bold")
            bottom += vals
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylabel("Probability"); ax.set_ylim(0, 1.08)
        ax.set_title(B_DISP[ds]); ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Bootstrap Rank Distribution (10,000 iterations)",
                 fontsize=17, fontweight="bold", y=1.01)
    fig.tight_layout(); save(fig, "fig4_rank_distribution.png")


def fig5_two_layer(data, analyses):
    """MMLU-Pro two-layer accuracy."""
    if "two_layer_accuracy" not in analyses.get("mmlu", {}):
        return
    fig, ax = plt.subplots(figsize=(10, 5.5))
    tl = analyses["mmlu"]["two_layer_accuracy"]
    x, w = np.arange(len(tl)), .35
    ax.bar(x - w/2, [r["acc_all_150"]*100 for r in tl], w,
           label="All 150 (PF=wrong)", color="#4A90D9", alpha=.75, edgecolor="white")
    ax.bar(x + w/2, [r["acc_clean"]*100 for r in tl], w,
           label=f"Clean {analyses['mmlu']['n_evaluable']}", color="#45B87F", alpha=.85, edgecolor="white")
    for i, r in enumerate(tl):
        if r["n_pf"] > 0:
            ax.text(i - w/2, r["acc_all_150"]*100 - 3, f"PF:{r['n_pf']}",
                    ha="center", fontsize=9, color="#C0392B", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([M_DISP[r["model"]] for r in tl])
    ax.set_ylabel("Accuracy (%)"); ax.legend(fontsize=10)
    ax.set_title("MMLU-Pro: All-150 vs Clean Subset", fontsize=17, fontweight="bold")
    fig.tight_layout(); save(fig, "fig5_two_layer_mmlu.png")


def fig6_cross_flip(analyses):
    """Three-bar cross-experiment flip rate comparison."""
    cross_arc = analyses.get("arc", {}).get("cross_experiment", {}).get("flip_comparison", [])
    cross_mmlu = analyses.get("mmlu", {}).get("cross_experiment", {}).get("flip_comparison", [])
    if not cross_arc:
        return
    fig, ax = plt.subplots(figsize=(16, 6))
    labels, e1, e2 = [], [], []
    for ds, cross in [("ARC", cross_arc), ("MMLU", cross_mmlu)]:
        for c in cross:
            labels.append(f"{M_DISP[c['model']]}\n{ds}")
            e1.append(c["exp1_all"])
            e2.append(c["exp2"])
    x, w = np.arange(len(labels)), .35
    bars1 = ax.bar(x - w/2, e1, w, label="Exp I (18 prompts)", color="#E8565C", alpha=.85, edgecolor="white")
    bars2 = ax.bar(x + w/2, e2, w, label="Exp II (4 paraphrases)", color="#4A90D9", alpha=.85, edgecolor="white")
    for bars, vals in [(bars1, e1), (bars2, e2)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.6,
                    f"{v:.1f}%", ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Item Flip Rate (%)"); ax.legend(fontsize=10)
    ax.set_title("Cross-Experiment Flip Rate: Prompt vs Paraphrase",
                 fontsize=17, fontweight="bold")
    fig.tight_layout(); save(fig, "fig6_cross_flip_rate.png")


def fig7_cross_variance(analyses):
    """Variance decomposition ALL vs NO-EXPL."""
    vd_all = []
    for ds in ["arc", "mmlu"]:
        vd = analyses.get(ds, {}).get("cross_experiment", {}).get("variance_decomposition", [])
        for v in vd:
            v["bench"] = B_DISP[ds]
        vd_all.extend(vd)
    if not vd_all:
        return
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    for ax, tag, title in zip(axes, ["ALL", "NO-EXPL"],
                               ["All 18 Variants", "Excluding with_explanation (14)"]):
        sub = [v for v in vd_all if v["version"] == tag]
        labels = [f"{M_DISP[v['model']]}\n{v['bench']}" for v in sub]
        x, w = np.arange(len(sub)), .55
        pp = [v["pct_prompt"] for v in sub]
        ps = [v["pct_sampling"] for v in sub]
        pt = [v["pct_testset"] for v in sub]
        ax.bar(x, pp, w, label="Var(prompt)", color="#E8565C", alpha=.85, edgecolor="white")
        ax.bar(x, ps, w, bottom=pp, label="Var(sampling)", color="#95A5A6", alpha=.85, edgecolor="white")
        ax.bar(x, pt, w, bottom=[a+b for a,b in zip(pp,ps)], label="Var(testset)", color="#4A90D9", alpha=.85, edgecolor="white")
        for i in range(len(sub)):
            cum = 0
            for pct in [pp[i], ps[i], pt[i]]:
                if pct > 6:
                    ax.text(i, cum+pct/2, f"{pct:.0f}%", ha="center", va="center",
                            fontsize=9, fontweight="bold", color="white")
                cum += pct
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Variance Share (%)"); ax.set_ylim(0, 108)
        ax.legend(fontsize=9); ax.set_title(title)
    fig.suptitle("Three-Way Variance Decomposition", fontsize=17, fontweight="bold", y=1.01)
    fig.tight_layout(); save(fig, "fig7_variance_decomposition.png")


# ============================================================
# Main
# ============================================================

def main():
    print("Loading data...")
    data = {}
    for ds in ["arc", "mmlu"]:
        df = load_raw(ds)
        clean = get_clean_qids(df)
        data[ds] = df[df.question_id.isin(clean)].copy()

    analyses = {}
    for ds in ["arc", "mmlu"]:
        p = ANALYSIS / f"analysis_{ds}.json"
        if p.exists():
            analyses[ds] = json.loads(p.read_text())

    if not analyses:
        print("Run analyze_experiment2.py first!")
        return

    print(f"Generating figures to {FIGDIR}/")
    fig1_accuracy_by_version(data)
    fig2_flip_rate(analyses)
    fig3_pairwise_ci(analyses)
    fig4_rank_distribution(analyses)
    fig5_two_layer(data, analyses)
    fig6_cross_flip(analyses)
    fig7_cross_variance(analyses)

    n = len(list(FIGDIR.glob("*.png")))
    print(f"\nDone — {n} figures saved to {FIGDIR}/")


if __name__ == "__main__":
    main()
