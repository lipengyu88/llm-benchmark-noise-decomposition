"""
Experiment I-Extended: Publication-quality Visualizations (100 variants)

Figures:
  fig1  — Accuracy distribution across 100 variants (box + swarm)
  fig2  — OFAT main effects by dimension (5 dimensions)
  fig3  — Variance decomposition: Var(prompt) vs Var(sampling)
  fig4  — Dimension-level variance attribution (pie/bar)
  fig5  — Pairwise ranking stability (violin + jitter)
  fig6  — Scale analysis: robustness vs model size
  fig7  — Noise removal impact on accuracy std
  fig8  — Category sensitivity heatmap (MMLU-Pro)
  fig9  — Ranking reversal frequency
  fig10 — Regression coefficients (main effects)
  fig11 — Summary dashboard
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import seaborn as sns
from pathlib import Path
from itertools import combinations


sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
})

COLORS = {
    "llama":   "#4C72B0",
    "qwen7b":  "#C44E52",
    "qwen32b": "#DD8452",
    "qwen72b": "#55A868",
}
MODEL_LABELS = {
    "llama":   "Llama-3.1-8B",
    "qwen7b":  "Qwen2.5-7B",
    "qwen32b": "Qwen3-32B",
    "qwen72b": "Qwen2.5-72B",
}
MODELS = ["llama", "qwen7b", "qwen32b", "qwen72b"]
DATASET_LABELS = {"arc": "ARC-Challenge", "mmlu": "MMLU-Pro"}
DIM_NAMES = ["instruction", "answer_format", "option_format", "framing", "delimiter"]
DIM_SHORT = ["Instruction", "Ans. Format", "Opt. Format", "Framing", "Delimiter"]

OUTPUT_DIR = Path("figures_exp1")
OUTPUT_DIR.mkdir(exist_ok=True)

ANALYSIS_DIR = Path("analysis_exp1")


def load_analysis(dataset):
    with open(ANALYSIS_DIR / f"analysis_{dataset}.json") as f:
        return json.load(f)



def fig1_accuracy_distribution():
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax_idx, dataset in enumerate(["arc", "mmlu"]):
        ax = axes[ax_idx]
        data = load_analysis(dataset)

        positions = []
        box_data = []
        box_colors = []
        tick_labels = []

        for i, model in enumerate(MODELS):
            if model not in data:
                continue
            accs = data[model]["accuracy_stats"]["per_variant"]
            box_data.append(accs)
            positions.append(i)
            tick_labels.append(MODEL_LABELS[model])
            box_colors.append(COLORS[model])

        bp = ax.boxplot(box_data, positions=positions, widths=0.5, patch_artist=True,
                        showfliers=False)
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
            patch.set_edgecolor(color)

        for i, (accs, color) in enumerate(zip(box_data, box_colors)):
            jitter = np.random.RandomState(42).uniform(-0.12, 0.12, len(accs))
            ax.scatter([i] * len(accs) + jitter, accs,
                       color=color, s=12, alpha=0.5, edgecolors="none", zorder=3)

        ax.set_xticks(positions)
        ax.set_xticklabels(tick_labels, fontsize=10)
        ax.set_ylabel("Accuracy")
        ax.set_title(DATASET_LABELS[dataset])

        for i, (accs, model) in enumerate(zip(box_data, MODELS)):
            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            ax.text(i, ax.get_ylim()[1] * 0.98, f"{mean_acc:.3f}\n+/-{std_acc:.3f}",
                    ha="center", va="top", fontsize=8, fontweight="bold",
                    color=COLORS[model])

    fig.suptitle("Accuracy Distribution Across 100 Prompt Variants",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig1_accuracy_distribution.png")
    plt.close()
    print("  Saved fig1_accuracy_distribution.png")



def fig2_ofat_main_effects():
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes_flat = axes.flatten()

    for d_idx, (dim, title) in enumerate(zip(DIM_NAMES, DIM_SHORT)):
        ax = axes_flat[d_idx]
        bar_groups = []
        for dataset in ["arc", "mmlu"]:
            data = load_analysis(dataset)
            for model in MODELS:
                if model not in data:
                    continue
                effects = data[model]["ofat_effects"][dim]
                for name, acc, delta in effects[1:]:
                    bar_groups.append({
                        "dataset": DATASET_LABELS[dataset],
                        "model": MODEL_LABELS[model],
                        "model_key": model,
                        "variant": name,
                        "delta": delta,
                    })

        if not bar_groups:
            continue

        variant_names = sorted(set(b["variant"] for b in bar_groups))
        width = 0.1
        offsets = np.arange(len(variant_names))

        i = 0
        for dataset in ["arc", "mmlu"]:
            for model in MODELS:
                deltas = []
                for vn in variant_names:
                    match = [b for b in bar_groups
                             if b["variant"] == vn and b["model_key"] == model
                             and b["dataset"] == DATASET_LABELS[dataset]]
                    deltas.append(match[0]["delta"] if match else 0)

                hatch = "" if dataset == "arc" else "///"
                label = f"{MODEL_LABELS[model]} ({DATASET_LABELS[dataset][:3]})"
                ax.bar(offsets + i * width, deltas, width * 0.9,
                       color=COLORS[model], alpha=0.7 if dataset == "arc" else 0.4,
                       hatch=hatch, label=label, edgecolor="white", linewidth=0.5)
                i += 1

        ax.set_xticks(offsets + width * 3.5)
        ax.set_xticklabels(variant_names, fontsize=8)
        ax.set_ylabel("Delta Acc from Base")
        ax.set_title(title)
        ax.axhline(y=0, color="black", linewidth=0.8)

        if d_idx == 0:
            ax.legend(fontsize=6, ncol=2, loc="upper right")

    axes_flat[5].set_visible(False)

    fig.suptitle("OFAT Main Effects by Dimension (5 Dimensions)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig2_ofat_main_effects.png")
    plt.close()
    print("  Saved fig2_ofat_main_effects.png")



def fig3_variance_decomposition():
    fig, ax = plt.subplots(figsize=(12, 5.5))

    labels = []
    var_prompt_vals = []
    var_sampling_vals = []
    colors_bars = []

    for dataset in ["arc", "mmlu"]:
        data = load_analysis(dataset)
        for model in MODELS:
            if model not in data:
                continue
            vd = data[model]["variance_decomposition"]
            labels.append(f"{MODEL_LABELS[model]}\n({DATASET_LABELS[dataset][:3]})")
            var_prompt_vals.append(vd["var_prompt"])
            var_sampling_vals.append(vd["var_sampling"])
            colors_bars.append(COLORS[model])

    x = np.arange(len(labels))
    w = 0.55

    ax.bar(x, var_prompt_vals, w, label="Var(prompt)", color=colors_bars, alpha=0.85,
           edgecolor="white", linewidth=1.2)
    ax.bar(x, var_sampling_vals, w, bottom=var_prompt_vals,
           label="Var(sampling)", color=colors_bars, alpha=0.3,
           edgecolor="white", linewidth=1.2, hatch="///")

    for i in range(len(labels)):
        ratio = var_prompt_vals[i] / var_sampling_vals[i] if var_sampling_vals[i] > 0 else 0
        total = var_prompt_vals[i] + var_sampling_vals[i]
        ax.text(i, total + total * 0.08, f"x{ratio:.1f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Variance")
    ax.set_title("Variance Decomposition: Prompt Variation vs. Sampling Noise (100 Variants)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-5)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig3_variance_decomposition.png")
    plt.close()
    print("  Saved fig3_variance_decomposition.png")



def fig4_dimension_variance():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    dataset_model_pairs = [
        ("arc", "All Models (ARC)"),
        ("mmlu", "All Models (MMLU)"),
    ]

    for ax_row, dataset in enumerate(["arc", "mmlu"]):
        data = load_analysis(dataset)

        ax_bar = axes[ax_row][0]
        dim_keys = DIM_NAMES + ["interaction"]
        dim_colors = ["#4C72B0", "#C44E52", "#DD8452", "#55A868", "#8172B2", "#CCCCCC"]

        x_pos = np.arange(len(MODELS))
        bottoms = np.zeros(len(MODELS))

        for d_idx, dim_key in enumerate(dim_keys):
            pcts = []
            for model in MODELS:
                if model in data and "dimension_variance" in data[model]:
                    pcts.append(data[model]["dimension_variance"]["percentages"].get(dim_key, 0))
                else:
                    pcts.append(0)
            ax_bar.bar(x_pos, pcts, 0.6, bottom=bottoms, label=dim_key.replace("_", " ").title(),
                       color=dim_colors[d_idx], alpha=0.8, edgecolor="white")
            bottoms += np.array(pcts)

        ax_bar.set_xticks(x_pos)
        ax_bar.set_xticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=9)
        ax_bar.set_ylabel("% of Var(prompt)")
        ax_bar.set_title(f"{DATASET_LABELS[dataset]}: Variance Attribution by Dimension")
        ax_bar.set_ylim(0, 105)
        if ax_row == 0:
            ax_bar.legend(fontsize=8, loc="upper right")

        ax_pie = axes[ax_row][1]
        avg_pcts = []
        for dim_key in dim_keys:
            vals = []
            for model in MODELS:
                if model in data and "dimension_variance" in data[model]:
                    vals.append(data[model]["dimension_variance"]["percentages"].get(dim_key, 0))
            avg_pcts.append(np.mean(vals) if vals else 0)

        labels_pie = [dk.replace("_", " ").title() if p > 1 else ""
                      for dk, p in zip(dim_keys, avg_pcts)]
        wedges, texts, autotexts = ax_pie.pie(
            avg_pcts, labels=labels_pie, autopct=lambda p: f"{p:.1f}%" if p > 1 else "",
            colors=dim_colors, startangle=90, pctdistance=0.75,
            textprops={"fontsize": 9}
        )
        ax_pie.set_title(f"{DATASET_LABELS[dataset]}: Avg. Variance Share")

    fig.suptitle("Dimension-Level Variance Attribution (100 Variants)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig4_dimension_variance.png")
    plt.close()
    print("  Saved fig4_dimension_variance.png")



def fig5_ranking_stability():
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    pair_colors = ["#4C72B0", "#DD8452", "#C44E52", "#55A868", "#8172B2", "#CCB974"]

    for ax_idx, dataset in enumerate(["arc", "mmlu"]):
        ax = axes[ax_idx]
        data = load_analysis(dataset)
        ranking = data.get("ranking", {})
        gaps = ranking.get("pairwise_gaps", {})

        pair_names = []
        all_gap_data = []
        for i, (pair_key, gap_info) in enumerate(gaps.items()):
            per_variant_gaps = gap_info["per_variant_gaps"]
            m_a, m_b = pair_key.split("_vs_")
            label = f"{MODEL_LABELS.get(m_a, m_a)}\nvs {MODEL_LABELS.get(m_b, m_b)}"
            pair_names.append(label)
            all_gap_data.append(per_variant_gaps)

        if not all_gap_data:
            continue

        parts = ax.violinplot(all_gap_data, positions=range(len(all_gap_data)),
                              showmeans=True, showmedians=False)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(pair_colors[i % len(pair_colors)])
            pc.set_alpha(0.5)

        for i, gaps_list in enumerate(all_gap_data):
            jitter = np.random.RandomState(42).uniform(-0.15, 0.15, len(gaps_list))
            ax.scatter([i] * len(gaps_list) + jitter, gaps_list,
                       color=pair_colors[i % len(pair_colors)], s=8, alpha=0.4,
                       edgecolors="none", zorder=3)

        ax.axhline(y=0, color="red", linewidth=1.2, linestyle="--", alpha=0.7,
                   label="No difference")
        ax.set_xticks(range(len(pair_names)))
        ax.set_xticklabels(pair_names, fontsize=8)
        ax.set_ylabel("Accuracy Gap (A - B)")
        ax.set_title(DATASET_LABELS[dataset])
        ax.legend(fontsize=9)

    fig.suptitle("Pairwise Performance Gap Distribution (100 Prompt Variants)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig5_ranking_stability.png")
    plt.close()
    print("  Saved fig5_ranking_stability.png")



def fig6_scale_analysis():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metrics = [
        ("std", "Accuracy Std"),
        ("range", "Accuracy Range"),
        ("flip_rate", "Item Flip Rate"),
    ]
    markers = {"arc": "o", "mmlu": "s"}
    linestyles = {"arc": "-", "mmlu": "--"}

    for ax_idx, (metric_key, ylabel) in enumerate(metrics):
        ax = axes[ax_idx]
        for dataset in ["arc", "mmlu"]:
            data = load_analysis(dataset)
            sizes = []
            values = []
            for model in MODELS:
                if model not in data:
                    continue
                sizes.append({"llama": 8, "qwen7b": 7, "qwen32b": 32, "qwen72b": 72}[model])
                stats = data[model]["accuracy_stats"]
                if metric_key == "flip_rate":
                    values.append(data[model]["item_flip_rate"])
                else:
                    values.append(stats[metric_key])

            ax.plot(sizes, values,
                    marker=markers[dataset], linestyle=linestyles[dataset],
                    color="#333333", linewidth=2, markersize=10,
                    label=DATASET_LABELS[dataset], alpha=0.8)
            for s, v, model in zip(sizes, values, MODELS):
                ax.scatter([s], [v], color=COLORS[model], s=120, zorder=5,
                           edgecolors="white", linewidth=1.5)

        ax.set_xlabel("Model Size (B)")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log")
        ax.set_xticks([8, 32, 72])
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.legend(fontsize=9)

    fig.suptitle("Prompt Robustness vs. Model Scale (100 Variants)",
                 fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig6_scale_analysis.png")
    plt.close()
    print("  Saved fig6_scale_analysis.png")



def fig7_noise_removal():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax_idx, dataset in enumerate(["arc", "mmlu"]):
        ax = axes[ax_idx]
        data = load_analysis(dataset)
        noise = data.get("noise_analysis", {})
        removal = noise.get("removal_analysis", {})

        pcts = [0, 10, 20, 30]
        for model in MODELS:
            if model not in data:
                continue
            stds = [data[model]["accuracy_stats"]["std"]]
            for pct in [10, 20, 30]:
                key = f"remove_{pct}pct"
                if key in removal and model in removal[key]["per_model"]:
                    stds.append(removal[key]["per_model"][model]["std"])
                else:
                    stds.append(np.nan)

            ax.plot(pcts, stds, "o-", color=COLORS[model],
                    label=MODEL_LABELS[model], linewidth=2, markersize=8)

        ax.set_xlabel("% Noisiest Items Removed")
        ax.set_ylabel("Accuracy Std Across Prompts")
        ax.set_title(DATASET_LABELS[dataset])
        ax.set_xticks(pcts)
        ax.set_xticklabels(["0%", "10%", "20%", "30%"])
        ax.legend(fontsize=9)

    fig.suptitle("Effect of Removing High-Noise Items (100 Variants)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig7_noise_removal.png")
    plt.close()
    print("  Saved fig7_noise_removal.png")



def fig8_category_heatmap():
    data = load_analysis("mmlu")

    all_cats = set()
    for model in MODELS:
        if model in data and "category_analysis" in data[model]:
            all_cats.update(data[model]["category_analysis"].keys())

    cats = sorted(all_cats)
    if not cats:
        return

    range_matrix = np.full((len(cats), len(MODELS)), np.nan)
    n_questions = {}

    for j, model in enumerate(MODELS):
        if model not in data or "category_analysis" not in data[model]:
            continue
        cat_data = data[model]["category_analysis"]
        for i, cat in enumerate(cats):
            if cat in cat_data:
                range_matrix[i, j] = cat_data[cat]["range"]
                n_questions[cat] = cat_data[cat]["n_questions"]

    mean_range = np.nanmean(range_matrix, axis=1)
    sort_idx = np.argsort(mean_range)[::-1]
    range_matrix = range_matrix[sort_idx]
    cats = [cats[i] for i in sort_idx]
    cat_labels = [f"{c} (n={n_questions.get(c, '?')})" for c in cats]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(range_matrix, cmap="YlOrRd", aspect="auto", vmin=0)

    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=10)
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels(cat_labels, fontsize=9)

    for i in range(len(cats)):
        for j in range(len(MODELS)):
            val = range_matrix[i, j]
            if not np.isnan(val):
                color = "white" if val > 0.35 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color=color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Accuracy Range Across 100 Prompts", fontsize=10)
    ax.set_title("MMLU-Pro: Prompt Sensitivity by Category (100 Variants)",
                 fontsize=13, fontweight="bold")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig8_category_heatmap.png")
    plt.close()
    print("  Saved fig8_category_heatmap.png")



def fig9_reversal_summary():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, dataset in enumerate(["arc", "mmlu"]):
        ax = axes[ax_idx]
        data = load_analysis(dataset)
        ranking = data.get("ranking", {})
        reversals = ranking.get("reversals_all", {})

        pair_labels = []
        rev_rates = []
        bar_colors = []
        pair_colors_list = ["#4C72B0", "#DD8452", "#C44E52", "#55A868", "#8172B2", "#CCB974"]

        for i, (pair_key, rev_info) in enumerate(reversals.items()):
            m_a, m_b = pair_key.split("_vs_")
            pair_labels.append(f"{MODEL_LABELS.get(m_a, m_a)}\nvs {MODEL_LABELS.get(m_b, m_b)}")
            rev_rates.append(rev_info["reversal_rate"] * 100)
            bar_colors.append(pair_colors_list[i % len(pair_colors_list)])

        bars = ax.bar(range(len(pair_labels)), rev_rates, color=bar_colors,
                      alpha=0.8, edgecolor="white", linewidth=1.5, width=0.6)

        for bar, rate in zip(bars, rev_rates):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{rate:.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax.set_xticks(range(len(pair_labels)))
        ax.set_xticklabels(pair_labels, fontsize=8)
        ax.set_ylabel("Ranking Reversal Rate (%)")
        ax.set_title(DATASET_LABELS[dataset])
        ax.set_ylim(0, 100)
        ax.axhline(y=50, color="gray", linewidth=1, linestyle=":", alpha=0.5,
                   label="Random (50%)")
        ax.legend(fontsize=9)

    fig.suptitle("Pairwise Ranking Reversal Frequency (100 Prompt Variants)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig9_reversal_summary.png")
    plt.close()
    print("  Saved fig9_reversal_summary.png")



def fig10_regression_coefficients():
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    for row, dataset in enumerate(["arc", "mmlu"]):
        data = load_analysis(dataset)

        ax_all = axes[row][0]
        all_coefs = {}
        for model in MODELS:
            if model not in data:
                continue
            interact = data[model].get("interaction_effects", {})
            coeffs = interact.get("coefficients", {})
            for k, v in coeffs.items():
                if k == "intercept":
                    continue
                if k not in all_coefs:
                    all_coefs[k] = {}
                all_coefs[k][model] = v

        if not all_coefs:
            continue

        coef_names = sorted(all_coefs.keys())
        x = np.arange(len(coef_names))
        width = 0.18

        for i, model in enumerate(MODELS):
            vals = [all_coefs.get(c, {}).get(model, 0) for c in coef_names]
            ax_all.barh(x + i * width, vals, width * 0.9, color=COLORS[model],
                        alpha=0.8, label=MODEL_LABELS[model])

        ax_all.set_yticks(x + width * 1.5)
        ax_all.set_yticklabels([c.replace("_", " ") for c in coef_names], fontsize=8)
        ax_all.axvline(x=0, color="black", linewidth=0.8)
        ax_all.set_xlabel("Coefficient (Delta Accuracy)")
        ax_all.set_title(f"{DATASET_LABELS[dataset]}: Main Effect Coefficients")
        if row == 0:
            ax_all.legend(fontsize=8, loc="lower right")

        ax_r2 = axes[row][1]
        r2_main = []
        r2_full = []
        model_names = []
        for model in MODELS:
            if model not in data:
                continue
            interact = data[model].get("interaction_effects", {})
            r2_main.append(interact.get("r_squared_adj", 0))
            r2_full.append(interact.get("r_squared_adj_full", 0))
            model_names.append(MODEL_LABELS[model])

        x_r2 = np.arange(len(model_names))
        ax_r2.bar(x_r2 - 0.15, r2_main, 0.28, label="Main Effects Only",
                  color="#4C72B0", alpha=0.8)
        ax_r2.bar(x_r2 + 0.15, r2_full, 0.28, label="Main + Interactions",
                  color="#DD8452", alpha=0.8)
        ax_r2.set_xticks(x_r2)
        ax_r2.set_xticklabels(model_names, fontsize=9)
        ax_r2.set_ylabel("R-squared (adj)")
        ax_r2.set_title(f"{DATASET_LABELS[dataset]}: Model Fit")
        ax_r2.set_ylim(0, 1.05)
        ax_r2.legend(fontsize=8)

        for xi, (r1, r2) in enumerate(zip(r2_main, r2_full)):
            ax_r2.text(xi - 0.15, r1 + 0.02, f"{r1:.2f}", ha="center", fontsize=8)
            ax_r2.text(xi + 0.15, r2 + 0.02, f"{r2:.2f}", ha="center", fontsize=8)

    fig.suptitle("OLS Regression Analysis (100 Variants, 5 Dimensions)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig10_regression_coefficients.png")
    plt.close()
    print("  Saved fig10_regression_coefficients.png")



def fig11_summary_dashboard():
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    positions = []
    box_data = []
    tick_labels = []
    box_colors = []
    pos = 0
    for dataset in ["arc", "mmlu"]:
        data = load_analysis(dataset)
        for model in MODELS:
            if model not in data:
                continue
            accs = data[model]["accuracy_stats"]["per_variant"]
            box_data.append(accs)
            positions.append(pos)
            tick_labels.append(f"{MODEL_LABELS[model][:8]}\n({dataset.upper()[:3]})")
            box_colors.append(COLORS[model])
            pos += 1
        pos += 0.5

    bp = ax1.boxplot(box_data, positions=positions, widths=0.55, patch_artist=True,
                     showfliers=False)
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax1.set_xticks(positions)
    ax1.set_xticklabels(tick_labels, fontsize=6.5)
    ax1.set_ylabel("Accuracy")
    ax1.set_title("(A) Accuracy Distribution", fontsize=10)

    ax2 = fig.add_subplot(gs[0, 1])
    labels_vr = []
    ratios = []
    colors_vr = []
    for dataset in ["arc", "mmlu"]:
        data = load_analysis(dataset)
        for model in MODELS:
            if model not in data:
                continue
            vd = data[model]["variance_decomposition"]
            labels_vr.append(f"{MODEL_LABELS[model][:8]}\n({dataset.upper()[:3]})")
            ratios.append(vd["ratio"])
            colors_vr.append(COLORS[model])

    ax2.bar(range(len(labels_vr)), ratios, color=colors_vr, alpha=0.75, edgecolor="white")
    ax2.axhline(y=1, color="red", linestyle="--", linewidth=1,
                label="Var_prompt = Var_sampling")
    ax2.set_xticks(range(len(labels_vr)))
    ax2.set_xticklabels(labels_vr, fontsize=6.5)
    ax2.set_ylabel("Var_prompt / Var_sampling")
    ax2.set_title("(B) Variance Ratio", fontsize=10)
    ax2.set_yscale("log")
    ax2.legend(fontsize=7)

    ax3 = fig.add_subplot(gs[0, 2])
    labels_fr = []
    flips = []
    colors_fr = []
    for dataset in ["arc", "mmlu"]:
        data = load_analysis(dataset)
        for model in MODELS:
            if model not in data:
                continue
            labels_fr.append(f"{MODEL_LABELS[model][:8]}\n({dataset.upper()[:3]})")
            flips.append(data[model]["item_flip_rate"] * 100)
            colors_fr.append(COLORS[model])

    ax3.bar(range(len(labels_fr)), flips, color=colors_fr, alpha=0.75, edgecolor="white")
    ax3.set_xticks(range(len(labels_fr)))
    ax3.set_xticklabels(labels_fr, fontsize=6.5)
    ax3.set_ylabel("Item Flip Rate (%)")
    ax3.set_title("(C) Item Flip Rate", fontsize=10)

    ax4 = fig.add_subplot(gs[1, 0])
    dim_keys = DIM_NAMES + ["interaction"]
    dim_colors = ["#4C72B0", "#C44E52", "#DD8452", "#55A868", "#8172B2", "#CCCCCC"]
    x_dim = np.arange(len(dim_keys))
    width_d = 0.35

    for d_idx, dataset in enumerate(["arc", "mmlu"]):
        data = load_analysis(dataset)
        avg_pcts = []
        for dk in dim_keys:
            vals = []
            for model in MODELS:
                if model in data and "dimension_variance" in data[model]:
                    vals.append(data[model]["dimension_variance"]["percentages"].get(dk, 0))
            avg_pcts.append(np.mean(vals) if vals else 0)
        ax4.bar(x_dim + d_idx * width_d, avg_pcts, width_d * 0.9,
                color=dim_colors, alpha=0.8 if d_idx == 0 else 0.5,
                edgecolor="white",
                label=DATASET_LABELS[dataset] if d_idx < 1 else None)

    ax4.set_xticks(x_dim + width_d * 0.5)
    ax4.set_xticklabels([dk.replace("_", "\n") for dk in dim_keys], fontsize=7)
    ax4.set_ylabel("% of Var(prompt)")
    ax4.set_title("(D) Avg Dimension Variance Share", fontsize=10)

    ax5 = fig.add_subplot(gs[1, 1])
    pair_labels_all = []
    rev_rates_all = []
    rev_colors_all = []
    pair_c = ["#4C72B0", "#DD8452", "#C44E52", "#55A868", "#8172B2", "#CCB974"]
    for dataset in ["arc", "mmlu"]:
        data = load_analysis(dataset)
        ranking = data.get("ranking", {})
        reversals = ranking.get("reversals_all", {})
        for pi, (pk, rv) in enumerate(reversals.items()):
            m_a, m_b = pk.split("_vs_")
            pair_labels_all.append(
                f"{MODEL_LABELS.get(m_a, m_a)[:5]}v{MODEL_LABELS.get(m_b, m_b)[:5]}\n({dataset[:3]})")
            rev_rates_all.append(rv["reversal_rate"] * 100)
            rev_colors_all.append(pair_c[pi % len(pair_c)])

    ax5.bar(range(len(pair_labels_all)), rev_rates_all, color=rev_colors_all,
            alpha=0.75, edgecolor="white")
    ax5.axhline(y=50, color="gray", linestyle=":", alpha=0.5)
    ax5.set_xticks(range(len(pair_labels_all)))
    ax5.set_xticklabels(pair_labels_all, fontsize=5.5)
    ax5.set_ylabel("Reversal Rate (%)")
    ax5.set_title("(E) Ranking Reversals", fontsize=10)
    ax5.set_ylim(0, 100)

    ax6 = fig.add_subplot(gs[1, 2])
    for dataset in ["arc", "mmlu"]:
        data = load_analysis(dataset)
        sizes = []
        stds = []
        for model in MODELS:
            if model not in data:
                continue
            sizes.append({"llama": 8, "qwen7b": 7, "qwen32b": 32, "qwen72b": 72}[model])
            stds.append(data[model]["accuracy_stats"]["std"])
        ls = "-" if dataset == "arc" else "--"
        ax6.plot(sizes, stds, f"o{ls}", color="#333", linewidth=2,
                 label=DATASET_LABELS[dataset], alpha=0.8)
        for s, v, model in zip(sizes, stds, MODELS):
            ax6.scatter([s], [v], color=COLORS[model], s=100, zorder=5,
                        edgecolors="white", linewidth=1.5)

    ax6.set_xlabel("Model Size (B)")
    ax6.set_ylabel("Accuracy Std")
    ax6.set_title("(F) Robustness vs Scale", fontsize=10)
    ax6.legend(fontsize=8)

    fig.suptitle("Experiment I-Extended: 100 Prompt Variants — Overview",
                 fontsize=15, fontweight="bold", y=1.01)
    fig.savefig(OUTPUT_DIR / "fig11_summary_dashboard.png")
    plt.close()
    print("  Saved fig11_summary_dashboard.png")



if __name__ == "__main__":
    print("Generating figures for Experiment I-Extended (100 variants)...")
    fig1_accuracy_distribution()
    fig2_ofat_main_effects()
    fig3_variance_decomposition()
    fig4_dimension_variance()
    fig5_ranking_stability()
    fig6_scale_analysis()
    fig7_noise_removal()
    fig8_category_heatmap()
    fig9_reversal_summary()
    fig10_regression_coefficients()
    fig11_summary_dashboard()
    print(f"\nAll figures saved to {OUTPUT_DIR}/")
