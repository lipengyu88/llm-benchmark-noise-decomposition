"""
Experiment I: Publication-quality Visualizations for PPT
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from pathlib import Path
from itertools import combinations

# ============================================================
# Style setup
# ============================================================

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
    "llama":   "#4C72B0",   # blue
    "qwen7b":  "#C44E52",   # red
    "qwen32b": "#DD8452",   # orange
    "qwen72b": "#55A868",   # green
}
MODEL_LABELS = {
    "llama":   "Llama-3.1-8B",
    "qwen7b":  "Qwen2.5-7B",
    "qwen32b": "Qwen3-32B",
    "qwen72b": "Qwen2.5-72B",
}
MODELS = ["llama", "qwen7b", "qwen32b", "qwen72b"]
DATASET_LABELS = {"arc": "ARC-Challenge", "mmlu": "MMLU-Pro"}

OUTPUT_DIR = Path("figures_exp1")
OUTPUT_DIR.mkdir(exist_ok=True)

def load_analysis(dataset):
    with open(f"analysis_exp1/analysis_{dataset}.json") as f:
        return json.load(f)

# ============================================================
# Figure 1: Per-variant accuracy across models (dot + line)
# ============================================================

def fig1_accuracy_per_variant():
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

    for ax_idx, dataset in enumerate(["arc", "mmlu"]):
        ax = axes[ax_idx]
        data = load_analysis(dataset)

        from prompt_variants import get_all_variants
        variants = get_all_variants()
        variant_labels = [v[0] for v in variants]
        x = np.arange(len(variant_labels))

        for model in MODELS:
            if model not in data:
                continue
            accs = data[model]["accuracy_stats"]["per_variant"]
            ax.plot(x, accs, "o-", color=COLORS[model], label=MODEL_LABELS[model],
                    markersize=6, linewidth=1.8, alpha=0.85)
            # Shade the range
            ax.fill_between(x, min(accs), max(accs), color=COLORS[model], alpha=0.06)

        ax.set_xticks(x)
        ax.set_xticklabels(variant_labels, rotation=55, ha="right", fontsize=8)
        ax.set_ylabel("Accuracy")
        ax.set_title(DATASET_LABELS[dataset])
        ax.legend(loc="lower left", fontsize=9)
        ax.set_ylim(bottom=max(0, ax.get_ylim()[0] - 0.05))

    fig.suptitle("Accuracy Across 18 Prompt Variants", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig1_accuracy_per_variant.png")
    plt.close()
    print("  Saved fig1_accuracy_per_variant.png")

# ============================================================
# Figure 2: OFAT Main Effects (bar chart)
# ============================================================

def fig2_ofat_main_effects():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    dims = ["instruction", "answer_format", "option_format", "framing"]
    dim_titles = ["Instruction Phrasing", "Answer Format", "Option Formatting", "Contextual Framing"]

    for d_idx, (dim, title) in enumerate(zip(dims, dim_titles)):
        ax = axes[d_idx // 2][d_idx % 2]

        bar_groups = []
        for dataset in ["arc", "mmlu"]:
            data = load_analysis(dataset)
            for model in MODELS:
                if model not in data:
                    continue
                effects = data[model]["ofat_effects"][dim]
                # effects: list of (name, acc, delta)
                for name, acc, delta in effects[1:]:  # skip base
                    bar_groups.append({
                        "dataset": DATASET_LABELS[dataset],
                        "model": MODEL_LABELS[model],
                        "model_key": model,
                        "variant": name,
                        "delta": delta,
                    })

        if not bar_groups:
            continue

        # Group by variant
        variant_names = sorted(set(b["variant"] for b in bar_groups))
        n_bars = len(MODELS) * 2  # models x datasets
        width = 0.12
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

        ax.set_xticks(offsets + width * (n_bars - 1) / 2)
        ax.set_xticklabels(variant_names, fontsize=9)
        ax.set_ylabel("Δ Accuracy from Base")
        ax.set_title(title)
        ax.axhline(y=0, color="black", linewidth=0.8, linestyle="-")

        if d_idx == 0:
            ax.legend(fontsize=7, ncol=2, loc="upper right")

    fig.suptitle("OFAT Main Effects by Dimension", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig2_ofat_main_effects.png")
    plt.close()
    print("  Saved fig2_ofat_main_effects.png")

# ============================================================
# Figure 3: Variance Decomposition (stacked bar)
# ============================================================

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

    bars1 = ax.bar(x, var_prompt_vals, w, label="Var(prompt)", color=colors_bars, alpha=0.85,
                   edgecolor="white", linewidth=1.2)
    bars2 = ax.bar(x, var_sampling_vals, w, bottom=var_prompt_vals,
                   label="Var(sampling)", color=colors_bars, alpha=0.3,
                   edgecolor="white", linewidth=1.2, hatch="///")

    # Add ratio annotations
    for i in range(len(labels)):
        ratio = var_prompt_vals[i] / var_sampling_vals[i] if var_sampling_vals[i] > 0 else 0
        total = var_prompt_vals[i] + var_sampling_vals[i]
        ax.text(i, total + total * 0.05, f"×{ratio:.1f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Variance")
    ax.set_title("Variance Decomposition: Prompt Variation vs. Sampling Noise",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-5)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig3_variance_decomposition.png")
    plt.close()
    print("  Saved fig3_variance_decomposition.png")

# ============================================================
# Figure 4: Pairwise Ranking Stability (gap distribution)
# ============================================================

def fig4_ranking_stability():
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    pair_colors = ["#4C72B0", "#DD8452", "#C44E52"]

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
            label = f"{MODEL_LABELS.get(m_a, m_a)} vs\n{MODEL_LABELS.get(m_b, m_b)}"
            pair_names.append(label)
            all_gap_data.append(per_variant_gaps)

        if not all_gap_data:
            continue

        parts = ax.violinplot(all_gap_data, positions=range(len(all_gap_data)),
                              showmeans=True, showmedians=False)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(pair_colors[i % len(pair_colors)])
            pc.set_alpha(0.6)

        # Overlay individual points
        for i, gaps_list in enumerate(all_gap_data):
            jitter = np.random.RandomState(42).uniform(-0.08, 0.08, len(gaps_list))
            ax.scatter([i] * len(gaps_list) + jitter, gaps_list,
                       color=pair_colors[i % len(pair_colors)], s=30, alpha=0.7,
                       edgecolors="white", linewidth=0.5, zorder=3)

        ax.axhline(y=0, color="red", linewidth=1.2, linestyle="--", alpha=0.7, label="No difference")
        ax.set_xticks(range(len(pair_names)))
        ax.set_xticklabels(pair_names, fontsize=9)
        ax.set_ylabel("Accuracy Gap (A − B)")
        ax.set_title(DATASET_LABELS[dataset])
        ax.legend(fontsize=9)

    fig.suptitle("Pairwise Performance Gap Distribution Across Prompt Variants",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig4_ranking_stability.png")
    plt.close()
    print("  Saved fig4_ranking_stability.png")

# ============================================================
# Figure 5: Scale Analysis (trend plot)
# ============================================================

def fig5_scale_analysis():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metrics = [
        ("std", "Accuracy Std (↓ = more robust)"),
        ("range", "Accuracy Range (↓ = more robust)"),
        ("flip_rate", "Item Flip Rate (↓ = more robust)"),
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
            # Color the markers
            for s, v, model in zip(sizes, values, MODELS):
                ax.scatter([s], [v], color=COLORS[model], s=120, zorder=5,
                           edgecolors="white", linewidth=1.5)

        ax.set_xlabel("Model Size (B)")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log")
        ax.set_xticks([8, 32, 72])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.legend(fontsize=9)

    fig.suptitle("Prompt Robustness vs. Model Scale",
                 fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig5_scale_analysis.png")
    plt.close()
    print("  Saved fig5_scale_analysis.png")

# ============================================================
# Figure 6: Noise Removal Impact
# ============================================================

def fig6_noise_removal():
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
        ax.set_ylabel("Accuracy Std Across Prompts (↓ = more stable)")
        ax.set_title(DATASET_LABELS[dataset])
        ax.set_xticks(pcts)
        ax.set_xticklabels(["0%", "10%", "20%", "30%"])
        ax.legend(fontsize=9)

    fig.suptitle("Effect of Removing High-Noise Items on Evaluation Stability",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig6_noise_removal.png")
    plt.close()
    print("  Saved fig6_noise_removal.png")

# ============================================================
# Figure 7: Category Sensitivity Heatmap (MMLU-Pro)
# ============================================================

def fig7_category_heatmap():
    data = load_analysis("mmlu")

    # Collect categories across all models
    all_cats = set()
    for model in MODELS:
        if model in data and "category_analysis" in data[model]:
            all_cats.update(data[model]["category_analysis"].keys())

    cats = sorted(all_cats)
    if not cats:
        return

    # Build matrix: rows = categories, cols = models, values = range
    range_matrix = np.full((len(cats), len(MODELS)), np.nan)
    acc_matrix = np.full((len(cats), len(MODELS)), np.nan)
    n_questions = {}

    for j, model in enumerate(MODELS):
        if model not in data or "category_analysis" not in data[model]:
            continue
        cat_data = data[model]["category_analysis"]
        for i, cat in enumerate(cats):
            if cat in cat_data:
                range_matrix[i, j] = cat_data[cat]["range"]
                acc_matrix[i, j] = cat_data[cat]["mean_acc"]
                n_questions[cat] = cat_data[cat]["n_questions"]

    # Sort by mean range across models
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

    # Annotate cells
    for i in range(len(cats)):
        for j in range(len(MODELS)):
            val = range_matrix[i, j]
            if not np.isnan(val):
                color = "white" if val > 0.35 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color=color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Accuracy Range Across Prompts (↑ = more sensitive)", fontsize=10)
    ax.set_title("MMLU-Pro: Prompt Sensitivity by Subject Category",
                 fontsize=13, fontweight="bold")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig7_category_heatmap.png")
    plt.close()
    print("  Saved fig7_category_heatmap.png")

# ============================================================
# Figure 8: Reversal Frequency Summary
# ============================================================

def fig8_reversal_summary():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, dataset in enumerate(["arc", "mmlu"]):
        ax = axes[ax_idx]
        data = load_analysis(dataset)
        ranking = data.get("ranking", {})
        reversals = ranking.get("reversals_all", {})

        pair_labels = []
        rev_rates = []
        bar_colors = []
        pair_colors_list = ["#4C72B0", "#DD8452", "#C44E52"]

        for i, (pair_key, rev_info) in enumerate(reversals.items()):
            m_a, m_b = pair_key.split("_vs_")
            pair_labels.append(f"{MODEL_LABELS.get(m_a, m_a)}\nvs {MODEL_LABELS.get(m_b, m_b)}")
            rev_rates.append(rev_info["reversal_rate"] * 100)
            bar_colors.append(pair_colors_list[i % len(pair_colors_list)])

        bars = ax.bar(range(len(pair_labels)), rev_rates, color=bar_colors,
                      alpha=0.8, edgecolor="white", linewidth=1.5, width=0.6)

        # Add percentage labels
        for bar, rate in zip(bars, rev_rates):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{rate:.0f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

        ax.set_xticks(range(len(pair_labels)))
        ax.set_xticklabels(pair_labels, fontsize=9)
        ax.set_ylabel("Ranking Reversal Rate (%)")
        ax.set_title(DATASET_LABELS[dataset])
        ax.set_ylim(0, 100)
        ax.axhline(y=50, color="gray", linewidth=1, linestyle=":", alpha=0.5, label="Random (50%)")
        ax.legend(fontsize=9)

    fig.suptitle("Pairwise Ranking Reversal Frequency Across Prompt Variants",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig8_reversal_summary.png")
    plt.close()
    print("  Saved fig8_reversal_summary.png")

# ============================================================
# Figure 9: Summary Dashboard
# ============================================================

def fig9_summary_dashboard():
    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.35)

    # Panel A: Accuracy distribution (box-like)
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

    bp = ax1.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True)
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax1.set_xticks(positions)
    ax1.set_xticklabels(tick_labels, fontsize=7)
    ax1.set_ylabel("Accuracy")
    ax1.set_title("(A) Accuracy Distribution", fontsize=10)

    # Panel B: Variance ratio
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

    bars = ax2.bar(range(len(labels_vr)), ratios, color=colors_vr, alpha=0.75,
                   edgecolor="white")
    ax2.axhline(y=1, color="red", linestyle="--", linewidth=1, label="Var_prompt = Var_sampling")
    ax2.set_xticks(range(len(labels_vr)))
    ax2.set_xticklabels(labels_vr, fontsize=7)
    ax2.set_ylabel("Var_prompt / Var_sampling")
    ax2.set_title("(B) Variance Ratio", fontsize=10)
    ax2.set_yscale("log")
    ax2.legend(fontsize=7)

    # Panel C: Flip rate
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
    ax3.set_xticklabels(labels_fr, fontsize=7)
    ax3.set_ylabel("Item Flip Rate (%)")
    ax3.set_title("(C) Item Flip Rate", fontsize=10)

    # Panel D: OFAT effect magnitude
    ax4 = fig.add_subplot(gs[1, 0])
    dim_names = ["instruction", "answer_format", "option_format", "framing"]
    dim_short = ["Instruction", "Ans. Format", "Opt. Format", "Framing"]
    x_dim = np.arange(len(dim_names))
    w = 0.13
    i = 0
    for dataset in ["arc", "mmlu"]:
        data = load_analysis(dataset)
        for model in MODELS:
            if model not in data:
                continue
            max_deltas = []
            for dim in dim_names:
                effects = data[model]["ofat_effects"][dim]
                deltas = [abs(e[2]) for e in effects[1:]]
                max_deltas.append(max(deltas) if deltas else 0)
            hatch = "" if dataset == "arc" else "///"
            ax4.bar(x_dim + i * w, max_deltas, w * 0.9, color=COLORS[model],
                    alpha=0.7 if dataset == "arc" else 0.4, hatch=hatch,
                    edgecolor="white")
            i += 1

    ax4.set_xticks(x_dim + w * 2.5)
    ax4.set_xticklabels(dim_short, fontsize=8)
    ax4.set_ylabel("|Max Δ Accuracy|")
    ax4.set_title("(D) Max OFAT Effect by Dimension", fontsize=10)

    # Panel E: Reversal rates
    ax5 = fig.add_subplot(gs[1, 1])
    pair_labels_all = []
    rev_rates_all = []
    rev_colors_all = []
    dataset_markers = []
    pair_c = ["#4C72B0", "#DD8452", "#C44E52"]
    for dataset in ["arc", "mmlu"]:
        data = load_analysis(dataset)
        ranking = data.get("ranking", {})
        reversals = ranking.get("reversals_all", {})
        for pi, (pk, rv) in enumerate(reversals.items()):
            m_a, m_b = pk.split("_vs_")
            pair_labels_all.append(f"{MODEL_LABELS.get(m_a, m_a)[:5]}v{MODEL_LABELS.get(m_b, m_b)[:5]}\n({dataset[:3]})")
            rev_rates_all.append(rv["reversal_rate"] * 100)
            rev_colors_all.append(pair_c[pi % 3])

    ax5.bar(range(len(pair_labels_all)), rev_rates_all, color=rev_colors_all,
            alpha=0.75, edgecolor="white")
    ax5.axhline(y=50, color="gray", linestyle=":", alpha=0.5)
    ax5.set_xticks(range(len(pair_labels_all)))
    ax5.set_xticklabels(pair_labels_all, fontsize=6)
    ax5.set_ylabel("Reversal Rate (%)")
    ax5.set_title("(E) Ranking Reversals", fontsize=10)
    ax5.set_ylim(0, 100)

    # Panel F: Scale trend
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

    fig.suptitle("Experiment I: Prompt Perturbation — Overview",
                 fontsize=15, fontweight="bold", y=1.01)
    fig.savefig(OUTPUT_DIR / "fig9_summary_dashboard.png")
    plt.close()
    print("  Saved fig9_summary_dashboard.png")

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("Generating figures...")
    fig1_accuracy_per_variant()
    fig2_ofat_main_effects()
    fig3_variance_decomposition()
    fig4_ranking_stability()
    fig5_scale_analysis()
    fig6_noise_removal()
    fig7_category_heatmap()
    fig8_reversal_summary()
    fig9_summary_dashboard()
    print(f"\nAll figures saved to {OUTPUT_DIR}/")
