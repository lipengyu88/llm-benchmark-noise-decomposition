"""
Generate appendix figures for the report.

Outputs:
  fig_a1_ols_coefficients.png  - heatmap of main-effects OLS coefficients
                                 across all 8 model-benchmark regressions

Each figure is saved at 600 DPI.
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
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.labelweight": "bold",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
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

COEF_NAMES = [
    "instruction_L1", "instruction_L2",
    "answer_format_L1", "answer_format_L2",
    "option_format_L1", "option_format_L2",
    "framing_L1",
    "delimiter_L1", "delimiter_L2",
]
COEF_LABELS = [
    "instruction\nL1", "instruction\nL2",
    "answer\nfmt L1", "answer\nfmt L2",
    "option\nfmt L1", "option\nfmt L2",
    "framing\nL1",
    "delimiter\nL1", "delimiter\nL2",
]



def fig_a1_ols_coefficients():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    all_vals = []
    matrices = {}
    for ds_key in ["arc", "mmlu"]:
        data = json.loads(
            (ROOT / "exp1" / "analysis_exp1" / f"analysis_{ds_key}.json").read_text()
        )
        mat = np.zeros((len(MODELS), len(COEF_NAMES)))
        for i, m in enumerate(MODELS):
            coefs = data[m]["interaction_effects"]["coefficients"]
            for j, name in enumerate(COEF_NAMES):
                mat[i, j] = coefs.get(name, 0.0)
        matrices[ds_key] = mat
        all_vals.extend(mat.flatten().tolist())

    vmax = max(abs(min(all_vals)), abs(max(all_vals)))
    vmin = -vmax

    for ax_idx, ds_key in enumerate(["arc", "mmlu"]):
        ax = axes[ax_idx]
        mat = matrices[ds_key]

        im = ax.imshow(
            mat,
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
        )

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                color = "white" if abs(val) > 0.08 else "black"
                ax.text(
                    j, i, f"{val:+.2f}",
                    ha="center", va="center",
                    fontsize=8, color=color,
                    fontweight="bold" if abs(val) >= 0.05 else "normal",
                )

        ax.set_xticks(np.arange(len(COEF_NAMES)))
        ax.set_xticklabels(COEF_LABELS, fontsize=8, rotation=0)
        ax.set_yticks(np.arange(len(MODELS)))
        ax.set_yticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=9)
        ax.set_title(DATASETS[ds_key], fontsize=12)

        ax.set_xticks(np.arange(-0.5, len(COEF_NAMES), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(MODELS), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.5)
        ax.tick_params(which="minor", length=0)

    cbar = fig.colorbar(
        im,
        ax=axes,
        orientation="vertical",
        fraction=0.025,
        pad=0.02,
        shrink=0.85,
    )
    cbar.set_label("Coefficient ($\\Delta$ accuracy)", fontsize=10)

    fig.suptitle(
        "Main-effects OLS coefficients across the 8 model-benchmark regressions",
        fontsize=13, fontweight="bold", y=1.03,
    )

    fig.savefig(OUT / "fig_a1_ols_coefficients.png")
    plt.close(fig)
    print("  saved fig_a1_ols_coefficients.png")



def fig_a2_dim_variance_per_model():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    dim_keys = [
        "answer_format", "delimiter", "instruction",
        "framing", "option_format", "interaction",
    ]
    dim_labels = {
        "answer_format": "Answer format",
        "delimiter":     "Delimiter",
        "instruction":   "Instruction",
        "framing":       "Framing",
        "option_format": "Option format",
        "interaction":   "Interaction (residual)",
    }
    dim_colors = {
        "answer_format": "#C44E52",
        "delimiter":     "#DD8452",
        "instruction":   "#55A868",
        "framing":       "#8172B2",
        "option_format": "#937860",
        "interaction":   "#CCCCCC",
    }

    for ax_idx, ds_key in enumerate(["arc", "mmlu"]):
        ax = axes[ax_idx]
        data = json.loads(
            (ROOT / "exp1" / "analysis_exp1" / f"analysis_{ds_key}.json").read_text()
        )

        positions = np.arange(len(MODELS))
        bottoms = np.zeros(len(MODELS))

        for dim in dim_keys:
            vals = np.array([
                data[m]["dimension_variance"]["percentages"][dim]
                for m in MODELS
            ])
            ax.barh(
                positions, vals,
                left=bottoms, height=0.62,
                color=dim_colors[dim],
                edgecolor="white", linewidth=1.0,
                label=dim_labels[dim] if ax_idx == 1 else None,
            )
            for i, v in enumerate(vals):
                if v >= 6:
                    ax.text(
                        bottoms[i] + v / 2, positions[i],
                        f"{v:.0f}%",
                        ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold",
                    )
            bottoms += vals

        ax.set_yticks(positions)
        ax.set_yticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.set_xlabel("Share of total Var(prompt)", fontsize=10)
        ax.set_title(DATASETS[ds_key], fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="x", alpha=0.25)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=6,
        bbox_to_anchor=(0.5, -0.04),
        frameon=False,
        fontsize=9,
    )

    fig.suptitle(
        "Per-model dimension-level variance attribution",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig_a2_dim_variance_per_model.png")
    plt.close(fig)
    print("  saved fig_a2_dim_variance_per_model.png")



def fig_a3_surface_vs_full():
    """
    Paired bar chart comparing four sensitivity metrics
    (sigma, range, flip rate, var ratio) under the full 100-variant
    set vs the 68-variant surface-only subset (no with_explanation).
    """
    metrics = [
        ("std",       "Accuracy std $\\sigma$",        "accuracy_stats",        "std"),
        ("range",     "Accuracy range $\\Delta$",      "accuracy_stats",        "range"),
        ("flip",      "Item flip rate",                "item_flip_rate",         None),
        ("var_ratio", "Variance ratio (log scale)",    "variance_decomposition", "ratio"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(15, 6.6))

    for row_idx, ds_key in enumerate(["arc", "mmlu"]):
        data = json.loads(
            (ROOT / "exp1" / "analysis_exp1" / f"analysis_{ds_key}.json").read_text()
        )
        for col_idx, (mkey, mlabel, section, field) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            full_vals = []
            surf_vals = []
            for m in MODELS:
                full_section = data[m][section]
                surf_section = data[m][section + "_surface"] if section != "item_flip_rate" \
                               else data[m]["item_flip_rate_surface"]
                if field is None:
                    full_vals.append(data[m]["item_flip_rate"])
                    surf_vals.append(data[m]["item_flip_rate_surface"])
                else:
                    full_vals.append(full_section[field])
                    surf_vals.append(surf_section[field])

            positions = np.arange(len(MODELS))
            width = 0.36
            b1 = ax.bar(
                positions - width / 2, full_vals, width,
                color="#C44E52", alpha=0.85,
                label="Full (100 variants)",
                edgecolor="white", linewidth=0.8,
            )
            b2 = ax.bar(
                positions + width / 2, surf_vals, width,
                color="#4C72B0", alpha=0.85,
                label="Surface (68 variants)",
                edgecolor="white", linewidth=0.8,
            )

            for i, (full, surf) in enumerate(zip(full_vals, surf_vals)):
                if full > 1e-9:
                    pct_drop = (1 - surf / full) * 100
                    label = f"$-${pct_drop:.0f}\\%"
                    color = "#2A6F2A" if pct_drop > 0 else "#999999"
                    ax.text(
                        positions[i],
                        max(full, surf) * (1.18 if mkey != "var_ratio" else 1.4),
                        label,
                        ha="center", va="bottom",
                        fontsize=8, color=color, fontweight="bold",
                    )

            ax.set_xticks(positions)
            ax.set_xticklabels(
                [MODEL_LABELS[m].replace("LLaMA-3.1-", "LLaMA-")
                                 .replace("Qwen2.5-", "Q-")
                                 .replace("Qwen3-", "Q3-") for m in MODELS],
                fontsize=8, rotation=18, ha="right",
            )
            if mkey == "var_ratio":
                ax.set_yscale("log")
                ax.set_ylim(0.05, 80)
            else:
                ax.set_ylim(top=max(full_vals) * 1.45)

            if col_idx == 0:
                ax.set_ylabel(DATASETS[ds_key], fontsize=11, fontweight="bold")

            if row_idx == 0:
                ax.set_title(mlabel, fontsize=11)

            if row_idx == 0 and col_idx == 0:
                ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle(
        "Effect of removing the 32 with-explanation variants on prompt sensitivity",
        fontsize=13, fontweight="bold", y=1.00,
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig_a3_surface_vs_full.png")
    plt.close(fig)
    print("  saved fig_a3_surface_vs_full.png")



def fig_a4_cross_source():
    """
    Side-by-side comparison of paraphrase results from the two
    independent sources (GPT-4o and Qwen2.5-72B). For each model and
    benchmark we plot the per-version accuracy under each source.
    Identical conclusions across sources should produce nearly identical
    bar heights.
    """
    from collections import defaultdict

    MODELS_E2 = ["llama-3.1-8b", "qwen2.5-7b", "qwen3-32b", "qwen2.5-72b"]
    MODEL_LABELS_E2 = {
        "llama-3.1-8b": "LLaMA-3.1-8B",
        "qwen2.5-7b":   "Qwen2.5-7B",
        "qwen3-32b":    "Qwen3-32B",
        "qwen2.5-72b":  "Qwen2.5-72B",
    }
    DATASETS_BENCH = {"arc": "arc_challenge", "mmlu": "mmlu_pro"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    for ax_idx, (ds_key, bench) in enumerate(DATASETS_BENCH.items()):
        ax = axes[ax_idx]

        bars_per_model = {}
        for src in ["gpt4o", "qwen"]:
            for m in MODELS_E2:
                path = ROOT / "exp2" / f"exp2_{bench}_{m}_{src}.json"
                if not path.exists():
                    continue
                data = json.loads(path.read_text())
                by_version = defaultdict(list)
                for r in data:
                    if r.get("is_correct") is not None:
                        by_version[r["version"]].append(int(r["is_correct"]))
                v_accs = [np.mean(by_version[v]) for v in sorted(by_version.keys())]
                bars_per_model.setdefault(m, {})[src] = v_accs

        positions = np.arange(len(MODELS_E2))
        group_width = 0.86
        n_versions = 4
        n_sources = 2
        n_bars = n_versions * n_sources
        bar_w = group_width / n_bars

        for m_idx, m in enumerate(MODELS_E2):
            if m not in bars_per_model:
                continue
            base_x = positions[m_idx] - group_width / 2 + bar_w / 2
            for v in range(n_versions):
                gpt_acc = bars_per_model[m]["gpt4o"][v] if v < len(bars_per_model[m].get("gpt4o", [])) else 0
                qwen_acc = bars_per_model[m]["qwen"][v] if v < len(bars_per_model[m].get("qwen", [])) else 0
                ax.bar(
                    base_x + (2 * v) * bar_w, gpt_acc,
                    bar_w * 0.92, color="#4C72B0",
                    edgecolor="white", linewidth=0.4,
                    alpha=0.85,
                )
                ax.bar(
                    base_x + (2 * v + 1) * bar_w, qwen_acc,
                    bar_w * 0.92, color="#DD8452",
                    edgecolor="white", linewidth=0.4,
                    alpha=0.85,
                )

        for m_idx, m in enumerate(MODELS_E2):
            all_vals = bars_per_model[m]["gpt4o"] + bars_per_model[m]["qwen"]
            mean_acc = float(np.mean(all_vals))
            ax.text(
                positions[m_idx], mean_acc + 0.02,
                f"{mean_acc:.3f}",
                ha="center", va="bottom",
                fontsize=8, fontweight="bold", color="#333",
            )

        ax.set_xticks(positions)
        ax.set_xticklabels(
            [MODEL_LABELS_E2[m] for m in MODELS_E2],
            fontsize=9, rotation=12, ha="right",
        )
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_title(DATASETS[ds_key], fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", alpha=0.25)

        if ax_idx == 0:
            handles = [
                mpatches.Patch(color="#4C72B0", alpha=0.85, label="GPT-4o paraphrase source"),
                mpatches.Patch(color="#DD8452", alpha=0.85, label="Qwen2.5-72B paraphrase source"),
            ]
            ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.9)

    fig.suptitle(
        "Cross-source comparison of paraphrase resampling (4 versions per model per source)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig_a4_cross_source.png")
    plt.close(fig)
    print("  saved fig_a4_cross_source.png")



def fig_a5_noise_correlation():
    """
    4x4 noise correlation heatmap per benchmark.
    Diagonals are 1 by definition; off-diagonals are the Pearson r
    of per-question noise scores between two models.
    """
    MODELS_E2 = ["llama-3.1-8b", "qwen2.5-7b", "qwen3-32b", "qwen2.5-72b"]
    MODEL_LABELS_SHORT = {
        "llama-3.1-8b": "LLaMA-8B",
        "qwen2.5-7b":   "Qwen-7B",
        "qwen3-32b":    "Qwen3-32B",
        "qwen2.5-72b":  "Qwen-72B",
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    for ax_idx, ds_key in enumerate(["arc", "mmlu"]):
        ax = axes[ax_idx]
        d = json.loads(
            (ROOT / "exp3" / "analysis_exp3" / f"analysis_{ds_key}_shared150.json").read_text()
        )
        nc = d["noise_correlation"]

        n = len(MODELS_E2)
        mat = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                k1 = f"{MODELS_E2[i]}_vs_{MODELS_E2[j]}"
                k2 = f"{MODELS_E2[j]}_vs_{MODELS_E2[i]}"
                r = nc.get(k1, nc.get(k2, 0.0))
                mat[i, j] = r
                mat[j, i] = r

        im = ax.imshow(
            mat, cmap="RdYlBu_r",
            vmin=0.0, vmax=1.0,
            aspect="equal",
        )

        for i in range(n):
            for j in range(n):
                v = mat[i, j]
                color = "white" if (v > 0.65 or v < 0.10) else "black"
                txt = f"{v:.2f}" if i != j else "—"
                ax.text(
                    j, i, txt,
                    ha="center", va="center",
                    fontsize=11, color=color,
                    fontweight="bold" if i != j else "normal",
                )

        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels([MODEL_LABELS_SHORT[m] for m in MODELS_E2],
                           fontsize=9, rotation=18, ha="right")
        ax.set_yticklabels([MODEL_LABELS_SHORT[m] for m in MODELS_E2],
                           fontsize=9)
        ax.set_title(DATASETS[ds_key], fontsize=12)

        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="minor", length=0)

    cbar = fig.colorbar(
        im, ax=axes,
        orientation="vertical",
        fraction=0.04, pad=0.03, shrink=0.85,
    )
    cbar.set_label("Pearson $r$", fontsize=10)

    fig.suptitle(
        "Pairwise correlation of per-question noise scores between models",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.savefig(OUT / "fig_a5_noise_correlation.png")
    plt.close(fig)
    print("  saved fig_a5_noise_correlation.png")



def fig_a6_category_sensitivity():
    """
    14 categories x 4 models heatmap of per-category accuracy range
    across the 100 prompt variants. Categories are ordered by mean
    range across the four models (most sensitive at top).
    """
    d = json.loads(
        (ROOT / "exp1" / "analysis_exp1" / "analysis_mmlu.json").read_text()
    )

    cats = set()
    for m in MODELS:
        for c in d[m].get("category_analysis", {}):
            cats.add(c)
    cats = sorted(cats)

    range_mat = np.full((len(cats), len(MODELS)), np.nan)
    n_q_per_cat = {}
    for j, m in enumerate(MODELS):
        ca = d[m].get("category_analysis", {})
        for i, c in enumerate(cats):
            if c in ca:
                range_mat[i, j] = ca[c]["range"]
                n_q_per_cat[c] = ca[c]["n_questions"]

    mean_ranges = np.nanmean(range_mat, axis=1)
    sort_idx = np.argsort(mean_ranges)[::-1]
    range_mat = range_mat[sort_idx]
    cats_sorted = [cats[i] for i in sort_idx]
    cat_labels = [
        f"{c} (n={n_q_per_cat.get(c, '?')})"
        for c in cats_sorted
    ]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(
        range_mat, cmap="YlOrRd",
        vmin=0, vmax=0.8,
        aspect="auto",
    )

    for i in range(range_mat.shape[0]):
        for j in range(range_mat.shape[1]):
            v = range_mat[i, j]
            if not np.isnan(v):
                color = "white" if v > 0.45 else "black"
                ax.text(
                    j, i, f"{v:.2f}",
                    ha="center", va="center",
                    fontsize=9, color=color,
                    fontweight="bold" if v >= 0.6 else "normal",
                )

    ax.set_xticks(np.arange(len(MODELS)))
    ax.set_xticklabels(
        [MODEL_LABELS[m] for m in MODELS],
        fontsize=10, rotation=20, ha="right",
    )
    ax.set_yticks(np.arange(len(cats_sorted)))
    ax.set_yticklabels(cat_labels, fontsize=9)

    ax.set_xticks(np.arange(-0.5, len(MODELS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(cats_sorted), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03, shrink=0.85)
    cbar.set_label(
        "Accuracy range across 100 prompt variants",
        fontsize=10,
    )

    ax.set_title(
        "MMLU-Pro per-category prompt sensitivity",
        fontsize=12, fontweight="bold", pad=10,
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig_a6_category_sensitivity.png")
    plt.close(fig)
    print("  saved fig_a6_category_sensitivity.png")



def fig_a7_bt_posterior():
    """
    Top row: forest plot of BT log-strengths with 95% bootstrap CIs,
             one panel per benchmark.
    Bottom row: 4x4 rank posterior heatmap (rows = models, cols = ranks),
                one panel per benchmark.
    """
    MODELS_E2 = ["llama-3.1-8b", "qwen2.5-7b", "qwen3-32b", "qwen2.5-72b"]
    MODEL_LABELS_E2 = {
        "llama-3.1-8b": "LLaMA-3.1-8B",
        "qwen2.5-7b":   "Qwen2.5-7B",
        "qwen3-32b":    "Qwen3-32B",
        "qwen2.5-72b":  "Qwen2.5-72B",
    }
    MC = {
        "llama-3.1-8b": "#4C72B0",
        "qwen2.5-7b":   "#C44E52",
        "qwen3-32b":    "#DD8452",
        "qwen2.5-72b":  "#55A868",
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 6.5))

    for col_idx, ds_key in enumerate(["arc", "mmlu"]):
        d = json.loads(
            (ROOT / "exp4" / f"bt_results_{ds_key}.json").read_text()
        )
        models = d["models"]
        log_r = d["log_ratings"]
        ci_lo = d["bootstrap_ci_low"]
        ci_hi = d["bootstrap_ci_high"]
        rp = d["rank_posterior"]

        order = sorted(range(len(models)), key=lambda i: -log_r[i])
        models_o = [models[i] for i in order]
        log_o = [log_r[i] for i in order]
        lo_o = [ci_lo[i] for i in order]
        hi_o = [ci_hi[i] for i in order]
        rp_o = [rp[i] for i in order]

        ax = axes[0, col_idx]
        y = np.arange(len(models_o))
        for i, m in enumerate(models_o):
            ax.errorbar(
                log_o[i], y[i],
                xerr=[[log_o[i] - lo_o[i]], [hi_o[i] - log_o[i]]],
                fmt="o",
                color=MC[m],
                markersize=10,
                markeredgecolor="white",
                markeredgewidth=1.0,
                ecolor=MC[m],
                elinewidth=2.0,
                capsize=5,
                capthick=1.5,
            )
            ax.text(
                hi_o[i] + 0.06, y[i],
                f"{log_o[i]:.2f}",
                va="center", ha="left",
                fontsize=9, color=MC[m], fontweight="bold",
            )
        ax.set_yticks(y)
        ax.set_yticklabels([MODEL_LABELS_E2[m] for m in models_o], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Bradley Terry log-strength $\\lambda$", fontsize=10)
        ax.set_title(DATASETS[ds_key], fontsize=12)
        ax.grid(True, axis="x", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        x_min = min(lo_o) - 0.1
        x_max = max(hi_o) + 0.5
        ax.set_xlim(x_min, x_max)

        ax = axes[1, col_idx]
        rp_mat = np.array(rp_o)
        im = ax.imshow(
            rp_mat, cmap="Purples",
            vmin=0, vmax=1, aspect="auto",
        )
        for i in range(rp_mat.shape[0]):
            for j in range(rp_mat.shape[1]):
                v = rp_mat[i, j]
                color = "white" if v > 0.55 else "black"
                if v >= 0.001:
                    label = f"{v:.3f}" if v < 0.999 else "1.000"
                else:
                    label = "0"
                ax.text(
                    j, i, label,
                    ha="center", va="center",
                    fontsize=9, color=color,
                    fontweight="bold" if v > 0.5 else "normal",
                )
        ax.set_xticks(range(rp_mat.shape[1]))
        ax.set_xticklabels([f"Rank {k+1}" for k in range(rp_mat.shape[1])],
                           fontsize=9)
        ax.set_yticks(range(rp_mat.shape[0]))
        ax.set_yticklabels([MODEL_LABELS_E2[m] for m in models_o], fontsize=9)
        ax.set_xticks(np.arange(-0.5, rp_mat.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, rp_mat.shape[0], 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.5)
        ax.tick_params(which="minor", length=0)

    fig.suptitle(
        "Bradley Terry log-strengths with 95\\% bootstrap CIs (top) and rank posterior (bottom)",
        fontsize=12, fontweight="bold", y=1.00,
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig_a7_bt_posterior.png")
    plt.close(fig)
    print("  saved fig_a7_bt_posterior.png")



def fig_a8_threshold_sweep():
    """
    Top row: per-model accuracy std vs removal threshold (one panel
             per benchmark).
    Bottom row: per-pair reversal rate vs removal threshold (one panel
                per benchmark).
    """
    thresholds = [0, 10, 20, 30]
    threshold_keys = ["baseline", "remove_10pct", "remove_20pct", "remove_30pct"]

    MODELS_E1_TO_E2 = {
        "llama":   "llama-3.1-8b",
        "qwen7b":  "qwen2.5-7b",
        "qwen32b": "qwen3-32b",
        "qwen72b": "qwen2.5-72b",
    }

    fig, axes = plt.subplots(2, 2, figsize=(13, 7.0))

    for col_idx, ds_key in enumerate(["arc", "mmlu"]):
        d = json.loads(
            (ROOT / "exp3" / "analysis_exp3" / f"analysis_{ds_key}_shared150.json").read_text()
        )
        tr = d["threshold_results"]

        ax = axes[0, col_idx]
        for m in MODELS:
            stds = []
            for tk in threshold_keys:
                stds.append(tr[tk]["exp1"][m]["accuracy"]["std"])
            ax.plot(
                thresholds, stds,
                marker="o", linewidth=2, markersize=8,
                color=COLORS[m],
                label=MODEL_LABELS[m],
                markeredgecolor="white", markeredgewidth=0.8,
            )
        ax.set_xticks(thresholds)
        ax.set_xticklabels([f"{t}\\%" for t in thresholds])
        ax.set_xlabel("Noisiest items removed", fontsize=10)
        ax.set_ylabel("Accuracy std $\\sigma$", fontsize=10)
        ax.set_title(f"{DATASETS[ds_key]} -- accuracy std", fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.25)
        if col_idx == 0:
            ax.legend(loc="best", fontsize=8, framealpha=0.9, ncol=2)

        ax = axes[1, col_idx]
        pair_styles = {
            "qwen32b_vs_qwen72b": ("Qwen3-32B vs Qwen2.5-72B", "#C44E52", "o", "-",  2.4),
            "llama_vs_qwen7b":    ("LLaMA vs Qwen2.5-7B",     "#4C72B0", "s", "--", 1.6),
            "llama_vs_qwen72b":   ("LLaMA vs Qwen2.5-72B",    "#55A868", "^", "--", 1.6),
            "qwen7b_vs_qwen72b":  ("Qwen2.5-7B vs Qwen2.5-72B", "#DD8452", "D", "--", 1.6),
        }
        for pair_key, (label, color, marker, ls, lw) in pair_styles.items():
            rates = []
            for tk in threshold_keys:
                rev = tr[tk].get("exp1_ranking", {}).get("reversals", {})
                rates.append(rev.get(pair_key, {}).get("reversal_rate", 0) * 100)
            ax.plot(
                thresholds, rates,
                marker=marker, linewidth=lw, linestyle=ls,
                markersize=8, color=color, alpha=0.95,
                label=label,
                markeredgecolor="white", markeredgewidth=0.8,
            )
        ax.set_xticks(thresholds)
        ax.set_xticklabels([f"{t}\\%" for t in thresholds])
        ax.set_xlabel("Noisiest items removed", fontsize=10)
        ax.set_ylabel("Pairwise reversal rate (\\%)", fontsize=10)
        ax.set_title(f"{DATASETS[ds_key]} -- reversal rate", fontsize=11)
        ax.set_ylim(-3, 75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.25)
        if col_idx == 0:
            ax.legend(loc="upper right", fontsize=7, framealpha=0.9)

    fig.suptitle(
        "Effect of noise-item removal on per-model std (top) and pairwise reversal rate (bottom)",
        fontsize=12, fontweight="bold", y=1.00,
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig_a8_threshold_sweep.png")
    plt.close(fig)
    print("  saved fig_a8_threshold_sweep.png")



def fig_a10_tarr_distribution():
    """
    Per-question TARr@5 distribution shown as a stacked bar of 4 buckets
    (perfect, high, medium, low) for each (model, benchmark) cell.
    Highlights how MMLU-Pro has a much longer tail of unstable items
    compared to ARC.
    """
    from collections import defaultdict
    from itertools import combinations as combn

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    bucket_edges = [(1.0, 1.001, "Perfect (TARr=1.0)"),
                    (0.7, 1.0,   "High (0.7--1.0)"),
                    (0.5, 0.7,   "Medium (0.5--0.7)"),
                    (0.0, 0.5,   "Low (<0.5)")]
    bucket_colors = ["#55A868", "#85C68A", "#F1C470", "#C44E52"]

    for ax_idx, ds_key in enumerate(["arc", "mmlu"]):
        ax = axes[ax_idx]
        d = json.loads(
            (ROOT / "exp5" / "results_exp5" / f"stability_{ds_key}.json").read_text()
        )
        groups = defaultdict(list)
        for r in d:
            groups[(r["qid"], r["model"])].append(r)

        per_model = {m: [] for m in MODELS}
        for (qid, m), trials in groups.items():
            trials.sort(key=lambda x: x["repeat"])
            parsed = [t["parsed_answer"] for t in trials]
            pairs = list(combn(range(5), 2))
            agree = sum(1 for i, j in pairs if parsed[i] == parsed[j])
            tar = agree / len(pairs)
            per_model[m].append(tar)

        positions = np.arange(len(MODELS))
        width = 0.62
        bottoms = np.zeros(len(MODELS))

        for (lo, hi, label), color in zip(bucket_edges, bucket_colors):
            counts = []
            for m in MODELS:
                arr = np.array(per_model[m])
                if hi == 1.001:
                    cnt = int(np.sum(arr == 1.0))
                else:
                    cnt = int(np.sum((arr >= lo) & (arr < hi)))
                counts.append(cnt)
            counts = np.array(counts)
            ax.bar(
                positions, counts, width=width, bottom=bottoms,
                color=color, edgecolor="white", linewidth=1.0,
                label=label,
            )
            for i, c in enumerate(counts):
                if c >= 3:
                    ax.text(
                        positions[i], bottoms[i] + c / 2,
                        str(c),
                        ha="center", va="center",
                        fontsize=9, color="white",
                        fontweight="bold",
                    )
            bottoms += counts

        ax.set_xticks(positions)
        ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS],
                           fontsize=9, rotation=12, ha="right")
        ax.set_ylabel("Number of questions (out of 50)", fontsize=10)
        ax.set_title(DATASETS[ds_key], fontsize=12)
        ax.set_ylim(0, 55)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", alpha=0.25)

        if ax_idx == 0:
            ax.legend(
                loc="upper left",
                fontsize=8, framealpha=0.95,
                bbox_to_anchor=(0.0, 1.0),
            )

    fig.suptitle(
        "Per-question TARr@5 distribution under repeated runs at temperature 0",
        fontsize=12, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig_a10_tarr_distribution.png")
    plt.close(fig)
    print("  saved fig_a10_tarr_distribution.png")



if __name__ == "__main__":
    print("Generating appendix figures...")
    fig_a1_ols_coefficients()
    fig_a2_dim_variance_per_model()
    fig_a3_surface_vs_full()
    fig_a4_cross_source()
    fig_a5_noise_correlation()
    fig_a6_category_sensitivity()
    fig_a7_bt_posterior()
    fig_a8_threshold_sweep()
    fig_a10_tarr_distribution()
    print("Done.")
