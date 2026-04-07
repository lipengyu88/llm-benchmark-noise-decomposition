"""
Experiment I-Extended: 100 Prompt Variants — Analysis

Adapted from exp1/analyze_experiment1.py for the extended 100-variant,
150-question design with 5 dimensions (instruction, answer_format,
option_format, framing, delimiter).

Metrics:
  Accuracy-level: Mean/Std, Max-Min Range, Item Flip Rate
  Ranking-level: Pairwise Gap Stability, Reversal Frequency, Rank Distribution
  Variance decomposition: Var_prompt vs Var_sampling
  Dimension-level variance attribution (5 dimensions)
  Interaction effects: main-effects OLS regression
  Category-level analysis: MMLU-Pro sensitivity by subject category
  Noise analysis: per-item noise score, removal impact
  Scale analysis: robustness trend across model sizes

Usage:
    cd exp1
    python analyze_experiment1_extended.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations

from prompt_variants import (
    get_all_variants, get_ofat_variants, describe_variant,
    INSTRUCTIONS, ANSWER_FORMATS, OPTION_FORMATS, FRAMINGS, DELIMITERS,
    N_LEVELS,
)

RESULTS_DIR = Path("results_exp1")
OUTPUT_DIR = Path("analysis_exp1")
OUTPUT_DIR.mkdir(exist_ok=True)

MODELS = ["llama", "qwen7b", "qwen32b", "qwen72b"]
MODEL_LABELS = {
    "llama":   "Llama-3.1-8B",
    "qwen7b":  "Qwen2.5-7B",
    "qwen32b": "Qwen3-32B",
    "qwen72b": "Qwen2.5-72B",
}
MODEL_SIZES = {"llama": 8, "qwen7b": 7, "qwen32b": 32, "qwen72b": 72}
DATASETS = ["arc", "mmlu"]
DATASET_LABELS = {"arc": "ARC-Challenge", "mmlu": "MMLU-Pro"}

N_BOOTSTRAP = 10000
RNG = np.random.RandomState(42)

# 5 dimensions for the extended design
DIM_NAMES = ["instruction", "answer_format", "option_format", "framing", "delimiter"]
DIM_LABELS = ["Instruction", "Answer Format", "Option Format", "Framing", "Delimiter"]

# ============================================================
# Data loading
# ============================================================

def load_results(model, dataset):
    path = RESULTS_DIR / f"results_{model}_{dataset}.json"
    if not path.exists():
        path = RESULTS_DIR / f"checkpoint_{model}_{dataset}.json"
    with open(path) as f:
        return json.load(f)


def load_dataset_items(dataset):
    """Load raw dataset items for category analysis."""
    base = Path(__file__).parent.parent / "exp1"
    files = {"arc": base / "arc_challenge_300.json", "mmlu": base / "mmlu_pro_300.json"}
    items = {}
    with open(files[dataset]) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if dataset == "arc":
                items[item["id"]] = item
            else:
                items[str(item["question_id"])] = item
    return items


def build_matrix(model, dataset, variants):
    results = load_results(model, dataset)
    variant_ids = [v[0] for v in variants]
    qids = sorted(results.keys())
    matrix = np.full((len(qids), len(variant_ids)), np.nan)
    for i, qid in enumerate(qids):
        for j, vid in enumerate(variant_ids):
            if vid in results[qid] and results[qid][vid]["is_correct"] is not None:
                matrix[i, j] = results[qid][vid]["is_correct"]
    return qids, variant_ids, matrix


# ============================================================
# Accuracy-level metrics
# ============================================================

def accuracy_per_variant(matrix):
    return np.nanmean(matrix, axis=0)


def accuracy_stats(matrix):
    accs = accuracy_per_variant(matrix)
    return {
        "mean": float(np.mean(accs)),
        "std": float(np.std(accs)),
        "max": float(np.max(accs)),
        "min": float(np.min(accs)),
        "range": float(np.max(accs) - np.min(accs)),
        "per_variant": accs.tolist(),
    }


def item_flip_rate(matrix):
    n_questions = matrix.shape[0]
    flip_count = 0
    noise_scores = []
    for i in range(n_questions):
        row = matrix[i, ~np.isnan(matrix[i])]
        n = len(row)
        if n == 0:
            noise_scores.append(0.0)
            continue
        c = np.sum(row)
        noise = 1.0 - abs(2 * c - n) / n
        noise_scores.append(float(noise))
        if np.min(row) != np.max(row):
            flip_count += 1
    return {
        "overall_flip_rate": flip_count / n_questions,
        "noise_scores": noise_scores,
    }


# ============================================================
# OFAT main effect analysis (5 dimensions)
# ============================================================

def ofat_main_effects(matrix, variants):
    """Compute OFAT main effects for all 5 dimensions."""
    variant_ids = [v[0] for v in variants]
    base_acc = float(np.nanmean(matrix[:, 0]))

    # Map OFAT variant IDs to dimensions
    # OFAT order: instruction(2), answer_format(2), option_format(2), framing(1), delimiter(2) = 9
    ofat_groups = {
        "instruction":   ["ofat_1", "ofat_2"],
        "answer_format": ["ofat_3", "ofat_4"],
        "option_format": ["ofat_5", "ofat_6"],
        "framing":       ["ofat_7"],
        "delimiter":     ["ofat_8", "ofat_9"],
    }

    effects = {}
    for dim, ofat_ids in ofat_groups.items():
        effects[dim] = [("base", base_acc, 0.0)]
        for oid in ofat_ids:
            if oid in variant_ids:
                idx = variant_ids.index(oid)
                acc = float(np.nanmean(matrix[:, idx]))
                effects[dim].append((oid, acc, acc - base_acc))
    return effects


# ============================================================
# Interaction effect analysis (5 dimensions)
# ============================================================

def interaction_analysis(matrix, variants):
    """
    Main-effects OLS regression with dummy-coded 5 dimensions.
    Parameters: 1 intercept + sum(n_levels_d - 1) = 1 + 2+2+2+1+2 = 10.
    Observations: 100 variants.
    """
    variant_indices = [v[1] for v in variants]
    accs = accuracy_per_variant(matrix)
    n = len(accs)

    n_levels = list(N_LEVELS)  # (3, 3, 3, 2, 3)

    # Build design matrix with dummy coding (drop level 0 as reference)
    X_main_cols = []
    main_names = []
    raw = np.array(variant_indices, dtype=int)  # (n_variants, 5)
    for d in range(5):
        for lvl in range(1, n_levels[d]):
            col = (raw[:, d] == lvl).astype(float)
            X_main_cols.append(col)
            main_names.append(f"{DIM_NAMES[d]}_L{lvl}")

    X_main = np.column_stack(X_main_cols)
    X_main_full = np.column_stack([np.ones(n), X_main])
    main_feature_names = ["intercept"] + main_names

    try:
        beta_main, _, _, _ = np.linalg.lstsq(X_main_full, accs, rcond=None)
    except np.linalg.LinAlgError:
        return {"error": "lstsq failed"}

    y_pred_main = X_main_full @ beta_main
    ss_res_main = np.sum((accs - y_pred_main) ** 2)
    ss_tot = np.sum((accs - np.mean(accs)) ** 2)
    r2_main = 1 - ss_res_main / ss_tot if ss_tot > 0 else 0.0

    p_main = len(main_feature_names)
    r2_adj = 1 - (1 - r2_main) * (n - 1) / (n - p_main) if n > p_main else 0.0

    main_coefficients = {name: float(b) for name, b in zip(main_feature_names, beta_main)}
    notable_main = {k: v for k, v in main_coefficients.items()
                    if k != "intercept" and abs(v) > 0.01}

    # Interaction model (main + 2-way interactions)
    dim_col_ranges = []
    offset = 0
    for d in range(5):
        k = n_levels[d] - 1
        dim_col_ranges.append(list(range(offset, offset + k)))
        offset += k

    interaction_names = []
    interaction_cols = []
    for di, dj in combinations(range(5), 2):
        for ci in dim_col_ranges[di]:
            for cj in dim_col_ranges[dj]:
                interaction_names.append(f"{main_names[ci]}*{main_names[cj]}")
                interaction_cols.append(X_main[:, ci] * X_main[:, cj])

    X_interact = np.column_stack(interaction_cols) if interaction_cols else np.empty((n, 0))
    X_full = np.column_stack([np.ones(n), X_main, X_interact])
    full_feature_names = ["intercept"] + main_names + interaction_names

    try:
        beta_full, _, _, _ = np.linalg.lstsq(X_full, accs, rcond=None)
    except np.linalg.LinAlgError:
        beta_full = np.zeros(len(full_feature_names))

    y_pred_full = X_full @ beta_full
    ss_res_full = np.sum((accs - y_pred_full) ** 2)
    r2_full = 1 - ss_res_full / ss_tot if ss_tot > 0 else 0.0
    p_full = len(full_feature_names)
    r2_adj_full = 1 - (1 - r2_full) * (n - 1) / (n - p_full) if n > p_full else 0.0

    full_coefficients = {name: float(b) for name, b in zip(full_feature_names, beta_full)}

    interaction_ss_pct = ss_res_main / ss_tot * 100 if ss_tot > 0 else 0.0

    return {
        "coefficients": main_coefficients,
        "r_squared": float(r2_main),
        "r_squared_adj": float(r2_adj),
        "r_squared_full": float(r2_full),
        "r_squared_adj_full": float(r2_adj_full),
        "n_observations": n,
        "n_parameters_main": p_main,
        "n_parameters_full": p_full,
        "notable_effects": notable_main,
        "interaction_pct_of_variance": float(interaction_ss_pct),
        "interaction_coefficients": {k: v for k, v in full_coefficients.items()
                                     if "*" in k and abs(v) > 0.01},
        "note": (f"Main model: {p_main} params / {n} obs. "
                 f"Full model: {p_full} params / {n} obs."),
    }


# ============================================================
# Variance decomposition
# ============================================================

def variance_decomposition(matrix, n_bootstrap=N_BOOTSTRAP):
    n_questions, n_variants = matrix.shape
    acc_per_variant = np.nanmean(matrix, axis=0)
    var_prompt = float(np.var(acc_per_variant))

    var_sampling_per_variant = []
    for v in range(n_variants):
        col = matrix[:, v]
        col = col[~np.isnan(col)]
        if len(col) == 0:
            continue
        idx = RNG.randint(0, len(col), size=(n_bootstrap, len(col)))
        boot_accs = col[idx].mean(axis=1)
        var_sampling_per_variant.append(float(np.var(boot_accs)))

    var_sampling = float(np.mean(var_sampling_per_variant))
    ratio = var_prompt / var_sampling if var_sampling > 0 else float('inf')

    return {
        "var_prompt": var_prompt,
        "var_sampling": var_sampling,
        "ratio": ratio,
        "interpretation": (
            "prompt variation dominates" if ratio > 2 else
            "comparable" if ratio > 0.5 else
            "sampling noise dominates"
        ),
    }


# ============================================================
# Dimension-level variance decomposition (5 dimensions)
# ============================================================

def dimension_variance_decomposition(matrix, variants):
    """Decompose Var_prompt into per-dimension contributions.

    For each dimension, group variants by their level on that dimension,
    compute group-mean accuracy, then compute the variance of those group means.
    """
    variant_indices = np.array([v[1] for v in variants], dtype=int)
    accs_all = np.nanmean(matrix, axis=0)  # (n_variants,)
    var_total = float(np.var(accs_all))

    dim_vars = {}
    for d, dim_name in enumerate(DIM_NAMES):
        group_means = []
        for lvl in range(N_LEVELS[d]):
            mask = variant_indices[:, d] == lvl
            if mask.sum() > 0:
                group_means.append(float(np.mean(accs_all[mask])))
        dim_vars[dim_name] = float(np.var(group_means)) if len(group_means) > 1 else 0.0

    var_explained = sum(dim_vars.values())
    var_inter = max(0.0, var_total - var_explained)
    dim_vars["interaction"] = var_inter
    dim_vars["total"] = var_total

    dim_pcts = {}
    for k, v in dim_vars.items():
        if k != "total":
            dim_pcts[k] = v / var_total * 100 if var_total > 0 else 0.0

    return {"variances": dim_vars, "percentages": dim_pcts}


# ============================================================
# Ranking-level metrics
# ============================================================

def pairwise_gap_bootstrap(matrices, model_names, n_bootstrap=N_BOOTSTRAP):
    pairs = list(combinations(model_names, 2))
    n_variants = list(matrices.values())[0].shape[1]
    results = {}

    for m_a, m_b in pairs:
        per_variant_gaps = []
        per_variant_cis = []

        for v in range(n_variants):
            col_a = matrices[m_a][:, v]
            col_b = matrices[m_b][:, v]
            gap_obs = float(np.nanmean(col_a) - np.nanmean(col_b))
            per_variant_gaps.append(gap_obs)

            valid_mask = ~(np.isnan(col_a) | np.isnan(col_b))
            a_valid = col_a[valid_mask]
            b_valid = col_b[valid_mask]
            n_valid = len(a_valid)
            if n_valid == 0:
                per_variant_cis.append((0.0, 0.0))
                continue
            idx = RNG.randint(0, n_valid, size=(n_bootstrap, n_valid))
            boot_gaps = a_valid[idx].mean(axis=1) - b_valid[idx].mean(axis=1)
            ci_low, ci_high = np.percentile(boot_gaps, [2.5, 97.5])
            per_variant_cis.append((float(ci_low), float(ci_high)))

        gap_arr = np.array(per_variant_gaps)
        avg_boot_std = float(np.mean([(hi - lo) / 3.92 for lo, hi in per_variant_cis]))
        cross_prompt_std = float(np.std(gap_arr))
        n_ci_contains_zero = sum(1 for lo, hi in per_variant_cis if lo <= 0 <= hi)
        mean_ci_width = float(np.mean([hi - lo for lo, hi in per_variant_cis]))

        results[(m_a, m_b)] = {
            "mean_gap": float(np.mean(gap_arr)),
            "std_gap_across_prompts": cross_prompt_std,
            "min_gap": float(np.min(gap_arr)),
            "max_gap": float(np.max(gap_arr)),
            "per_variant_gaps": [float(g) for g in per_variant_gaps],
            "per_variant_cis": per_variant_cis,
            "n_ci_contains_zero": n_ci_contains_zero,
            "mean_ci_width": mean_ci_width,
            "avg_boot_std": avg_boot_std,
            "cross_prompt_std": cross_prompt_std,
        }

    return results


def reversal_frequency(matrices, model_names):
    n_variants = list(matrices.values())[0].shape[1]
    acc_per_variant = {m: accuracy_per_variant(matrices[m]) for m in model_names}
    pairs = list(combinations(model_names, 2))
    pairwise_reversals = {}
    for m_a, m_b in pairs:
        mean_a = np.mean(acc_per_variant[m_a])
        mean_b = np.mean(acc_per_variant[m_b])
        expected_better = m_a if mean_a >= mean_b else m_b
        expected_worse = m_b if expected_better == m_a else m_a
        reversals = sum(
            1 for v in range(n_variants)
            if acc_per_variant[expected_worse][v] > acc_per_variant[expected_better][v]
        )
        pairwise_reversals[(m_a, m_b)] = {
            "reversal_count": reversals,
            "reversal_rate": reversals / n_variants,
            "expected_order": f"{MODEL_LABELS[expected_better]} > {MODEL_LABELS[expected_worse]}",
        }
    return pairwise_reversals


def rank_distribution_bootstrap(matrices, model_names, n_bootstrap=5000):
    """Bootstrap rank distribution — reduced n_bootstrap for 100 variants."""
    n_questions = list(matrices.values())[0].shape[0]
    n_variants = list(matrices.values())[0].shape[1]
    n_models = len(model_names)

    per_variant_rank_dist = {m: np.zeros((n_variants, n_models)) for m in model_names}

    for v in range(n_variants):
        rank_counts = {m: np.zeros(n_models) for m in model_names}
        all_idx = RNG.randint(0, n_questions, size=(n_bootstrap, n_questions))
        cols = {m: matrices[m][:, v] for m in model_names}
        for b in range(n_bootstrap):
            idx = all_idx[b]
            accs = {m: np.nanmean(cols[m][idx]) for m in model_names}
            sorted_models = sorted(model_names, key=lambda m: accs[m], reverse=True)
            for rank, m in enumerate(sorted_models):
                rank_counts[m][rank] += 1
        for m in model_names:
            per_variant_rank_dist[m][v, :] = rank_counts[m] / n_bootstrap

    avg_rank_dist = {m: per_variant_rank_dist[m].mean(axis=0).tolist() for m in model_names}
    rank_dist_std = {m: per_variant_rank_dist[m].std(axis=0).tolist() for m in model_names}
    return {"mean": avg_rank_dist, "std_across_prompts": rank_dist_std}


# ============================================================
# MMLU-Pro category-level analysis
# ============================================================

def category_analysis(matrix, qids, dataset):
    if dataset != "mmlu":
        return None
    items = load_dataset_items("mmlu")
    cat_indices = {}
    for i, qid in enumerate(qids):
        cat = items.get(qid, {}).get("category", "unknown")
        cat_indices.setdefault(cat, []).append(i)

    cat_results = {}
    for cat, indices in sorted(cat_indices.items()):
        if len(indices) < 3:
            continue
        sub_matrix = matrix[indices, :]
        accs = np.nanmean(sub_matrix, axis=0)
        cat_results[cat] = {
            "n_questions": len(indices),
            "mean_acc": float(np.mean(accs)),
            "std_acc": float(np.std(accs)),
            "range": float(np.max(accs) - np.min(accs)),
        }
    sorted_cats = sorted(cat_results.items(), key=lambda x: x[1]["range"], reverse=True)
    return {cat: info for cat, info in sorted_cats}


# ============================================================
# Noise analysis
# ============================================================

def noise_analysis(matrices, qids, dataset, available_models):
    all_cols = [matrices[m] for m in MODELS if m in matrices]
    if not all_cols:
        return {}

    combined = np.hstack(all_cols)
    n_questions = combined.shape[0]

    noise_scores = []
    for i in range(n_questions):
        row = combined[i, ~np.isnan(combined[i])]
        n = len(row)
        if n == 0:
            noise_scores.append(0.0)
            continue
        c = np.sum(row)
        noise_scores.append(float(1.0 - abs(2 * c - n) / n))

    sorted_idx = np.argsort(noise_scores)[::-1]
    top_noisy = [
        {"rank": r + 1, "qid": qids[i], "noise_score": noise_scores[i]}
        for r, i in enumerate(sorted_idx[:30])
    ]
    qid_noise_map = {qids[i]: noise_scores[i] for i in range(n_questions)}

    thresholds = {}
    for pct in [10, 20, 30]:
        cutoff_idx = int(n_questions * pct / 100)
        remove_set = set(sorted_idx[:cutoff_idx])
        keep_mask = np.array([i not in remove_set for i in range(n_questions)])

        per_model = {}
        for m in MODELS:
            if m not in matrices:
                continue
            filtered = matrices[m][keep_mask, :]
            accs = np.nanmean(filtered, axis=0)
            per_model[m] = {
                "mean": float(np.mean(accs)),
                "std": float(np.std(accs)),
                "range": float(np.max(accs) - np.min(accs)),
            }

        reversals_after = None
        if len(available_models) >= 2:
            filtered_matrices = {m: matrices[m][keep_mask, :] for m in available_models}
            reversals_after = {}
            rev = reversal_frequency(filtered_matrices, available_models)
            for (m_a, m_b), rev_info in rev.items():
                reversals_after[f"{m_a}_vs_{m_b}"] = rev_info

        thresholds[f"remove_{pct}pct"] = {
            "n_removed": cutoff_idx,
            "n_remaining": n_questions - cutoff_idx,
            "per_model": per_model,
            "reversals_after_removal": reversals_after,
        }

    return {
        "noise_scores": noise_scores,
        "qid_noise_map": qid_noise_map,
        "top_noisy_items": top_noisy,
        "removal_analysis": thresholds,
    }


# ============================================================
# Scale analysis
# ============================================================

def scale_analysis(all_results, dataset):
    available = [m for m in MODELS if m in all_results]
    if len(available) < 2:
        return None

    rows = []
    for m in available:
        r = all_results[m]
        vd = r["variance_decomposition"]
        rows.append({
            "model": MODEL_LABELS[m],
            "size_B": MODEL_SIZES[m],
            "mean_acc": r["accuracy_stats"]["mean"],
            "acc_std": r["accuracy_stats"]["std"],
            "acc_range": r["accuracy_stats"]["range"],
            "flip_rate": r["item_flip_rate"],
            "var_prompt": vd["var_prompt"],
            "var_ratio": vd["ratio"],
        })

    sizes = [r["size_B"] for r in rows]
    stds = [r["acc_std"] for r in rows]
    ranges = [r["acc_range"] for r in rows]
    flips = [r["flip_rate"] for r in rows]

    def trend_direction(sizes, metric):
        if len(sizes) < 2:
            return "insufficient data"
        corr = np.corrcoef(sizes, metric)[0, 1]
        if corr < -0.3:
            return "robustness increases with scale"
        elif corr > 0.3:
            return "robustness decreases with scale"
        return "no clear trend"

    return {
        "per_model": rows,
        "std_trend": trend_direction(sizes, stds),
        "range_trend": trend_direction(sizes, ranges),
        "flip_rate_trend": trend_direction(sizes, flips),
    }


# ============================================================
# Token usage
# ============================================================

def token_usage_summary(model, dataset):
    results = load_results(model, dataset)
    total_prompt = 0
    total_completion = 0
    count = 0
    for qid, variants in results.items():
        for vid, entry in variants.items():
            usage = entry.get("usage", {})
            total_prompt += usage.get("prompt_tokens", 0)
            total_completion += usage.get("completion_tokens", 0)
            count += 1
    return {
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "total_tokens": total_prompt + total_completion,
        "n_calls": count,
    }


# ============================================================
# Main analysis
# ============================================================

def analyze_single_dataset(dataset):
    variants = get_all_variants()
    n_variants = len(variants)
    print(f"\n{'='*60}")
    print(f"Dataset: {DATASET_LABELS[dataset]} ({n_variants} variants)")
    print(f"{'='*60}")

    matrices = {}
    all_results = {}
    qids_ref = None

    # Surface indices: exclude with_explanation (answer_format level 2)
    surface_indices = [i for i, (_, vidx) in enumerate(variants) if vidx[1] != 2]

    for model in MODELS:
        try:
            qids, variant_ids, matrix = build_matrix(model, dataset, variants)
        except FileNotFoundError:
            print(f"  WARNING: No results for {model}/{dataset}, skipping.")
            continue

        matrices[model] = matrix
        if qids_ref is None:
            qids_ref = qids

        stats = accuracy_stats(matrix)
        flip = item_flip_rate(matrix)
        effects = ofat_main_effects(matrix, variants)
        interact = interaction_analysis(matrix, variants)
        var_decomp = variance_decomposition(matrix, n_bootstrap=5000)
        dim_var = dimension_variance_decomposition(matrix, variants)
        usage = token_usage_summary(model, dataset)

        # Surface-only metrics
        m_surf = matrix[:, surface_indices]
        stats_surf = accuracy_stats(m_surf)
        flip_surf = item_flip_rate(m_surf)
        var_decomp_surf = variance_decomposition(m_surf, n_bootstrap=3000)

        print(f"\n--- {MODEL_LABELS[model]} ---")
        print(f"  [ALL {n_variants}]  Acc: {stats['mean']:.4f} +/- {stats['std']:.4f}, "
              f"Range: {stats['range']:.4f}, Flip: {flip['overall_flip_rate']:.4f}, "
              f"VarRatio: {var_decomp['ratio']:.2f}")
        print(f"  [NO-EXPL]  Acc: {stats_surf['mean']:.4f} +/- {stats_surf['std']:.4f}, "
              f"Range: {stats_surf['range']:.4f}, Flip: {flip_surf['overall_flip_rate']:.4f}, "
              f"VarRatio: {var_decomp_surf['ratio']:.2f}")

        print(f"  Dimension Var% (of Var_total):")
        for dim_name in DIM_NAMES + ["interaction"]:
            pct = dim_var["percentages"][dim_name]
            print(f"    {dim_name}: {pct:.1f}%")

        print(f"  OFAT Main Effects:")
        for dim, eff_list in effects.items():
            deltas = [f"{e[2]:+.4f}" for e in eff_list[1:]]
            print(f"    {dim}: {', '.join(deltas)}")

        if interact.get("notable_effects"):
            r2 = interact['r_squared']
            r2a = interact.get('r_squared_adj', r2)
            print(f"  Main Effects Model: R2={r2:.3f}, R2_adj={r2a:.3f}")
            for name, coef in sorted(interact["notable_effects"].items(),
                                     key=lambda x: abs(x[1]), reverse=True)[:10]:
                print(f"    {name}: {coef:+.4f}")

        cat = category_analysis(matrix, qids, dataset)
        if cat:
            print(f"  Category Sensitivity (top 5):")
            for i, (cat_name, info) in enumerate(cat.items()):
                if i >= 5:
                    break
                print(f"    {cat_name} (n={info['n_questions']}): "
                      f"range={info['range']:.4f}, mean={info['mean_acc']:.4f}")

        all_results[model] = {
            "accuracy_stats": stats,
            "accuracy_stats_surface": stats_surf,
            "item_flip_rate": flip["overall_flip_rate"],
            "item_flip_rate_surface": flip_surf["overall_flip_rate"],
            "noise_scores": flip["noise_scores"],
            "ofat_effects": {
                dim: [(name, float(acc), float(delta)) for name, acc, delta in eff_list]
                for dim, eff_list in effects.items()
            },
            "interaction_effects": interact,
            "variance_decomposition": var_decomp,
            "variance_decomposition_surface": var_decomp_surf,
            "dimension_variance": dim_var,
            "token_usage": usage,
        }
        if cat:
            all_results[model]["category_analysis"] = cat

    # Ranking analysis
    available_models = [m for m in MODELS if m in matrices]
    if len(available_models) >= 2:
        print(f"\n--- Ranking Analysis (ALL {n_variants}) ---")
        reversals_all = reversal_frequency(matrices, available_models)
        for (m_a, m_b), rev in reversals_all.items():
            print(f"  {MODEL_LABELS[m_a]} vs {MODEL_LABELS[m_b]}: "
                  f"{rev['reversal_rate']:.2%} ({rev['reversal_count']}/{n_variants})")

        surf_matrices = {m: matrices[m][:, surface_indices] for m in available_models}
        n_surf = len(surface_indices)
        print(f"\n--- Ranking Analysis (NO-EXPL, {n_surf} variants) ---")
        reversals_surf = reversal_frequency(surf_matrices, available_models)
        for (m_a, m_b), rev in reversals_surf.items():
            print(f"  {MODEL_LABELS[m_a]} vs {MODEL_LABELS[m_b]}: "
                  f"{rev['reversal_rate']:.2%} ({rev['reversal_count']}/{n_surf})")

        print(f"\n--- Pairwise Gap Bootstrap (may take a while for {n_variants} variants) ---")
        gaps = pairwise_gap_bootstrap(matrices, available_models, n_bootstrap=5000)
        for (m_a, m_b), g in gaps.items():
            print(f"  {MODEL_LABELS[m_a]} vs {MODEL_LABELS[m_b]}: "
                  f"mean_gap={g['mean_gap']:.4f}, std={g['std_gap_across_prompts']:.4f}")

        print(f"\n--- Rank Distribution Bootstrap ---")
        rank_dist = rank_distribution_bootstrap(matrices, available_models, n_bootstrap=3000)

        all_results["ranking"] = {
            "pairwise_gaps": {
                f"{m_a}_vs_{m_b}": {
                    k: v for k, v in g.items() if k != "per_variant_cis"
                }
                for (m_a, m_b), g in gaps.items()
            },
            "reversals_all": {
                f"{m_a}_vs_{m_b}": rev for (m_a, m_b), rev in reversals_all.items()
            },
            "reversals_surface": {
                f"{m_a}_vs_{m_b}": rev for (m_a, m_b), rev in reversals_surf.items()
            },
            "rank_distribution": rank_dist,
        }

    # Scale analysis
    if len(available_models) >= 2:
        sa = scale_analysis(all_results, dataset)
        if sa:
            sizes = [MODEL_SIZES[m] for m in available_models]
            stds_s = [all_results[m]["accuracy_stats_surface"]["std"] for m in available_models]
            flips_s = [all_results[m]["item_flip_rate_surface"] for m in available_models]

            def trend(s, metric):
                corr = np.corrcoef(s, metric)[0, 1]
                return ("increases with scale" if corr < -0.3
                        else "decreases with scale" if corr > 0.3
                        else "no clear trend")

            sa["std_trend_surface"] = f"robustness {trend(sizes, stds_s)}"
            sa["flip_trend_surface"] = f"robustness {trend(sizes, flips_s)}"
            print(f"\n--- Scale Analysis ---")
            print(f"  ALL:     Std trend={sa['std_trend']}, Flip trend={sa['flip_rate_trend']}")
            print(f"  NO-EXPL: Std trend={sa['std_trend_surface']}, Flip trend={sa['flip_trend_surface']}")
            all_results["scale_analysis"] = sa

    # Noise analysis
    if qids_ref and len(available_models) >= 1:
        print(f"\n--- Noise Analysis ---")
        noise = noise_analysis(matrices, qids_ref, dataset, available_models)
        print(f"  Top 5 noisiest items:")
        for item in noise.get("top_noisy_items", [])[:5]:
            print(f"    {item['qid']}: noise={item['noise_score']:.3f}")
        noise_map_path = OUTPUT_DIR / f"noise_map_{dataset}.json"
        with open(noise_map_path, "w") as f:
            json.dump(noise.get("qid_noise_map", {}), f, indent=2)
        all_results["noise_analysis"] = {
            k: v for k, v in noise.items() if k not in ("noise_scores", "qid_noise_map")
        }

    # Save
    out_path = OUTPUT_DIR / f"analysis_{dataset}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")
    return all_results


def generate_summary_table(results_arc, results_mmlu):
    rows = []
    for dataset, label, results in [("arc", "ARC-Challenge", results_arc),
                                     ("mmlu", "MMLU-Pro", results_mmlu)]:
        for model in MODELS:
            if model not in results:
                continue
            r = results[model]
            vd = r["variance_decomposition"]
            vds = r["variance_decomposition_surface"]
            rows.append({
                "Dataset": label,
                "Model": MODEL_LABELS[model],
                "Size_B": MODEL_SIZES[model],
                "Acc_ALL": f"{r['accuracy_stats']['mean']:.4f}",
                "Std_ALL": f"{r['accuracy_stats']['std']:.4f}",
                "Range_ALL": f"{r['accuracy_stats']['range']:.4f}",
                "Flip_ALL": f"{r['item_flip_rate']:.4f}",
                "VarR_ALL": f"{vd['ratio']:.2f}",
                "Acc_SURF": f"{r['accuracy_stats_surface']['mean']:.4f}",
                "Std_SURF": f"{r['accuracy_stats_surface']['std']:.4f}",
                "Range_SURF": f"{r['accuracy_stats_surface']['range']:.4f}",
                "Flip_SURF": f"{r['item_flip_rate_surface']:.4f}",
                "VarR_SURF": f"{vds['ratio']:.2f}",
            })
    df = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "summary_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSummary table saved to {csv_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    results_arc = analyze_single_dataset("arc")
    results_mmlu = analyze_single_dataset("mmlu")
    generate_summary_table(results_arc, results_mmlu)
    print("\nAnalysis complete!")
