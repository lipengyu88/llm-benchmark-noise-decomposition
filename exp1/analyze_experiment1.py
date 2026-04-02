"""
Experiment I: Prompt Perturbation - Analysis

Metrics:
  Accuracy-level: Mean/Std, Max-Min Range, Item Flip Rate
  Ranking-level: Pairwise Gap Stability, Reversal Frequency, Rank Distribution
  Variance decomposition: Var_prompt vs Var_sampling
  Interaction effects: factorial regression on dimension encodings
  Category-level analysis: MMLU-Pro sensitivity by subject category
  Noise analysis: per-item noise score, removal impact on accuracy + rankings
  Scale analysis: robustness trend across model sizes

Usage:
    python analyze_experiment1.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from prompt_variants import (
    get_all_variants, describe_variant,
    INSTRUCTIONS, ANSWER_FORMATS, OPTION_FORMATS, FRAMINGS,
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
    files = {"arc": "arc_challenge_300.json", "mmlu": "mmlu_pro_300.json"}
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
# OFAT main effect analysis
# ============================================================

def ofat_main_effects(matrix, variants):
    variant_ids = [v[0] for v in variants]
    base_acc = float(np.nanmean(matrix[:, 0]))
    ofat_groups = {
        "instruction":   ["ofat_1", "ofat_2"],
        "answer_format": ["ofat_3", "ofat_4"],
        "option_format": ["ofat_5", "ofat_6"],
        "framing":       ["ofat_7"],
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
# Interaction effect analysis (using factorial variants)
# ============================================================

def interaction_analysis(matrix, variants):
    """
    Fit a linear model on per-variant accuracy with dimension encodings
    as features, including pairwise interaction terms.
    Uses all 18 variants to estimate main + interaction effects.

    Dimensions are nominal (unordered), so we use dummy coding
    (one-hot with reference level dropped) instead of linear encoding.
    """
    from itertools import combinations as combos

    variant_ids = [v[0] for v in variants]
    variant_indices = [v[1] for v in variants]
    accs = accuracy_per_variant(matrix)  # (n_variants,)

    dim_names = ["instruction", "answer_format", "option_format", "framing"]
    n_levels = [3, 3, 3, 2]  # levels per dimension

    # Build design matrix with dummy coding (drop level 0 as reference)
    X_main_cols = []
    main_names = []
    raw = np.array(variant_indices, dtype=int)  # (n_variants, 4)
    for d in range(4):
        for lvl in range(1, n_levels[d]):
            col = (raw[:, d] == lvl).astype(float)
            X_main_cols.append(col)
            main_names.append(f"{dim_names[d]}_L{lvl}")

    X_main = np.column_stack(X_main_cols)  # (n_variants, sum(n_levels)-4)

    # Pairwise interactions between dummy columns of different dimensions
    interaction_names = []
    interaction_cols = []
    # Track which main columns belong to which dimension
    dim_col_ranges = []
    offset = 0
    for d in range(4):
        k = n_levels[d] - 1  # number of dummies for this dimension
        dim_col_ranges.append(list(range(offset, offset + k)))
        offset += k

    for di, dj in combos(range(4), 2):
        for ci in dim_col_ranges[di]:
            for cj in dim_col_ranges[dj]:
                interaction_names.append(f"{main_names[ci]}*{main_names[cj]}")
                interaction_cols.append(X_main[:, ci] * X_main[:, cj])

    X_interact = np.column_stack(interaction_cols) if interaction_cols else np.empty((len(accs), 0))
    X_full = np.column_stack([np.ones(len(accs)), X_main, X_interact])
    feature_names = ["intercept"] + main_names + interaction_names

    # OLS fit
    try:
        beta, residuals, rank, sv = np.linalg.lstsq(X_full, accs, rcond=None)
    except np.linalg.LinAlgError:
        return {"error": "lstsq failed"}

    y_pred = X_full @ beta
    ss_res = np.sum((accs - y_pred) ** 2)
    ss_tot = np.sum((accs - np.mean(accs)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    coefficients = {name: float(b) for name, b in zip(feature_names, beta)}

    # Identify which effects are large (> 1% accuracy impact)
    notable = {k: v for k, v in coefficients.items()
               if k != "intercept" and abs(v) > 0.01}

    return {
        "coefficients": coefficients,
        "r_squared": float(r_squared),
        "notable_effects": notable,
    }

# ============================================================
# Variance decomposition (vectorized)
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
        # Vectorized bootstrap
        idx = RNG.randint(0, len(col), size=(n_bootstrap, len(col)))
        boot_accs = col[idx].mean(axis=1)
        var_sampling_per_variant.append(float(np.var(boot_accs)))

    var_sampling = float(np.mean(var_sampling_per_variant))
    ratio = var_prompt / var_sampling if var_sampling > 0 else float('inf')

    return {
        "var_prompt": var_prompt,
        "var_sampling": var_sampling,
        "ratio": ratio,
        # Interpretation thresholds follow a conventional signal-to-noise
        # heuristic: ratio > 2 means prompt variance is at least twice
        # sampling noise (a clear signal), ratio < 0.5 means sampling noise
        # is at least twice prompt variance, and values in between are
        # treated as comparable.  These symmetric log-scale boundaries
        # (2 and 1/2) mirror the "rule of 2" used in ANOVA effect-size
        # screening (see e.g., Gelman & Hill 2007, §21.4) and ensure
        # symmetric treatment of both directions.
        "interpretation": (
            "prompt variation dominates" if ratio > 2 else
            "comparable" if ratio > 0.5 else
            "sampling noise dominates"
        ),
    }

# ============================================================
# Ranking-level metrics (vectorized bootstrap)
# ============================================================

def pairwise_gap_bootstrap(matrices, model_names, n_bootstrap=N_BOOTSTRAP):
    pairs = list(combinations(model_names, 2))
    n_questions = list(matrices.values())[0].shape[0]
    n_variants = list(matrices.values())[0].shape[1]
    results = {}

    for m_a, m_b in pairs:
        per_variant_cis = []
        per_variant_gaps = []

        for v in range(n_variants):
            col_a = matrices[m_a][:, v]
            col_b = matrices[m_b][:, v]
            gap_obs = float(np.nanmean(col_a) - np.nanmean(col_b))
            per_variant_gaps.append(gap_obs)

            # Vectorized bootstrap
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
        # Use bootstrap std (CI_width / 3.92) for proper comparison
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
            "cross_prompt_var_vs_sampling": (
                "prompt variation exceeds sampling noise"
                if cross_prompt_std > avg_boot_std
                else "sampling noise dominates gap variation"
            ),
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


def rank_distribution_bootstrap(matrices, model_names, n_bootstrap=N_BOOTSTRAP):
    n_questions = list(matrices.values())[0].shape[0]
    n_variants = list(matrices.values())[0].shape[1]
    n_models = len(model_names)

    per_variant_rank_dist = {m: np.zeros((n_variants, n_models)) for m in model_names}

    for v in range(n_variants):
        rank_counts = {m: np.zeros(n_models) for m in model_names}
        # Vectorized: generate all bootstrap indices at once
        all_idx = RNG.randint(0, n_questions, size=(n_bootstrap, n_questions))
        # Precompute columns for this variant
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
    """Analyze prompt sensitivity by MMLU-Pro subject category."""
    if dataset != "mmlu":
        return None

    items = load_dataset_items("mmlu")
    # Group question indices by category
    cat_indices = {}
    for i, qid in enumerate(qids):
        cat = items.get(qid, {}).get("category", "unknown")
        cat_indices.setdefault(cat, []).append(i)

    cat_results = {}
    for cat, indices in sorted(cat_indices.items()):
        if len(indices) < 3:  # skip tiny categories
            continue
        sub_matrix = matrix[indices, :]
        accs = np.nanmean(sub_matrix, axis=0)
        cat_results[cat] = {
            "n_questions": len(indices),
            "mean_acc": float(np.mean(accs)),
            "std_acc": float(np.std(accs)),
            "range": float(np.max(accs) - np.min(accs)),
        }

    # Sort by range (most prompt-sensitive first)
    sorted_cats = sorted(cat_results.items(), key=lambda x: x[1]["range"], reverse=True)
    return {cat: info for cat, info in sorted_cats}

# ============================================================
# Noise analysis (with ranking impact)
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

    # qid -> noise_score map for cross-experiment use
    qid_noise_map = {qids[i]: noise_scores[i] for i in range(n_questions)}

    # Removal analysis with ranking impact
    thresholds = {}
    for pct in [10, 20, 30]:
        cutoff_idx = int(n_questions * pct / 100)
        remove_set = set(sorted_idx[:cutoff_idx])
        keep_mask = np.array([i not in remove_set for i in range(n_questions)])

        # Re-compute accuracy stats
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

        # Re-compute ranking reversals after removal
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
    """Analyze whether prompt robustness increases with model scale."""
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

    # Check if robustness (lower std/range/flip_rate) increases with scale
    sizes = [r["size_B"] for r in rows]
    stds = [r["acc_std"] for r in rows]
    ranges = [r["acc_range"] for r in rows]
    flips = [r["flip_rate"] for r in rows]

    # Simple correlation: negative = larger models more robust
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

def dimension_variance_decomposition(matrix, variants):
    """Decompose Var_prompt into per-dimension contributions using OFAT."""
    dim_groups = {
        "instruction":   [0, 1, 2],
        "answer_format": [0, 3, 4],
        "option_format": [0, 5, 6],
        "framing":       [0, 7],
    }
    accs_all = np.nanmean(matrix, axis=0)
    var_total = float(np.var(accs_all))

    dim_vars = {}
    for dim, cols in dim_groups.items():
        accs = [float(np.nanmean(matrix[:, c])) for c in cols]
        dim_vars[dim] = float(np.var(accs))

    var_inter = max(0.0, var_total - sum(dim_vars.values()))
    dim_vars["interaction"] = var_inter
    dim_vars["total"] = var_total

    # Percentages
    dim_pcts = {}
    for k, v in dim_vars.items():
        if k != "total":
            dim_pcts[k] = v / var_total * 100 if var_total > 0 else 0.0

    return {"variances": dim_vars, "percentages": dim_pcts}


def analyze_one_variant_set(matrices, available_models, variants, col_indices, set_name):
    """Run all metrics on a subset of variant columns. Returns dict of results."""
    n_variants = len(col_indices)
    sub_matrices = {m: matrices[m][:, col_indices] for m in available_models}
    sub_variants = [variants[i] for i in col_indices]

    results = {}

    # Per-model metrics
    for model in available_models:
        m = sub_matrices[model]
        stats = accuracy_stats(m)
        flip = item_flip_rate(m)
        var_decomp = variance_decomposition(m, n_bootstrap=5000)
        results[model] = {
            "accuracy_stats": stats,
            "item_flip_rate": flip["overall_flip_rate"],
            "variance_decomposition": var_decomp,
        }

    # Ranking
    if len(available_models) >= 2:
        rev = reversal_frequency(sub_matrices, available_models)
        results["reversals"] = {
            f"{m_a}_vs_{m_b}": info for (m_a, m_b), info in rev.items()
        }

    return results


def analyze_single_dataset(dataset):
    variants = get_all_variants()
    print(f"\n{'='*60}")
    print(f"Dataset: {DATASET_LABELS[dataset]}")
    print(f"{'='*60}")

    matrices = {}
    all_results = {}
    qids_ref = None

    # Variant set indices
    all_indices = list(range(len(variants)))
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
        var_decomp = variance_decomposition(matrix)
        dim_var = dimension_variance_decomposition(matrix, variants)
        usage = token_usage_summary(model, dataset)

        # Surface-only metrics
        m_surf = matrix[:, surface_indices]
        stats_surf = accuracy_stats(m_surf)
        flip_surf = item_flip_rate(m_surf)
        var_decomp_surf = variance_decomposition(m_surf, n_bootstrap=5000)

        print(f"\n--- {MODEL_LABELS[model]} ---")
        print(f"  [ALL 18]   Acc: {stats['mean']:.4f}±{stats['std']:.4f}, "
              f"Range: {stats['range']:.4f}, Flip: {flip['overall_flip_rate']:.4f}, "
              f"VarRatio: {var_decomp['ratio']:.2f}")
        print(f"  [NO-EXPL]  Acc: {stats_surf['mean']:.4f}±{stats_surf['std']:.4f}, "
              f"Range: {stats_surf['range']:.4f}, Flip: {flip_surf['overall_flip_rate']:.4f}, "
              f"VarRatio: {var_decomp_surf['ratio']:.2f}")

        print(f"  Dimension Var% (of Var_total):")
        for dim_name in ["instruction", "answer_format", "option_format", "framing", "interaction"]:
            pct = dim_var["percentages"][dim_name]
            print(f"    {dim_name}: {pct:.1f}%")

        print(f"  OFAT Main Effects:")
        for dim, eff_list in effects.items():
            deltas = [f"{e[2]:+.4f}" for e in eff_list[1:]]
            print(f"    {dim}: {', '.join(deltas)}")

        if interact.get("notable_effects"):
            print(f"  Interaction Effects (R²={interact['r_squared']:.3f}):")
            for name, coef in interact["notable_effects"].items():
                print(f"    {name}: {coef:+.4f}")

        cat = category_analysis(matrix, qids, dataset)
        if cat:
            print(f"  Category Sensitivity (top 5):")
            for i, (cat_name, info) in enumerate(cat.items()):
                if i >= 5: break
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

    # Ranking analysis (both sets)
    available_models = [m for m in MODELS if m in matrices]
    if len(available_models) >= 2:
        print(f"\n--- Ranking Analysis (ALL 18) ---")
        reversals_all = reversal_frequency(matrices, available_models)
        for (m_a, m_b), rev in reversals_all.items():
            print(f"  {MODEL_LABELS[m_a]} vs {MODEL_LABELS[m_b]}: "
                  f"{rev['reversal_rate']:.2%} ({rev['reversal_count']}/18)")

        surf_matrices = {m: matrices[m][:, surface_indices] for m in available_models}
        print(f"\n--- Ranking Analysis (NO-EXPL, {len(surface_indices)} variants) ---")
        reversals_surf = reversal_frequency(surf_matrices, available_models)
        for (m_a, m_b), rev in reversals_surf.items():
            print(f"  {MODEL_LABELS[m_a]} vs {MODEL_LABELS[m_b]}: "
                  f"{rev['reversal_rate']:.2%} ({rev['reversal_count']}/{len(surface_indices)})")

        gaps = pairwise_gap_bootstrap(matrices, available_models)
        rank_dist = rank_distribution_bootstrap(matrices, available_models)

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
            # Also compute surface-only scale trend
            sizes = [MODEL_SIZES[m] for m in available_models]
            stds_s = [all_results[m]["accuracy_stats_surface"]["std"] for m in available_models]
            flips_s = [all_results[m]["item_flip_rate_surface"] for m in available_models]
            def trend(s, metric):
                corr = np.corrcoef(s, metric)[0,1]
                return "increases with scale" if corr < -0.3 else "decreases with scale" if corr > 0.3 else "no clear trend"
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
