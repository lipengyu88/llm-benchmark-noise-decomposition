"""
Experiment III: High-Noise Item Analysis — Full Re-evaluation

After removing the noisiest items at 10%/20%/30% thresholds, re-compute
all accuracy-level and ranking-level metrics from Experiments I & II.

Metrics re-evaluated:
  Accuracy-level: Mean/Std, Max-Min Range, Item Flip Rate
  Ranking-level: Pairwise Gap Stability (bootstrap CIs), Reversal Frequency,
                 Rank Distribution
  Cross-experiment: Variance decomposition (prompt vs sampling vs test-set)

Also computes:
  - Noise source attribution (which experiment contributes more noise)
  - Noise correlation across models (do models agree on noisy items?)
  - Scale analysis (does noise sensitivity change with model size?)

Usage:
    python analyze_experiment3.py
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent
EXP1_DIR = ROOT.parent / "exp1"
EXP2_DIR = ROOT.parent / "exp2"
NOISE_DIR = ROOT / "noise_data"
ANALYSIS_DIR = ROOT / "analysis_exp3"
ANALYSIS_DIR.mkdir(exist_ok=True)

# Consistent model definitions
MODELS_EXP1 = ["llama", "qwen7b", "qwen32b", "qwen72b"]
MODELS_EXP2 = ["llama-3.1-8b", "qwen2.5-7b", "qwen3-32b", "qwen2.5-72b"]
MODEL_MAP_E1_TO_E2 = {
    "llama": "llama-3.1-8b",
    "qwen7b": "qwen2.5-7b",
    "qwen32b": "qwen3-32b",
    "qwen72b": "qwen2.5-72b",
}
MODEL_LABELS = {
    "llama-3.1-8b": "LLaMA-3.1-8B",
    "qwen2.5-7b": "Qwen2.5-7B",
    "qwen3-32b": "Qwen3-32B",
    "qwen2.5-72b": "Qwen2.5-72B",
}
MODEL_SIZES = {
    "llama-3.1-8b": 8,
    "qwen2.5-7b": 7,
    "qwen3-32b": 32,
    "qwen2.5-72b": 72,
}
DATASETS = {"arc": "arc_challenge", "mmlu": "mmlu_pro"}

N_BOOTSTRAP = 10_000
RNG = np.random.RandomState(42)


# ============================================================
# Data loading helpers
# ============================================================

def load_noise_data(dataset_key: str) -> dict:
    """Load pre-computed noise data from run_experiment3.py."""
    path = NOISE_DIR / f"noise_{dataset_key}.json"
    if not path.exists():
        raise FileNotFoundError(f"Run run_experiment3.py first: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_exp1_matrix(model_e1: str, dataset_key: str) -> tuple:
    """Load Exp I results as (qids, variant_ids, matrix)."""
    # Import variant info from exp1
    import sys
    sys.path.insert(0, str(EXP1_DIR))
    from prompt_variants import get_all_variants
    sys.path.pop(0)

    variants = get_all_variants()
    variant_ids = [v[0] for v in variants]

    results_dir = EXP1_DIR / "results_exp1"
    path = results_dir / f"results_{model_e1}_{dataset_key}.json"
    if not path.exists():
        path = results_dir / f"checkpoint_{model_e1}_{dataset_key}.json"
    if not path.exists():
        return [], [], np.array([])

    with open(path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    qids = sorted(results.keys())
    matrix = np.full((len(qids), len(variant_ids)), np.nan)
    for i, qid in enumerate(qids):
        for j, vid in enumerate(variant_ids):
            if vid in results[qid] and results[qid][vid]["is_correct"] is not None:
                matrix[i, j] = results[qid][vid]["is_correct"]

    return qids, variant_ids, matrix


def load_exp2_dataframe(dataset_key: str) -> pd.DataFrame:
    """Load Exp II results as a DataFrame.
    Tries new naming convention (with source suffix) first, then falls back."""
    bench = DATASETS[dataset_key]
    frames = []
    for m in MODELS_EXP2:
        loaded = False
        for suffix in ["_gpt4o", "_qwen", ""]:
            p = EXP2_DIR / f"exp2_{bench}_{m}{suffix}.json"
            if p.exists():
                frames.append(pd.DataFrame(json.loads(p.read_text(encoding="utf-8"))))
                loaded = True
                break
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if "parse_failure" not in df.columns:
        df["parse_failure"] = False
    return df


# ============================================================
# Filtered metric computation — Experiment I
# ============================================================

def exp1_accuracy_stats(matrix: np.ndarray) -> dict:
    """Compute accuracy-level metrics for an Exp I matrix."""
    accs = np.nanmean(matrix, axis=0)
    return {
        "mean": float(np.mean(accs)),
        "std": float(np.std(accs)),
        "max": float(np.max(accs)),
        "min": float(np.min(accs)),
        "range": float(np.max(accs) - np.min(accs)),
        "per_variant": accs.tolist(),
    }


def exp1_flip_rate(matrix: np.ndarray) -> float:
    """Item flip rate for Exp I matrix."""
    n = matrix.shape[0]
    flips = 0
    for i in range(n):
        row = matrix[i, ~np.isnan(matrix[i])]
        if len(row) > 0 and np.min(row) != np.max(row):
            flips += 1
    return flips / n if n > 0 else 0.0


def exp1_variance_decomposition(matrix: np.ndarray) -> dict:
    """Var(prompt) vs Var(sampling) for Exp I."""
    acc_per_variant = np.nanmean(matrix, axis=0)
    var_prompt = float(np.var(acc_per_variant))

    var_sampling_list = []
    for v in range(matrix.shape[1]):
        col = matrix[:, v]
        col = col[~np.isnan(col)]
        if len(col) == 0:
            continue
        idx = RNG.randint(0, len(col), size=(N_BOOTSTRAP, len(col)))
        boot_accs = col[idx].mean(axis=1)
        var_sampling_list.append(float(np.var(boot_accs)))

    var_sampling = float(np.mean(var_sampling_list)) if var_sampling_list else 0.0
    ratio = var_prompt / var_sampling if var_sampling > 0 else float("inf")

    return {
        "var_prompt": var_prompt,
        "var_sampling": var_sampling,
        "ratio": ratio,
    }


def exp1_pairwise_gaps(matrices: dict, model_keys: list) -> dict:
    """Pairwise gap bootstrap for Exp I matrices."""
    results = {}
    for m_a, m_b in combinations(model_keys, 2):
        n_variants = matrices[m_a].shape[1]
        per_variant_gaps = []
        per_variant_cis = []

        for v in range(n_variants):
            col_a = matrices[m_a][:, v]
            col_b = matrices[m_b][:, v]
            gap = float(np.nanmean(col_a) - np.nanmean(col_b))
            per_variant_gaps.append(gap)

            valid = ~(np.isnan(col_a) | np.isnan(col_b))
            a_v, b_v = col_a[valid], col_b[valid]
            n_v = len(a_v)
            if n_v == 0:
                per_variant_cis.append((0.0, 0.0))
                continue
            idx = RNG.randint(0, n_v, size=(N_BOOTSTRAP, n_v))
            boot = a_v[idx].mean(axis=1) - b_v[idx].mean(axis=1)
            ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
            per_variant_cis.append(ci)

        gap_arr = np.array(per_variant_gaps)
        n_contains_zero = sum(1 for lo, hi in per_variant_cis if lo <= 0 <= hi)

        results[f"{m_a}_vs_{m_b}"] = {
            "mean_gap": float(np.mean(gap_arr)),
            "std_gap": float(np.std(gap_arr)),
            "min_gap": float(np.min(gap_arr)),
            "max_gap": float(np.max(gap_arr)),
            "n_ci_contains_zero": n_contains_zero,
            "per_variant_gaps": [float(g) for g in per_variant_gaps],
        }

    return results


def exp1_reversal_frequency(matrices: dict, model_keys: list) -> dict:
    """Reversal frequency for Exp I."""
    n_variants = list(matrices.values())[0].shape[1]
    accs = {m: np.nanmean(matrices[m], axis=0) for m in model_keys}
    results = {}
    for m_a, m_b in combinations(model_keys, 2):
        mean_a, mean_b = np.mean(accs[m_a]), np.mean(accs[m_b])
        better = m_a if mean_a >= mean_b else m_b
        worse = m_b if better == m_a else m_a
        reversals = sum(1 for v in range(n_variants) if accs[worse][v] > accs[better][v])
        results[f"{m_a}_vs_{m_b}"] = {
            "reversal_count": reversals,
            "reversal_rate": reversals / n_variants,
            "total_variants": n_variants,
            "expected_order": f"{MODEL_LABELS.get(better, better)} > {MODEL_LABELS.get(worse, worse)}",
        }
    return results


def exp1_rank_distribution(matrices: dict, model_keys: list) -> dict:
    """Bootstrap rank distribution for Exp I."""
    n_questions = list(matrices.values())[0].shape[0]
    n_variants = list(matrices.values())[0].shape[1]
    n_models = len(model_keys)
    counts = {m: np.zeros(n_models) for m in model_keys}

    for v in range(n_variants):
        cols = {m: matrices[m][:, v] for m in model_keys}
        all_idx = RNG.randint(0, n_questions, size=(N_BOOTSTRAP, n_questions))
        for b in range(N_BOOTSTRAP):
            idx = all_idx[b]
            model_accs = {m: np.nanmean(cols[m][idx]) for m in model_keys}
            ranked = sorted(model_keys, key=lambda m: model_accs[m], reverse=True)
            for rank, m in enumerate(ranked):
                counts[m][rank] += 1

    total = N_BOOTSTRAP * n_variants
    return {m: (counts[m] / total).tolist() for m in model_keys}


# ============================================================
# Filtered metric computation — Experiment II
# ============================================================

def exp2_accuracy_stats(df: pd.DataFrame) -> list[dict]:
    """Accuracy stats per model for Exp II."""
    acc = (df.groupby(["model_short", "version"])
           .agg(n_correct=("is_correct", "sum"), n_total=("is_correct", "count"))
           .reset_index())
    acc["accuracy"] = acc.n_correct / acc.n_total
    summary = (acc.groupby("model_short")["accuracy"]
               .agg(mean="mean", std="std", min="min", max="max")
               .reset_index())
    summary["range"] = summary["max"] - summary["min"]
    return summary.to_dict("records")


def exp2_flip_rate(df: pd.DataFrame) -> dict:
    """Item flip rate per model for Exp II."""
    grp = (df.groupby(["model_short", "question_id"])
           .agg(cc=("is_correct", "sum"), tt=("is_correct", "count"))
           .reset_index())
    grp["flipped"] = grp.apply(lambda r: int(r.cc > 0 and r.cc < r.tt), axis=1)
    out = {}
    for m in MODELS_EXP2:
        md = grp[grp.model_short == m]
        n_flip = int(md.flipped.sum())
        n_tot = len(md)
        out[m] = {"n_flipped": n_flip, "n_total": n_tot,
                  "rate": n_flip / n_tot if n_tot else 0}
    return out


def exp2_pairwise_bootstrap(df: pd.DataFrame) -> list[dict]:
    """Pairwise bootstrap gaps for Exp II."""
    q_acc = {}
    for m in MODELS_EXP2:
        mdf = df[df.model_short == m]
        q_acc[m] = mdf.groupby("question_id")["is_correct"].mean()

    rows = []
    for m1, m2 in combinations(MODELS_EXP2, 2):
        common = q_acc[m1].index.intersection(q_acc[m2].index)
        if len(common) == 0:
            continue
        d = q_acc[m1].loc[common].values - q_acc[m2].loc[common].values
        boot = np.array([d[RNG.choice(len(d), len(d), replace=True)].mean()
                         for _ in range(N_BOOTSTRAP)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        sig = (lo > 0) or (hi < 0)
        rows.append({
            "model_1": m1, "model_2": m2,
            "mean_gap": float(d.mean()),
            "ci_lower": float(lo), "ci_upper": float(hi),
            "significant": bool(sig), "n": int(len(common)),
        })
    return rows


def exp2_reversal_frequency(df: pd.DataFrame) -> list[dict]:
    """Reversal frequency for Exp II."""
    acc_mv = df.groupby(["model_short", "version"])["is_correct"].mean().reset_index()
    acc_mv.columns = ["model", "version", "accuracy"]
    rows = []
    for m1, m2 in combinations(MODELS_EXP2, 2):
        a1 = acc_mv[acc_mv.model == m1].set_index("version")["accuracy"]
        a2 = acc_mv[acc_mv.model == m2].set_index("version")["accuracy"]
        common_v = sorted(set(a1.index) & set(a2.index))
        overall_diff = a1.loc[common_v].mean() - a2.loc[common_v].mean()
        m1_better = overall_diff > 0
        n_rev = sum(1 for v in common_v if (a1.loc[v] > a2.loc[v]) != m1_better)
        rows.append({
            "pair": f"{m1} vs {m2}",
            "reversal_count": n_rev,
            "total_versions": len(common_v),
            "reversal_rate": n_rev / len(common_v) if common_v else 0,
        })
    return rows


def exp2_rank_distribution(df: pd.DataFrame) -> list[dict]:
    """Bootstrap rank distribution for Exp II."""
    q_acc = {}
    for m in MODELS_EXP2:
        q_acc[m] = df[df.model_short == m].groupby("question_id")["is_correct"].mean()
    common = q_acc[MODELS_EXP2[0]].index
    for m in MODELS_EXP2[1:]:
        common = common.intersection(q_acc[m].index)
    if len(common) == 0:
        return []

    counts = {m: np.zeros(len(MODELS_EXP2)) for m in MODELS_EXP2}
    for _ in range(N_BOOTSTRAP):
        idx = RNG.choice(len(common), len(common), replace=True)
        qids = common[idx]
        accs = {m: q_acc[m].loc[qids].mean() for m in MODELS_EXP2}
        ranked = sorted(accs, key=lambda m: -accs[m])
        for rank, m in enumerate(ranked):
            counts[m][rank] += 1

    return [{
        "model": m,
        **{f"rank_{r+1}_prob": float(counts[m][r] / N_BOOTSTRAP) for r in range(len(MODELS_EXP2))}
    } for m in MODELS_EXP2]


# ============================================================
# Noise correlation analysis
# ============================================================

def noise_correlation_across_models(noise_data: dict) -> dict:
    """Compute correlation of per-model noise scores across questions."""
    qids = sorted(noise_data["noise_scores"].keys())
    per_model_noise = {m: [] for m in MODELS_EXP2}

    for qid in qids:
        nd = noise_data["noise_scores"][qid]
        for m in MODELS_EXP2:
            if m in nd["per_model"] and nd["per_model"][m]["total"] > 0:
                per_model_noise[m].append(nd["per_model"][m]["noise"])
            else:
                per_model_noise[m].append(np.nan)

    # Pairwise correlation
    corrs = {}
    for m1, m2 in combinations(MODELS_EXP2, 2):
        a1 = np.array(per_model_noise[m1])
        a2 = np.array(per_model_noise[m2])
        valid = ~(np.isnan(a1) | np.isnan(a2))
        if valid.sum() < 3:
            continue
        corr = np.corrcoef(a1[valid], a2[valid])[0, 1]
        corrs[f"{m1}_vs_{m2}"] = round(float(corr), 4)

    return corrs


def noise_vs_difficulty(noise_data: dict) -> dict:
    """Analyze relationship between noise and item difficulty."""
    difficulties = []
    noises = []

    for qid, nd in noise_data["noise_scores"].items():
        if nd["total"] > 0:
            diff = 1.0 - nd["correct"] / nd["total"]  # higher = harder
            difficulties.append(diff)
            noises.append(nd["noise_score"])

    arr_d = np.array(difficulties)
    arr_n = np.array(noises)
    corr = float(np.corrcoef(arr_d, arr_n)[0, 1])

    # Bin by difficulty
    bins = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    binned = {}
    for lo, hi in bins:
        mask = (arr_d >= lo) & (arr_d < hi)
        if mask.sum() > 0:
            binned[f"{lo:.1f}-{hi:.1f}"] = {
                "n": int(mask.sum()),
                "mean_noise": round(float(arr_n[mask].mean()), 4),
                "mean_difficulty": round(float(arr_d[mask].mean()), 4),
            }

    return {
        "correlation": corr,
        "interpretation": (
            "strong positive" if corr > 0.5 else
            "moderate positive" if corr > 0.3 else
            "weak" if abs(corr) < 0.3 else
            "moderate negative" if corr < -0.3 else
            "strong negative"
        ),
        "binned": binned,
    }


# ============================================================
# Three-way variance decomposition
# ============================================================

def three_way_variance_decomposition(
    exp1_matrices: dict,
    exp2_df: pd.DataFrame,
    kept_qids: set,
) -> list[dict]:
    """
    Decompose total evaluation variance into:
      Var(prompt) — from Exp I prompt variants
      Var(sampling) — bootstrap resampling noise
      Var(test-set) — from Exp II paraphrase versions
    """
    results = []
    for m_e1, m_e2 in MODEL_MAP_E1_TO_E2.items():
        if m_e1 not in exp1_matrices:
            continue
        mat = exp1_matrices[m_e1]
        qids = exp1_matrices[f"{m_e1}_qids"]
        keep_mask = np.array([q in kept_qids for q in qids])

        if keep_mask.sum() == 0:
            continue

        filtered = mat[keep_mask, :]
        accs_prompt = np.nanmean(filtered, axis=0)
        var_prompt = float(np.var(accs_prompt))

        # Var(sampling) via bootstrap
        var_samp_list = []
        for v in range(filtered.shape[1]):
            col = filtered[:, v]
            col = col[~np.isnan(col)]
            if len(col) == 0:
                continue
            idx = RNG.randint(0, len(col), size=(5000, len(col)))
            var_samp_list.append(float(np.var(col[idx].mean(axis=1))))
        var_sampling = float(np.mean(var_samp_list)) if var_samp_list else 0.0

        # Var(test-set) from Exp II
        mdf = exp2_df[
            (exp2_df.model_short == m_e2) &
            (exp2_df.question_id.astype(str).isin(kept_qids))
        ]
        if len(mdf) > 0:
            version_accs = mdf.groupby("version")["is_correct"].mean()
            var_testset = float(version_accs.var()) if len(version_accs) > 1 else 0.0
        else:
            var_testset = 0.0

        total = var_prompt + var_sampling + var_testset
        results.append({
            "model": m_e2,
            "var_prompt": var_prompt,
            "var_sampling": var_sampling,
            "var_testset": var_testset,
            "var_total": total,
            "pct_prompt": 100 * var_prompt / total if total > 0 else 0,
            "pct_sampling": 100 * var_sampling / total if total > 0 else 0,
            "pct_testset": 100 * var_testset / total if total > 0 else 0,
        })

    return results


# ============================================================
# Scale analysis
# ============================================================

def scale_analysis(threshold_results: dict) -> dict:
    """Analyze how noise removal impact varies with model scale."""
    scale_data = {}

    for pct_key, tr in threshold_results.items():
        rows = []
        exp1 = tr.get("exp1", {})
        for m_e2 in MODELS_EXP2:
            m_e1 = {v: k for k, v in MODEL_MAP_E1_TO_E2.items()}.get(m_e2)
            if m_e1 and m_e1 in exp1:
                e1_stats = exp1[m_e1]
                rows.append({
                    "model": m_e2,
                    "size_B": MODEL_SIZES[m_e2],
                    "acc_std": e1_stats["accuracy"]["std"],
                    "acc_range": e1_stats["accuracy"]["range"],
                    "flip_rate": e1_stats["flip_rate"],
                })

        if len(rows) >= 2:
            sizes = [r["size_B"] for r in rows]
            stds = [r["acc_std"] for r in rows]
            flips = [r["flip_rate"] for r in rows]
            corr_std = float(np.corrcoef(sizes, stds)[0, 1]) if len(sizes) > 1 else 0
            corr_flip = float(np.corrcoef(sizes, flips)[0, 1]) if len(sizes) > 1 else 0
            scale_data[pct_key] = {
                "per_model": rows,
                "std_scale_corr": corr_std,
                "flip_scale_corr": corr_flip,
            }

    return scale_data


# ============================================================
# Main analysis pipeline
# ============================================================

def analyze_dataset(dataset_key: str) -> dict:
    log.info(f"\n{'='*60}")
    log.info(f"Analyzing {DATASETS[dataset_key]}")
    log.info(f"{'='*60}")

    # Load noise data
    noise_data = load_noise_data(dataset_key)
    noise_scores = noise_data["noise_scores"]
    all_qids = set(noise_scores.keys())

    # Load Exp I matrices
    exp1_matrices = {}
    exp1_available = []
    for m_e1 in MODELS_EXP1:
        qids, vids, matrix = load_exp1_matrix(m_e1, dataset_key)
        if len(qids) > 0:
            exp1_matrices[m_e1] = matrix
            exp1_matrices[f"{m_e1}_qids"] = qids
            exp1_matrices[f"{m_e1}_vids"] = vids
            exp1_available.append(m_e1)

    # Load Exp II data
    exp2_df = load_exp2_dataframe(dataset_key)
    exp2_df["question_id"] = exp2_df["question_id"].astype(str)

    # Baseline (no removal) metrics
    log.info("\n--- Baseline (no removal) ---")
    baseline = compute_threshold_metrics(
        exp1_matrices, exp1_available, exp2_df, all_qids, dataset_key
    )

    # Threshold-based analysis
    threshold_results = {"baseline": baseline}
    for pct_str, rs in noise_data["removal_sets"].items():
        pct = int(pct_str)
        kept_qids = set(rs["kept_qids"])
        log.info(f"\n--- Remove {pct}% (keep {rs['n_kept']}/{noise_data['n_questions']}) ---")
        tr = compute_threshold_metrics(
            exp1_matrices, exp1_available, exp2_df, kept_qids, dataset_key
        )
        threshold_results[f"remove_{pct}pct"] = tr

    # Noise correlation
    log.info("\n--- Noise Correlation Across Models ---")
    corrs = noise_correlation_across_models(noise_data)
    for pair, c in corrs.items():
        log.info(f"  {pair}: r={c:.4f}")

    # Noise vs difficulty
    log.info("\n--- Noise vs Difficulty ---")
    nvd = noise_vs_difficulty(noise_data)
    log.info(f"  Correlation: {nvd['correlation']:.4f} ({nvd['interpretation']})")

    # Three-way variance decomposition at each threshold
    log.info("\n--- Three-Way Variance Decomposition ---")
    var_decomp = {}
    for pct_str in ["baseline"] + [f"remove_{p}pct" for p in [10, 20, 30]]:
        if pct_str == "baseline":
            kept = all_qids
        else:
            pct = int(pct_str.split("_")[1].replace("pct", ""))
            kept = set(noise_data["removal_sets"][str(pct)]["kept_qids"])
        vd = three_way_variance_decomposition(exp1_matrices, exp2_df, kept)
        var_decomp[pct_str] = vd
        for v in vd:
            log.info(f"  [{pct_str}] {v['model']}: prompt={v['pct_prompt']:.1f}%, "
                     f"sampling={v['pct_sampling']:.1f}%, testset={v['pct_testset']:.1f}%")

    # Scale analysis
    scale = scale_analysis(threshold_results)

    # Compile results
    analysis = {
        "dataset": DATASETS[dataset_key],
        "n_total_questions": noise_data["n_questions"],
        "threshold_results": threshold_results,
        "noise_correlation": corrs,
        "noise_vs_difficulty": nvd,
        "variance_decomposition_3way": var_decomp,
        "scale_analysis": scale,
        "source_stats": noise_data["source_stats"],
        "qualitative": noise_data["qualitative_analysis"],
    }

    return analysis


def compute_threshold_metrics(
    exp1_matrices: dict,
    exp1_available: list,
    exp2_df: pd.DataFrame,
    kept_qids: set,
    dataset_key: str,
) -> dict:
    """Compute all metrics for a given set of kept question IDs."""
    result = {"exp1": {}, "exp2": {}}

    # --- Exp I metrics ---
    filtered_matrices = {}
    for m_e1 in exp1_available:
        qids = exp1_matrices[f"{m_e1}_qids"]
        keep_mask = np.array([q in kept_qids for q in qids])
        if keep_mask.sum() == 0:
            continue
        mat = exp1_matrices[m_e1][keep_mask, :]
        filtered_matrices[m_e1] = mat

        stats = exp1_accuracy_stats(mat)
        flip = exp1_flip_rate(mat)
        var_d = exp1_variance_decomposition(mat)

        m_e2 = MODEL_MAP_E1_TO_E2[m_e1]
        log.info(f"  [Exp I] {MODEL_LABELS[m_e2]}: acc={stats['mean']:.4f}±{stats['std']:.4f}, "
                 f"range={stats['range']:.4f}, flip={flip:.4f}, var_ratio={var_d['ratio']:.2f}")

        result["exp1"][m_e1] = {
            "accuracy": stats,
            "flip_rate": flip,
            "variance_decomposition": var_d,
        }

    # Exp I ranking metrics
    if len(filtered_matrices) >= 2:
        avail = [m for m in exp1_available if m in filtered_matrices]
        gaps = exp1_pairwise_gaps(filtered_matrices, avail)
        reversals = exp1_reversal_frequency(filtered_matrices, avail)
        rank_dist = exp1_rank_distribution(filtered_matrices, avail)

        result["exp1_ranking"] = {
            "pairwise_gaps": gaps,
            "reversals": reversals,
            "rank_distribution": rank_dist,
        }

        for pair, rev in reversals.items():
            log.info(f"  [Exp I] Reversal {pair}: {rev['reversal_rate']:.2%} "
                     f"({rev['reversal_count']}/{rev['total_variants']})")

    # --- Exp II metrics ---
    if len(exp2_df) > 0:
        df_filtered = exp2_df[exp2_df.question_id.isin(kept_qids)].copy()
        if len(df_filtered) > 0:
            acc_summary = exp2_accuracy_stats(df_filtered)
            flip = exp2_flip_rate(df_filtered)
            gaps = exp2_pairwise_bootstrap(df_filtered)
            reversals = exp2_reversal_frequency(df_filtered)
            rank_dist = exp2_rank_distribution(df_filtered)

            for a in acc_summary:
                m = a["model_short"]
                log.info(f"  [Exp II] {MODEL_LABELS.get(m, m)}: "
                         f"acc={a['mean']:.4f}±{a['std']:.4f}, "
                         f"flip={flip.get(m, {}).get('rate', 0):.4f}")

            result["exp2"] = {
                "accuracy_summary": acc_summary,
                "item_flip_rate": flip,
                "pairwise_gaps": gaps,
                "reversals": reversals,
                "rank_distribution": rank_dist,
            }

    return result


# ============================================================
# Summary tables
# ============================================================

def generate_summary(all_results: dict):
    """Generate cross-dataset summary tables."""
    rows = []
    for ds_key, analysis in all_results.items():
        for threshold_key in ["baseline", "remove_10pct", "remove_20pct", "remove_30pct"]:
            tr = analysis["threshold_results"].get(threshold_key, {})
            exp1 = tr.get("exp1", {})
            for m_e1 in MODELS_EXP1:
                m_e2 = MODEL_MAP_E1_TO_E2[m_e1]
                if m_e1 not in exp1:
                    continue
                e1 = exp1[m_e1]
                rows.append({
                    "dataset": DATASETS[ds_key],
                    "threshold": threshold_key,
                    "model": MODEL_LABELS[m_e2],
                    "size_B": MODEL_SIZES[m_e2],
                    "exp1_acc_mean": e1["accuracy"]["mean"],
                    "exp1_acc_std": e1["accuracy"]["std"],
                    "exp1_acc_range": e1["accuracy"]["range"],
                    "exp1_flip_rate": e1["flip_rate"],
                    "exp1_var_ratio": e1["variance_decomposition"]["ratio"],
                })

    if rows:
        df = pd.DataFrame(rows)
        csv_path = ANALYSIS_DIR / "summary_table_exp3.csv"
        df.to_csv(csv_path, index=False)
        log.info(f"\nSummary table saved to {csv_path}")
        print("\n" + df.to_string(index=False))


# ============================================================
# Main
# ============================================================

def main():
    all_results = {}

    for ds_key in DATASETS:
        try:
            analysis = analyze_dataset(ds_key)
            all_results[ds_key] = analysis

            # Save
            outpath = ANALYSIS_DIR / f"analysis_{ds_key}.json"
            outpath.write_text(json.dumps(analysis, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
            log.info(f"Saved to {outpath}")
        except FileNotFoundError as e:
            log.warning(f"Skipping {ds_key}: {e}")

    if all_results:
        generate_summary(all_results)

    print("\n" + "=" * 70)
    print("EXPERIMENT III ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
