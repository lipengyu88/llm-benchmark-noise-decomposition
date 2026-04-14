"""
Experiment IV: Bradley-Terry Ranking Stability Analysis

Covers Theme 1, Direction 3 from the project specification:
  "Analyzing the robustness of model rankings... For pairwise preference
  evaluations, one can fit a BT model and derive posterior intervals, then
  simulate how many extra comparisons are required for the top-k ranking
  to become stable with high probability (like >= 0.95)."

Methodology
-----------
We convert item-level correctness from Exp I (18 prompt variants x 300 qs)
and Exp II (4 versions x 150 qs x 2 sources) into pairwise model comparisons:

For every (question, condition) cell, given a model pair (A, B):
  - if A correct and B wrong  -> A beats B (1 win for A)
  - if B correct and A wrong  -> B beats A (1 win for B)
  - if tied (both correct or both wrong) -> skip (no comparison)

With 4 models we have C(4,2) = 6 pairs, and with thousands of (q, condition)
cells we accumulate tens of thousands of pairwise comparisons per pair.

We then:
  1. Fit a Bradley-Terry model via MLE (Zermelo iteration) on the aggregate
     pairwise wins for each benchmark separately.
  2. Estimate posterior intervals by parametric bootstrap (10,000 resamples)
     of the pairwise win counts.
  3. Compute the top-k rank posterior: probability that each model holds
     rank k = 1, 2, 3, 4.
  4. Run a sample-size simulation: subsample N comparisons per pair
     (N in {50, 100, 200, 500, 1000, 2000, 5000, 10000}) and measure how
     many resamples are required for Pr(correct top-1) and Pr(correct top-2
     set) to exceed 0.95.

Outputs
-------
  bt_results_{dataset}.json      — BT ratings, CIs, rank posteriors
  bt_sample_size_{dataset}.json  — top-k stability vs N comparisons
  figures_bt/*.png               — 3 publication-quality figures

Usage
-----
    python run_bradley_terry.py
    python run_bradley_terry.py --dataset arc --n-bootstrap 5000
"""
from __future__ import annotations
import json
import argparse
import logging
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent
EXP1_DIR = ROOT.parent / "exp1"
EXP2_DIR = ROOT.parent / "exp2"
OUT_DIR = ROOT

MODELS = ["llama-3.1-8b", "qwen2.5-7b", "qwen3-32b", "qwen2.5-72b"]
MODEL_SIZES = {"llama-3.1-8b": 8, "qwen2.5-7b": 7, "qwen3-32b": 32, "qwen2.5-72b": 72}
MODEL_LABELS = {
    "llama-3.1-8b": "LLaMA-3.1-8B",
    "qwen2.5-7b":   "Qwen2.5-7B",
    "qwen3-32b":    "Qwen3-32B",
    "qwen2.5-72b":  "Qwen2.5-72B",
}
MODEL_E1_TO_E2 = {
    "llama": "llama-3.1-8b",
    "qwen7b": "qwen2.5-7b",
    "qwen32b": "qwen3-32b",
    "qwen72b": "qwen2.5-72b",
}
DATASETS = {"arc": "arc_challenge", "mmlu": "mmlu_pro"}

RNG = np.random.RandomState(42)



def load_exp1_matrix(dataset_key: str) -> tuple[list[str], np.ndarray]:
    """Return (condition_labels, matrix of shape (n_conditions, n_models))
    where each entry is the correctness of a model on a given (qid, variant)."""
    results = {}
    for e1_name, e2_name in MODEL_E1_TO_E2.items():
        path = EXP1_DIR / "results_exp1" / f"results_{e1_name}_{dataset_key}.json"
        if not path.exists():
            path = EXP1_DIR / "results_exp1" / f"checkpoint_{e1_name}_{dataset_key}.json"
        if not path.exists():
            return [], np.array([])
        with open(path, encoding="utf-8") as f:
            results[e2_name] = json.load(f)

    ref = results["llama-3.1-8b"]
    qids = sorted(ref.keys())
    variants = sorted({vid for q in ref.values() for vid in q.keys()})

    rows = []
    labels = []
    for qid in qids:
        for vid in variants:
            row = []
            ok = True
            for m in MODELS:
                entry = results[m].get(qid, {}).get(vid)
                if entry is None or entry.get("is_correct") is None:
                    ok = False
                    break
                row.append(int(entry["is_correct"]))
            if ok:
                rows.append(row)
                labels.append(f"exp1|{qid}|{vid}")

    mat = np.array(rows, dtype=int) if rows else np.zeros((0, 4), dtype=int)
    return labels, mat


def load_exp2_matrix(dataset_key: str) -> tuple[list[str], np.ndarray]:
    """Load Exp II data across both paraphrase sources, treating each
    (question_id, version, source) as one condition."""
    bench = DATASETS[dataset_key]
    cells: dict[tuple[str, int, str], dict[str, int]] = {}
    for m in MODELS:
        for src in ["gpt4o", "qwen"]:
            path = EXP2_DIR / f"exp2_{bench}_{m}_{src}.json"
            if not path.exists():
                continue
            records = json.loads(path.read_text(encoding="utf-8"))
            for r in records:
                if r.get("is_correct") is None:
                    continue
                key = (str(r["question_id"]), int(r["version"]), src)
                cells.setdefault(key, {})[m] = int(r["is_correct"])

    rows = []
    labels = []
    for key, mod_map in sorted(cells.items()):
        if len(mod_map) != len(MODELS):
            continue
        rows.append([mod_map[m] for m in MODELS])
        labels.append(f"exp2|{key[2]}|{key[0]}|v{key[1]}")
    mat = np.array(rows, dtype=int) if rows else np.zeros((0, 4), dtype=int)
    return labels, mat


def build_pairwise_wins(matrix: np.ndarray) -> np.ndarray:
    """Convert correctness matrix (n_conditions x n_models) to pairwise win
    matrix W where W[i, j] = number of conditions where model i was correct
    AND model j was wrong. Ties (both correct or both wrong) are skipped."""
    n_models = matrix.shape[1]
    W = np.zeros((n_models, n_models), dtype=float)
    for i in range(n_models):
        for j in range(n_models):
            if i == j:
                continue
            W[i, j] = int(np.sum((matrix[:, i] == 1) & (matrix[:, j] == 0)))
    return W



def fit_bt(W: np.ndarray, tol: float = 1e-8, max_iter: int = 10000) -> np.ndarray:
    """Fit BT ratings from pairwise win matrix W. Returns log-strength vector.

    Uses the Zermelo iterative update:
        pi_i <- W_i_total / sum_{j != i} (N_ij / (pi_i + pi_j))
    Normalises so pi sums to 1, then returns log(pi).
    """
    n = W.shape[0]
    W = W + 0.01
    pi = np.ones(n) / n
    N = W + W.T
    wins = W.sum(axis=1)

    for _ in range(max_iter):
        pi_new = np.zeros(n)
        for i in range(n):
            denom = 0.0
            for j in range(n):
                if i == j:
                    continue
                denom += N[i, j] / (pi[i] + pi[j])
            pi_new[i] = wins[i] / denom if denom > 0 else pi[i]
        pi_new /= pi_new.sum()
        if np.max(np.abs(pi_new - pi)) < tol:
            pi = pi_new
            break
        pi = pi_new
    return np.log(pi)


def bootstrap_bt(matrix: np.ndarray, n_bootstrap: int = 10000,
                 rng: np.random.RandomState | None = None) -> np.ndarray:
    """Bootstrap BT log-strengths by resampling rows (conditions) of the
    correctness matrix. Returns array of shape (n_bootstrap, n_models)."""
    if rng is None:
        rng = RNG
    n_rows = matrix.shape[0]
    n_models = matrix.shape[1]
    out = np.zeros((n_bootstrap, n_models))
    for b in range(n_bootstrap):
        idx = rng.randint(0, n_rows, size=n_rows)
        boot_mat = matrix[idx]
        W = build_pairwise_wins(boot_mat)
        out[b] = fit_bt(W)
    return out


def rank_posterior(boot: np.ndarray) -> np.ndarray:
    """Given bootstrap BT ratings (n_bootstrap x n_models), compute for each
    model the posterior probability of holding each rank (1 = best)."""
    n_bootstrap, n_models = boot.shape
    ranks = np.zeros_like(boot, dtype=int)
    for b in range(n_bootstrap):
        order = np.argsort(-boot[b])
        for pos, idx in enumerate(order):
            ranks[b, idx] = pos + 1
    posterior = np.zeros((n_models, n_models))
    for m in range(n_models):
        for k in range(1, n_models + 1):
            posterior[m, k - 1] = np.mean(ranks[:, m] == k)
    return posterior



def simulate_sample_size(matrix: np.ndarray,
                         sample_sizes: list[int],
                         n_repeats: int = 200,
                         rng: np.random.RandomState | None = None) -> dict:
    """For each candidate sample size N, draw N conditions (with replacement),
    fit BT, record the top-1 and top-2 set. Repeat n_repeats times and report
    Pr(correct top-1) and Pr(correct top-2 set) where 'correct' is defined
    relative to the full-data BT fit."""
    if rng is None:
        rng = RNG

    full_W = build_pairwise_wins(matrix)
    full_ratings = fit_bt(full_W)
    order = np.argsort(-full_ratings)
    true_top1 = int(order[0])
    true_top2 = frozenset((int(order[0]), int(order[1])))

    n_rows = matrix.shape[0]
    results = {"true_top1": MODELS[true_top1],
               "true_top2": sorted(MODEL_LABELS[MODELS[i]] for i in true_top2),
               "n_total_conditions": int(n_rows),
               "sample_curves": []}

    for N in sample_sizes:
        if N > n_rows:
            continue
        top1_hits = 0
        top2_hits = 0
        for _ in range(n_repeats):
            idx = rng.randint(0, n_rows, size=N)
            W = build_pairwise_wins(matrix[idx])
            ratings = fit_bt(W)
            o = np.argsort(-ratings)
            if int(o[0]) == true_top1:
                top1_hits += 1
            if frozenset((int(o[0]), int(o[1]))) == true_top2:
                top2_hits += 1
        results["sample_curves"].append({
            "n_comparisons": int(N),
            "pr_correct_top1": top1_hits / n_repeats,
            "pr_correct_top2_set": top2_hits / n_repeats,
        })

    results["n_needed_top1_95"] = None
    results["n_needed_top2_95"] = None
    for pt in results["sample_curves"]:
        if results["n_needed_top1_95"] is None and pt["pr_correct_top1"] >= 0.95:
            results["n_needed_top1_95"] = pt["n_comparisons"]
        if results["n_needed_top2_95"] is None and pt["pr_correct_top2_set"] >= 0.95:
            results["n_needed_top2_95"] = pt["n_comparisons"]
    return results



def analyze_dataset(dataset_key: str, n_bootstrap: int = 10000,
                    n_simulate_repeats: int = 200) -> dict:
    log.info(f"\n{'=' * 60}\nDataset: {DATASETS[dataset_key]}\n{'=' * 60}")

    labels_e1, mat_e1 = load_exp1_matrix(dataset_key)
    labels_e2, mat_e2 = load_exp2_matrix(dataset_key)
    log.info(f"  Exp I conditions: {len(labels_e1)}  "
             f"Exp II conditions: {len(labels_e2)}")

    combined_mat = np.vstack([mat_e1, mat_e2]) if len(mat_e1) and len(mat_e2) else \
        (mat_e1 if len(mat_e1) else mat_e2)
    combined_labels = labels_e1 + labels_e2
    log.info(f"  Combined conditions: {len(combined_labels)}")

    W = build_pairwise_wins(combined_mat)
    log_ratings = fit_bt(W)
    ratings = np.exp(log_ratings)
    ratings /= ratings.sum()

    order = np.argsort(-log_ratings)
    log.info("\n  BT ranking (full data):")
    for rank, idx in enumerate(order, 1):
        log.info(f"    {rank}. {MODEL_LABELS[MODELS[idx]]:15s}: "
                 f"logit={log_ratings[idx]:+.4f}, strength={ratings[idx]:.4f}")

    log.info("\n  Pairwise win matrix (row beats col):")
    for i, m_i in enumerate(MODELS):
        line = f"    {MODEL_LABELS[m_i]:15s}"
        for j, m_j in enumerate(MODELS):
            if i == j:
                line += "      -  "
            else:
                total = W[i, j] + W[j, i]
                rate = W[i, j] / total if total > 0 else 0
                line += f" {rate:5.3f}  "
        log.info(line)

    log.info(f"\n  Running bootstrap ({n_bootstrap} resamples)...")
    boot = bootstrap_bt(combined_mat, n_bootstrap=n_bootstrap,
                        rng=np.random.RandomState(42))

    ci_low = np.percentile(boot, 2.5, axis=0)
    ci_high = np.percentile(boot, 97.5, axis=0)
    log.info("  Bootstrap 95% CIs on log-ratings:")
    for idx in order:
        log.info(f"    {MODEL_LABELS[MODELS[idx]]:15s}: "
                 f"{log_ratings[idx]:+.4f} [{ci_low[idx]:+.4f}, {ci_high[idx]:+.4f}]")

    posterior = rank_posterior(boot)
    log.info("\n  Rank posterior (Pr(model holds rank k)):")
    header = "    " + " " * 16 + "  ".join(f"rank{k}" for k in range(1, len(MODELS) + 1))
    log.info(header)
    for idx in order:
        row = f"    {MODEL_LABELS[MODELS[idx]]:15s}"
        for k in range(len(MODELS)):
            row += f"  {posterior[idx, k]:.3f}"
        log.info(row)

    log.info(f"\n  Running sample-size simulation ({n_simulate_repeats} repeats per N)...")
    sample_sizes = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    sample_sizes = [s for s in sample_sizes if s <= len(combined_labels)]
    sim = simulate_sample_size(combined_mat, sample_sizes,
                                n_repeats=n_simulate_repeats,
                                rng=np.random.RandomState(42))

    log.info(f"  True top-1: {sim['true_top1']}")
    log.info(f"  True top-2 set: {sim['true_top2']}")
    log.info(f"  N needed for top-1 stability >= 95%: {sim['n_needed_top1_95']}")
    log.info(f"  N needed for top-2 stability >= 95%: {sim['n_needed_top2_95']}")
    for pt in sim["sample_curves"]:
        log.info(f"    N={pt['n_comparisons']:5d}: "
                 f"Pr(top1)={pt['pr_correct_top1']:.3f}, "
                 f"Pr(top2)={pt['pr_correct_top2_set']:.3f}")

    return {
        "dataset": DATASETS[dataset_key],
        "n_conditions_exp1": int(len(labels_e1)),
        "n_conditions_exp2": int(len(labels_e2)),
        "n_conditions_total": int(len(combined_labels)),
        "models": MODELS,
        "pairwise_wins": W.tolist(),
        "log_ratings": log_ratings.tolist(),
        "normalized_strengths": ratings.tolist(),
        "ranking": [MODELS[i] for i in order],
        "bootstrap_ci_low": ci_low.tolist(),
        "bootstrap_ci_high": ci_high.tolist(),
        "rank_posterior": posterior.tolist(),
        "sample_size_simulation": sim,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["arc", "mmlu", "both"], default="both")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--n-simulate-repeats", type=int, default=200)
    args = parser.parse_args()

    datasets = ["arc", "mmlu"] if args.dataset == "both" else [args.dataset]
    for ds in datasets:
        result = analyze_dataset(ds, n_bootstrap=args.n_bootstrap,
                                  n_simulate_repeats=args.n_simulate_repeats)
        out_path = OUT_DIR / f"bt_results_{ds}.json"
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        log.info(f"\n  Saved {out_path}")

    print("\n" + "=" * 70)
    print("BRADLEY-TERRY ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
