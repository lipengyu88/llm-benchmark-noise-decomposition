"""
Experiment III: High-Noise Item Analysis

Combines item-level results from Experiments I and II to compute per-question
noise scores, identify high-noise items, and generate filtered datasets at
different removal thresholds (10%, 20%, 30%).

No new API calls — this is an *analysis phase* that operates entirely on
existing results from Exp I and Exp II.  It is positioned as a separate
experiment for organizational clarity, but methodologically it is downstream
analysis rather than an independent data-collection step.

Noise score formula:
    Noise(q) = 1 - |2c(q) - N(q)| / N(q)
where c(q) = number of conditions answered correctly, N(q) = total conditions.

Properties:
  - Noise = 0  when c(q) = 0 or c(q) = N(q)  (perfectly consistent)
  - Noise = 1  when c(q) = N(q)/2             (maximally inconsistent)

Known limitation:
  Items that ALL models get WRONG under ALL conditions receive Noise = 0,
  treating them as "perfectly consistent" — the same score as items all
  models always get right.  Such items may include mislabeled ground-truth
  answers or questions beyond all tested models' capabilities.  They are
  retained as "low noise" by the filtering procedure, which is correct
  from a *measurement stability* perspective (they don't flip) but
  potentially misleading from a *benchmark quality* perspective.  The
  report discusses this distinction and recommends complementing noise
  filtering with difficulty-stratified analysis (see analyze_experiment3.py
  noise-vs-difficulty section).

Usage:
    python run_experiment3.py                    # run all
    python run_experiment3.py --dataset arc       # one dataset
    python run_experiment3.py --exp1-only         # noise from Exp I only
    python run_experiment3.py --exp2-only         # noise from Exp II only
"""
from __future__ import annotations
import json
import argparse
import logging
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent
EXP1_DIR = ROOT.parent / "exp1"
EXP2_DIR = ROOT.parent / "exp2"
OUTPUT_DIR = ROOT / "noise_data"
OUTPUT_DIR.mkdir(exist_ok=True)

MODELS_EXP1 = ["llama", "qwen7b", "qwen32b", "qwen72b"]
MODELS_EXP2 = ["llama-3.1-8b", "qwen2.5-7b", "qwen3-32b", "qwen2.5-72b"]
MODEL_MAP_E1_TO_E2 = {
    "llama": "llama-3.1-8b",
    "qwen7b": "qwen2.5-7b",
    "qwen32b": "qwen3-32b",
    "qwen72b": "qwen2.5-72b",
}
MODEL_MAP_E2_TO_E1 = {v: k for k, v in MODEL_MAP_E1_TO_E2.items()}

DATASETS = {"arc": "arc_challenge", "mmlu": "mmlu_pro"}
REMOVAL_THRESHOLDS = [10, 20, 30]


# ============================================================
# Data loading
# ============================================================

def load_exp1_results(model_e1: str, dataset_key: str) -> dict:
    """Load Exp I results: nested dict qid -> variant_id -> {is_correct, ...}."""
    results_dir = EXP1_DIR / "results_exp1"
    path = results_dir / f"results_{model_e1}_{dataset_key}.json"
    if not path.exists():
        path = results_dir / f"checkpoint_{model_e1}_{dataset_key}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def load_exp2_results(dataset_key: str, model_e2: str) -> list[dict]:
    """Load Exp II results: list of record dicts."""
    bench = DATASETS[dataset_key]
    path = EXP2_DIR / f"exp2_{bench}_{model_e2}.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())


def load_dataset_items(dataset_key: str) -> dict:
    """Load raw dataset items for qualitative analysis."""
    if dataset_key == "arc":
        path = EXP1_DIR / "arc_challenge_300.json"
    else:
        path = EXP1_DIR / "mmlu_pro_300.json"
    items = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            qid = item["id"] if dataset_key == "arc" else str(item["question_id"])
            items[qid] = item
    return items


def load_exp2_paraphrased(dataset_key: str) -> dict:
    """Load paraphrased question data for qualitative analysis."""
    bench = DATASETS[dataset_key]
    path = EXP2_DIR / f"{bench}_paraphrased.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {str(q["question_id"]): q for q in data}


# ============================================================
# Noise score computation
# ============================================================

def compute_noise_scores(
    dataset_key: str,
    use_exp1: bool = True,
    use_exp2: bool = True,
) -> dict:
    """
    Compute per-question noise scores combining Exp I and Exp II results.

    Returns dict: qid -> {
        "noise_score": float,
        "correct": int,
        "total": int,
        "per_model": {model: {"correct": int, "total": int, "noise": float}},
        "exp1_correct": int, "exp1_total": int,
        "exp2_correct": int, "exp2_total": int,
        "sources": ["exp1", "exp2"],
    }
    """
    # Aggregate: qid -> model -> list of (is_correct, source)
    item_data = defaultdict(lambda: defaultdict(list))

    # Load Exp I results
    if use_exp1:
        for model_e1 in MODELS_EXP1:
            results = load_exp1_results(model_e1, dataset_key)
            model_e2 = MODEL_MAP_E1_TO_E2[model_e1]
            for qid, variants in results.items():
                for vid, entry in variants.items():
                    ic = entry.get("is_correct")
                    if ic is not None:
                        item_data[qid][model_e2].append((int(ic), "exp1"))

    # Load Exp II results
    if use_exp2:
        for model_e2 in MODELS_EXP2:
            records = load_exp2_results(dataset_key, model_e2)
            for rec in records:
                qid = str(rec["question_id"])
                ic = rec.get("is_correct")
                if ic is not None:
                    item_data[qid][model_e2].append((int(ic), "exp2"))

    # Compute noise scores
    noise_data = {}
    for qid in sorted(item_data.keys()):
        total_correct = 0
        total_count = 0
        exp1_correct = 0
        exp1_total = 0
        exp2_correct = 0
        exp2_total = 0
        per_model = {}
        sources = set()

        for model, entries in item_data[qid].items():
            m_correct = sum(e[0] for e in entries)
            m_total = len(entries)
            m_noise = 1.0 - abs(2 * m_correct - m_total) / m_total if m_total > 0 else 0.0
            per_model[model] = {
                "correct": m_correct,
                "total": m_total,
                "noise": round(m_noise, 6),
            }
            total_correct += m_correct
            total_count += m_total
            for c, src in entries:
                sources.add(src)
                if src == "exp1":
                    exp1_correct += c
                    exp1_total += 1
                else:
                    exp2_correct += c
                    exp2_total += 1

        noise_score = 1.0 - abs(2 * total_correct - total_count) / total_count if total_count > 0 else 0.0

        noise_data[qid] = {
            "noise_score": round(noise_score, 6),
            "correct": total_correct,
            "total": total_count,
            "per_model": per_model,
            "exp1_correct": exp1_correct,
            "exp1_total": exp1_total,
            "exp2_correct": exp2_correct,
            "exp2_total": exp2_total,
            "sources": sorted(sources),
        }

    return noise_data


# ============================================================
# Threshold-based filtering
# ============================================================

def compute_removal_sets(noise_data: dict, thresholds: list[int] = REMOVAL_THRESHOLDS) -> dict:
    """
    For each threshold (percent), determine which question IDs to remove.

    Returns dict: threshold_pct -> {
        "removed_qids": list,
        "kept_qids": list,
        "noise_cutoff": float,
        "n_removed": int,
        "n_kept": int,
    }
    """
    # Sort by noise score descending
    sorted_items = sorted(noise_data.items(), key=lambda x: x[1]["noise_score"], reverse=True)
    n_total = len(sorted_items)

    removal_sets = {}
    for pct in thresholds:
        n_remove = int(n_total * pct / 100)
        removed = [qid for qid, _ in sorted_items[:n_remove]]
        kept = [qid for qid, _ in sorted_items[n_remove:]]
        cutoff = sorted_items[n_remove - 1][1]["noise_score"] if n_remove > 0 else 0.0

        removal_sets[pct] = {
            "removed_qids": removed,
            "kept_qids": kept,
            "noise_cutoff": cutoff,
            "n_removed": n_remove,
            "n_kept": n_total - n_remove,
        }

    return removal_sets


# ============================================================
# Qualitative analysis of noisy items
# ============================================================

def analyze_noisy_items(
    noise_data: dict,
    dataset_key: str,
    top_n: int = 30,
) -> dict:
    """Qualitative analysis of the noisiest items."""
    items = load_dataset_items(dataset_key)
    paraphrased = load_exp2_paraphrased(dataset_key)

    sorted_items = sorted(noise_data.items(), key=lambda x: x[1]["noise_score"], reverse=True)
    top_noisy = sorted_items[:top_n]

    analysis = {
        "top_noisy_items": [],
        "category_distribution": defaultdict(int),
        "noise_score_distribution": {
            "bins": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "counts": [0] * 10,
        },
        "model_agreement": {
            "all_agree_correct": 0,
            "all_agree_wrong": 0,
            "mixed": 0,
        },
    }

    # Noise score histogram
    for qid, nd in noise_data.items():
        ns = nd["noise_score"]
        bin_idx = min(int(ns * 10), 9)
        analysis["noise_score_distribution"]["counts"][bin_idx] += 1

    # Model agreement analysis
    for qid, nd in noise_data.items():
        per_model = nd["per_model"]
        if not per_model:
            continue
        model_acc = [m["correct"] / m["total"] for m in per_model.values() if m["total"] > 0]
        if all(a > 0.9 for a in model_acc):
            analysis["model_agreement"]["all_agree_correct"] += 1
        elif all(a < 0.1 for a in model_acc):
            analysis["model_agreement"]["all_agree_wrong"] += 1
        else:
            analysis["model_agreement"]["mixed"] += 1

    # Top noisy items detail
    for qid, nd in top_noisy:
        item_info = items.get(qid, {})
        para_info = paraphrased.get(qid, {})

        detail = {
            "qid": qid,
            "noise_score": nd["noise_score"],
            "correct": nd["correct"],
            "total": nd["total"],
            "accuracy": nd["correct"] / nd["total"] if nd["total"] > 0 else 0,
        }

        if dataset_key == "arc":
            detail["question"] = item_info.get("question", "")
            detail["n_choices"] = len(item_info.get("choices", {}).get("label", []))
            detail["answer_key"] = item_info.get("answerKey", "")
        else:
            detail["question"] = item_info.get("question", "")
            detail["category"] = item_info.get("category", "unknown")
            detail["n_options"] = len(item_info.get("options", []))
            detail["answer"] = item_info.get("answer", "")
            analysis["category_distribution"][detail["category"]] += 1

        detail["has_paraphrases"] = qid in paraphrased
        detail["per_model_accuracy"] = {
            m: round(info["correct"] / info["total"], 4) if info["total"] > 0 else None
            for m, info in nd["per_model"].items()
        }

        analysis["top_noisy_items"].append(detail)

    # Category analysis for MMLU
    if dataset_key == "mmlu":
        cat_noise = defaultdict(list)
        for qid, nd in noise_data.items():
            cat = items.get(qid, {}).get("category", "unknown")
            cat_noise[cat].append(nd["noise_score"])

        analysis["category_noise"] = {
            cat: {
                "n_questions": len(scores),
                "mean_noise": round(sum(scores) / len(scores), 4),
                "n_noisy_above_0.5": sum(1 for s in scores if s > 0.5),
            }
            for cat, scores in sorted(cat_noise.items(), key=lambda x: -sum(x[1]) / len(x[1]))
        }

    # Convert defaultdict
    analysis["category_distribution"] = dict(analysis["category_distribution"])

    return analysis


# ============================================================
# Per-model noise scores (for Exp I and Exp II separately)
# ============================================================

def compute_per_source_noise(noise_data: dict) -> dict:
    """Compute aggregate stats for exp1 vs exp2 noise contributions."""
    exp1_scores = []
    exp2_scores = []
    combined_scores = []

    for qid, nd in noise_data.items():
        combined_scores.append(nd["noise_score"])

        if nd["exp1_total"] > 0:
            e1_noise = 1.0 - abs(2 * nd["exp1_correct"] - nd["exp1_total"]) / nd["exp1_total"]
            exp1_scores.append(e1_noise)
        if nd["exp2_total"] > 0:
            e2_noise = 1.0 - abs(2 * nd["exp2_correct"] - nd["exp2_total"]) / nd["exp2_total"]
            exp2_scores.append(e2_noise)

    def stats(scores):
        if not scores:
            return {}
        import numpy as np
        arr = np.array(scores)
        return {
            "n": len(scores),
            "mean": round(float(arr.mean()), 4),
            "std": round(float(arr.std()), 4),
            "median": round(float(np.median(arr)), 4),
            "pct_above_0.5": round(sum(1 for s in scores if s > 0.5) / len(scores) * 100, 2),
        }

    return {
        "combined": stats(combined_scores),
        "exp1_only": stats(exp1_scores),
        "exp2_only": stats(exp2_scores),
    }


# ============================================================
# Main
# ============================================================

def run(dataset_keys: list[str], use_exp1: bool, use_exp2: bool):
    all_outputs = {}

    for ds_key in dataset_keys:
        bench = DATASETS[ds_key]
        log.info(f"\n{'='*60}")
        log.info(f"Dataset: {bench}")
        log.info(f"Sources: {'Exp I' if use_exp1 else ''} {'Exp II' if use_exp2 else ''}")
        log.info(f"{'='*60}")

        # Compute noise scores
        noise_data = compute_noise_scores(ds_key, use_exp1=use_exp1, use_exp2=use_exp2)
        log.info(f"Computed noise scores for {len(noise_data)} questions")

        if not noise_data:
            log.warning(f"No data for {bench}, skipping.")
            continue

        # Per-source stats
        source_stats = compute_per_source_noise(noise_data)
        log.info(f"Combined: mean_noise={source_stats['combined']['mean']:.4f}, "
                 f"pct_above_0.5={source_stats['combined']['pct_above_0.5']:.1f}%")
        if source_stats.get("exp1_only"):
            log.info(f"Exp I:    mean_noise={source_stats['exp1_only']['mean']:.4f}")
        if source_stats.get("exp2_only"):
            log.info(f"Exp II:   mean_noise={source_stats['exp2_only']['mean']:.4f}")

        # Compute removal sets
        removal_sets = compute_removal_sets(noise_data)
        for pct, rs in removal_sets.items():
            log.info(f"  Remove {pct}%: {rs['n_removed']} items, "
                     f"keep {rs['n_kept']}, cutoff={rs['noise_cutoff']:.4f}")

        # Qualitative analysis
        qualitative = analyze_noisy_items(noise_data, ds_key)
        log.info(f"Model agreement: {qualitative['model_agreement']}")
        log.info(f"Top 5 noisiest items:")
        for item in qualitative["top_noisy_items"][:5]:
            log.info(f"  {item['qid']}: noise={item['noise_score']:.4f}, "
                     f"acc={item['accuracy']:.2f}, "
                     f"q={item['question'][:60]}...")

        # Package output
        output = {
            "dataset": bench,
            "n_questions": len(noise_data),
            "sources": {"exp1": use_exp1, "exp2": use_exp2},
            "source_stats": source_stats,
            "noise_scores": noise_data,
            "removal_sets": {
                str(pct): {
                    "removed_qids": rs["removed_qids"],
                    "kept_qids": rs["kept_qids"],
                    "noise_cutoff": rs["noise_cutoff"],
                    "n_removed": rs["n_removed"],
                    "n_kept": rs["n_kept"],
                }
                for pct, rs in removal_sets.items()
            },
            "qualitative_analysis": qualitative,
        }
        all_outputs[ds_key] = output

        # Save
        tag = ""
        if use_exp1 and not use_exp2:
            tag = "_exp1only"
        elif use_exp2 and not use_exp1:
            tag = "_exp2only"
        outpath = OUTPUT_DIR / f"noise_{ds_key}{tag}.json"
        outpath.write_text(json.dumps(output, indent=2, ensure_ascii=False))
        log.info(f"Saved to {outpath}")

    return all_outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment III: High-Noise Item Analysis")
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), default=None)
    parser.add_argument("--exp1-only", action="store_true", help="Use only Exp I results")
    parser.add_argument("--exp2-only", action="store_true", help="Use only Exp II results")
    args = parser.parse_args()

    dataset_keys = [args.dataset] if args.dataset else list(DATASETS.keys())
    use_exp1 = not args.exp2_only
    use_exp2 = not args.exp1_only

    run(dataset_keys, use_exp1=use_exp1, use_exp2=use_exp2)
