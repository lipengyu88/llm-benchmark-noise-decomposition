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



def load_exp1_results(model_e1: str, dataset_key: str) -> dict:
    """Load Exp I results: nested dict qid -> variant_id -> {is_correct, ...}."""
    results_dir = EXP1_DIR / "results_exp1"
    path = results_dir / f"results_{model_e1}_{dataset_key}.json"
    if not path.exists():
        path = results_dir / f"checkpoint_{model_e1}_{dataset_key}.json"
    if not path.exists():
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_exp2_results(dataset_key: str, model_e2: str,
                      exp2_source: str = "both") -> list[dict]:
    """Load Exp II results: list of record dicts.

    exp2_source: "gpt4o", "qwen", or "both" (merge both sources).
    """
    bench = DATASETS[dataset_key]

    if exp2_source == "both":
        all_records = []
        for src in ["gpt4o", "qwen"]:
            path = EXP2_DIR / f"exp2_{bench}_{model_e2}_{src}.json"
            if path.exists():
                all_records.extend(json.loads(path.read_text(encoding="utf-8")))
        if all_records:
            return all_records
        path = EXP2_DIR / f"exp2_{bench}_{model_e2}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return []
    else:
        path = EXP2_DIR / f"exp2_{bench}_{model_e2}_{exp2_source}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        path = EXP2_DIR / f"exp2_{bench}_{model_e2}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return []


def load_dataset_items(dataset_key: str) -> dict:
    """Load raw dataset items for qualitative analysis."""
    orig_exp1 = EXP1_DIR
    if dataset_key == "arc":
        path = orig_exp1 / "arc_challenge_300.json"
    else:
        path = orig_exp1 / "mmlu_pro_300.json"
    items = {}
    with open(path, encoding="utf-8") as f:
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
    if dataset_key == "arc":
        for name in ["arc_challenge_paraphrased_gpt4o.json",
                      "arc_challenge_paraphrased_qwen.json",
                      "arc_challenge_paraphrased.json"]:
            path = EXP2_DIR / name
            if path.exists():
                break
    else:
        for name in ["mmlu_pro_paraphrased_gpt4o.json",
                      "mmlu_pro_paraphrased_qwen.json",
                      "mmlu_pro_paraphrased.json"]:
            path = EXP2_DIR / name
            if path.exists():
                break
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(q["question_id"]): q for q in data}



def compute_noise_scores(
    dataset_key: str,
    use_exp1: bool = True,
    use_exp2: bool = True,
    exp2_source: str = "both",
    shared_only: bool = False,
) -> dict:
    """
    Compute per-question noise scores combining Exp I and Exp II results.

    Args:
        exp2_source: "gpt4o", "qwen", or "both" — which Exp II source(s) to use.
        shared_only: If True, only include questions that appear in BOTH Exp I
                     and Exp II.  This avoids the bias where the 150 questions
                     without Exp II data receive systematically lower noise
                     scores due to fewer observation dimensions.

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
    item_data = defaultdict(lambda: defaultdict(list))
    exp2_qids = set()

    if use_exp1:
        for model_e1 in MODELS_EXP1:
            results = load_exp1_results(model_e1, dataset_key)
            model_e2 = MODEL_MAP_E1_TO_E2[model_e1]
            for qid, variants in results.items():
                for vid, entry in variants.items():
                    ic = entry.get("is_correct")
                    if ic is not None:
                        item_data[qid][model_e2].append((int(ic), "exp1"))

    if use_exp2:
        for model_e2 in MODELS_EXP2:
            records = load_exp2_results(dataset_key, model_e2,
                                        exp2_source=exp2_source)
            for rec in records:
                qid = str(rec["question_id"])
                ic = rec.get("is_correct")
                if ic is not None:
                    item_data[qid][model_e2].append((int(ic), "exp2"))
                    exp2_qids.add(qid)

    if shared_only and use_exp1 and use_exp2:
        exp1_qids = {qid for qid in item_data
                     if any(src == "exp1" for entries in item_data[qid].values()
                            for _, src in entries)}
        shared = exp1_qids & exp2_qids
        item_data = {qid: v for qid, v in item_data.items() if qid in shared}
        log.info(f"  shared_only: keeping {len(shared)} questions "
                 f"(exp1={len(exp1_qids)}, exp2={len(exp2_qids)})")

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
    sorted_items = sorted(noise_data.items(), key=lambda x: (x[1]["noise_score"], x[0]), reverse=True)
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



def analyze_noisy_items(
    noise_data: dict,
    dataset_key: str,
    top_n: int = 30,
) -> dict:
    """Qualitative analysis of the noisiest items."""
    items = load_dataset_items(dataset_key)
    paraphrased = load_exp2_paraphrased(dataset_key)

    sorted_items = sorted(noise_data.items(), key=lambda x: (x[1]["noise_score"], x[0]), reverse=True)
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

    for qid, nd in noise_data.items():
        ns = nd["noise_score"]
        bin_idx = min(int(ns * 10), 9)
        analysis["noise_score_distribution"]["counts"][bin_idx] += 1

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

    analysis["category_distribution"] = dict(analysis["category_distribution"])

    return analysis



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



def run(dataset_keys: list[str], use_exp1: bool, use_exp2: bool,
        exp2_source: str = "both", shared_only: bool = False):
    all_outputs = {}

    for ds_key in dataset_keys:
        bench = DATASETS[ds_key]
        log.info(f"\n{'='*60}")
        log.info(f"Dataset: {bench}")
        log.info(f"Sources: {'Exp I' if use_exp1 else ''} {'Exp II' if use_exp2 else ''}"
                 f" | exp2_source={exp2_source} | shared_only={shared_only}")
        log.info(f"{'='*60}")

        noise_data = compute_noise_scores(
            ds_key, use_exp1=use_exp1, use_exp2=use_exp2,
            exp2_source=exp2_source, shared_only=shared_only,
        )
        log.info(f"Computed noise scores for {len(noise_data)} questions")

        if not noise_data:
            log.warning(f"No data for {bench}, skipping.")
            continue

        source_stats = compute_per_source_noise(noise_data)
        log.info(f"Combined: mean_noise={source_stats['combined']['mean']:.4f}, "
                 f"pct_above_0.5={source_stats['combined']['pct_above_0.5']:.1f}%")
        if source_stats.get("exp1_only"):
            log.info(f"Exp I:    mean_noise={source_stats['exp1_only']['mean']:.4f}")
        if source_stats.get("exp2_only"):
            log.info(f"Exp II:   mean_noise={source_stats['exp2_only']['mean']:.4f}")

        removal_sets = compute_removal_sets(noise_data)
        for pct, rs in removal_sets.items():
            log.info(f"  Remove {pct}%: {rs['n_removed']} items, "
                     f"keep {rs['n_kept']}, cutoff={rs['noise_cutoff']:.4f}")

        qualitative = analyze_noisy_items(noise_data, ds_key)
        log.info(f"Model agreement: {qualitative['model_agreement']}")
        log.info(f"Top 5 noisiest items:")
        for item in qualitative["top_noisy_items"][:5]:
            log.info(f"  {item['qid']}: noise={item['noise_score']:.4f}, "
                     f"acc={item['accuracy']:.2f}, "
                     f"q={item['question'][:60]}...")

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

        tag = ""
        if use_exp1 and not use_exp2:
            tag = "_exp1only"
        elif use_exp2 and not use_exp1:
            tag = "_exp2only"
        if shared_only:
            tag += "_shared150"
        if exp2_source != "both" and use_exp2:
            tag += f"_{exp2_source}"
        outpath = OUTPUT_DIR / f"noise_{ds_key}{tag}.json"
        outpath.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info(f"Saved to {outpath}")

    return all_outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment III: High-Noise Item Analysis")
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), default=None)
    parser.add_argument("--exp1-only", action="store_true", help="Use only Exp I results")
    parser.add_argument("--exp2-only", action="store_true", help="Use only Exp II results")
    parser.add_argument("--exp2-source", choices=["gpt4o", "qwen", "both"], default="both",
                        help="Which Exp II paraphrase source to use (default: both)")
    parser.add_argument("--shared-only", action="store_true",
                        help="Only include questions present in both Exp I and Exp II "
                             "(avoids bias from unequal coverage)")
    args = parser.parse_args()

    dataset_keys = [args.dataset] if args.dataset else list(DATASETS.keys())
    use_exp1 = not args.exp2_only
    use_exp2 = not args.exp1_only

    run(dataset_keys, use_exp1=use_exp1, use_exp2=use_exp2,
        exp2_source=args.exp2_source, shared_only=args.shared_only)
