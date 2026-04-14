"""
Experiment II: Automated Paraphrase Semantic Faithfulness Check

Uses GPT-4o as a natural-language-inference judge to compute bidirectional
entailment scores for every paraphrase in Experiment II. This complements
the 50-sample manual QC: it scales to all 1800 paraphrases and gives each
paraphrase an objective, reproducible faithfulness grade.

For each paraphrase p and original question q, we ask GPT-4o:
  direction_1: Does p entail q? (does paraphrase preserve q's meaning?)
  direction_2: Does q entail p? (does paraphrase add unstated info?)

Both directions use a 3-way entailment label {entailment, neutral, contradiction}
plus a 1-5 confidence score.

A paraphrase is "faithful" iff BOTH directions return entailment.
Neutral in either direction => info added/lost (fails strict equivalence).
Contradiction in either direction => meaning changed.

Scope:
  4 files x 150 questions x 3 paraphrases = 1800 paraphrases
  1800 x 2 directions = 3600 GPT-4o calls
  Cost: ~$2 total

Usage:
    python semantic_faithfulness.py                 # all 4 files
    python semantic_faithfulness.py --source gpt4o  # only gpt4o source
    python semantic_faithfulness.py --max-items 50  # quick sanity check
"""
from __future__ import annotations
import asyncio
import json
import os
import re
import argparse
import logging
from pathlib import Path
from collections import Counter

import httpx


API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
if not API_KEY:
    _api_file = Path(__file__).parent.parent / "api.txt"
    if _api_file.exists():
        for line in _api_file.read_text().splitlines():
            if line.strip().startswith("sk-or-"):
                API_KEY = line.strip()
                break
    if not API_KEY:
        raise ValueError("Set OPENROUTER_API_KEY env var or place key in api.txt")

API_URL = "https://openrouter.ai/api/v1/chat/completions"
JUDGE_MODEL = "openai/gpt-4o"
TEMPERATURE = 0.0
MAX_TOKENS = 200
CONCURRENCY = 10
MAX_RETRIES = 5
CHECKPOINT_EVERY = 200

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

ROOT = Path(__file__).parent.parent
OUT_DIR = Path(__file__).parent

PARAPHRASE_FILES = {
    ("gpt4o", "arc"):  "arc_challenge_paraphrased_gpt4o.json",
    ("gpt4o", "mmlu"): "mmlu_pro_paraphrased_gpt4o.json",
    ("qwen", "arc"):   "arc_challenge_paraphrased_qwen.json",
    ("qwen", "mmlu"):  "mmlu_pro_paraphrased_qwen.json",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)



NLI_PROMPT = """\
You are an expert in natural language inference. Given a PREMISE and a \
HYPOTHESIS, decide whether the PREMISE entails the HYPOTHESIS.

Output exactly one of:
- entailment:    The premise clearly implies the hypothesis
- neutral:       The premise neither implies nor contradicts the hypothesis
- contradiction: The premise contradicts the hypothesis

Also give a confidence score (1-5) for your judgment.

Both premise and hypothesis are questions or statements about the same topic. \
If they are semantically identical (asking exactly the same thing), the label \
is entailment. If one asks about something the other does not mention, the \
label is neutral. If one asks something that contradicts the other, the label \
is contradiction.

PREMISE:
{premise}

HYPOTHESIS:
{hypothesis}

Output ONLY valid JSON in this exact format:
{{"label": "<entailment|neutral|contradiction>", "confidence": <1-5>}}"""



async def call_nli(client, sem, premise: str, hypothesis: str) -> dict | None:
    prompt = NLI_PROMPT.format(premise=premise, hypothesis=hypothesis)
    async with sem:
        for attempt in range(MAX_RETRIES):
            try:
                resp = await client.post(
                    API_URL,
                    headers=HEADERS,
                    json={
                        "model": JUDGE_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": TEMPERATURE,
                        "max_tokens": MAX_TOKENS,
                    },
                    timeout=60.0,
                )
                if resp.status_code == 429:
                    await asyncio.sleep(min(2 ** (attempt + 1), 30))
                    continue
                resp.raise_for_status()
                data = resp.json()
                raw = data["choices"][0]["message"]["content"].strip()
                m = re.search(r'\{.*\}', raw, re.DOTALL)
                if not m:
                    raise ValueError(f"No JSON: {raw[:150]}")
                parsed = json.loads(m.group())
                label = str(parsed["label"]).lower().strip()
                if label not in {"entailment", "neutral", "contradiction"}:
                    raise ValueError(f"Invalid label: {label}")
                return {
                    "label": label,
                    "confidence": int(parsed.get("confidence", 3)),
                }
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** (attempt + 1))
                else:
                    log.error(f"NLI call failed: {e}")
                    return None



async def evaluate_file(source: str, dataset: str, max_items: int | None = None):
    out_path = OUT_DIR / f"faithfulness_{dataset}_{source}.json"

    results = {}
    if out_path.exists():
        results = json.loads(out_path.read_text())
        log.info(f"  Resuming from checkpoint with {len(results)} entries")

    para_path = ROOT / PARAPHRASE_FILES[(source, dataset)]
    paraphrases = json.loads(para_path.read_text())
    if max_items:
        paraphrases = paraphrases[:max_items]

    tasks = []
    for q in paraphrases:
        qid = str(q["question_id"])
        original = q["question"]
        for pi, para in enumerate(q["paraphrases"]):
            key = f"{qid}_{pi}"
            if key in results and "bidirectional" in results[key]:
                continue
            tasks.append({
                "key": key,
                "question_id": qid,
                "paraphrase_index": pi,
                "original": original,
                "paraphrase": para,
            })

    log.info(f"  [{source}/{dataset}] {len(tasks)} paraphrases to check "
             f"(skipping {len(results)} from checkpoint)")

    if not tasks:
        log.info("  Already complete.")
        return results

    sem = asyncio.Semaphore(CONCURRENCY)
    done = 0

    async with httpx.AsyncClient() as client:
        async def process(task):
            nonlocal done
            dir1 = await call_nli(client, sem, premise=task["paraphrase"], hypothesis=task["original"])
            dir2 = await call_nli(client, sem, premise=task["original"], hypothesis=task["paraphrase"])

            bidirectional = (dir1["label"] == "entailment" and dir2["label"] == "entailment") \
                if (dir1 and dir2) else None

            done += 1
            if done % 50 == 0:
                log.info(f"    Progress: {done}/{len(tasks)}")
            return task["key"], {
                "question_id": task["question_id"],
                "paraphrase_index": task["paraphrase_index"],
                "dir_para_to_orig": dir1,
                "dir_orig_to_para": dir2,
                "bidirectional": bool(bidirectional) if bidirectional is not None else None,
                "original_snippet": task["original"][:150],
                "paraphrase_snippet": task["paraphrase"][:150],
            }

        for batch_start in range(0, len(tasks), CHECKPOINT_EVERY):
            batch = tasks[batch_start:batch_start + CHECKPOINT_EVERY]
            batch_results = await asyncio.gather(*[process(t) for t in batch])
            for key, entry in batch_results:
                results[key] = entry
            out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
            log.info(f"    Checkpoint saved: {len(results)} entries")

    return results


def summarize(all_results: dict) -> dict:
    """Aggregate faithfulness across sources and datasets."""
    summary = {
        "judge_model": JUDGE_MODEL,
        "judge_temperature": TEMPERATURE,
        "methodology": (
            "Bidirectional NLI via GPT-4o. For each paraphrase, we check: "
            "(1) paraphrase -> original (meaning preserved?) and "
            "(2) original -> paraphrase (info added?). A paraphrase is 'faithful' "
            "iff BOTH directions return entailment."
        ),
        "by_source_dataset": {},
        "by_source": {},
        "overall": {},
    }

    by_src = {"gpt4o": [], "qwen": []}
    overall = []

    for (source, dataset), results in all_results.items():
        key = f"{source}_{dataset}"
        valid = [r for r in results.values() if r.get("bidirectional") is not None]
        if not valid:
            continue

        n = len(valid)
        n_faithful = sum(1 for r in valid if r["bidirectional"])

        d1_labels = Counter(r["dir_para_to_orig"]["label"] for r in valid)
        d2_labels = Counter(r["dir_orig_to_para"]["label"] for r in valid)

        summary["by_source_dataset"][key] = {
            "n": n,
            "n_faithful_bidirectional": n_faithful,
            "faithful_rate": n_faithful / n,
            "dir1_para_to_orig": dict(d1_labels),
            "dir2_orig_to_para": dict(d2_labels),
        }

        by_src[source].extend(valid)
        overall.extend(valid)

    for src, items in by_src.items():
        if not items:
            continue
        n_faithful = sum(1 for r in items if r["bidirectional"])
        summary["by_source"][src] = {
            "n": len(items),
            "n_faithful_bidirectional": n_faithful,
            "faithful_rate": n_faithful / len(items),
        }

    if overall:
        n_faithful = sum(1 for r in overall if r["bidirectional"])
        summary["overall"] = {
            "n": len(overall),
            "n_faithful_bidirectional": n_faithful,
            "faithful_rate": n_faithful / len(overall),
        }

    return summary


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["gpt4o", "qwen", "both"], default="both")
    parser.add_argument("--dataset", choices=["arc", "mmlu", "both"], default="both")
    parser.add_argument("--max-items", type=int, default=None,
                        help="Limit questions per file (for sanity checking)")
    args = parser.parse_args()

    sources = ["gpt4o", "qwen"] if args.source == "both" else [args.source]
    datasets = ["arc", "mmlu"] if args.dataset == "both" else [args.dataset]

    all_results = {}
    for source in sources:
        for dataset in datasets:
            log.info(f"\n--- {source} / {dataset} ---")
            results = await evaluate_file(source, dataset, max_items=args.max_items)
            all_results[(source, dataset)] = results

    summary = summarize(all_results)
    summary_path = OUT_DIR / "faithfulness_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    log.info(f"\nSaved summary: {summary_path}")

    print("\n" + "=" * 70)
    print("BIDIRECTIONAL ENTAILMENT FAITHFULNESS SUMMARY (GPT-4o)")
    print("=" * 70)
    for key, stats in summary["by_source_dataset"].items():
        print(f"\n{key}: {stats['n_faithful_bidirectional']}/{stats['n']} "
              f"faithful ({stats['faithful_rate']*100:.1f}%)")
        print(f"  direction 1 (para->orig): {stats['dir1_para_to_orig']}")
        print(f"  direction 2 (orig->para): {stats['dir2_orig_to_para']}")
    if summary["by_source"]:
        print("\nBy source:")
        for src, s in summary["by_source"].items():
            print(f"  {src}: {s['n_faithful_bidirectional']}/{s['n']} "
                  f"({s['faithful_rate']*100:.1f}%)")
    if summary["overall"]:
        o = summary["overall"]
        print(f"\nOVERALL: {o['n_faithful_bidirectional']}/{o['n']} "
              f"({o['faithful_rate']*100:.1f}%)")


if __name__ == "__main__":
    asyncio.run(main())
