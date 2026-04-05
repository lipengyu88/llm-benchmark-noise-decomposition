"""
Experiment II: Paraphrase Quality Check — Manual-Style Review of 50 Samples

Implements the QC commitment from the project proposal: "We manually inspect
50 random samples to verify semantic and label correctness."

Since human manual annotation is impractical for this pipeline, we use
GPT-4o as a rigorous LLM-as-judge with an explicit rubric and chain-of-
thought reasoning. This is a standard practice in recent evaluation
literature (e.g., G-Eval, LLM-as-a-Judge), but it is clearly labelled as
LLM-assisted review throughout the outputs so readers can calibrate their
trust accordingly.

Sampling strategy:
  - 25 paraphrases from GPT-4o source + 25 from Qwen source = 50 total
  - Stratified across the two benchmarks (13 ARC + 12 MMLU per source)
  - Within each (source, benchmark) stratum, random with seed=42
  - We sample ONE paraphrase per question to avoid question-level coupling
  - The paraphrase index (0/1/2) is also randomised

For each sample, GPT-4o rates three dimensions on a 1–5 scale:
  1. semantic_equivalence: Is the paraphrase semantically equivalent?
  2. answer_invariance:    Would the correct answer stay the same?
  3. information_preservation: Is all task-relevant info preserved?

A paraphrase PASSES QC if all three scores are >= 4 (i.e. "good" or "excellent").

Output: paraphrase_qc_manual.csv with per-sample scores, rationales, and
overall pass/fail flags. Aggregate pass rates are printed to stdout and
also written to paraphrase_qc_summary.json.

Cost estimate: 50 GPT-4o calls with ~500 input + ~300 output tokens each
≈ $0.20 total.
"""
from __future__ import annotations
import asyncio
import json
import os
import random
import re
import csv
import logging
from pathlib import Path

import httpx

# ============================================================
# Configuration
# ============================================================

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
MAX_TOKENS = 800
CONCURRENCY = 4
MAX_RETRIES = 5

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

ROOT = Path(__file__).parent.parent
OUT_DIR = Path(__file__).parent
SEED = 42

# 25 per source, split across datasets
SAMPLE_PLAN = [
    ("gpt4o", "arc", 13),
    ("gpt4o", "mmlu", 12),
    ("qwen", "arc", 13),
    ("qwen", "mmlu", 12),
]

PARAPHRASE_FILES = {
    "gpt4o": {
        "arc":  "arc_challenge_paraphrased_gpt4o.json",
        "mmlu": "mmlu_pro_paraphrased_gpt4o.json",
    },
    "qwen": {
        "arc":  "arc_challenge_paraphrased_qwen.json",
        "mmlu": "mmlu_pro_paraphrased_qwen.json",
    },
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ============================================================
# Sampling
# ============================================================

def sample_paraphrases() -> list[dict]:
    """Stratified random sampling of 50 paraphrases."""
    rng = random.Random(SEED)
    samples = []

    for source, dataset, n in SAMPLE_PLAN:
        path = ROOT / PARAPHRASE_FILES[source][dataset]
        data = json.loads(path.read_text())
        # Sample n distinct questions
        selected_questions = rng.sample(data, n)
        for q in selected_questions:
            # Randomly pick one of the 3 paraphrases
            pi = rng.randrange(3)
            samples.append({
                "source": source,
                "dataset": dataset,
                "question_id": q["question_id"],
                "paraphrase_index": pi,
                "original_question": q["question"],
                "paraphrase": q["paraphrases"][pi],
                "choices": q["choices"],
                "labels": q["labels"],
                "correct_answer": q["answer"],
            })
    return samples


# ============================================================
# Judge prompt
# ============================================================

JUDGE_PROMPT = """\
You are an expert test-item reviewer. Evaluate whether a paraphrased version \
of a multiple-choice question preserves the original meaning, correct answer, \
and all task-relevant information.

ORIGINAL QUESTION:
{original}

PARAPHRASED QUESTION:
{paraphrase}

ANSWER CHOICES (unchanged):
{choices}

CORRECT ANSWER: {answer}

Rate the paraphrase on three dimensions using a 1-5 scale:

1. SEMANTIC_EQUIVALENCE (1-5):
   5 = Perfect semantic match; meaning fully preserved
   4 = Minor surface variations but same meaning
   3 = Mostly equivalent with small nuance changes
   2 = Noticeable meaning drift
   1 = Substantially different meaning

2. ANSWER_INVARIANCE (1-5):
   5 = The same answer is clearly correct given the paraphrase
   4 = The same answer is correct, but reasoning is slightly different
   3 = The same answer is probably still correct
   2 = The correct answer might change
   1 = The correct answer clearly changes

3. INFORMATION_PRESERVATION (1-5):
   5 = All task-relevant info preserved; nothing added or removed
   4 = Minor non-essential info changed
   3 = Some relevant info restructured or slightly lost
   2 = Important info missing or added
   1 = Substantial information loss or addition

Output ONLY valid JSON in this exact format:
{{"semantic_equivalence": <1-5>, "answer_invariance": <1-5>, "information_preservation": <1-5>, "rationale": "<one short sentence>"}}"""


# ============================================================
# Judge API call
# ============================================================

async def judge_sample(client, sem, sample: dict) -> dict:
    choices_str = "\n".join(f"{lb}. {ch}" for lb, ch in zip(sample["labels"], sample["choices"]))
    prompt = JUDGE_PROMPT.format(
        original=sample["original_question"],
        paraphrase=sample["paraphrase"],
        choices=choices_str,
        answer=sample["correct_answer"],
    )

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
                    timeout=90.0,
                )
                if resp.status_code == 429:
                    await asyncio.sleep(min(2 ** (attempt + 1), 30))
                    continue
                resp.raise_for_status()
                data = resp.json()
                raw = data["choices"][0]["message"]["content"].strip()

                # Extract JSON
                m = re.search(r'\{.*\}', raw, re.DOTALL)
                if not m:
                    raise ValueError(f"No JSON in response: {raw[:200]}")
                parsed = json.loads(m.group())

                return {
                    **sample,
                    "semantic_equivalence": int(parsed["semantic_equivalence"]),
                    "answer_invariance": int(parsed["answer_invariance"]),
                    "information_preservation": int(parsed["information_preservation"]),
                    "rationale": str(parsed.get("rationale", "")),
                    "raw_judge_response": raw,
                }
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** (attempt + 1))
                else:
                    log.error(f"Failed: {e}")
                    return {
                        **sample,
                        "semantic_equivalence": None,
                        "answer_invariance": None,
                        "information_preservation": None,
                        "rationale": f"ERROR: {e}",
                        "raw_judge_response": None,
                    }


# ============================================================
# Main
# ============================================================

async def main():
    log.info("Sampling 50 paraphrases (stratified by source x dataset)...")
    samples = sample_paraphrases()
    log.info(f"Sampled {len(samples)} paraphrases")

    log.info(f"Judging with {JUDGE_MODEL} (temperature={TEMPERATURE})...")
    sem = asyncio.Semaphore(CONCURRENCY)
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*[judge_sample(client, sem, s) for s in samples])

    # Add pass/fail flag
    for r in results:
        scores = [r.get("semantic_equivalence"), r.get("answer_invariance"),
                  r.get("information_preservation")]
        if None in scores:
            r["passed"] = None
            r["min_score"] = None
        else:
            r["min_score"] = min(scores)
            r["passed"] = int(r["min_score"] >= 4)

    # Save CSV
    csv_path = OUT_DIR / "paraphrase_qc_manual.csv"
    fieldnames = ["source", "dataset", "question_id", "paraphrase_index",
                  "semantic_equivalence", "answer_invariance",
                  "information_preservation", "min_score", "passed",
                  "rationale", "original_question", "paraphrase"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    log.info(f"Saved {csv_path}")

    # Aggregate by source
    summary = {
        "judge_model": JUDGE_MODEL,
        "judge_temperature": TEMPERATURE,
        "n_samples": len(results),
        "by_source": {},
        "overall": {},
        "note": ("This is LLM-assisted QC using GPT-4o as judge. "
                 "A sample PASSES if all three scores (semantic_equivalence, "
                 "answer_invariance, information_preservation) are >= 4/5."),
    }

    for source in ["gpt4o", "qwen"]:
        sub = [r for r in results if r["source"] == source and r["passed"] is not None]
        if not sub:
            continue
        n_pass = sum(r["passed"] for r in sub)
        summary["by_source"][source] = {
            "n": len(sub),
            "n_passed": n_pass,
            "pass_rate": n_pass / len(sub),
            "mean_semantic_equivalence": sum(r["semantic_equivalence"] for r in sub) / len(sub),
            "mean_answer_invariance": sum(r["answer_invariance"] for r in sub) / len(sub),
            "mean_information_preservation": sum(r["information_preservation"] for r in sub) / len(sub),
        }

    # Overall
    valid = [r for r in results if r["passed"] is not None]
    if valid:
        n_pass = sum(r["passed"] for r in valid)
        summary["overall"] = {
            "n": len(valid),
            "n_passed": n_pass,
            "pass_rate": n_pass / len(valid),
            "mean_semantic_equivalence": sum(r["semantic_equivalence"] for r in valid) / len(valid),
            "mean_answer_invariance": sum(r["answer_invariance"] for r in valid) / len(valid),
            "mean_information_preservation": sum(r["information_preservation"] for r in valid) / len(valid),
        }

    json_path = OUT_DIR / "paraphrase_qc_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    log.info(f"Saved {json_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("PARAPHRASE MANUAL QC SUMMARY (GPT-4o as judge)")
    print("=" * 70)
    for source, stats in summary["by_source"].items():
        print(f"\n{source.upper()} source (n={stats['n']}):")
        print(f"  Pass rate: {stats['n_passed']}/{stats['n']} ({stats['pass_rate']*100:.1f}%)")
        print(f"  Mean semantic_equivalence:     {stats['mean_semantic_equivalence']:.2f}/5")
        print(f"  Mean answer_invariance:        {stats['mean_answer_invariance']:.2f}/5")
        print(f"  Mean information_preservation: {stats['mean_information_preservation']:.2f}/5")
    if summary["overall"]:
        o = summary["overall"]
        print(f"\nOVERALL (n={o['n']}):")
        print(f"  Pass rate: {o['n_passed']}/{o['n']} ({o['pass_rate']*100:.1f}%)")


if __name__ == "__main__":
    asyncio.run(main())
