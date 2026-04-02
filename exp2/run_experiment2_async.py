"""
Experiment II: Test-Set Resampling via Paraphrasing — Async Runner

Evaluates 4 models on original + 3 paraphrased versions of 150 questions
per benchmark, using the base prompt template (v00) to isolate test-set
variation from prompt variation.

Supports multiple paraphrase sources (GPT-4o and Qwen2.5-72B) for
cross-source comparison.

Usage:
    python run_experiment2_async.py                          # GPT-4o source, all
    python run_experiment2_async.py --source qwen            # Qwen source
    python run_experiment2_async.py --source both            # both sources
    python run_experiment2_async.py --model llama --dataset arc
"""
from __future__ import annotations
import asyncio
import json
import re
import os
import time
import argparse
import logging
from pathlib import Path

import httpx

# ============================================================
# Configuration
# ============================================================

API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
if not API_KEY:
    _api_file = Path(__file__).parent / "api.txt"
    if _api_file.exists():
        for line in _api_file.read_text().splitlines():
            if line.strip().startswith("sk-or-"):
                API_KEY = line.strip()
                break
    if not API_KEY:
        raise ValueError("Set OPENROUTER_API_KEY env var or place key in api.txt")

API_URL = "https://openrouter.ai/api/v1/chat/completions"

MODELS = {
    "llama":   ("meta-llama/llama-3.1-8b-instruct", "llama-3.1-8b"),
    "qwen7b":  ("qwen/qwen-2.5-7b-instruct",       "qwen2.5-7b"),
    "qwen32b": ("qwen/qwen3-32b",                   "qwen3-32b"),
    "qwen72b": ("qwen/qwen-2.5-72b-instruct",       "qwen2.5-72b"),
}

DATASETS = {
    "arc":  "arc_challenge",
    "mmlu": "mmlu_pro",
}

# Paraphrase source -> file name pattern
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

# Qwen3 thinking mode
THINKING_MODELS = {"qwen/qwen3-32b"}

TEMPERATURE = 0.0
MAX_TOKENS = 128
CONCURRENCY = 10
MAX_RETRIES = 5

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ============================================================
# Prompt rendering (v00_base)
# ============================================================

def render_prompt(question: str, choices: list, labels: list) -> str:
    options = "\n".join(f"{lb}. {ch}" for lb, ch in zip(labels, choices))
    return (
        "Choose the correct answer.\n\n"
        f"{question}\n\n"
        f"{options}\n\n"
        "Respond with only the letter of the correct option (e.g., A)."
    )


# ============================================================
# Answer extraction
# ============================================================

def extract_answer(response: str, valid_labels: list) -> str | None:
    if not response:
        return None
    text = response.strip()
    valid = set(lb.upper() for lb in valid_labels)
    vr = "".join(sorted(valid))  # e.g. "ABCDEFGHIJ"

    # Pattern 1: explicit "Answer: X" or "Answer is X"
    for pat in [r"[Aa]nswer\s*:\s*\(?([A-Za-z])\)?",
                r"[Aa]nswer\s+is\s+\(?([A-Za-z])\)?",
                r"[Ff]inal\s+[Aa]nswer\s*[:\s]\s*\(?([A-Za-z])\)?",
                r"\*\*([A-Za-z])\*\*"]:
        m = re.search(pat, text)
        if m and m.group(1).upper() in valid:
            return m.group(1).upper()

    # Pattern 2: "X. option text" at start
    m = re.match(r"^\(?([A-Za-z])\)?[\.\)]\s+", text)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()

    # Pattern 3: "X is the correct answer" or "X is correct"
    m = re.search(rf'\b([{vr}])\s+is\s+(?:the\s+)?correct', text, re.IGNORECASE)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()

    # Pattern 4: "the correct answer is X" or "the correct option is X"
    m = re.search(rf'correct\s+(?:answer|option)\s+is\s+\(?([{vr}a-{vr[-1].lower()}])\)?',
                  text, re.IGNORECASE)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()

    # Pattern 5: "option X" or "choice X"
    m = re.search(rf'(?:option|choice)\s+\(?([{vr}])\)?', text, re.IGNORECASE)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()

    # Pattern 6: standalone letter on a line (scan from bottom)
    for line in reversed(text.split("\n")):
        line = line.strip().rstrip(".)")
        if len(line) == 1 and line.upper() in valid:
            return line.upper()

    # Pattern 7: last bold/parenthesized letter
    m = re.search(rf'\*\*\(?([{vr}])\)?\*\*', text)
    if m:
        return m.group(1).upper()

    # Pattern 8: first letter in very short response
    if len(text) <= 5:
        for ch in text:
            if ch.upper() in valid:
                return ch.upper()

    return None


# ============================================================
# API client
# ============================================================

async def call_api(client, sem, model_id, prompt):
    async with sem:
        for attempt in range(MAX_RETRIES):
            try:
                content = prompt
                if model_id in THINKING_MODELS:
                    content = prompt + "\n/nothink"

                resp = await client.post(
                    API_URL,
                    headers=HEADERS,
                    json={
                        "model": model_id,
                        "messages": [{"role": "user", "content": content}],
                        "temperature": TEMPERATURE,
                        "max_tokens": MAX_TOKENS,
                        "top_p": 1.0,
                    },
                    timeout=60.0,
                )
                if resp.status_code == 429:
                    wait = min(2 ** (attempt + 1), 30)
                    log.warning(f"Rate limited, waiting {wait}s...")
                    await asyncio.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                if "error" in data:
                    raise Exception(data["error"].get("message", "API error"))
                raw = data["choices"][0]["message"]["content"].strip()
                # Strip thinking blocks
                return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** (attempt + 1))
                else:
                    log.error(f"Failed after {MAX_RETRIES} retries: {e}")
                    return None


# ============================================================
# Main evaluation
# ============================================================

async def evaluate(model_key, dataset_key, source):
    model_id, model_short = MODELS[model_key]
    bench = DATASETS[dataset_key]

    # Output file includes source tag
    outpath = Path(__file__).parent / f"exp2_{bench}_{model_short}_{source}.json"
    if outpath.exists():
        log.info(f"Already exists: {outpath}, skipping.")
        return

    # Load paraphrased data
    para_file = Path(__file__).parent / PARAPHRASE_FILES[source][dataset_key]
    if not para_file.exists():
        log.error(f"Paraphrase file not found: {para_file}")
        log.error(f"Run generate_paraphrases_gpt4o.py --model {source} first.")
        return

    questions = json.loads(para_file.read_text())
    log.info(f"Loaded {len(questions)} questions from {para_file}")

    # Build tasks: original (v0) + 3 paraphrases (v1-v3) per question
    tasks = []
    for q in questions:
        for v, qtext in enumerate([q["question"]] + q.get("paraphrases", [])):
            if qtext is None:
                continue
            tasks.append({
                "prompt": render_prompt(qtext, q["choices"], q["labels"]),
                "version": v,
                "question_id": q["question_id"],
                "correct_answer": q["answer"],
                "labels": q["labels"],
            })

    log.info(f"{model_short} x {bench} x {source}: {len(tasks)} tasks")

    sem = asyncio.Semaphore(CONCURRENCY)
    done = 0
    results = []

    async with httpx.AsyncClient() as client:
        async def process(task):
            nonlocal done
            response = await call_api(client, sem, model_id, task["prompt"])
            ext = extract_answer(response or "", task["labels"])
            is_correct = (ext == task["correct_answer"]) if ext else None
            done += 1
            if done % 100 == 0:
                log.info(f"  Progress: {done}/{len(tasks)}")
            return {
                "benchmark": bench,
                "model_short": model_short,
                "paraphrase_source": source,
                "version": task["version"],
                "question_id": task["question_id"],
                "correct_answer": task["correct_answer"],
                "labels": task["labels"],
                "response": response,
                "extracted_answer": ext,
                "is_correct": is_correct,
                "parse_failure": ext is None,
                "error": None if response else "no_response",
            }

        results = list(await asyncio.gather(*[process(t) for t in tasks]))

    outpath.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    n_ok = sum(1 for r in results if r["is_correct"] is True)
    n_pf = sum(1 for r in results if r["parse_failure"])
    n_valid = sum(1 for r in results if r["is_correct"] is not None)
    log.info(f"  Saved {outpath}: acc={n_ok}/{n_valid} ({100*n_ok/n_valid:.1f}%), PF={n_pf}")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS.keys()), default=None)
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), default=None)
    parser.add_argument("--source", choices=["gpt4o", "qwen", "both"], default="gpt4o",
                        help="Paraphrase source (default: gpt4o)")
    args = parser.parse_args()

    models = [args.model] if args.model else list(MODELS.keys())
    datasets = [args.dataset] if args.dataset else list(DATASETS.keys())
    sources = ["gpt4o", "qwen"] if args.source == "both" else [args.source]

    for source in sources:
        for ds in datasets:
            for md in models:
                await evaluate(md, ds, source)

    log.info("Done!")


if __name__ == "__main__":
    asyncio.run(main())
