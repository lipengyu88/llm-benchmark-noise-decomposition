from __future__ import annotations
import asyncio
import json
import os
import re
import argparse
import logging
from pathlib import Path

import httpx


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
    "gpt4o": "openai/gpt-4o",
    "qwen":  "qwen/qwen-2.5-72b-instruct",
}

TEMPERATURE = 0.7
MAX_TOKENS = 1024
CONCURRENCY = 8
MAX_RETRIES = 5
N_PARAPHRASES = 3
N_QUESTIONS = 150

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

DATASETS = {
    "arc":  {"input": "arc_challenge_300.json", "type": "arc"},
    "mmlu": {"input": "mmlu_pro_300.json",      "type": "mmlu"},
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)



PARAPHRASE_PROMPT = """\
You are a careful test-item paraphraser. Given a multiple-choice question, \
produce exactly 3 DISTINCT paraphrased versions of the question stem.

Rules:
- Each paraphrase must preserve the EXACT meaning, difficulty, and all technical details.
- The 3 paraphrases must be clearly DIFFERENT from each other in wording and sentence structure.
- Do NOT modify the answer choices — only rephrase the question stem.
- Do NOT add or remove any information.
- Output valid JSON: a list of exactly 3 strings.

Original question:
{question}

Answer choices (for context only — do NOT include in output):
{choices}

Output format (JSON array of 3 strings):
["paraphrase 1", "paraphrase 2", "paraphrase 3"]"""



async def call_api(client: httpx.AsyncClient, sem: asyncio.Semaphore,
                   model_id: str, prompt: str) -> str | None:
    async with sem:
        for attempt in range(MAX_RETRIES):
            try:
                resp = await client.post(
                    API_URL,
                    headers=HEADERS,
                    json={
                        "model": model_id,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": TEMPERATURE,
                        "max_tokens": MAX_TOKENS,
                        "top_p": 1.0,
                    },
                    timeout=90.0,
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
                raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
                return raw
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** (attempt + 1))
                else:
                    log.error(f"Failed after {MAX_RETRIES} retries: {e}")
                    return None


def parse_paraphrases(raw: str, original: str) -> list[str]:
    """Parse the JSON array of 3 paraphrases from model output."""
    if not raw:
        return [original] * N_PARAPHRASES

    try:
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
            if isinstance(parsed, list) and len(parsed) >= N_PARAPHRASES:
                result = [str(p).strip() for p in parsed[:N_PARAPHRASES]]
                if all(len(p) > 10 for p in result):
                    return result
    except (json.JSONDecodeError, ValueError):
        pass

    lines = [l.strip() for l in raw.split('\n') if l.strip()]
    paras = []
    for line in lines:
        cleaned = re.sub(r'^[\d]+[\.\)]\s*', '', line)
        cleaned = re.sub(r'^[-\*]\s*', '', cleaned)
        cleaned = cleaned.strip().strip('"').strip("'")
        if len(cleaned) > 10 and cleaned != original:
            paras.append(cleaned)

    if len(paras) >= N_PARAPHRASES:
        return paras[:N_PARAPHRASES]

    while len(paras) < N_PARAPHRASES:
        paras.append(original)
    return paras



def load_questions(dataset_key: str) -> list[dict]:
    """Load N_QUESTIONS via stratified random sampling (seed=42).

    For MMLU-Pro, stratifies by 'category' so the sampled subset mirrors the
    overall subject distribution.  For ARC (uniform structure), falls back to
    simple random sampling.  The fixed seed guarantees reproducibility.

    If N_QUESTIONS >= total items, all items are returned (original order).
    """
    import random as _random

    cfg = DATASETS[dataset_key]
    input_path = Path(__file__).parent / cfg["input"]

    items = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    if N_QUESTIONS >= len(items):
        return items

    rng = _random.Random(42)

    if cfg["type"] == "mmlu":
        from collections import defaultdict
        by_cat = defaultdict(list)
        for item in items:
            by_cat[item.get("category", "unknown")].append(item)

        sampled = []
        total = len(items)
        for cat, cat_items in sorted(by_cat.items()):
            n_cat = max(1, round(len(cat_items) / total * N_QUESTIONS))
            rng.shuffle(cat_items)
            sampled.extend(cat_items[:n_cat])

        rng.shuffle(sampled)
        if len(sampled) > N_QUESTIONS:
            sampled = sampled[:N_QUESTIONS]
        elif len(sampled) < N_QUESTIONS:
            remaining = [it for it in items if it not in sampled]
            rng.shuffle(remaining)
            sampled.extend(remaining[:N_QUESTIONS - len(sampled)])

        return sampled
    else:
        rng.shuffle(items)
        return items[:N_QUESTIONS]


def format_choices(q: dict, dataset_type: str) -> str:
    """Format answer choices for inclusion in prompt context."""
    if dataset_type == "arc":
        labels = q["choices"]["label"]
        texts = q["choices"]["text"]
    else:
        texts = q.get("options", q.get("choices", []))
        labels = [chr(65 + i) for i in range(len(texts))]
    return "\n".join(f"{lb}. {tx}" for lb, tx in zip(labels, texts))


def build_output_record(q: dict, dataset_type: str, idx: int, paraphrases: list[str]) -> dict:
    """Build the output record in the expected format."""
    record = {
        "question_id": q.get("id", q.get("question_id", str(idx))),
        "question": q["question"],
    }

    if dataset_type == "arc":
        label_map = {"1": "A", "2": "B", "3": "C", "4": "D"}
        labels = [label_map.get(lb, lb) for lb in q["choices"]["label"]]
        record["choices"] = q["choices"]["text"]
        record["labels"] = labels
        answer = q.get("answerKey", q.get("answer", ""))
        record["answer"] = label_map.get(answer, answer)
        record["source"] = "arc_challenge"
        record["original_idx"] = idx
        record["subject"] = None
    else:
        record["choices"] = q.get("options", q.get("choices", []))
        record["labels"] = [chr(65 + j) for j in range(len(record["choices"]))]
        record["answer"] = q["answer"]
        record["source"] = "mmlu_pro"
        record["original_idx"] = idx
        record["subject"] = q.get("category", None)

    record["paraphrases"] = paraphrases
    return record



async def generate_for_model(model_key: str, dataset_key: str):
    cfg = DATASETS[dataset_key]
    model_id = MODELS[model_key]
    output_path = Path(__file__).parent / f"{cfg['type']}_challenge_paraphrased_{model_key}.json" \
        if cfg["type"] == "arc" else Path(__file__).parent / f"mmlu_pro_paraphrased_{model_key}.json"

    questions = load_questions(dataset_key)
    log.info(f"[{model_key}] Loaded {len(questions)} questions for {dataset_key}")

    sem = asyncio.Semaphore(CONCURRENCY)
    results = []
    n_parse_ok = 0
    n_all_unique = 0

    async with httpx.AsyncClient() as client:
        tasks = []
        for i, q in enumerate(questions):
            choices_str = format_choices(q, cfg["type"])
            prompt = PARAPHRASE_PROMPT.format(
                question=q["question"],
                choices=choices_str,
            )
            tasks.append((i, q, prompt))

        batch_size = 25
        for batch_start in range(0, len(tasks), batch_size):
            batch = tasks[batch_start:batch_start + batch_size]

            async def process_one(idx, q, prompt):
                raw = await call_api(client, sem, model_id, prompt)
                paraphrases = parse_paraphrases(raw, q["question"])
                return idx, q, paraphrases, raw

            coros = [process_one(idx, q, prompt) for idx, q, prompt in batch]
            batch_results = await asyncio.gather(*coros)

            for idx, q, paraphrases, raw in batch_results:
                record = build_output_record(q, cfg["type"], idx, paraphrases)
                results.append(record)

                if all(p != q["question"] for p in paraphrases):
                    n_parse_ok += 1
                if len(set(paraphrases)) == N_PARAPHRASES:
                    n_all_unique += 1

            done = min(batch_start + batch_size, len(tasks))
            log.info(f"  [{model_key}/{dataset_key}] Progress: {done}/{len(tasks)}")

    results.sort(key=lambda r: r["original_idx"])

    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    log.info(f"  [{model_key}/{dataset_key}] Saved {len(results)} questions to {output_path}")
    log.info(f"  [{model_key}/{dataset_key}] Parse success (all 3 different from original): "
             f"{n_parse_ok}/{len(results)} ({100*n_parse_ok/len(results):.1f}%)")
    log.info(f"  [{model_key}/{dataset_key}] All 3 unique: "
             f"{n_all_unique}/{len(results)} ({100*n_all_unique/len(results):.1f}%)")

    return output_path


async def main():
    parser = argparse.ArgumentParser(description="Generate paraphrases with GPT-4o (and optionally Qwen)")
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), default=None)
    parser.add_argument("--model", choices=["gpt4o", "qwen", "both"], default="gpt4o",
                        help="Which model to use for paraphrasing (default: gpt4o)")
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else list(DATASETS.keys())
    models = ["gpt4o", "qwen"] if args.model == "both" else [args.model]

    for model_key in models:
        for ds in datasets:
            await generate_for_model(model_key, ds)

    log.info("All done!")


if __name__ == "__main__":
    asyncio.run(main())
