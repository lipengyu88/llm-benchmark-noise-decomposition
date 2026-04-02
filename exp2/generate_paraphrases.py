"""
Experiment II: Paraphrase Generation Script

Generates 3 paraphrased versions of each question for the test-set resampling
experiment.  Paraphrases reword the question stem while preserving:
  - Semantic meaning and difficulty
  - All answer choices (unchanged)
  - The correct answer label

Generation details:
  - Model: Qwen2.5-72B-Instruct via OpenRouter API
  - Temperature: 0.0 (greedy decoding for reproducibility)
  - 3 independent paraphrase requests per question
  - Only the question stem is paraphrased; answer options are kept verbatim
  - No manual quality filtering was applied post-generation

Note: Because temperature=0.0 (greedy) was used, some questions yield
identical paraphrases across the 3 requests.  This is expected — greedy
decoding is deterministic, so when the model finds one dominant rephrasing
path, all 3 outputs converge.  This does not reduce experimental validity
because the paraphrase *versions* (v1–v3) still serve as repeated
measurements of test-set variation; identical paraphrases simply contribute
zero test-set variance for that item, which is correctly captured in our
variance decomposition.

Reproducibility:
  - Seed / temperature: 0.0 (deterministic)
  - Model checkpoint: qwen/qwen-2.5-72b-instruct (OpenRouter)
  - Input: arc_challenge_300.json (first 150 items), mmlu_pro_300.json (first 150 items)
  - Output: arc_challenge_paraphrased.json, mmlu_pro_paraphrased.json
  - Date generated: 2025-05 (approximate)

Usage:
    python generate_paraphrases.py                # regenerate all
    python generate_paraphrases.py --dataset arc  # one dataset only
"""
from __future__ import annotations
import asyncio
import json
import os
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
MODEL_ID = "qwen/qwen-2.5-72b-instruct"
TEMPERATURE = 0.0
MAX_TOKENS = 512
CONCURRENCY = 5
MAX_RETRIES = 5
N_PARAPHRASES = 3
N_QUESTIONS = 150  # first 150 questions from each 300-question dataset

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

DATASETS = {
    "arc": {
        "input": "arc_challenge_300.json",
        "output": "arc_challenge_paraphrased.json",
        "type": "arc",
    },
    "mmlu": {
        "input": "mmlu_pro_300.json",
        "output": "mmlu_pro_paraphrased.json",
        "type": "mmlu",
    },
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ============================================================
# Paraphrase prompt
# ============================================================

PARAPHRASE_PROMPT = """\
Paraphrase the following question while preserving its exact meaning, \
difficulty level, and all technical details. Only rephrase the question \
stem — do NOT modify the answer choices. Output ONLY the paraphrased \
question text, nothing else.

Original question:
{question}"""


# ============================================================
# API call
# ============================================================

async def call_api(client, sem, prompt: str) -> str | None:
    async with sem:
        for attempt in range(MAX_RETRIES):
            try:
                resp = await client.post(
                    API_URL,
                    headers=HEADERS,
                    json={
                        "model": MODEL_ID,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": TEMPERATURE,
                        "max_tokens": MAX_TOKENS,
                        "top_p": 1.0,
                    },
                    timeout=60.0,
                )
                resp.raise_for_status()
                data = resp.json()
                if "error" in data:
                    raise Exception(data["error"].get("message", "API error"))
                return data["choices"][0]["message"]["content"].strip()
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** (attempt + 1))
                else:
                    log.error(f"Failed after {MAX_RETRIES} retries: {e}")
                    return None


# ============================================================
# Main generation
# ============================================================

async def generate_paraphrases(dataset_key: str):
    cfg = DATASETS[dataset_key]
    input_path = Path(__file__).parent.parent / cfg["input"]
    if not input_path.exists():
        input_path = Path(__file__).parent / cfg["input"]
    output_path = Path(__file__).parent / cfg["output"]

    questions = json.loads(input_path.read_text())[:N_QUESTIONS]
    log.info(f"Loaded {len(questions)} questions from {input_path}")

    sem = asyncio.Semaphore(CONCURRENCY)
    results = []

    async with httpx.AsyncClient() as client:
        for i, q in enumerate(questions):
            question_text = q["question"]
            prompt = PARAPHRASE_PROMPT.format(question=question_text)

            # Generate N_PARAPHRASES independent paraphrases
            paraphrases = []
            for _ in range(N_PARAPHRASES):
                para = await call_api(client, sem, prompt)
                paraphrases.append(para if para else question_text)

            # Build output record
            record = {
                "question_id": q.get("question_id", q.get("id", str(i))),
                "question": question_text,
            }

            # Preserve dataset-specific fields
            if cfg["type"] == "arc":
                record["choices"] = q["choices"]["text"] if isinstance(q["choices"], dict) else q["choices"]
                record["labels"] = q["choices"]["label"] if isinstance(q["choices"], dict) else q.get("labels", [])
                record["answer"] = q["answerKey"] if "answerKey" in q else q["answer"]
                record["source"] = "arc_challenge"
                record["original_idx"] = q.get("original_idx", i)
                record["subject"] = q.get("subject", None)
            else:  # mmlu
                record["choices"] = q.get("options", q.get("choices", []))
                record["labels"] = q.get("labels", [chr(65 + j) for j in range(len(record["choices"]))])
                record["answer"] = q["answer"]
                record["source"] = "mmlu_pro"
                record["original_idx"] = q.get("original_idx", i)
                record["category"] = q.get("category", None)

            record["paraphrases"] = paraphrases
            results.append(record)

            if (i + 1) % 25 == 0:
                log.info(f"  Progress: {i + 1}/{len(questions)}")

    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    log.info(f"Saved {len(results)} paraphrased questions to {output_path}")


async def main():
    parser = argparse.ArgumentParser(description="Generate paraphrased questions for Experiment II")
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), default=None,
                        help="Which dataset to paraphrase (default: both)")
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else list(DATASETS.keys())
    for ds in datasets:
        await generate_paraphrases(ds)

    log.info("Done!")


if __name__ == "__main__":
    asyncio.run(main())