from __future__ import annotations
import json
import os
import re
import time
import asyncio
import argparse
import logging
from pathlib import Path

import httpx

import sys
EXP1_DIR = Path(__file__).parent.parent / "exp1"
sys.path.insert(0, str(EXP1_DIR))
from prompt_variants import build_prompt, BASE_INDEX
sys.path.pop(0)


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

MODELS = {
    "llama":   "meta-llama/llama-3.1-8b-instruct",
    "qwen7b":  "qwen/qwen-2.5-7b-instruct",
    "qwen32b": "qwen/qwen3-32b",
    "qwen72b": "qwen/qwen-2.5-72b-instruct",
}

EXP2_DIR = Path(__file__).parent.parent / "exp2"
DATASET_SUBSET_FILES = {
    "arc":  EXP2_DIR / "arc_challenge_paraphrased_gpt4o.json",
    "mmlu": EXP2_DIR / "mmlu_pro_paraphrased_gpt4o.json",
}
FULL_DATASET_FILES = {
    "arc":  EXP1_DIR / "arc_challenge_300.json",
    "mmlu": EXP1_DIR / "mmlu_pro_300.json",
}

RESULTS_DIR = Path(__file__).parent / "results_exp5"
RESULTS_DIR.mkdir(exist_ok=True)

TEMPERATURE = 0.0
MAX_TOKENS = 200
CONCURRENCY = 20
MAX_RETRIES = 5

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)



def normalize_arc_labels(item):
    label_map = {"1": "A", "2": "B", "3": "C", "4": "D"}
    item["choices"]["label"] = [label_map.get(lb, lb) for lb in item["choices"]["label"]]
    if item["answerKey"] in label_map:
        item["answerKey"] = label_map[item["answerKey"]]
    return item


def load_subset(dataset_name: str, n_questions: int) -> list[dict]:
    """Load the first n_questions from the shared 150-question subset.

    The first n questions in deterministic order are taken so that re-runs
    pick the same items. The 150-subset itself is randomly stratified, so
    taking the first 50 yields a balanced sample.
    """
    subset_data = json.loads(DATASET_SUBSET_FILES[dataset_name].read_text(encoding="utf-8"))
    subset_qids = [str(q["question_id"]) for q in subset_data][:n_questions]
    subset_qid_set = set(subset_qids)

    items = []
    with open(FULL_DATASET_FILES[dataset_name], encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                if dataset_name == "arc":
                    normalize_arc_labels(item)
                items.append(item)

    qid_to_item = {}
    for item in items:
        qid = item["id"] if dataset_name == "arc" else str(item["question_id"])
        qid_to_item[qid] = item

    filtered = [qid_to_item[qid] for qid in subset_qids if qid in qid_to_item]
    log.info(f"  Loaded {len(filtered)} questions from {dataset_name}")
    return filtered


def get_question_id(item, dataset_name):
    return item["id"] if dataset_name == "arc" else str(item["question_id"])


def get_correct_answer(item, dataset_name):
    return item["answerKey"].upper() if dataset_name == "arc" else item["answer"].upper()


def get_num_options(item, dataset_name):
    return len(item["choices"]["label"]) if dataset_name == "arc" else len(item["options"])



def parse_answer(response_text, num_options=4):
    text = (response_text or "").strip()
    if not text:
        return None
    max_letter = chr(64 + num_options)
    vr = f"A-{max_letter}"
    vrl = f"a-{max_letter.lower()}"

    m = re.search(rf'[Aa]nswer\s*:\s*\[?\s*([{vr}{vrl}])\s*\]?', text)
    if m:
        return m.group(1).upper()
    m = re.search(rf'[Ff]inal\s+[Aa]nswer\s*[:\s]?\s*\(?([{vr}{vrl}])\)?', text)
    if m:
        return m.group(1).upper()
    first_line = text.split('\n')[0].strip()
    m = re.match(rf'^([{vr}{vrl}])\s*[\.\)\:]?\s*$', first_line)
    if m:
        return m.group(1).upper()
    m = re.search(rf'(?:the\s+)?answer\s+is\s+\(?([{vr}{vrl}])\)?', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    matches = re.findall(rf'\b([{vr}])\b', text)
    if matches:
        return matches[0]
    return None



async def call_api(client, model_id, system_msg, user_msg, semaphore):
    messages = []
    if system_msg:
        messages.append({"role": "system", "content": system_msg})
    if "qwen3" in model_id:
        user_msg = user_msg + "\n/no_think"
    messages.append({"role": "user", "content": user_msg})

    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }

    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                resp = await client.post(API_URL, json=payload, headers=HEADERS, timeout=90.0)
                if resp.status_code == 429:
                    await asyncio.sleep(min(2 ** attempt * 2, 30))
                    continue
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                content = re.sub(r"<think>.*?</think>", "", content or "", flags=re.DOTALL).strip()
                return content
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(min(2 ** attempt, 20))
                else:
                    log.error(f"  API failed after {MAX_RETRIES} attempts: {e}")
                    return None
    return None


async def process_one(client, semaphore, model_name, model_id, item, dataset_name,
                       repeat_idx, correct_answer):
    """Run one (question, model, repeat) trial."""
    system_msg, user_msg = build_prompt(item, dataset_name, BASE_INDEX)
    raw = await call_api(client, model_id, system_msg, user_msg, semaphore)
    num_opts = get_num_options(item, dataset_name)
    parsed = parse_answer(raw, num_opts) if raw else None
    is_correct = int(parsed == correct_answer) if parsed else None
    qid = get_question_id(item, dataset_name)
    return {
        "qid": qid,
        "model": model_name,
        "repeat": repeat_idx,
        "parsed_answer": parsed,
        "correct_answer": correct_answer,
        "is_correct": is_correct,
        "raw_response": raw,
    }



async def run_dataset(dataset_name, n_questions, n_repeats, concurrency):
    log.info(f"\n{'='*60}")
    log.info(f"Stability run: {dataset_name}, n={n_questions}, repeats={n_repeats}")
    log.info(f"{'='*60}")

    items = load_subset(dataset_name, n_questions)

    tasks_meta = []
    for item in items:
        correct = get_correct_answer(item, dataset_name)
        for model_name in MODELS:
            for r in range(n_repeats):
                tasks_meta.append((item, model_name, r, correct))

    log.info(f"Total trials: {len(tasks_meta)}")

    semaphore = asyncio.Semaphore(concurrency)
    results = []
    start = time.time()

    async with httpx.AsyncClient() as client:
        coros = [
            process_one(client, semaphore, mn, MODELS[mn], item, dataset_name, r, c)
            for item, mn, r, c in tasks_meta
        ]
        chunk = 100
        for i in range(0, len(coros), chunk):
            batch = await asyncio.gather(*coros[i:i + chunk])
            results.extend(batch)
            elapsed = time.time() - start
            done = len(results)
            rate = done / elapsed if elapsed > 0 else 0
            eta = (len(coros) - done) / rate if rate > 0 else 0
            log.info(f"  {done}/{len(coros)} ({done/len(coros)*100:.0f}%), "
                     f"{rate:.1f}/s, ETA {eta/60:.1f}m")

    out_path = RESULTS_DIR / f"stability_{dataset_name}.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"Saved {len(results)} trials to {out_path}")

    none_ct = sum(1 for r in results if r["is_correct"] is None)
    log.info(f"  None rate: {none_ct}/{len(results)} ({none_ct/len(results)*100:.1f}%)")
    return results


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["arc", "mmlu"], default=None,
                        help="Run only one dataset (default: both)")
    parser.add_argument("--n-questions", type=int, default=50)
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else ["arc", "mmlu"]
    for ds in datasets:
        await run_dataset(ds, args.n_questions, args.n_repeats, args.concurrency)


if __name__ == "__main__":
    asyncio.run(main())
