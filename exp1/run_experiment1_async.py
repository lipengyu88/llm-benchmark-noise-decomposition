"""
Experiment I: Prompt Perturbation — Async Runner

Two-phase pipeline:
  Phase 1: Run all (model, dataset, variant) combinations with max_tokens=200
  Phase 2: Re-run entries that failed to parse (is_correct=None) with max_tokens=1024

Usage:
    python run_experiment1_async.py                      # run all
    python run_experiment1_async.py --model llama         # one model
    python run_experiment1_async.py --dataset arc         # one dataset
    python run_experiment1_async.py --concurrency 30      # adjust concurrency
    python run_experiment1_async.py --skip-rerun          # skip phase 2
"""

import json
import os
import re
import time
import asyncio
import argparse
from pathlib import Path

import httpx
from prompt_variants import get_all_variants, build_prompt

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
    "llama":   "meta-llama/llama-3.1-8b-instruct",
    "qwen7b":  "qwen/qwen-2.5-7b-instruct",
    "qwen32b": "qwen/qwen3-32b",
    "qwen72b": "qwen/qwen-2.5-72b-instruct",
}

DATASETS = {
    "arc":  "arc_challenge_300.json",
    "mmlu": "mmlu_pro_300.json",
}

RESULTS_DIR = Path("results_exp1")
RESULTS_DIR.mkdir(exist_ok=True)

MAX_TOKENS_PHASE1 = 200
MAX_TOKENS_PHASE2 = 1024   # longer for with_explanation truncations
TEMPERATURE = 0.0
CONCURRENCY = 20
MAX_RETRIES = 5
CHECKPOINT_INTERVAL = 50

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

# ============================================================
# Data loading
# ============================================================

def normalize_arc_labels(item):
    label_map = {"1": "A", "2": "B", "3": "C", "4": "D"}
    item["choices"]["label"] = [label_map.get(lb, lb) for lb in item["choices"]["label"]]
    if item["answerKey"] in label_map:
        item["answerKey"] = label_map[item["answerKey"]]
    return item


def load_dataset(dataset_name):
    filepath = DATASETS[dataset_name]
    items = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                if dataset_name == "arc":
                    normalize_arc_labels(item)
                items.append(item)
    return items


def get_question_id(item, dataset_name):
    return item["id"] if dataset_name == "arc" else str(item["question_id"])


def get_correct_answer(item, dataset_name):
    return item["answerKey"].upper() if dataset_name == "arc" else item["answer"].upper()


def get_num_options(item, dataset_name):
    return len(item["choices"]["label"]) if dataset_name == "arc" else len(item["options"])

# ============================================================
# Answer parsing (comprehensive)
# ============================================================

def parse_answer(response_text, answer_format_idx, num_options=4):
    """
    Extract answer letter from model response.
    Handles: standalone letter, "Answer: [X]", "the answer is X",
    "Final answer: X", markdown bold **X**, and fallback to uppercase letters.
    """
    text = response_text.strip()
    if not text:
        return None

    max_letter = chr(64 + num_options)
    vr = f"A-{max_letter}"
    vrl = f"a-{max_letter.lower()}"

    # 1. "Answer: [X]" or "Answer: X"
    m = re.search(rf'[Aa]nswer\s*:\s*\[?\s*([{vr}{vrl}])\s*\]?', text)
    if m:
        return m.group(1).upper()

    # 2. "Final answer: X" or "final answer is X"
    m = re.search(rf'[Ff]inal\s+[Aa]nswer\s*[:\s]?\s*\(?([{vr}{vrl}])\)?', text)
    if m:
        return m.group(1).upper()

    # 3. Standalone letter on first line
    first_line = text.split('\n')[0].strip()
    m = re.match(rf'^([{vr}{vrl}])\s*[\.\)\:]?\s*$', first_line)
    if m:
        return m.group(1).upper()

    # 4. "the answer is X"
    m = re.search(rf'(?:the\s+)?answer\s+is\s+\(?([{vr}{vrl}])\)?', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # 5. Markdown bold: **X**
    m = re.search(rf'\*\*([{vr}])\*\*', text)
    if m:
        return m.group(1)

    # 6. "option X" or "choice X"
    m = re.search(rf'(?:option|choice)\s+\(?([{vr}{vrl}])\)?', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # 7. Last line is just a letter
    last_line = text.strip().split('\n')[-1].strip()
    m = re.match(rf'^\*?\*?\(?([{vr}{vrl}])\)?\*?\*?[\.\,]?\s*$', last_line)
    if m:
        return m.group(1).upper()

    # 8. Fallback: uppercase option letters only (avoid "a", "I")
    matches = re.findall(rf'\b([{vr}])\b', text)
    if matches:
        return matches[-1] if answer_format_idx == 2 else matches[0]

    return None

# ============================================================
# Checkpoint
# ============================================================

def get_checkpoint_path(model_name, dataset_name):
    return RESULTS_DIR / f"checkpoint_{model_name}_{dataset_name}.json"


def load_checkpoint(model_name, dataset_name):
    path = get_checkpoint_path(model_name, dataset_name)
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}


def save_checkpoint(results, model_name, dataset_name):
    path = get_checkpoint_path(model_name, dataset_name)
    with open(path, "w") as f:
        json.dump(results, f, ensure_ascii=False)

# ============================================================
# Async API
# ============================================================

async def call_api(client, model_id, system_msg, user_msg, semaphore, max_tokens):
    messages = []
    if system_msg:
        messages.append({"role": "system", "content": system_msg})
    if "qwen3" in model_id:
        user_msg = user_msg + "\n/no_think"  # Qwen3 soft switch to disable thinking
    messages.append({"role": "user", "content": user_msg})

    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": max_tokens,
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
                usage = data.get("usage", {})
                return content, usage
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(min(2 ** attempt, 20))
                else:
                    print(f"  API failed after {MAX_RETRIES} attempts: {e}")
                    return None, {}


async def process_task(client, semaphore, model_id, item, dataset_name,
                       variant_id, variant_index, correct_answer, max_tokens):
    system_msg, user_msg = build_prompt(item, dataset_name, variant_index)
    raw_response, usage = await call_api(client, model_id, system_msg, user_msg, semaphore, max_tokens)

    ans_fmt_idx = variant_index[1]
    num_opts = get_num_options(item, dataset_name)
    parsed = parse_answer(raw_response, ans_fmt_idx, num_opts) if raw_response else None
    is_correct = int(parsed == correct_answer) if parsed else None

    qid = get_question_id(item, dataset_name)
    return qid, variant_id, {
        "variant_index": list(variant_index),
        "correct_answer": correct_answer,
        "parsed_answer": parsed,
        "is_correct": is_correct,
        "raw_response": raw_response,
        "usage": usage,
    }

# ============================================================
# Phase 1: Main run
# ============================================================

async def run_phase1(model_name, model_id, dataset_name, concurrency):
    print(f"\n{'='*60}")
    print(f"Phase 1: {model_name} / {dataset_name}")
    print(f"{'='*60}")

    data = load_dataset(dataset_name)
    variants = get_all_variants()
    results = load_checkpoint(model_name, dataset_name)

    tasks_to_run = []
    for item in data:
        qid = get_question_id(item, dataset_name)
        correct = get_correct_answer(item, dataset_name)
        if qid not in results:
            results[qid] = {}
        for variant_id, variant_index in variants:
            if variant_id not in results[qid]:
                tasks_to_run.append((item, variant_id, variant_index, correct))

    total = len(data) * len(variants)
    done = total - len(tasks_to_run)
    print(f"Total: {total}, Done: {done}, Remaining: {len(tasks_to_run)}")

    if not tasks_to_run:
        print("Phase 1 complete.")
        return results

    semaphore = asyncio.Semaphore(concurrency)
    completed = 0
    start_time = time.time()

    async with httpx.AsyncClient() as client:
        for batch_start in range(0, len(tasks_to_run), CHECKPOINT_INTERVAL):
            batch = tasks_to_run[batch_start:batch_start + CHECKPOINT_INTERVAL]
            coros = [
                process_task(client, semaphore, model_id, item, dataset_name,
                             vid, vidx, correct, MAX_TOKENS_PHASE1)
                for item, vid, vidx, correct in batch
            ]
            batch_results = await asyncio.gather(*coros)
            for qid, variant_id, result_data in batch_results:
                results[qid][variant_id] = result_data

            completed += len(batch)
            elapsed = time.time() - start_time
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (len(tasks_to_run) - completed) / rate if rate > 0 else 0
            print(f"  Phase1: {done + completed}/{total} "
                  f"({(done + completed)/total*100:.1f}%) | "
                  f"{rate:.1f}/sec | ETA: {eta/60:.1f}m")
            save_checkpoint(results, model_name, dataset_name)

    elapsed = time.time() - start_time
    print(f"Phase 1 done in {elapsed/60:.1f}m ({len(tasks_to_run)} calls, {len(tasks_to_run)/elapsed:.1f}/sec)")
    return results

# ============================================================
# Phase 2: Rerun None entries with larger max_tokens
# ============================================================

async def run_phase2(model_name, model_id, dataset_name, concurrency, results, data_items):
    variants = get_all_variants()
    variant_map = {v[0]: v[1] for v in variants}

    # Collect None entries
    none_entries = []
    for qid, vdata in results.items():
        for vid, entry in vdata.items():
            if entry.get("is_correct") is not None:
                continue
            raw = entry.get("raw_response", "")
            item = data_items.get(qid)
            if not item:
                continue
            vidx = variant_map.get(vid)
            if not vidx:
                continue
            correct = entry["correct_answer"]
            num_opts = get_num_options(item, dataset_name)

            # Try improved parser on existing response first
            if raw:
                parsed = parse_answer(raw, vidx[1], num_opts)
                if parsed:
                    entry["parsed_answer"] = parsed
                    entry["is_correct"] = int(parsed == correct)
                    continue

            none_entries.append((qid, vid, item, vidx, correct))

    if not none_entries:
        print(f"  Phase 2: No None entries to rerun.")
        return results

    print(f"  Phase 2: {len(none_entries)} entries to rerun with max_tokens={MAX_TOKENS_PHASE2}")

    semaphore = asyncio.Semaphore(concurrency)
    fixed = 0
    start_time = time.time()

    async with httpx.AsyncClient() as client:
        for batch_start in range(0, len(none_entries), CHECKPOINT_INTERVAL):
            batch = none_entries[batch_start:batch_start + CHECKPOINT_INTERVAL]

            async def _process(qid, vid, item, vidx, correct):
                system_msg, user_msg = build_prompt(item, dataset_name, vidx)
                raw, usage = await call_api(client, MODELS[model_name], system_msg, user_msg, semaphore, MAX_TOKENS_PHASE2)
                num_opts = get_num_options(item, dataset_name)
                parsed = parse_answer(raw, vidx[1], num_opts) if raw else None
                is_correct = int(parsed == correct) if parsed else None
                return qid, vid, raw, parsed, is_correct, usage

            coros = [_process(*args) for args in batch]
            batch_results = await asyncio.gather(*coros)

            for qid, vid, raw, parsed, is_correct, usage in batch_results:
                results[qid][vid]["raw_response"] = raw
                results[qid][vid]["parsed_answer"] = parsed
                results[qid][vid]["is_correct"] = is_correct
                results[qid][vid]["usage"] = usage
                if is_correct is not None:
                    fixed += 1

            done = min(batch_start + CHECKPOINT_INTERVAL, len(none_entries))
            print(f"    Rerun: {done}/{len(none_entries)}, fixed: {fixed}")
            save_checkpoint(results, model_name, dataset_name)

    elapsed = time.time() - start_time
    print(f"  Phase 2 done: fixed {fixed}/{len(none_entries)} in {elapsed/60:.1f}m")
    return results

# ============================================================
# Main
# ============================================================

async def run_all(model_names, dataset_names, concurrency, skip_rerun):
    for model_name in model_names:
        model_id = MODELS[model_name]
        for dataset_name in dataset_names:
            # Phase 1
            results = await run_phase1(model_name, model_id, dataset_name, concurrency)

            # Phase 2 (rerun truncated entries)
            if not skip_rerun:
                data = load_dataset(dataset_name)
                data_items = {}
                for item in data:
                    qid = get_question_id(item, dataset_name)
                    data_items[qid] = item
                results = await run_phase2(model_name, model_id, dataset_name, concurrency, results, data_items)

            # Save final
            final_path = RESULTS_DIR / f"results_{model_name}_{dataset_name}.json"
            with open(final_path, "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"Final results saved to {final_path}")

            # Report None rate
            total = sum(len(v) for v in results.values())
            none_c = sum(1 for v in results.values() for e in v.values() if e.get("is_correct") is None)
            print(f"Final None rate: {none_c}/{total} ({none_c/total*100:.1f}%)\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment I: Prompt Perturbation")
    parser.add_argument("--model", type=str, choices=list(MODELS.keys()))
    parser.add_argument("--dataset", type=str, choices=list(DATASETS.keys()))
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("--skip-rerun", action="store_true", help="Skip phase 2 rerun")
    args = parser.parse_args()

    model_names = [args.model] if args.model else list(MODELS.keys())
    dataset_names = [args.dataset] if args.dataset else list(DATASETS.keys())

    asyncio.run(run_all(model_names, dataset_names, args.concurrency, args.skip_rerun))
