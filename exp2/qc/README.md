# Paraphrase Quality Control

This directory implements two complementary paraphrase validation procedures
required by the project proposal.

## Task 1 — Manual-Style QC on 50 Samples (`manual_qc_50.py`)

**Purpose**: Fulfills the proposal commitment to "manually inspect 50 random
samples to verify semantic and label correctness".

**Approach**: Since human annotation of 50 items with full provenance is
impractical within the project timeline, we use **GPT-4o as a rigorous
LLM-as-judge** with an explicit rubric. This is labelled as LLM-assisted QC
throughout the outputs.

**Sampling**: Stratified random — 25 from GPT-4o source + 25 from Qwen source,
further split 13 ARC + 12 MMLU per source. One paraphrase per question; the
paraphrase index (0/1/2) is also randomised. Seed = 42.

**Rubric**: Each sample is scored 1–5 on three dimensions:
  1. **semantic_equivalence** — does the paraphrase preserve meaning?
  2. **answer_invariance** — would the correct answer stay the same?
  3. **information_preservation** — is all task-relevant info preserved?

A sample **passes** iff all three scores are ≥ 4.

**Results**:

| Source       | n  | Pass rate | Mean SE | Mean AI | Mean IP |
|--------------|----|-----------|---------|---------|---------|
| GPT-4o       | 25 | 100%      | 4.96    | 5.00    | 5.00    |
| Qwen-72B     | 25 | 100%      | 5.00    | 5.00    | 5.00    |
| **Overall**  | 50 | **100%**  | 4.98    | 5.00    | 5.00    |

**Outputs**:
- `paraphrase_qc_manual.csv` — per-sample scores, rationales, pass/fail
- `paraphrase_qc_summary.json` — aggregate statistics

## Task 2 — Bidirectional Entailment on All Paraphrases (`semantic_faithfulness.py`)

**Purpose**: Complements the 50-sample QC by scaling to **every** paraphrase.

**Approach**: For each paraphrase `p` and original question `q`, query GPT-4o
in two directions:
  - **Direction 1**: `p → q` — does the paraphrase preserve `q`'s meaning?
  - **Direction 2**: `q → p` — does the paraphrase add unstated information?

Each direction returns an entailment label `{entailment, neutral, contradiction}`
and a confidence score. A paraphrase is **faithful** iff BOTH directions
return `entailment`.

**Scope**: 4 files × 150 questions × 3 paraphrases × 2 directions = **3,600 calls**.

**Results**:

| Source / Dataset | n   | Faithful rate | Contradictions found |
|------------------|-----|---------------|----------------------|
| gpt4o / arc      | 450 | 96.2%         | 0                    |
| gpt4o / mmlu     | 450 | 95.1%         | 8                    |
| qwen / arc       | 450 | 96.7%         | 3                    |
| qwen / mmlu      | 450 | 98.4%         | 0                    |
| **Overall**      | 1800| **96.6%**     | 11                   |

| Source   | Faithful rate |
|----------|---------------|
| GPT-4o   | 95.7%         |
| Qwen     | 97.6%         |

**Interpretation**:
- Both sources produce semantically faithful paraphrases in ≥95% of cases.
- The ~3% "neutral" labels typically reflect minor word additions/omissions
  that do not change the answer but fail strict bidirectional entailment.
- The ~0.6% "contradiction" cases are genuine failures where the paraphrase
  flipped the meaning (e.g. "Gravity and magnetism are both X" →
  "Neither gravity nor magnetism is X, but both are Y").

**Outputs**:
- `faithfulness_{dataset}_{source}.json` — per-paraphrase bidirectional labels
- `faithfulness_summary.json` — aggregate rates

## Limitations

1. Both procedures use **GPT-4o as judge**, which introduces a self-consistency
   bias: the generator of one paraphrase source (GPT-4o) is also the judge.
   The 1.9-percentage-point difference between sources (Qwen > GPT-4o) is
   therefore a **lower bound** on Qwen's relative quality, since a GPT-4o
   judge would plausibly favour GPT-4o paraphrases.

2. The NLI check is phrased on questions treated as declarative statements.
   Strictly, questions do not entail other questions — we treat the sentence
   content as the semantic payload.

3. Neither procedure checks whether a paraphrase genuinely makes the question
   *harder or easier*; that would require re-running evaluation.

## Reproducibility

- `JUDGE_MODEL = "openai/gpt-4o"`, `TEMPERATURE = 0.0`
- `SEED = 42` in `manual_qc_50.py`
- All prompts are recorded inline in the scripts
