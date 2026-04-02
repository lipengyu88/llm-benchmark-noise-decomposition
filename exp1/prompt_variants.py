"""
Experiment I: Prompt Perturbation - Variant Generation

4 orthogonal dimensions:
  - Instruction (3 levels)
  - Answer Format (3 levels)
  - Option Format (3 levels)
  - Framing (2 levels)

Total factor space: 3x3x3x2 = 54
We select: 1 base + 7 OFAT + 10 random factorial = 18 variants
"""

import random
import itertools

# ============================================================
# Dimension definitions
# ============================================================

INSTRUCTIONS = [
    "Choose the correct answer.",                  # base (level 0)
    "Select the best answer.",                     # level 1
    "Which is correct?",                           # level 2
]

# Answer format: defines how we instruct the model to format its answer
ANSWER_FORMATS = [
    "letter_only",       # base: "Only output the letter of the answer."
    "answer_x",          # "Answer with the format 'Answer: [X]' where X is the letter."
    "with_explanation",  # "Provide a brief explanation, then give your answer."
]

ANSWER_FORMAT_INSTRUCTIONS = {
    "letter_only":      "Only output the letter of your answer (e.g., A).",
    "answer_x":         "Answer with the format 'Answer: [X]' where X is the letter of your answer.",
    "with_explanation": "Provide a brief explanation, then state your final answer letter.",
}

OPTION_FORMATS = [
    "dot",        # base: "A. text"
    "paren",      # "(A) text"
    "half_paren", # "A) text"
]

FRAMINGS = [
    "",                                              # base: no prefix
    "You are a knowledgeable assistant.",             # role prefix
]

# ============================================================
# Base prompt index: (0, 0, 0, 0)
# ============================================================

BASE_INDEX = (0, 0, 0, 0)

def get_ofat_variants():
    """
    One-Factor-At-a-Time: change exactly one dimension from the base.
    Instruction: 2 non-base levels -> 2 variants
    Answer Format: 2 non-base levels -> 2 variants
    Option Format: 2 non-base levels -> 2 variants
    Framing: 1 non-base level -> 1 variant
    Total: 7 OFAT variants
    """
    variants = []
    # Instruction
    variants.append((1, 0, 0, 0))
    variants.append((2, 0, 0, 0))
    # Answer Format
    variants.append((0, 1, 0, 0))
    variants.append((0, 2, 0, 0))
    # Option Format
    variants.append((0, 0, 1, 0))
    variants.append((0, 0, 2, 0))
    # Framing
    variants.append((0, 0, 0, 1))
    return variants


def get_factorial_variants(n=10, seed=42):
    """
    Random sample from the full factorial space, excluding base and OFAT variants.

    The full factorial space has 54 combinations; after removing 1 base + 7 OFAT = 8,
    there are 46 candidates. We sample 10 (~22%) to keep total API cost manageable
    (18 variants × 4 models × 2 datasets × 300 questions = 43,200 calls) while still
    providing enough interaction terms for OLS regression on the 4 factors. With 18
    total variants (8 main-effect + 10 interaction) and only 4 main effects plus up
    to 6 two-way interactions, the design is well-identified (df_residual >= 2).
    Seed is fixed for full reproducibility.
    """
    rng = random.Random(seed)
    all_combos = set(itertools.product(range(3), range(3), range(3), range(2)))
    exclude = {BASE_INDEX} | set(get_ofat_variants())
    candidates = sorted(all_combos - exclude)  # sort for reproducibility
    sampled = rng.sample(candidates, min(n, len(candidates)))
    return sampled


def get_all_variants():
    """
    Returns list of (variant_id, index_tuple) for all 18 prompt variants.
    """
    variants = []
    variants.append(("base", BASE_INDEX))
    for i, idx in enumerate(get_ofat_variants()):
        variants.append((f"ofat_{i+1}", idx))
    for i, idx in enumerate(get_factorial_variants(n=10, seed=42)):
        variants.append((f"fact_{i+1}", idx))
    return variants


def format_options(question_data, dataset_type, option_format_idx):
    """
    Format the answer options according to the option format dimension.

    dataset_type: "arc" or "mmlu"
    """
    if dataset_type == "arc":
        labels = question_data["choices"]["label"]
        texts = question_data["choices"]["text"]
    else:  # mmlu
        # MMLU-Pro uses letter labels A-J (up to 10 options)
        texts = question_data["options"]
        labels = [chr(65 + i) for i in range(len(texts))]  # A, B, C, ...

    fmt = OPTION_FORMATS[option_format_idx]
    lines = []
    for label, text in zip(labels, texts):
        if fmt == "dot":
            lines.append(f"{label}. {text}")
        elif fmt == "paren":
            lines.append(f"({label}) {text}")
        elif fmt == "half_paren":
            lines.append(f"{label}) {text}")
    return "\n".join(lines)


def build_prompt(question_data, dataset_type, variant_index):
    """
    Build the full prompt for a given question and variant index.

    Returns: (system_message, user_message)
    """
    instr_idx, ans_fmt_idx, opt_fmt_idx, framing_idx = variant_index

    # System message (framing)
    system_msg = FRAMINGS[framing_idx] if FRAMINGS[framing_idx] else None

    # Question stem
    if dataset_type == "arc":
        question_stem = question_data["question"]
    else:
        question_stem = question_data["question"]

    # Format options
    options_str = format_options(question_data, dataset_type, opt_fmt_idx)

    # Instruction
    instruction = INSTRUCTIONS[instr_idx]

    # Answer format instruction
    ans_fmt_instr = ANSWER_FORMAT_INSTRUCTIONS[ANSWER_FORMATS[ans_fmt_idx]]

    # Compose user message
    user_msg = f"{instruction}\n\n{question_stem}\n\n{options_str}\n\n{ans_fmt_instr}"

    return system_msg, user_msg


def describe_variant(variant_id, variant_index):
    """Human-readable description of a variant."""
    instr_idx, ans_fmt_idx, opt_fmt_idx, framing_idx = variant_index
    return (
        f"{variant_id}: instruction={INSTRUCTIONS[instr_idx]!r}, "
        f"answer_format={ANSWER_FORMATS[ans_fmt_idx]!r}, "
        f"option_format={OPTION_FORMATS[opt_fmt_idx]!r}, "
        f"framing={'role_prefix' if framing_idx == 1 else 'none'}"
    )


if __name__ == "__main__":
    variants = get_all_variants()
    print(f"Total variants: {len(variants)}\n")
    for vid, vidx in variants:
        print(describe_variant(vid, vidx))
