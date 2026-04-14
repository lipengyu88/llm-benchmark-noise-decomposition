import random
import itertools



INSTRUCTIONS = [
    "Choose the correct answer.",
    "Select the best answer.",
    "Which is correct?",
]

ANSWER_FORMATS = [
    "letter_only",
    "answer_x",
    "with_explanation",
]

ANSWER_FORMAT_INSTRUCTIONS = {
    "letter_only":      "Only output the letter of your answer (e.g., A).",
    "answer_x":         "Answer with the format 'Answer: [X]' where X is the letter of your answer.",
    "with_explanation": "Provide a brief explanation, then state your final answer letter.",
}

OPTION_FORMATS = [
    "dot",
    "paren",
    "half_paren",
]

FRAMINGS = [
    "",
    "You are a knowledgeable assistant.",
]

DELIMITERS = [
    "blank",
    "dashes",
    "headers",
]



BASE_INDEX = (0, 0, 0, 0, 0)
N_DIMENSIONS = 5
N_LEVELS = (3, 3, 3, 2, 3)


def get_ofat_variants():
    """All one-factor-at-a-time variants (base with exactly one dim non-zero).

    Non-zero levels per dimension:
      instruction(2) + answer_format(2) + option_format(2) + framing(1)
      + delimiter(2) = 9 + 1 = 10 OFAT variants.
    """
    variants = []
    for d in range(N_DIMENSIONS):
        for lvl in range(1, N_LEVELS[d]):
            v = list(BASE_INDEX)
            v[d] = lvl
            variants.append(tuple(v))
    return variants


def get_factorial_variants(n=90, seed=42):
    """Random sample from the full factorial space, excluding base + OFAT."""
    rng = random.Random(seed)
    all_combos = set(itertools.product(*(range(n) for n in N_LEVELS)))
    exclude = {BASE_INDEX} | set(get_ofat_variants())
    candidates = sorted(all_combos - exclude)
    sampled = rng.sample(candidates, min(n, len(candidates)))
    return sampled


def get_all_variants():
    """Returns list of (variant_id, index_tuple) for all 100 prompt variants."""
    variants = []
    variants.append(("base", BASE_INDEX))
    for i, idx in enumerate(get_ofat_variants()):
        variants.append((f"ofat_{i+1}", idx))
    for i, idx in enumerate(get_factorial_variants(n=90, seed=42)):
        variants.append((f"fact_{i+1}", idx))
    return variants



def format_options(question_data, dataset_type, option_format_idx):
    if dataset_type == "arc":
        labels = question_data["choices"]["label"]
        texts = question_data["choices"]["text"]
    else:
        texts = question_data["options"]
        labels = [chr(65 + i) for i in range(len(texts))]

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


def assemble_prompt(instruction, question_stem, options_str, ans_fmt_instr, delimiter_idx):
    """Assemble the user message using the chosen delimiter style."""
    delim = DELIMITERS[delimiter_idx]
    if delim == "blank":
        return f"{instruction}\n\n{question_stem}\n\n{options_str}\n\n{ans_fmt_instr}"
    elif delim == "dashes":
        return (f"{instruction}\n---\n{question_stem}\n---\n{options_str}\n---\n{ans_fmt_instr}")
    elif delim == "headers":
        return (f"### Task\n{instruction}\n\n"
                f"### Question\n{question_stem}\n\n"
                f"### Options\n{options_str}\n\n"
                f"### Response format\n{ans_fmt_instr}")
    return f"{instruction}\n\n{question_stem}\n\n{options_str}\n\n{ans_fmt_instr}"


def build_prompt(question_data, dataset_type, variant_index):
    """Build the full prompt for a given question and variant index.

    Returns: (system_message, user_message)
    """
    instr_idx, ans_fmt_idx, opt_fmt_idx, framing_idx, delim_idx = variant_index

    system_msg = FRAMINGS[framing_idx] if FRAMINGS[framing_idx] else None

    question_stem = question_data["question"]
    options_str = format_options(question_data, dataset_type, opt_fmt_idx)
    instruction = INSTRUCTIONS[instr_idx]
    ans_fmt_instr = ANSWER_FORMAT_INSTRUCTIONS[ANSWER_FORMATS[ans_fmt_idx]]

    user_msg = assemble_prompt(instruction, question_stem, options_str,
                                ans_fmt_instr, delim_idx)
    return system_msg, user_msg


def describe_variant(variant_id, variant_index):
    instr_idx, ans_fmt_idx, opt_fmt_idx, framing_idx, delim_idx = variant_index
    return (
        f"{variant_id}: instruction={INSTRUCTIONS[instr_idx]!r}, "
        f"answer_format={ANSWER_FORMATS[ans_fmt_idx]!r}, "
        f"option_format={OPTION_FORMATS[opt_fmt_idx]!r}, "
        f"framing={'role_prefix' if framing_idx == 1 else 'none'}, "
        f"delimiter={DELIMITERS[delim_idx]!r}"
    )


if __name__ == "__main__":
    variants = get_all_variants()
    print(f"Total variants: {len(variants)}")
    print(f"  base: 1")
    print(f"  OFAT: {len(get_ofat_variants())}")
    print(f"  factorial: 90")
    print()
    all_idx = [v[1] for v in variants]
    assert len(set(all_idx)) == len(all_idx), "Duplicate variants!"
    print(f"All {len(variants)} variants are unique. ✓")
    print()
    print("First 5 variants:")
    for vid, vidx in variants[:5]:
        print(f"  {describe_variant(vid, vidx)}")
    print("\nLast 3 variants:")
    for vid, vidx in variants[-3:]:
        print(f"  {describe_variant(vid, vidx)}")
