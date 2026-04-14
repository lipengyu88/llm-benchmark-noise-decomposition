import matplotlib.pyplot as plt
import numpy as np

# Data
models = ["LLaMA-8B", "Qwen-7B", "Qwen-32B", "Qwen-72B"]

prompt_arc   = [4.8, 5.6, 3.9, 9.7]
para_arc     = [2.1, 1.2, 1.15, 0.55]

prompt_mmlu  = [4.9, 4.1, 4.7, 5.3]
para_mmlu    = [1.65, 1.6, 1.25, 1.75]

x = np.arange(len(models))
width = 0.32

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

benchmarks = [("ARC-Challenge", prompt_arc, para_arc),
              ("MMLU-Pro",      prompt_mmlu, para_mmlu)]

for ax, (title, prompt_vals, para_vals) in zip(axes, benchmarks):
    b1 = ax.bar(x - width/2, prompt_vals, width, label="Prompt (100 variants)",
                color="#E06050", edgecolor="white", linewidth=0.5)
    b2 = ax.bar(x + width/2, para_vals, width, label="Paraphrase (4 versions)",
                color="#4682B4", edgecolor="white", linewidth=0.5)

    # ratio annotations
    for i in range(len(models)):
        ratio = prompt_vals[i] / para_vals[i]
        top = max(prompt_vals[i], para_vals[i])
        ax.text(x[i], top + 0.35, f"\u00d7{ratio:.1f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold", color="#333333")

    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_ylabel("Accuracy \u03c3 (pp)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylim(0, max(max(prompt_vals), max(para_vals)) * 1.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=9)

axes[0].legend(fontsize=8.5, frameon=False, loc="upper left")

plt.tight_layout()
plt.savefig("/Users/bytedance/ST5230/figures/fig4_prompt_vs_paraphrase.png", dpi=300,
            bbox_inches="tight", facecolor="white")
plt.close()
print("Saved.")
