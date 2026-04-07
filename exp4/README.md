# Experiment IV: Bradley-Terry Ranking Stability Analysis

Covers **Theme 1, Direction 3** from the project specification:

> "Analyzing the robustness of model rankings... For pairwise preference
> evaluations, one can fit a BT model and derive posterior intervals, then
> simulate how many extra comparisons are required for the top-k ranking
> to become stable with high probability (like ≥ 0.95)."

## Methodology

We convert item-level correctness from Exp I and Exp II into pairwise model
comparisons. For every `(question, condition)` cell (where a condition is a
prompt variant in Exp I or a paraphrase version × source in Exp II), and for
each model pair `(A, B)`:

- A correct, B wrong → A beats B
- B correct, A wrong → B beats A
- Ties (both correct or both wrong) are skipped

This gives ~6,200 comparison cells per benchmark (4 models × `C(4,2)=6` pairs
= ~37,000 directed pairwise outcomes per benchmark).

We then:

1. **Fit BT model** via Zermelo MLE iteration on aggregated win counts.
2. **Bootstrap 95% CIs** on log-strengths by resampling 10,000 times over
   the comparison cells.
3. **Compute rank posterior** `Pr(model holds rank k)` for `k ∈ {1, 2, 3, 4}`.
4. **Sample-size simulation**: subsample `N ∈ {50, 100, ..., 10000, 20000}`
   comparisons, refit BT, and record how often the bootstrap reproduces the
   full-data top-1 and top-2 set. Report the smallest `N` for which each
   reaches ≥ 95% stability.

## Key Findings

### Full-Data BT Ratings

| Rank | Model | ARC log-strength | MMLU log-strength |
|------|-------|------------------|-------------------|
| 1 | Qwen3-32B | -0.90 | -0.92 |
| 2 | Qwen2.5-72B | -1.16 | -1.11 |
| 3 | Qwen2.5-7B | -1.63 | -1.86 |
| 4 | LLaMA-3.1-8B | -2.47 | -2.17 |

With ~6,000 conditions per benchmark, the **95% CIs are tight (~±0.05)**
and the ranking is locked in — every model holds its observed rank with
posterior probability 1.000.

### Sample-Size Requirements for ≥ 95% Stability

| Benchmark | N needed for top-1 | N needed for top-2 set |
|-----------|--------------------|------------------------|
| ARC-Challenge | 2,000 | 100 |
| MMLU-Pro | 2,000 | 100 |

**Interpretation**:
- The top-2 set (Qwen3-32B vs Qwen2.5-72B) locks in very quickly
  (~100 comparisons) on both benchmarks because these two models' strengths
  differ appreciably from the weaker pair.
- The **top-1 position** between Qwen3-32B and Qwen2.5-72B requires
  ~2,000 comparisons to stabilise at 95%. With fewer comparisons, the two
  models flip the top-1 slot in 5-30% of resamples.
- In practice, this means a Chatbot Arena-style leaderboard with only
  ~100-500 comparisons between two close competitors would be **substantially
  unreliable at the top** — consistent with the findings of Huang & Shen
  (2508.11847) that 0.02% data removal can change Arena top-1.

## Files

- `run_bradley_terry.py` — main analysis pipeline
- `visualize_bt.py` — generates 3 figures
- `bt_results_{arc,mmlu}.json` — BT ratings, CIs, rank posteriors, simulation
- `figures_bt/fig1_bt_ratings.png` — BT log-strengths with bootstrap CIs
- `figures_bt/fig2_rank_posterior.png` — rank posterior heatmap
- `figures_bt/fig3_sample_size_curves.png` — top-k stability vs N

## Reproducibility

- Seed: `np.random.RandomState(42)` for both bootstrap and simulation
- Bootstrap: 10,000 resamples over condition cells
- Simulation: 500 repeats per sample-size point
- BT fit: Zermelo iteration with tolerance 1e-8 and max 10,000 iterations
- Input data: Exp I results (18 prompts × 300 questions) + Exp II results
  (4 versions × 150 questions × 2 paraphrase sources), for each of the
  4 evaluated models

## Relation to Exp I-III

While Exp I-III establish that prompt and paraphrase noise affect per-condition
accuracy, this experiment answers the complementary question: **even after
pooling all 6,000+ conditions into a single BT ranking, how many comparisons
are actually needed to nail down the top-k?** The answer — ~2,000 for top-1,
~100 for top-2 — directly translates our noise findings into actionable sample-
size guidance for benchmark curators and leaderboard operators.
