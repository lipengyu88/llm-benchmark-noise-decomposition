"""
OLS regression analysis of prompt-variant accuracy data.

For each (model, benchmark) pair, fits an OLS model with 9 dummy-coded
prompt-design variables (5 dimensions, level 0 as reference) and produces:
  - F-statistic, F p-value, R^2, adjusted R^2
  - Per-coefficient t-statistics and p-values
  - Shapiro-Wilk normality test on residuals
  - Q-Q residual plot (saved as fig_a11_ols_diagnostics.png)
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats

# ── paths ────────────────────────────────────────────────────────────
sys.path.insert(0, "/Users/bytedance/ST5230/exp1")
from prompt_variants import get_all_variants

ARC_JSON  = "/Users/bytedance/ST5230/exp1/analysis_exp1/analysis_arc.json"
MMLU_JSON = "/Users/bytedance/ST5230/exp1/analysis_exp1/analysis_mmlu.json"
OUT_FIG   = "/Users/bytedance/ST5230/figures/fig_a11_ols_diagnostics.png"

MODELS     = ["llama", "qwen7b", "qwen32b", "qwen72b"]
MODEL_NICE = {"llama": "LLaMA-3-8B", "qwen7b": "Qwen-2.5-7B",
              "qwen32b": "Qwen-2.5-32B", "qwen72b": "Qwen-2.5-72B"}
BENCHMARKS = [("arc", "ARC-Challenge", ARC_JSON),
              ("mmlu", "MMLU-Pro", MMLU_JSON)]

# dimension: (name, n_levels)
DIM_INFO = [
    ("instruction",    3),
    ("answer_format",  3),
    ("option_format",  3),
    ("framing",        2),
    ("delimiter",      3),
]
# Total dummy variables: (3-1)+(3-1)+(3-1)+(2-1)+(3-1) = 2+2+2+1+2 = 9


# ── build design matrix (dummy coding, level 0 = reference) ─────────
def build_design_matrix():
    """Return (X, col_names) where X is 100 x 10 (with intercept)."""
    variants = get_all_variants()  # list of (id, (i,j,k,l,m))
    indices = [v[1] for v in variants]

    cols = []
    col_names = []
    for dim_idx, (dim_name, n_levels) in enumerate(DIM_INFO):
        for lvl in range(1, n_levels):
            col_names.append(f"{dim_name}_{lvl}")
            cols.append([1.0 if idx[dim_idx] == lvl else 0.0
                         for idx in indices])

    X = np.column_stack(cols)            # 100 x 9
    X = sm.add_constant(X, prepend=True) # 100 x 10
    col_names = ["const"] + col_names
    return X, col_names


# ── main ─────────────────────────────────────────────────────────────
def main():
    X, col_names = build_design_matrix()
    print(f"Design matrix shape: {X.shape}  columns: {col_names}\n")

    # load data
    data = {}
    for key, nice, path in BENCHMARKS:
        with open(path) as f:
            data[key] = json.load(f)

    results = {}  # (bench_key, model) -> OLS result

    # header
    sep = "=" * 88
    for bench_key, bench_nice, _ in BENCHMARKS:
        print(f"\n{sep}")
        print(f"  Benchmark: {bench_nice}")
        print(sep)
        for model in MODELS:
            y = np.array(data[bench_key][model]["accuracy_stats"]["per_variant"])
            ols = sm.OLS(y, X).fit()
            results[(bench_key, model)] = ols

            # Shapiro-Wilk on residuals
            sw_stat, sw_p = stats.shapiro(ols.resid)

            print(f"\n--- {MODEL_NICE[model]} ({bench_nice}) ---")
            print(f"  N = {int(ols.nobs)},  k = {int(ols.df_model)}")
            print(f"  R^2 = {ols.rsquared:.4f},  Adj R^2 = {ols.rsquared_adj:.4f}")
            print(f"  F-statistic = {ols.fvalue:.4f},  F p-value = {ols.f_pvalue:.4e}")
            print(f"  Shapiro-Wilk: W = {sw_stat:.4f},  p = {sw_p:.4e}")
            print()
            print(f"  {'Coefficient':<20s} {'Estimate':>10s} {'Std Err':>10s} "
                  f"{'t-stat':>10s} {'p-value':>12s}")
            print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")
            for i, name in enumerate(col_names):
                print(f"  {name:<20s} {ols.params[i]:>10.6f} "
                      f"{ols.bse[i]:>10.6f} {ols.tvalues[i]:>10.4f} "
                      f"{ols.pvalues[i]:>12.4e}")

    # ── LaTeX-friendly summary table ─────────────────────────────────
    print(f"\n\n{'=' * 88}")
    print("  LaTeX-ready summary (copy-paste into tabular)")
    print("=" * 88)
    print()
    print("% Model & Benchmark & R^2 & Adj R^2 & F & F p-value & Shapiro-W & Shapiro-p")
    for bench_key, bench_nice, _ in BENCHMARKS:
        for model in MODELS:
            ols = results[(bench_key, model)]
            sw_stat, sw_p = stats.shapiro(ols.resid)
            print(f"{MODEL_NICE[model]} & {bench_nice} & "
                  f"{ols.rsquared:.4f} & {ols.rsquared_adj:.4f} & "
                  f"{ols.fvalue:.2f} & {ols.f_pvalue:.2e} & "
                  f"{sw_stat:.4f} & {sw_p:.2e} \\\\")

    print()
    print("% Per-coefficient table: Model & Benchmark & Coeff & Estimate & t & p")
    for bench_key, bench_nice, _ in BENCHMARKS:
        for model in MODELS:
            ols = results[(bench_key, model)]
            for i, name in enumerate(col_names):
                if name == "const":
                    continue
                print(f"{MODEL_NICE[model]} & {bench_nice} & {name} & "
                      f"{ols.params[i]:.4f} & {ols.tvalues[i]:.2f} & "
                      f"{ols.pvalues[i]:.2e} \\\\")

    # ── Q-Q plot grid ────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for row, (bench_key, bench_nice, _) in enumerate(BENCHMARKS):
        for col, model in enumerate(MODELS):
            ax = axes[row, col]
            ols = results[(bench_key, model)]
            resid = ols.resid
            sm.qqplot(resid, line="s", ax=ax, markersize=3,
                      markerfacecolor="steelblue", markeredgecolor="steelblue",
                      alpha=0.6)
            ax.set_title(MODEL_NICE[model], fontsize=11)
            if col == 0:
                ax.set_ylabel(bench_nice, fontsize=12, fontweight="bold")
            else:
                ax.set_ylabel("")

    fig.suptitle("OLS Residual Q-Q Plots", fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_FIG, dpi=200, bbox_inches="tight")
    print(f"\nFigure saved to {OUT_FIG}")


if __name__ == "__main__":
    main()
