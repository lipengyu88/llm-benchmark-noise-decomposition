"""
Experiment II: Analyze results — accuracy, flip rate, pairwise gaps, rank distribution.

Reads:   exp2_{dataset}_{model}.json
Writes:  analysis_exp2/analysis_{dataset}.json

Also performs cross-experiment comparison if Exp I analysis is available at ../exp1/.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent
ANALYSIS_DIR = ROOT / "analysis_exp2"
ANALYSIS_DIR.mkdir(exist_ok=True)

MODELS = ["llama-3.1-8b", "qwen2.5-7b", "qwen3-32b", "qwen2.5-72b"]
DATASETS = {"arc": "arc_challenge", "mmlu": "mmlu_pro"}
BOOTSTRAP_N = 10_000


# ============================================================
# Data loading
# ============================================================

def load_results(dataset_key):
    bench = DATASETS[dataset_key]
    frames = []
    for m in MODELS:
        p = ROOT / f"exp2_{bench}_{m}.json"
        if p.exists():
            frames.append(pd.DataFrame(json.loads(p.read_text())))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if "parse_failure" not in df.columns:
        df["parse_failure"] = False
    return df


def get_clean_qids(df):
    """Questions parsed successfully in ALL 4 versions for ALL models."""
    keep = set(df["question_id"].unique())
    for m in MODELS:
        mdf = df[df.model_short == m]
        keep -= set(mdf.loc[mdf.parse_failure == True, "question_id"])
    return keep


# ============================================================
# Metrics
# ============================================================

def accuracy_summary(df):
    acc = (df.groupby(["model_short", "version"])
           .agg(n_correct=("is_correct", "sum"), n_total=("is_correct", "count"))
           .reset_index())
    acc["accuracy"] = acc.n_correct / acc.n_total
    summary = (acc.groupby("model_short")["accuracy"]
               .agg(mean="mean", std="std", min="min", max="max")
               .reset_index())
    summary["range"] = summary["max"] - summary["min"]
    return summary.to_dict("records"), acc


def item_flip_rate(df):
    grp = (df.groupby(["model_short", "question_id"])
           .agg(cc=("is_correct", "sum"), tt=("is_correct", "count"))
           .reset_index())
    grp["flipped"] = grp.apply(lambda r: int(r.cc > 0 and r.cc < r.tt), axis=1)
    out = {}
    for m in MODELS:
        md = grp[grp.model_short == m]
        n_flip = int(md.flipped.sum())
        n_tot = len(md)
        out[m] = {"n_flipped": n_flip, "n_total": n_tot,
                  "rate": n_flip / n_tot if n_tot else 0}
    return out


def pairwise_bootstrap(df, n_boot=BOOTSTRAP_N):
    q_acc = {}
    for m in MODELS:
        mdf = df[df.model_short == m]
        q_acc[m] = mdf.groupby("question_id")["is_correct"].mean()
    rng = np.random.RandomState(42)
    rows = []
    for m1, m2 in combinations(MODELS, 2):
        common = q_acc[m1].index.intersection(q_acc[m2].index)
        if len(common) == 0:
            continue
        d = q_acc[m1].loc[common].values - q_acc[m2].loc[common].values
        boot = np.array([d[rng.choice(len(d), len(d), replace=True)].mean() for _ in range(n_boot)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        sig = (lo > 0) or (hi < 0)
        rows.append({"model_1": m1, "model_2": m2, "mean_gap": float(d.mean()),
                      "ci_lower": float(lo), "ci_upper": float(hi),
                      "significant": bool(sig), "n": int(len(common))})
    return rows


def rank_distribution(df, n_boot=BOOTSTRAP_N):
    q_acc = {}
    for m in MODELS:
        q_acc[m] = df[df.model_short == m].groupby("question_id")["is_correct"].mean()
    common = q_acc[MODELS[0]].index
    for m in MODELS[1:]:
        common = common.intersection(q_acc[m].index)
    rng = np.random.RandomState(42)
    counts = {m: np.zeros(len(MODELS)) for m in MODELS}
    for _ in range(n_boot):
        idx = rng.choice(len(common), len(common), replace=True)
        qids = common[idx]
        accs = {m: q_acc[m].loc[qids].mean() for m in MODELS}
        ranked = sorted(accs, key=lambda m: -accs[m])
        for rank, m in enumerate(ranked):
            counts[m][rank] += 1
    return [{
        "model": m,
        **{f"rank_{r+1}_prob": float(counts[m][r] / n_boot) for r in range(len(MODELS))}
    } for m in MODELS]


def reversal_frequency(df):
    acc_mv = df.groupby(["model_short", "version"])["is_correct"].mean().reset_index()
    acc_mv.columns = ["model", "version", "accuracy"]
    rows = []
    for m1, m2 in combinations(MODELS, 2):
        a1 = acc_mv[acc_mv.model == m1].set_index("version")["accuracy"]
        a2 = acc_mv[acc_mv.model == m2].set_index("version")["accuracy"]
        common_v = sorted(set(a1.index) & set(a2.index))
        overall_diff = a1.loc[common_v].mean() - a2.loc[common_v].mean()
        m1_better = overall_diff > 0
        n_rev = sum(1 for v in common_v if (a1.loc[v] > a2.loc[v]) != m1_better)
        rows.append({"pair": f"{m1} vs {m2}", "reversal_count": n_rev,
                      "total_versions": len(common_v),
                      "reversal_rate": n_rev / len(common_v) if common_v else 0,
                      "overall_better": m1 if m1_better else m2})
    return rows


# ============================================================
# Cross-experiment (if Exp I available)
# ============================================================

def cross_experiment(dataset_key, clean_flip, clean_acc_summary):
    exp1_path = ROOT.parent / "exp1" / "results_exp1"
    exp1_analysis = ROOT.parent / "exp1" / "analysis_exp1"
    if not exp1_analysis.exists():
        return None

    bench_map = {"arc": "arc", "mmlu": "mmlu"}
    b1 = bench_map[dataset_key]
    analysis = json.loads((exp1_analysis / f"analysis_{b1}.json").read_text())
    noise_map = json.loads((exp1_analysis / f"noise_map_{b1}.json").read_text())

    e1_map = {"llama": "llama-3.1-8b", "qwen7b": "qwen2.5-7b",
              "qwen32b": "qwen3-32b", "qwen72b": "qwen2.5-72b"}
    # Derive with_explanation indices: answer_format dim (index 1) == 2
    # Mirrors exp1/prompt_variants.py get_all_variants() with seed=42
    import sys, importlib
    sys.path.insert(0, str(ROOT.parent / "exp1"))
    pv_mod = importlib.import_module("prompt_variants")
    sys.path.pop(0)
    _all_variants = pv_mod.get_all_variants()
    EXPL_IDX = {i for i, (_, idx) in enumerate(_all_variants) if idx[1] == 2}

    cross = {}

    # Flip rate comparison
    flip_comp = []
    for m1, m2 in e1_map.items():
        if m1 not in analysis or m2 not in clean_flip:
            continue
        e1_all = analysis[m1]["item_flip_rate"] * 100
        e2 = clean_flip[m2]["rate"] * 100
        flip_comp.append({"model": m2, "exp1_all": e1_all, "exp2": e2})
    cross["flip_comparison"] = flip_comp

    # Three-way variance decomposition (ALL + NO-EXPL)
    var_decomp = []
    for m1, m2 in e1_map.items():
        if m1 not in analysis:
            continue
        pv = analysis[m1]["accuracy_stats"]["per_variant"]
        vs = analysis[m1].get("variance_decomposition", {}).get("var_sampling", 0)
        e2_acc = [s for s in clean_acc_summary if s["model_short"] == m2]
        if not e2_acc:
            continue
        vt = e2_acc[0]["std"] ** 2
        for tag, idx_set in [("ALL", set(range(len(pv)))),
                              ("NO-EXPL", set(range(len(pv))) - EXPL_IDX)]:
            vp = float(np.var([pv[i] for i in sorted(idx_set)]))
            total = vp + vs + vt
            var_decomp.append({
                "model": m2, "version": tag,
                "pct_prompt": 100 * vp / total if total else 0,
                "pct_sampling": 100 * vs / total if total else 0,
                "pct_testset": 100 * vt / total if total else 0,
            })
    cross["variance_decomposition"] = var_decomp

    return cross


# ============================================================
# Main
# ============================================================

def main():
    all_results = {}

    for ds_key, bench in DATASETS.items():
        log.info(f"\n{'='*60}\nAnalyzing {bench}\n{'='*60}")

        df_raw = load_results(ds_key)
        if len(df_raw) == 0:
            log.warning(f"No data for {bench}")
            continue

        stats = {}

        # Parse failure report
        pf = {}
        for m in MODELS:
            mdf = df_raw[df_raw.model_short == m]
            n_pf = int(mdf.parse_failure.sum())
            pf[m] = {"total": n_pf, "out_of": len(mdf), "rate": n_pf / len(mdf) if len(mdf) else 0}
            if n_pf > 0:
                log.info(f"  {m}: {n_pf}/{len(mdf)} parse failures ({n_pf/len(mdf)*100:.1f}%)")
        stats["parse_failure"] = pf

        # Filter to clean subset
        clean_qids = get_clean_qids(df_raw)
        df = df_raw[df_raw.question_id.isin(clean_qids)].copy()
        stats["n_evaluable"] = len(clean_qids)
        log.info(f"  Evaluable questions: {len(clean_qids)} (of {df_raw.question_id.nunique()})")

        # Two-layer accuracy (MMLU only if there are PFs)
        if any(p["total"] > 0 for p in pf.values()):
            two_layer = []
            for m in MODELS:
                mdf = df_raw[df_raw.model_short == m]
                acc_all = float(mdf.groupby("version")["is_correct"].mean().mean())
                mdf_c = mdf[mdf.question_id.isin(clean_qids)]
                acc_clean = float(mdf_c.groupby("version")["is_correct"].mean().mean())
                two_layer.append({"model": m, "acc_all_150": acc_all,
                                  "acc_clean": acc_clean, "n_pf": pf[m]["total"]})
            stats["two_layer_accuracy"] = two_layer

        # Accuracy
        acc_summary, acc_per_v = accuracy_summary(df)
        stats["accuracy_summary"] = acc_summary
        for a in sorted(acc_summary, key=lambda x: x["mean"]):
            log.info(f"  {a['model_short']:15s}: {a['mean']*100:.2f}% ± {a['std']*100:.2f}%")

        # Flip rate
        flip = item_flip_rate(df)
        stats["item_flip_rate"] = flip
        for m in MODELS:
            f = flip[m]
            log.info(f"  {m:15s} flip: {f['n_flipped']}/{f['n_total']} = {f['rate']*100:.1f}%")

        # Pairwise gaps
        gaps = pairwise_bootstrap(df)
        stats["pairwise_gaps"] = gaps
        for g in gaps:
            sig = "Sig." if g["significant"] else "N.S."
            log.info(f"  {g['model_1']:15s} vs {g['model_2']:15s}: "
                     f"{g['mean_gap']*100:+.2f}pp [{g['ci_lower']*100:+.2f}, {g['ci_upper']*100:+.2f}] {sig}")

        # Reversals
        rev = reversal_frequency(df)
        stats["reversals"] = rev

        # Rank distribution
        rankd = rank_distribution(df)
        stats["rank_distribution"] = rankd

        # Cross-experiment
        cross = cross_experiment(ds_key, flip, acc_summary)
        if cross:
            stats["cross_experiment"] = cross

        all_results[ds_key] = stats

        # Save
        outpath = ANALYSIS_DIR / f"analysis_{ds_key}.json"
        outpath.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
        log.info(f"  Saved {outpath}")

    # Print summary
    print("\n" + "=" * 70)
    print("EXPERIMENT II ANALYSIS SUMMARY")
    print("=" * 70)
    for ds_key, stats in all_results.items():
        bench = DATASETS[ds_key]
        print(f"\n{bench} (N={stats['n_evaluable']}):")
        for a in sorted(stats["accuracy_summary"], key=lambda x: x["mean"]):
            print(f"  {a['model_short']:15s}: {a['mean']*100:.2f}% ± {a['std']*100:.2f}%  "
                  f"flip={stats['item_flip_rate'][a['model_short']]['rate']*100:.1f}%")
        for g in stats["pairwise_gaps"]:
            sig = "✓" if g["significant"] else "✗"
            print(f"  {g['model_1']:15s} vs {g['model_2']:15s}: "
                  f"{g['mean_gap']*100:+.2f}pp [{g['ci_lower']*100:+.2f}, {g['ci_upper']*100:+.2f}] {sig}")


if __name__ == "__main__":
    main()
