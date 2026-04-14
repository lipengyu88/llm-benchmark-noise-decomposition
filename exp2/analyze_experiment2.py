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
SOURCES = ["gpt4o", "qwen"]
BOOTSTRAP_N = 10_000



def load_results(dataset_key: str, source: str) -> pd.DataFrame:
    bench = DATASETS[dataset_key]
    frames = []
    for m in MODELS:
        p = ROOT / f"exp2_{bench}_{m}_{source}.json"
        if p.exists():
            frames.append(pd.DataFrame(json.loads(p.read_text())))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if "parse_failure" not in df.columns:
        df["parse_failure"] = False
    if "paraphrase_source" not in df.columns:
        df["paraphrase_source"] = source
    return df


def get_clean_qids(df: pd.DataFrame) -> set:
    """Questions parsed successfully in ALL 4 versions for ALL models."""
    keep = set(df["question_id"].unique())
    for m in MODELS:
        mdf = df[df.model_short == m]
        keep -= set(mdf.loc[mdf.parse_failure == True, "question_id"])
    return keep


def prepare_analysis_views(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, set]:
    """Build analysis views.

    Full-set view keeps every attempt and treats parse failures / missing parses
    as incorrect. Clean view keeps only questions with no parse failures for any
    model. This mirrors Exp I's philosophy: retain the task and report parsing
    issues rather than silently dropping the item from the primary analysis.
    """
    df_full = df_raw.copy()
    df_full["is_correct"] = df_full["is_correct"].fillna(False).astype(int)

    clean_qids = get_clean_qids(df_raw)
    df_clean = df_raw[df_raw.question_id.isin(clean_qids)].copy()
    if len(df_clean) > 0:
        df_clean["is_correct"] = df_clean["is_correct"].fillna(False).astype(int)

    return df_full, df_clean, clean_qids



def accuracy_summary(df: pd.DataFrame):
    acc = (df.groupby(["model_short", "version"])
           .agg(n_correct=("is_correct", "sum"), n_total=("is_correct", "count"))
           .reset_index())
    acc["accuracy"] = acc.n_correct / acc.n_total
    summary = (acc.groupby("model_short")["accuracy"]
               .agg(mean="mean", std="std", min="min", max="max")
               .reset_index())
    summary["range"] = summary["max"] - summary["min"]
    return summary.to_dict("records"), acc


def item_flip_rate(df: pd.DataFrame) -> dict:
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


def pairwise_bootstrap(df: pd.DataFrame, n_boot=BOOTSTRAP_N):
    """Bootstrap pairwise gaps with p-value for BH correction."""
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
        boot = np.array([d[rng.choice(len(d), len(d), replace=True)].mean()
                         for _ in range(n_boot)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        sig = (lo > 0) or (hi < 0)

        if d.mean() > 0:
            p_value = float(np.mean(boot <= 0)) * 2
        else:
            p_value = float(np.mean(boot >= 0)) * 2
        p_value = min(p_value, 1.0)

        rows.append({
            "model_1": m1, "model_2": m2,
            "mean_gap": float(d.mean()),
            "ci_lower": float(lo), "ci_upper": float(hi),
            "significant": bool(sig),
            "p_value": p_value,
            "n": int(len(common)),
        })
    return rows


def apply_bh_correction(gaps: list[dict]) -> list[dict]:
    """Apply Benjamini-Hochberg FDR correction to pairwise comparisons."""
    if not gaps:
        return gaps

    p_values = [g["p_value"] for g in gaps]
    n = len(p_values)

    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]
    corrected = np.zeros(n)

    for i in range(n - 1, -1, -1):
        rank = i + 1
        if i == n - 1:
            corrected[i] = sorted_p[i]
        else:
            corrected[i] = min(sorted_p[i] * n / rank, corrected[i + 1])
        corrected[i] = min(corrected[i], 1.0)

    result_corrected = np.zeros(n)
    for i, orig_idx in enumerate(sorted_indices):
        result_corrected[orig_idx] = corrected[i]

    for i, g in enumerate(gaps):
        g["p_value_bh"] = float(result_corrected[i])
        g["significant_bh"] = bool(result_corrected[i] < 0.05)

    return gaps


def rank_distribution(df: pd.DataFrame, n_boot=BOOTSTRAP_N):
    q_acc = {}
    for m in MODELS:
        q_acc[m] = df[df.model_short == m].groupby("question_id")["is_correct"].mean()
    common = q_acc[MODELS[0]].index
    for m in MODELS[1:]:
        common = common.intersection(q_acc[m].index)
    if len(common) == 0:
        return []
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


def reversal_frequency(df: pd.DataFrame) -> list[dict]:
    acc_mv = df.groupby(["model_short", "version"])["is_correct"].mean().reset_index()
    acc_mv.columns = ["model", "version", "accuracy"]
    rows = []
    for m1, m2 in combinations(MODELS, 2):
        a1 = acc_mv[acc_mv.model == m1].set_index("version")["accuracy"]
        a2 = acc_mv[acc_mv.model == m2].set_index("version")["accuracy"]
        common_v = sorted(set(a1.index) & set(a2.index))
        if not common_v:
            continue
        overall_diff = a1.loc[common_v].mean() - a2.loc[common_v].mean()
        m1_better = overall_diff > 0
        n_rev = sum(1 for v in common_v if (a1.loc[v] > a2.loc[v]) != m1_better)
        rows.append({"pair": f"{m1} vs {m2}", "reversal_count": n_rev,
                      "total_versions": len(common_v),
                      "reversal_rate": n_rev / len(common_v) if common_v else 0,
                      "overall_better": m1 if m1_better else m2})
    return rows



def paraphrase_diversity(dataset_key: str, source: str) -> dict:
    """Measure paraphrase diversity: unique rate, avg edit distance."""
    if dataset_key == "arc":
        fname = f"arc_challenge_paraphrased_{source}.json"
    else:
        fname = f"mmlu_pro_paraphrased_{source}.json"

    path = ROOT / fname
    if not path.exists():
        return {}

    questions = json.loads(path.read_text())
    n_total = len(questions)
    n_all_unique = sum(1 for q in questions if len(set(q["paraphrases"])) == 3)
    n_any_dup = sum(1 for q in questions if len(set(q["paraphrases"])) < 3)

    edit_ratios = []
    for q in questions:
        for p in q["paraphrases"]:
            if p != q["question"]:
                ratio = abs(len(p) - len(q["question"])) / max(len(p), len(q["question"]))
                edit_ratios.append(ratio)

    return {
        "n_questions": n_total,
        "n_all_3_unique": n_all_unique,
        "n_any_duplicate": n_any_dup,
        "all_unique_rate": n_all_unique / n_total if n_total else 0,
        "mean_length_change_ratio": float(np.mean(edit_ratios)) if edit_ratios else 0,
    }



def cross_source_comparison(
    stats_gpt4o: dict, stats_qwen: dict, dataset_key: str
) -> dict:
    """Compare results between GPT-4o and Qwen paraphrase sources."""
    comparison = {}
    n_gpt4o = stats_gpt4o.get("n_questions_total", 0)
    n_qwen = stats_qwen.get("n_questions_total", 0)
    comparison["n_questions_gpt4o"] = n_gpt4o
    comparison["n_questions_qwen"] = n_qwen
    comparison["n_clean_gpt4o"] = stats_gpt4o.get("n_questions_clean", 0)
    comparison["n_clean_qwen"] = stats_qwen.get("n_questions_clean", 0)
    comparison["note"] = (
        "Primary comparisons use the full question set with parse failures "
        "counted as incorrect. Clean-subset counts are provided for sensitivity "
        "analysis."
    )

    acc_comp = []
    for m in MODELS:
        g = next((a for a in stats_gpt4o.get("accuracy_summary", [])
                  if a["model_short"] == m), None)
        q = next((a for a in stats_qwen.get("accuracy_summary", [])
                  if a["model_short"] == m), None)
        if g and q:
            acc_comp.append({
                "model": m,
                "gpt4o_mean": g["mean"], "gpt4o_std": g["std"],
                "qwen_mean": q["mean"], "qwen_std": q["std"],
                "mean_diff": g["mean"] - q["mean"],
            })
    comparison["accuracy"] = acc_comp

    flip_comp = []
    gfr = stats_gpt4o.get("item_flip_rate", {})
    qfr = stats_qwen.get("item_flip_rate", {})
    for m in MODELS:
        if m in gfr and m in qfr:
            flip_comp.append({
                "model": m,
                "gpt4o_flip": gfr[m]["rate"],
                "qwen_flip": qfr[m]["rate"],
                "diff": gfr[m]["rate"] - qfr[m]["rate"],
            })
    comparison["flip_rate"] = flip_comp

    g_gaps = {(g["model_1"], g["model_2"]): g
              for g in stats_gpt4o.get("pairwise_gaps", [])}
    q_gaps = {(g["model_1"], g["model_2"]): g
              for g in stats_qwen.get("pairwise_gaps", [])}
    sig_comp = []
    for pair in g_gaps:
        if pair in q_gaps:
            gg, qg = g_gaps[pair], q_gaps[pair]
            sig_comp.append({
                "pair": f"{pair[0]} vs {pair[1]}",
                "gpt4o_sig": gg.get("significant_bh", gg["significant"]),
                "qwen_sig": qg.get("significant_bh", qg["significant"]),
                "gpt4o_gap": gg["mean_gap"],
                "qwen_gap": qg["mean_gap"],
                "agree": (gg.get("significant_bh", gg["significant"])
                          == qg.get("significant_bh", qg["significant"])),
            })
    comparison["significance_agreement"] = sig_comp
    n_agree = sum(1 for s in sig_comp if s["agree"])
    comparison["agreement_rate"] = n_agree / len(sig_comp) if sig_comp else 0

    return comparison



def cross_experiment(dataset_key: str, source: str, flip_stats, acc_summary):
    """Compare with Experiment I results."""
    exp1_analysis_dir = ROOT.parent / "exp1" / "analysis_exp1"
    if not exp1_analysis_dir.exists():
        return None

    bench_map = {"arc": "arc", "mmlu": "mmlu"}
    b1 = bench_map[dataset_key]
    analysis_path = exp1_analysis_dir / f"analysis_{b1}.json"
    noise_map_path = exp1_analysis_dir / f"noise_map_{b1}.json"

    if not analysis_path.exists():
        return None

    analysis = json.loads(analysis_path.read_text())
    noise_map = json.loads(noise_map_path.read_text()) if noise_map_path.exists() else {}

    e1_map = {"llama": "llama-3.1-8b", "qwen7b": "qwen2.5-7b",
              "qwen32b": "qwen3-32b", "qwen72b": "qwen2.5-72b"}
    EXPL_IDX = {4, 8, 9, 14}

    cross = {"source": source}

    flip_comp = []
    for m1, m2 in e1_map.items():
        if m1 not in analysis or m2 not in flip_stats:
            continue
        e1_all = analysis[m1]["item_flip_rate"] * 100
        e2 = flip_stats[m2]["rate"] * 100
        flip_comp.append({"model": m2, "exp1_all": e1_all, "exp2": e2})
    cross["flip_comparison"] = flip_comp

    var_decomp = []
    for m1, m2 in e1_map.items():
        if m1 not in analysis:
            continue
        pv = analysis[m1]["accuracy_stats"]["per_variant"]
        vs = analysis[m1].get("variance_decomposition", {}).get("var_sampling", 0)
        e2_acc = [s for s in acc_summary if s["model_short"] == m2]
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



def analyze_source(ds_key: str, source: str) -> dict | None:
    bench = DATASETS[ds_key]
    df_raw = load_results(ds_key, source)
    if len(df_raw) == 0:
        log.warning(f"No data for {bench}/{source}")
        return None

    stats = {
        "source": source,
        "analysis_policy": {
            "primary": "full_set_parse_failures_count_as_incorrect",
            "supplemental": "clean_subset_no_parse_failures",
        },
    }

    pf = {}
    for m in MODELS:
        mdf = df_raw[df_raw.model_short == m]
        if len(mdf) == 0:
            continue
        n_pf = int(mdf.parse_failure.sum())
        pf[m] = {"total": n_pf, "out_of": len(mdf),
                  "rate": n_pf / len(mdf) if len(mdf) else 0}
        if n_pf > 0:
            log.info(f"  [{source}] {m}: {n_pf}/{len(mdf)} parse failures "
                     f"({n_pf/len(mdf)*100:.1f}%)")
    stats["parse_failure"] = pf

    df_full, df_clean, clean_qids = prepare_analysis_views(df_raw)
    stats["n_questions_total"] = int(df_raw.question_id.nunique())
    stats["n_questions_clean"] = len(clean_qids)
    stats["n_evaluable"] = len(clean_qids)
    log.info(f"  [{source}] Full-set questions: {stats['n_questions_total']}")
    log.info(f"  [{source}] Clean-subset questions: {len(clean_qids)} "
             f"(of {df_raw.question_id.nunique()})")

    two_layer = []
    for m in MODELS:
        mdf_all = df_full[df_full.model_short == m]
        if len(mdf_all) == 0:
            continue
        acc_all = float(mdf_all.groupby("version")["is_correct"].mean().mean())
        mdf_clean = df_clean[df_clean.model_short == m]
        acc_clean = float(mdf_clean.groupby("version")["is_correct"].mean().mean()) if len(mdf_clean) else 0.0
        two_layer.append({
            "model": m,
            "acc_all_questions": acc_all,
            "acc_all_150": acc_all,
            "acc_clean": acc_clean,
            "n_pf": pf.get(m, {}).get("total", 0),
        })
    stats["two_layer_accuracy"] = two_layer

    acc_summary, acc_per_v = accuracy_summary(df_full)
    stats["accuracy_summary"] = acc_summary
    for a in sorted(acc_summary, key=lambda x: x["mean"]):
        log.info(f"  [{source}] {a['model_short']:15s}: "
                 f"{a['mean']*100:.2f}% +/- {a['std']*100:.2f}% (full set)")

    flip = item_flip_rate(df_full)
    stats["item_flip_rate"] = flip

    gaps = pairwise_bootstrap(df_full)
    gaps = apply_bh_correction(gaps)
    stats["pairwise_gaps"] = gaps
    for g in gaps:
        sig_raw = "Sig." if g["significant"] else "N.S."
        sig_bh = "Sig.(BH)" if g["significant_bh"] else "N.S.(BH)"
        log.info(f"  [{source}] {g['model_1']:15s} vs {g['model_2']:15s}: "
                 f"{g['mean_gap']*100:+.2f}pp "
                 f"[{g['ci_lower']*100:+.2f}, {g['ci_upper']*100:+.2f}] "
                 f"{sig_raw} | p={g['p_value']:.4f} -> p_bh={g['p_value_bh']:.4f} {sig_bh}")

    rev = reversal_frequency(df_full)
    stats["reversals"] = rev

    rankd = rank_distribution(df_full)
    stats["rank_distribution"] = rankd

    clean_stats = {
        "n_questions": len(clean_qids),
    }
    if len(df_clean) > 0:
        clean_acc_summary, _ = accuracy_summary(df_clean)
        clean_stats["accuracy_summary"] = clean_acc_summary
        clean_stats["item_flip_rate"] = item_flip_rate(df_clean)
        clean_gaps = apply_bh_correction(pairwise_bootstrap(df_clean))
        clean_stats["pairwise_gaps"] = clean_gaps
        clean_stats["reversals"] = reversal_frequency(df_clean)
        clean_stats["rank_distribution"] = rank_distribution(df_clean)
    else:
        clean_stats["accuracy_summary"] = []
        clean_stats["item_flip_rate"] = {}
        clean_stats["pairwise_gaps"] = []
        clean_stats["reversals"] = []
        clean_stats["rank_distribution"] = []
    stats["clean_subset"] = clean_stats

    div = paraphrase_diversity(ds_key, source)
    stats["paraphrase_diversity"] = div
    if div:
        log.info(f"  [{source}] Paraphrase diversity: all_unique={div['all_unique_rate']*100:.1f}%")

    cross = cross_experiment(ds_key, source, flip, acc_summary)
    if cross:
        stats["cross_experiment"] = cross

    return stats



def main():
    all_results = {}

    for ds_key, bench in DATASETS.items():
        log.info(f"\n{'='*60}")
        log.info(f"Analyzing {bench}")
        log.info(f"{'='*60}")

        ds_results = {}

        for source in SOURCES:
            stats = analyze_source(ds_key, source)
            if stats:
                ds_results[source] = stats

        if not ds_results:
            log.warning(f"No data for {bench}")
            continue

        if "gpt4o" in ds_results and "qwen" in ds_results:
            log.info(f"\n--- Cross-Source Comparison ({bench}) ---")
            cross_src = cross_source_comparison(
                ds_results["gpt4o"], ds_results["qwen"], ds_key
            )
            ds_results["cross_source"] = cross_src
            log.info(f"  Significance agreement rate: {cross_src['agreement_rate']*100:.1f}%")
            for s in cross_src["significance_agreement"]:
                log.info(f"  {s['pair']}: gpt4o={s['gpt4o_sig']}, qwen={s['qwen_sig']}, "
                         f"agree={s['agree']}")

        all_results[ds_key] = ds_results

        outpath = ANALYSIS_DIR / f"analysis_{ds_key}.json"
        outpath.write_text(json.dumps(ds_results, indent=2, ensure_ascii=False, default=str))
        log.info(f"  Saved {outpath}")

    print("\n" + "=" * 70)
    print("EXPERIMENT II ANALYSIS SUMMARY")
    print("=" * 70)
    for ds_key, ds_results in all_results.items():
        bench = DATASETS[ds_key]
        for source, stats in ds_results.items():
            if source == "cross_source":
                continue
            print(
                f"\n{bench} / {source} "
                f"(full={stats.get('n_questions_total', 0)}, "
                f"clean={stats.get('n_questions_clean', stats.get('n_evaluable', 0))}):"
            )
            for a in sorted(stats["accuracy_summary"], key=lambda x: x["mean"]):
                fr = stats["item_flip_rate"].get(a["model_short"], {}).get("rate", 0)
                print(f"  {a['model_short']:15s}: {a['mean']*100:.2f}% +/- {a['std']*100:.2f}%  "
                      f"flip={fr*100:.1f}%")
            for g in stats["pairwise_gaps"]:
                sig_bh = "V" if g.get("significant_bh", g["significant"]) else "X"
                print(f"  {g['model_1']:15s} vs {g['model_2']:15s}: "
                      f"{g['mean_gap']*100:+.2f}pp "
                      f"[{g['ci_lower']*100:+.2f}, {g['ci_upper']*100:+.2f}] "
                      f"p_bh={g.get('p_value_bh', g.get('p_value', '?')):.4f} {sig_bh}")

        if "cross_source" in ds_results:
            cs = ds_results["cross_source"]
            print(f"\n  Cross-source agreement: {cs['agreement_rate']*100:.1f}%")


if __name__ == "__main__":
    main()
