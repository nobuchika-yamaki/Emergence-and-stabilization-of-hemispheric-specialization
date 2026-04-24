#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze outputs from revised_hemispheric_specialization.py.

Input folder:
~/Desktop/hemispheric_specialization_revised_outputs

Output folder:
~/Desktop/hemispheric_specialization_revised_outputs/manuscript_result_analysis

Main outputs:
- manuscript_results_report.txt
- manuscript_results_sentences.txt
- condition_summary_recomputed.csv
- key_numbers.csv
- key_tests.csv
- table_stage_main.csv
- table_controls.csv
- table_bias_sweep.csv
- table_decision_validity.csv
- table_landscape.csv
- analysis_figures/*.png
"""

from pathlib import Path
import os
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.stats import spearmanr
except Exception:
    spearmanr = None


# ============================================================
# 0. Paths
# ============================================================

OUTDIR = Path.home() / "Desktop" / "hemispheric_specialization_revised_outputs"
ANALYSIS_DIR = OUTDIR / "manuscript_result_analysis"
ANALYSIS_FIGDIR = ANALYSIS_DIR / "analysis_figures"

ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_FIGDIR.mkdir(parents=True, exist_ok=True)

SAVE_DPI = 300


# ============================================================
# 1. Utilities
# ============================================================

def safe_read_csv(name, required=False):
    path = OUTDIR / name
    if not path.exists():
        msg = f"Missing file: {path}"
        if required:
            raise FileNotFoundError(msg)
        print(msg)
        return pd.DataFrame()
    return pd.read_csv(path)


def numeric_series(df, col):
    if df.empty or col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce").dropna()


def fmt(x, digits=3):
    if x is None:
        return "NA"
    try:
        if pd.isna(x):
            return "NA"
    except Exception:
        pass

    x = float(x)

    if abs(x) >= 1000 or (abs(x) < 0.001 and x != 0):
        return f"{x:.{digits}e}"
    return f"{x:.{digits}f}"


def fmt_p(p):
    if p is None:
        return "NA"
    try:
        if pd.isna(p):
            return "NA"
    except Exception:
        pass

    p = float(p)
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.3f}"


def median_iqr(values):
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna().values
    if len(values) == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "sd": np.nan,
            "median": np.nan,
            "q1": np.nan,
            "q3": np.nan,
            "min": np.nan,
            "max": np.nan
        }

    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "median": float(np.median(values)),
        "q1": float(np.percentile(values, 25)),
        "q3": float(np.percentile(values, 75)),
        "min": float(np.min(values)),
        "max": float(np.max(values))
    }


def get_condition(df, condition):
    if df.empty or "condition" not in df.columns:
        return pd.DataFrame()
    return df[df["condition"] == condition].copy()


def polarity_counts(df):
    if df.empty or "polarity" not in df.columns:
        return {"right": 0, "left": 0, "symmetric": 0, "n_dir": 0, "prop_right": np.nan}

    right = int((df["polarity"] == "right_wider").sum())
    left = int((df["polarity"] == "left_wider").sum())
    symmetric = int((df["polarity"] == "symmetric").sum())
    n_dir = right + left
    prop_right = right / n_dir if n_dir > 0 else np.nan

    return {
        "right": right,
        "left": left,
        "symmetric": symmetric,
        "n_dir": n_dir,
        "prop_right": prop_right
    }


def find_test(test_df, comparison, metric):
    if test_df.empty:
        return pd.Series(dtype=object)
    sub = test_df[
        (test_df.get("comparison", "") == comparison) &
        (test_df.get("metric", "") == metric)
    ]
    if len(sub) == 0:
        return pd.Series(dtype=object)
    return sub.iloc[0]


def append_key(rows, section, item, value, note=""):
    rows.append({
        "section": section,
        "item": item,
        "value": value,
        "note": note
    })


def save_text(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ============================================================
# 2. Load files
# ============================================================

all_df = safe_read_csv("all_lineage_results.csv", required=True)
stat_summary = safe_read_csv("statistical_summary.csv")
polarity_summary = safe_read_csv("polarity_summary.csv")
paired_tests = safe_read_csv("paired_tests.csv")
independent_tests = safe_read_csv("independent_tests.csv")
decision_validity = safe_read_csv("decision_validity.csv")
decision_validity_stats = safe_read_csv("decision_validity_stats.csv")
bias_sweep_summary = safe_read_csv("bias_sweep_summary.csv")
bias_logistic = safe_read_csv("bias_logistic_regression.csv")
symmetry_controls = safe_read_csv("symmetry_controls.csv")
landscape_interpolation = safe_read_csv("landscape_interpolation.csv")
landscape_grid = safe_read_csv("landscape_grid.csv")
figure_index = safe_read_csv("figure_index.csv")

required_cols = [
    "condition", "lineage_id", "a_L", "a_R", "delta_a",
    "abs_delta_a", "polarity", "mean_rt", "fitness"
]
missing_cols = [c for c in required_cols if c not in all_df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns in all_lineage_results.csv: {missing_cols}")


# ============================================================
# 3. Recompute condition summary
# ============================================================

summary_rows = []
metrics = ["a_L", "a_R", "delta_a", "abs_delta_a", "mean_rt", "sd_rt", "fitness", "mean_abs_d"]

for condition, g in all_df.groupby("condition"):
    for metric in metrics:
        if metric not in g.columns:
            continue
        s = median_iqr(g[metric])
        summary_rows.append({
            "condition": condition,
            "metric": metric,
            **s
        })

condition_summary = pd.DataFrame(summary_rows)
condition_summary.to_csv(ANALYSIS_DIR / "condition_summary_recomputed.csv", index=False)


# ============================================================
# 4. Key condition tables
# ============================================================

main_conditions = [
    "stage1_symmetric",
    "bias_pos",
    "bias_neg",
    "persistence_pos_seed",
    "persistence_neg_seed",
    "persistence_pos_fresh",
    "persistence_neg_fresh",
    "env_structured_anomaly",
    "env_unstructured_anomaly"
]

control_conditions = [
    "control_forced_symmetry",
    "control_shared_parameter",
    "single_channel_baseline",
    "env_structured_no_anomaly",
    "env_unstructured_no_anomaly",
    "robust_pop40",
    "robust_trials64",
    "robust_mut003",
    "robust_mut007"
]

def make_condition_table(conditions):
    rows = []
    for c in conditions:
        g = get_condition(all_df, c)
        if g.empty:
            continue

        pc = polarity_counts(g)
        abs_s = median_iqr(g["abs_delta_a"]) if "abs_delta_a" in g.columns else {}
        rt_s = median_iqr(g["mean_rt"]) if "mean_rt" in g.columns else {}
        fit_s = median_iqr(g["fitness"]) if "fitness" in g.columns else {}

        rows.append({
            "condition": c,
            "n": len(g),
            "right_wider": pc["right"],
            "left_wider": pc["left"],
            "symmetric": pc["symmetric"],
            "prop_right": pc["prop_right"],
            "median_abs_delta_a": abs_s.get("median", np.nan),
            "q1_abs_delta_a": abs_s.get("q1", np.nan),
            "q3_abs_delta_a": abs_s.get("q3", np.nan),
            "mean_abs_delta_a": abs_s.get("mean", np.nan),
            "sd_abs_delta_a": abs_s.get("sd", np.nan),
            "median_mean_rt": rt_s.get("median", np.nan),
            "q1_mean_rt": rt_s.get("q1", np.nan),
            "q3_mean_rt": rt_s.get("q3", np.nan),
            "mean_mean_rt": rt_s.get("mean", np.nan),
            "sd_mean_rt": rt_s.get("sd", np.nan),
            "median_fitness": fit_s.get("median", np.nan),
            "q1_fitness": fit_s.get("q1", np.nan),
            "q3_fitness": fit_s.get("q3", np.nan)
        })

    return pd.DataFrame(rows)

table_stage_main = make_condition_table(main_conditions)
table_controls = make_condition_table(control_conditions)

table_stage_main.to_csv(ANALYSIS_DIR / "table_stage_main.csv", index=False)
table_controls.to_csv(ANALYSIS_DIR / "table_controls.csv", index=False)


# ============================================================
# 5. Bias sweep table
# ============================================================

bias_rows = []

if not bias_sweep_summary.empty:
    bias_table = bias_sweep_summary.copy()
else:
    sweep = all_df[all_df["condition"].astype(str).str.startswith("bias_sweep_delta")].copy()
    if not sweep.empty:
        for delta, g in sweep.groupby("delta"):
            pc = polarity_counts(g)
            bias_rows.append({
                "delta": float(delta),
                "right_wider": pc["right"],
                "left_wider": pc["left"],
                "n_directional": pc["n_dir"],
                "prop_right": pc["prop_right"]
            })
    bias_table = pd.DataFrame(bias_rows).sort_values("delta") if bias_rows else pd.DataFrame()

if not bias_table.empty:
    bias_table.to_csv(ANALYSIS_DIR / "table_bias_sweep.csv", index=False)


# ============================================================
# 6. Decision-validity table
# ============================================================

decision_table_rows = []

if not decision_validity.empty:
    for label, g in decision_validity.groupby("analysis_label"):
        for metric in ["mean_Q", "median_Q", "auc", "prop_Q_positive", "mean_abs_D_present", "mean_abs_D_absent"]:
            if metric not in g.columns:
                continue
            s = median_iqr(g[metric])
            decision_table_rows.append({
                "analysis_label": label,
                "metric": metric,
                **s
            })

decision_table = pd.DataFrame(decision_table_rows)
decision_table.to_csv(ANALYSIS_DIR / "table_decision_validity.csv", index=False)


# ============================================================
# 7. Landscape table
# ============================================================

landscape_rows = []

if not landscape_interpolation.empty:
    li = landscape_interpolation.copy()

    endpoint_left = li.iloc[0]
    endpoint_right = li.iloc[-1]
    midpoint = li.iloc[(li["t"] - 0.5).abs().idxmin()]
    best = li.iloc[li["fitness"].idxmax()]
    worst = li.iloc[li["fitness"].idxmin()]

    endpoint_mean_fitness = float(np.mean([endpoint_left["fitness"], endpoint_right["fitness"]]))
    endpoint_mean_rt = float(np.mean([endpoint_left["mean_rt"], endpoint_right["mean_rt"]]))

    landscape_rows.extend([
        {
            "quantity": "endpoint_0_fitness",
            "value": float(endpoint_left["fitness"]),
            "note": "t = 0 endpoint"
        },
        {
            "quantity": "endpoint_1_fitness",
            "value": float(endpoint_right["fitness"]),
            "note": "t = 1 endpoint"
        },
        {
            "quantity": "endpoint_mean_fitness",
            "value": endpoint_mean_fitness,
            "note": "mean of t = 0 and t = 1 endpoint fitness"
        },
        {
            "quantity": "midpoint_fitness",
            "value": float(midpoint["fitness"]),
            "note": "nearest t = 0.5"
        },
        {
            "quantity": "fitness_drop_endpoint_mean_minus_midpoint",
            "value": endpoint_mean_fitness - float(midpoint["fitness"]),
            "note": "positive value indicates midpoint lower fitness"
        },
        {
            "quantity": "midpoint_mean_rt",
            "value": float(midpoint["mean_rt"]),
            "note": "nearest t = 0.5"
        },
        {
            "quantity": "endpoint_mean_rt",
            "value": endpoint_mean_rt,
            "note": "mean of endpoint RT values"
        },
        {
            "quantity": "rt_increase_midpoint_minus_endpoint_mean",
            "value": float(midpoint["mean_rt"]) - endpoint_mean_rt,
            "note": "positive value indicates midpoint slower"
        },
        {
            "quantity": "best_path_fitness",
            "value": float(best["fitness"]),
            "note": f"t = {best['t']}"
        },
        {
            "quantity": "worst_path_fitness",
            "value": float(worst["fitness"]),
            "note": f"t = {worst['t']}"
        }
    ])

if not landscape_grid.empty:
    lg = landscape_grid.copy()
    best_grid = lg.iloc[lg["fitness"].idxmax()]
    worst_grid = lg.iloc[lg["fitness"].idxmin()]

    near_diag = lg[lg["abs_delta_a"] <= 0.05].copy()
    if not near_diag.empty:
        near_diag_s = median_iqr(near_diag["fitness"])
        landscape_rows.append({
            "quantity": "near_diagonal_median_fitness_abs_delta_a_le_0.05",
            "value": near_diag_s["median"],
            "note": "2D grid points with |Delta a| <= 0.05"
        })

    landscape_rows.extend([
        {
            "quantity": "best_grid_fitness",
            "value": float(best_grid["fitness"]),
            "note": f"a_L = {best_grid['a_L']}, a_R = {best_grid['a_R']}"
        },
        {
            "quantity": "worst_grid_fitness",
            "value": float(worst_grid["fitness"]),
            "note": f"a_L = {worst_grid['a_L']}, a_R = {worst_grid['a_R']}"
        }
    ])

landscape_table = pd.DataFrame(landscape_rows)
landscape_table.to_csv(ANALYSIS_DIR / "table_landscape.csv", index=False)


# ============================================================
# 8. Key tests table
# ============================================================

test_rows = []

def add_paired(comparison, metric, label):
    r = find_test(paired_tests, comparison, metric)
    if r.empty:
        return
    test_rows.append({
        "analysis": label,
        "comparison": comparison,
        "metric": metric,
        "n": r.get("n_pairs", np.nan),
        "median_x": r.get("median_x", np.nan),
        "median_y": r.get("median_y", np.nan),
        "median_difference_x_minus_y": r.get("median_difference_x_minus_y", np.nan),
        "ci_low": r.get("difference_ci_low", np.nan),
        "ci_high": r.get("difference_ci_high", np.nan),
        "p_value": r.get("p_value", np.nan),
        "effect_size": r.get("rank_biserial", np.nan),
        "effect_size_type": "rank_biserial"
    })

def add_independent(comparison, metric, label):
    r = find_test(independent_tests, comparison, metric)
    if r.empty:
        return
    test_rows.append({
        "analysis": label,
        "comparison": comparison,
        "metric": metric,
        "n": f"{r.get('n_x', 'NA')}/{r.get('n_y', 'NA')}",
        "median_x": r.get("median_x", np.nan),
        "median_y": r.get("median_y", np.nan),
        "median_difference_x_minus_y": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "p_value": r.get("p_value", np.nan),
        "effect_size": r.get("cliffs_delta", np.nan),
        "effect_size_type": "cliffs_delta"
    })

for m in ["abs_delta_a", "mean_rt", "fitness"]:
    add_paired(
        "structured_vs_unstructured_with_anomaly",
        m,
        "Environmental structure"
    )
    add_paired(
        "persistence_pos_seed_vs_fresh",
        m,
        "Persistence positive seed versus fresh symmetry"
    )
    add_paired(
        "persistence_neg_seed_vs_fresh",
        m,
        "Persistence negative seed versus fresh symmetry"
    )
    add_paired(
        "bias_pos_parent_vs_persistence",
        m,
        "Positive-bias parent versus persistence"
    )
    add_paired(
        "bias_neg_parent_vs_persistence",
        m,
        "Negative-bias parent versus persistence"
    )

for m in ["abs_delta_a", "mean_rt", "fitness"]:
    add_independent(
        "stage1_vs_forced_symmetry",
        m,
        "Free model versus forced symmetry"
    )
    add_independent(
        "stage1_vs_shared_parameter",
        m,
        "Free model versus shared parameter"
    )

for m in ["mean_rt", "fitness"]:
    add_independent(
        "stage1_vs_single_channel",
        m,
        "Free model versus single-channel baseline"
    )

key_tests = pd.DataFrame(test_rows)
key_tests.to_csv(ANALYSIS_DIR / "key_tests.csv", index=False)


# ============================================================
# 9. Key numbers and report text
# ============================================================

key_rows = []
report = []
sentences = []

report.append("Manuscript results analysis")
report.append("")
report.append(f"Input folder: {OUTDIR}")
report.append(f"Output folder: {ANALYSIS_DIR}")
report.append("")

# Stage 1
stage1 = get_condition(all_df, "stage1_symmetric")
if not stage1.empty:
    abs_s = median_iqr(stage1["abs_delta_a"])
    rt_s = median_iqr(stage1["mean_rt"])
    fit_s = median_iqr(stage1["fitness"])
    pc = polarity_counts(stage1)

    p_stage1 = np.nan
    if not polarity_summary.empty:
        sub = polarity_summary[polarity_summary["condition"] == "stage1_symmetric"]
        if len(sub) > 0:
            p_stage1 = sub.iloc[0].get("binomial_p", np.nan)

    report.append("1. Symmetric-development analysis")
    report.append(
        f"Stage 1 included n = {len(stage1)} lineages. "
        f"Median |Delta a| = {fmt(abs_s['median'])} "
        f"(IQR {fmt(abs_s['q1'])}-{fmt(abs_s['q3'])}); "
        f"mean RT = {fmt(rt_s['mean'])} (SD {fmt(rt_s['sd'])}); "
        f"median fitness = {fmt(fit_s['median'])}."
    )
    report.append(
        f"Polarity counts: right-wider = {pc['right']}, "
        f"left-wider = {pc['left']}, symmetric = {pc['symmetric']}; "
        f"proportion right-wider = {fmt(pc['prop_right'])}; "
        f"binomial p = {fmt_p(p_stage1)}."
    )
    report.append("")

    sentences.append(
        f"Under exact developmental symmetry, final solutions showed clear bilateral differentiation "
        f"(median |Delta a| = {fmt(abs_s['median'])}, IQR = {fmt(abs_s['q1'])}-{fmt(abs_s['q3'])}, n = {len(stage1)}). "
        f"Polarity was not perfectly balanced, with {pc['right']} right-wider and {pc['left']} left-wider lineages "
        f"(exact binomial test, p = {fmt_p(p_stage1)})."
    )

    append_key(key_rows, "Stage 1", "n_lineages", len(stage1))
    append_key(key_rows, "Stage 1", "median_abs_delta_a", abs_s["median"])
    append_key(key_rows, "Stage 1", "IQR_abs_delta_a", f"{fmt(abs_s['q1'])}-{fmt(abs_s['q3'])}")
    append_key(key_rows, "Stage 1", "right_wider", pc["right"])
    append_key(key_rows, "Stage 1", "left_wider", pc["left"])
    append_key(key_rows, "Stage 1", "binomial_p", p_stage1)

# Bias fixed conditions
for cond, name in [("bias_pos", "Positive developmental bias"), ("bias_neg", "Negative developmental bias")]:
    g = get_condition(all_df, cond)
    if g.empty:
        continue

    abs_s = median_iqr(g["abs_delta_a"])
    rt_s = median_iqr(g["mean_rt"])
    pc = polarity_counts(g)

    p_val = np.nan
    if not polarity_summary.empty:
        sub = polarity_summary[polarity_summary["condition"] == cond]
        if len(sub) > 0:
            p_val = sub.iloc[0].get("binomial_p", np.nan)

    report.append(f"2. {name}")
    report.append(
        f"{cond}: n = {len(g)}, median |Delta a| = {fmt(abs_s['median'])} "
        f"(IQR {fmt(abs_s['q1'])}-{fmt(abs_s['q3'])}), "
        f"right-wider = {pc['right']}, left-wider = {pc['left']}, "
        f"proportion right-wider = {fmt(pc['prop_right'])}, "
        f"mean RT = {fmt(rt_s['mean'])} (SD {fmt(rt_s['sd'])}), "
        f"binomial p = {fmt_p(p_val)}."
    )
    report.append("")

# Bias sweep
if not bias_table.empty:
    report.append("3. Bias sweep")
    report.append("delta, proportion right-wider:")
    for _, r in bias_table.sort_values("delta").iterrows():
        report.append(f"  delta = {fmt(r['delta'], 4)}: prop_right = {fmt(r['prop_right'])}")

    if spearmanr is not None and "delta" in bias_table.columns and "prop_right" in bias_table.columns:
        valid = bias_table[["delta", "prop_right"]].dropna()
        if len(valid) >= 3:
            rho, p = spearmanr(valid["delta"], valid["prop_right"])
            report.append(f"Spearman correlation between delta and prop_right: rho = {fmt(rho)}, p = {fmt_p(p)}.")
            append_key(key_rows, "Bias sweep", "spearman_rho_delta_prop_right", rho)
            append_key(key_rows, "Bias sweep", "spearman_p", p)

    if not bias_logistic.empty:
        slope = bias_logistic[bias_logistic["parameter"] == "slope_delta"]
        if len(slope) > 0:
            r = slope.iloc[0]
            report.append(
                f"Logistic slope for delta predicting right-wider polarity: "
                f"estimate = {fmt(r.get('estimate'))}, "
                f"95% CI {fmt(r.get('ci_low'))}-{fmt(r.get('ci_high'))}."
            )

    report.append("")

    sentences.append(
        "Across the bias-sweep analysis, the probability of right-wider polarity changed systematically with developmental gain asymmetry, supporting a graded biasing effect rather than deterministic polarity fixation at weak bias levels."
    )

# Persistence
report.append("4. Persistence analyses")
for comparison, label in [
    ("bias_pos_parent_vs_persistence", "Positive-bias parent versus reseeded persistence"),
    ("bias_neg_parent_vs_persistence", "Negative-bias parent versus reseeded persistence"),
    ("persistence_pos_seed_vs_fresh", "Positive-bias reseeded persistence versus fresh-start symmetry"),
    ("persistence_neg_seed_vs_fresh", "Negative-bias reseeded persistence versus fresh-start symmetry")
]:
    for metric in ["abs_delta_a", "mean_rt", "fitness"]:
        r = find_test(paired_tests, comparison, metric)
        if r.empty:
            continue
        report.append(
            f"{label}, {metric}: median_x = {fmt(r.get('median_x'))}, "
            f"median_y = {fmt(r.get('median_y'))}, "
            f"median difference = {fmt(r.get('median_difference_x_minus_y'))}, "
            f"95% CI {fmt(r.get('difference_ci_low'))}-{fmt(r.get('difference_ci_high'))}, "
            f"p = {fmt_p(r.get('p_value'))}, "
            f"rank-biserial = {fmt(r.get('rank_biserial'))}."
        )
report.append("")

# Controls
report.append("5. Architecture and baseline controls")
for comparison, label in [
    ("stage1_vs_forced_symmetry", "Free model versus forced-symmetry control"),
    ("stage1_vs_shared_parameter", "Free model versus shared-parameter control"),
    ("stage1_vs_single_channel", "Free model versus single-channel baseline")
]:
    for metric in ["abs_delta_a", "mean_rt", "fitness"]:
        r = find_test(independent_tests, comparison, metric)
        if r.empty:
            continue
        report.append(
            f"{label}, {metric}: median_x = {fmt(r.get('median_x'))}, "
            f"median_y = {fmt(r.get('median_y'))}, "
            f"p = {fmt_p(r.get('p_value'))}, "
            f"Cliff's delta = {fmt(r.get('cliffs_delta'))}."
        )
report.append("")

# Environment
report.append("6. Environmental-structure analysis")
for metric in ["abs_delta_a", "mean_rt", "fitness"]:
    r = find_test(paired_tests, "structured_vs_unstructured_with_anomaly", metric)
    if r.empty:
        continue
    report.append(
        f"Structured versus unstructured background with anomaly, {metric}: "
        f"structured median = {fmt(r.get('median_x'))}, "
        f"unstructured median = {fmt(r.get('median_y'))}, "
        f"median difference = {fmt(r.get('median_difference_x_minus_y'))}, "
        f"95% CI {fmt(r.get('difference_ci_low'))}-{fmt(r.get('difference_ci_high'))}, "
        f"p = {fmt_p(r.get('p_value'))}, "
        f"rank-biserial = {fmt(r.get('rank_biserial'))}."
    )
report.append("")

# Symmetry controls
if not symmetry_controls.empty:
    report.append("7. Symmetry controls")

    for metric in ["swap_fitness_diff", "mirror_fitness_diff", "swap_mean_rt_diff", "mirror_mean_rt_diff"]:
        if metric in symmetry_controls.columns:
            s = median_iqr(symmetry_controls[metric])
            report.append(
                f"{metric}: median = {fmt(s['median'])}, "
                f"IQR {fmt(s['q1'])}-{fmt(s['q3'])}, "
                f"max absolute = {fmt(np.max(np.abs(pd.to_numeric(symmetry_controls[metric], errors='coerce').dropna())))}."
            )
            append_key(key_rows, "Symmetry controls", metric + "_median", s["median"])

    report.append("")

# Decision validity
if not decision_table.empty:
    report.append("8. Decision-validity analysis")
    for label in sorted(decision_table["analysis_label"].unique()):
        sub = decision_table[decision_table["analysis_label"] == label]
        auc = sub[sub["metric"] == "auc"]
        q = sub[sub["metric"] == "mean_Q"]
        prop = sub[sub["metric"] == "prop_Q_positive"]

        line = f"{label}: "
        if len(q) > 0:
            line += f"median mean_Q = {fmt(q.iloc[0]['median'])}; "
        if len(prop) > 0:
            line += f"median proportion Q > 0 = {fmt(prop.iloc[0]['median'])}; "
        if len(auc) > 0:
            line += f"median AUC = {fmt(auc.iloc[0]['median'])}."
        report.append(line)

    if not decision_validity_stats.empty:
        report.append("Decision-validity statistical tests:")
        for _, r in decision_validity_stats.iterrows():
            if r.get("metric") in ["mean_Q", "median_Q", "auc"]:
                report.append(
                    f"{r.get('analysis_label')}, {r.get('metric')}: "
                    f"median = {fmt(r.get('median'))}, "
                    f"95% CI {fmt(r.get('median_ci_low'))}-{fmt(r.get('median_ci_high'))}, "
                    f"p = {fmt_p(r.get('p_value'))}."
                )

    report.append("")

# Landscape
if not landscape_table.empty:
    report.append("9. Fitness-landscape analysis")
    for _, r in landscape_table.iterrows():
        report.append(f"{r['quantity']}: {fmt(r['value'])}. {r.get('note', '')}")
    report.append("")

# Robustness
report.append("10. Robustness analyses")
for cond in ["robust_pop40", "robust_trials64", "robust_mut003", "robust_mut007"]:
    g = get_condition(all_df, cond)
    if g.empty:
        continue
    abs_s = median_iqr(g["abs_delta_a"])
    pc = polarity_counts(g)
    report.append(
        f"{cond}: n = {len(g)}, median |Delta a| = {fmt(abs_s['median'])} "
        f"(IQR {fmt(abs_s['q1'])}-{fmt(abs_s['q3'])}), "
        f"right-wider = {pc['right']}, left-wider = {pc['left']}."
    )
report.append("")

# Figures
if not figure_index.empty:
    report.append("11. Figure index")
    for _, r in figure_index.iterrows():
        report.append(f"{r.get('filename', 'NA')}: {r.get('title', '')}")
    report.append("")


# Save key numbers
key_numbers = pd.DataFrame(key_rows)
key_numbers.to_csv(ANALYSIS_DIR / "key_numbers.csv", index=False)

save_text(ANALYSIS_DIR / "manuscript_results_report.txt", report)
save_text(ANALYSIS_DIR / "manuscript_results_sentences.txt", sentences)


# ============================================================
# 10. Diagnostic analysis figures
# ============================================================

def savefig(name):
    plt.tight_layout()
    plt.savefig(ANALYSIS_FIGDIR / name, dpi=SAVE_DPI)
    plt.close()


# A. Condition-level |Delta a|
plot_conditions = [
    "stage1_symmetric",
    "bias_pos",
    "bias_neg",
    "persistence_pos_seed",
    "persistence_neg_seed",
    "control_forced_symmetry",
    "control_shared_parameter",
    "env_structured_anomaly",
    "env_unstructured_anomaly"
]

data = []
labels = []
for c in plot_conditions:
    g = get_condition(all_df, c)
    vals = numeric_series(g, "abs_delta_a")
    if len(vals) > 0:
        data.append(vals.values)
        labels.append(c)

if data:
    plt.figure(figsize=(max(8, len(labels) * 0.7), 5))
    plt.boxplot(data, labels=labels)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("|Delta a|")
    plt.title("Condition-level differentiation magnitude")
    savefig("analysis_abs_delta_by_condition.png")


# B. Condition-level mean RT
data = []
labels = []
for c in plot_conditions + ["single_channel_baseline"]:
    g = get_condition(all_df, c)
    vals = numeric_series(g, "mean_rt")
    if len(vals) > 0:
        data.append(vals.values)
        labels.append(c)

if data:
    plt.figure(figsize=(max(8, len(labels) * 0.7), 5))
    plt.boxplot(data, labels=labels)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean RT proxy")
    plt.title("Condition-level performance")
    savefig("analysis_mean_rt_by_condition.png")


# C. Bias sweep
if not bias_table.empty and "delta" in bias_table.columns and "prop_right" in bias_table.columns:
    bt = bias_table.sort_values("delta")
    plt.figure(figsize=(6, 5))
    plt.plot(bt["delta"], bt["prop_right"], marker="o")
    plt.axhline(0.5, linestyle="--")
    plt.axvline(0.0, linestyle="--")
    plt.xlabel("Developmental gain asymmetry delta")
    plt.ylabel("Proportion right-wider")
    plt.title("Bias sweep")
    savefig("analysis_bias_sweep.png")


# D. Decision validity AUC
if not decision_validity.empty and "auc" in decision_validity.columns:
    labels = []
    data = []
    for label, g in decision_validity.groupby("analysis_label"):
        vals = numeric_series(g, "auc")
        if len(vals) > 0:
            labels.append(label)
            data.append(vals.values)
    if data:
        plt.figure(figsize=(6, 5))
        plt.boxplot(data, labels=labels)
        plt.axhline(0.5, linestyle="--")
        plt.ylabel("AUC")
        plt.title("Decision-validity AUC")
        savefig("analysis_decision_validity_auc.png")


# E. Landscape interpolation
if not landscape_interpolation.empty:
    plt.figure(figsize=(6, 5))
    plt.plot(landscape_interpolation["t"], landscape_interpolation["fitness"])
    plt.xlabel("Interpolation parameter t")
    plt.ylabel("Fitness")
    plt.title("Fitness along interpolation path")
    savefig("analysis_landscape_interpolation_fitness.png")

    plt.figure(figsize=(6, 5))
    plt.plot(landscape_interpolation["t"], landscape_interpolation["mean_rt"])
    plt.xlabel("Interpolation parameter t")
    plt.ylabel("Mean RT proxy")
    plt.title("Mean RT along interpolation path")
    savefig("analysis_landscape_interpolation_rt.png")


# ============================================================
# 11. Console summary
# ============================================================

print("")
print("Analysis complete.")
print(f"Input folder:  {OUTDIR}")
print(f"Output folder: {ANALYSIS_DIR}")
print("")
print("Main files to open:")
print(f"- {ANALYSIS_DIR / 'manuscript_results_report.txt'}")
print(f"- {ANALYSIS_DIR / 'manuscript_results_sentences.txt'}")
print(f"- {ANALYSIS_DIR / 'table_stage_main.csv'}")
print(f"- {ANALYSIS_DIR / 'key_tests.csv'}")
print(f"- {ANALYSIS_DIR / 'table_decision_validity.csv'}")
print(f"- {ANALYSIS_DIR / 'table_landscape.csv'}")
print("")

