
#!/usr/bin/env python3
"""
Revised simulation suite for bilateral hemispheric specialization.

Main goals:
1) Clean spontaneous symmetry-breaking test under exact developmental symmetry
   with lambda fixed to zero in the main manuscript analyses.
2) Directional-bias sweep across developmental gain asymmetry.
3) Persistence tests from both positive- and negative-bias seeds.
4) Symmetry controls (left-right swap and mirrored environments).
5) Fitness-valley analysis across interpolations between polarized states.
6) Environmental necessity analysis (structured vs planar).
7) Basic robustness analyses across core evolutionary hyperparameters.

Default behavior writes all outputs to:
    ~/Desktop/evolution_spatial_brain_repaired_results

A fast smoke test can be run with:
    python3 run_evolution_spatial_brain_repaired.py --smoke-test
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.stats import binomtest, wilcoxon


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class EvoConfig:
    grid_size: int = 20
    sigma_bg: float = 1.0
    anomaly_offset_lo: float = 1.5
    anomaly_offset_hi: float = 2.0
    anomaly_radius: int = 1               # 3x3 patch
    truncate: float = 3.0
    eps: float = 1e-6
    rt_cap: float = 1e5
    min_a: float = 0.1

    pop_size: int = 20
    parent_frac: float = 0.5
    elite_count: int = 0
    trials_per_agent: int = 8
    mutation_sigma: float = 0.05
    mutation_decay: float = 1.0

    init_mode: str = "exact_symmetric"    # exact_symmetric, random_symmetric, random_independent
    init_a_center: float = 1.0
    init_a_lo: float = 1.4
    init_a_hi: float = 1.6
    init_lambda_lo: float = 0.0
    init_lambda_hi: float = 0.0

    allow_evolvable_lambda: bool = False
    fixed_lambda: float = 0.0

    final_eval_trials: int = 20
    divergence_threshold: float = 0.3
    save_dpi: int = 220
    master_seed: int = 20260403

    # Main manuscript sizes
    stage1_lineages: int = 40
    stage2_lineages: int = 20
    stage2r_lineages: int = 20
    stage_gens: int = 50
    stage3_gens: int = 30

    # Supplementary analyses
    bias_sweep_lineages: int = 40
    bias_sweep_deltas: Tuple[float, ...] = (-0.05, -0.03, -0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02, 0.03, 0.05)
    env_compare_lineages: int = 30
    robustness_lineages: int = 20
    fitness_valley_points: int = 41
    fitness_valley_trials: int = 128
    symmetry_control_eval_trials: int = 128

    smoke_test: bool = False

    def apply_smoke_test(self) -> "EvoConfig":
        self.smoke_test = True
        self.pop_size = 12
        self.trials_per_agent = 4
        self.final_eval_trials = 8
        self.stage1_lineages = 8
        self.stage2_lineages = 6
        self.stage2r_lineages = 6
        self.stage_gens = 10
        self.stage3_gens = 8
        self.bias_sweep_lineages = 10
        self.bias_sweep_deltas = (-0.02, 0.0, 0.02)
        self.env_compare_lineages = 8
        self.robustness_lineages = 8
        self.fitness_valley_points = 17
        self.fitness_valley_trials = 32
        self.symmetry_control_eval_trials = 32
        return self


@dataclass
class Agent:
    a_L: float
    a_R: float
    lambd: float = 0.0

    def clip(self, cfg: EvoConfig) -> "Agent":
        self.a_L = max(cfg.min_a, float(self.a_L))
        self.a_R = max(cfg.min_a, float(self.a_R))
        self.lambd = float(self.lambd)
        return self

    def swapped(self) -> "Agent":
        return Agent(self.a_R, self.a_L, -self.lambd)


# -----------------------------
# Output helpers
# -----------------------------
def make_outdir(user_outdir: Optional[str], smoke: bool) -> Path:
    if user_outdir:
        outdir = Path(user_outdir).expanduser().resolve()
    else:
        name = "evolution_spatial_brain_repaired_smoke" if smoke else "evolution_spatial_brain_repaired_results"
        outdir = Path.home() / "Desktop" / name
    outdir.mkdir(parents=True, exist_ok=True)
    for sub in ["figures", "tables", "generation_traces", "logs"]:
        (outdir / sub).mkdir(exist_ok=True)
    return outdir


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def stable_sign(x: float, tol: float = 1e-12) -> int:
    if x > tol:
        return 1
    if x < -tol:
        return -1
    return 0


# -----------------------------
# Environment generation
# -----------------------------
def generate_environment(rng: np.random.Generator, cfg: EvoConfig, mode: str = "structured", mirror: bool = False) -> np.ndarray:
    raw = rng.normal(0.0, 1.0, size=(cfg.grid_size, cfg.grid_size))

    if mode == "structured":
        env = gaussian_filter(raw, sigma=cfg.sigma_bg, mode="reflect", truncate=cfg.truncate)
        cx = int(rng.integers(0, cfg.grid_size))
        cy = int(rng.integers(0, cfg.grid_size))
        amp = float(rng.uniform(cfg.anomaly_offset_lo, cfg.anomaly_offset_hi))
        x0 = max(0, cx - cfg.anomaly_radius)
        x1 = min(cfg.grid_size, cx + cfg.anomaly_radius + 1)
        y0 = max(0, cy - cfg.anomaly_radius)
        y1 = min(cfg.grid_size, cy + cfg.anomaly_radius + 1)
        env[x0:x1, y0:y1] += amp
    elif mode == "planar":
        env = raw.copy()
        env -= env.mean()
        sd = env.std(ddof=0)
        if sd > 0:
            env /= sd
    else:
        raise ValueError(f"Unknown environment mode: {mode}")

    if mirror:
        env = np.fliplr(env)
    return env


# -----------------------------
# Core computations
# -----------------------------
def mismatch_signal(x: np.ndarray, a: float, cfg: EvoConfig) -> float:
    sigma = max(cfg.min_a, float(a))
    xhat = gaussian_filter(x, sigma=sigma, mode="reflect", truncate=cfg.truncate)
    return float(np.abs(x - xhat).sum())


def decision_value(agent: Agent, M_L: float, M_R: float, m_L: float, m_R: float, cfg: EvoConfig) -> float:
    lambd = agent.lambd if cfg.allow_evolvable_lambda else cfg.fixed_lambda
    return (m_R * M_R - m_L * M_L) + lambd


def evaluate_agent(
    agent: Agent,
    rng: np.random.Generator,
    cfg: EvoConfig,
    m_L: float,
    m_R: float,
    n_trials: int,
    env_mode: str = "structured",
    mirror_env: bool = False,
) -> Tuple[float, float, List[float]]:
    rts: List[float] = []
    for _ in range(n_trials):
        env = generate_environment(rng, cfg, mode=env_mode, mirror=mirror_env)
        M_L = mismatch_signal(env, agent.a_L, cfg)
        M_R = mismatch_signal(env, agent.a_R, cfg)
        D = decision_value(agent, M_L, M_R, m_L, m_R, cfg)
        rt = 1.0 / (abs(D) + cfg.eps)
        if rt > cfg.rt_cap:
            rt = cfg.rt_cap
        rts.append(rt)
    mean_rt = float(np.mean(rts))
    fitness = -mean_rt
    return fitness, mean_rt, rts


def initialize_population(cfg: EvoConfig, rng: np.random.Generator) -> List[Agent]:
    pop: List[Agent] = []
    for _ in range(cfg.pop_size):
        if cfg.init_mode == "exact_symmetric":
            a = cfg.init_a_center
            lambd = cfg.fixed_lambda if not cfg.allow_evolvable_lambda else 0.0
            pop.append(Agent(a, a, lambd).clip(cfg))
        elif cfg.init_mode == "random_symmetric":
            a = float(rng.uniform(cfg.init_a_lo, cfg.init_a_hi))
            lambd = float(rng.uniform(cfg.init_lambda_lo, cfg.init_lambda_hi)) if cfg.allow_evolvable_lambda else cfg.fixed_lambda
            pop.append(Agent(a, a, lambd).clip(cfg))
        elif cfg.init_mode == "random_independent":
            aL = float(rng.uniform(cfg.init_a_lo, cfg.init_a_hi))
            aR = float(rng.uniform(cfg.init_a_lo, cfg.init_a_hi))
            lambd = float(rng.uniform(cfg.init_lambda_lo, cfg.init_lambda_hi)) if cfg.allow_evolvable_lambda else cfg.fixed_lambda
            pop.append(Agent(aL, aR, lambd).clip(cfg))
        else:
            raise ValueError(f"Unknown init_mode: {cfg.init_mode}")
    return pop


def reseed_population_from_template(template: Agent, cfg: EvoConfig, rng: np.random.Generator) -> List[Agent]:
    pop = [Agent(template.a_L, template.a_R, template.lambd).clip(cfg)]
    for _ in range(cfg.pop_size - 1):
        if cfg.allow_evolvable_lambda:
            lambd = template.lambd + rng.normal(0.0, cfg.mutation_sigma)
        else:
            lambd = cfg.fixed_lambda
        pop.append(
            Agent(
                template.a_L + rng.normal(0.0, cfg.mutation_sigma),
                template.a_R + rng.normal(0.0, cfg.mutation_sigma),
                lambd,
            ).clip(cfg)
        )
    return pop


def evaluate_population(
    population: List[Agent],
    rng: np.random.Generator,
    cfg: EvoConfig,
    m_L: float,
    m_R: float,
    n_trials: int,
    env_mode: str,
    mirror_env: bool = False,
) -> pd.DataFrame:
    rows = []
    for idx, agent in enumerate(population):
        fitness, mean_rt, _ = evaluate_agent(agent, rng, cfg, m_L, m_R, n_trials, env_mode=env_mode, mirror_env=mirror_env)
        delta_a = agent.a_R - agent.a_L
        rows.append(
            {
                "agent_idx": idx,
                "a_L": agent.a_L,
                "a_R": agent.a_R,
                "lambda": agent.lambd if cfg.allow_evolvable_lambda else cfg.fixed_lambda,
                "Delta_a": delta_a,
                "abs_Delta_a": abs(delta_a),
                "polarity": stable_sign(delta_a),
                "fitness": fitness,
                "mean_rt": mean_rt,
            }
        )
    return pd.DataFrame(rows)


def select_and_reproduce(population: List[Agent], eval_df: pd.DataFrame, cfg: EvoConfig, rng: np.random.Generator, generation: int) -> List[Agent]:
    eval_df = eval_df.sort_values("fitness", ascending=False).reset_index(drop=True)
    elite_agents: List[Agent] = []
    if cfg.elite_count > 0:
        for _, row in eval_df.iloc[:cfg.elite_count].iterrows():
            elite_agents.append(Agent(float(row["a_L"]), float(row["a_R"]), float(row["lambda"])).clip(cfg))

    n_parents = max(2, int(round(cfg.pop_size * cfg.parent_frac)))
    parent_pool = eval_df.iloc[:n_parents].copy()

    mutation_sigma = cfg.mutation_sigma * (cfg.mutation_decay ** max(0, generation - 1))

    children: List[Agent] = elite_agents[:]
    while len(children) < cfg.pop_size:
        i, j = rng.integers(0, len(parent_pool), size=2)
        p1 = parent_pool.iloc[int(i)]
        p2 = parent_pool.iloc[int(j)]
        if cfg.allow_evolvable_lambda:
            new_lambda = 0.5 * (float(p1["lambda"]) + float(p2["lambda"])) + rng.normal(0.0, mutation_sigma)
        else:
            new_lambda = cfg.fixed_lambda
        child = Agent(
            0.5 * (float(p1["a_L"]) + float(p2["a_L"])) + rng.normal(0.0, mutation_sigma),
            0.5 * (float(p1["a_R"]) + float(p2["a_R"])) + rng.normal(0.0, mutation_sigma),
            new_lambda,
        ).clip(cfg)
        children.append(child)
    return children[:cfg.pop_size]


def run_lineage(
    lineage_seed: int,
    cfg: EvoConfig,
    n_gens: int,
    m_L: float,
    m_R: float,
    env_mode: str,
    stage_name: str,
    initial_population: Optional[List[Agent]] = None,
    mirror_env: bool = False,
) -> Tuple[pd.DataFrame, Agent, pd.Series]:
    rng = np.random.default_rng(lineage_seed)
    population = initialize_population(cfg, rng) if initial_population is None else [Agent(a.a_L, a.a_R, a.lambd).clip(cfg) for a in initial_population]

    gen_rows = []
    best_final_agent = None

    for gen in range(1, n_gens + 1):
        eval_df = evaluate_population(population, rng, cfg, m_L, m_R, cfg.trials_per_agent, env_mode=env_mode, mirror_env=mirror_env)
        eval_df = eval_df.sort_values("fitness", ascending=False).reset_index(drop=True)
        best = eval_df.iloc[0]
        gen_rows.append(
            {
                "stage": stage_name,
                "generation": gen,
                "lineage_seed": lineage_seed,
                "best_a_L": float(best["a_L"]),
                "best_a_R": float(best["a_R"]),
                "best_lambda": float(best["lambda"]),
                "best_Delta_a": float(best["Delta_a"]),
                "best_abs_Delta_a": float(best["abs_Delta_a"]),
                "best_polarity": int(best["polarity"]),
                "best_mean_rt": float(best["mean_rt"]),
                "best_fitness": float(best["fitness"]),
                "pop_mean_abs_Delta_a": float(eval_df["abs_Delta_a"].mean()),
                "pop_sem_abs_Delta_a": float(eval_df["abs_Delta_a"].std(ddof=1) / math.sqrt(len(eval_df))) if len(eval_df) > 1 else 0.0,
                "pop_mean_fitness": float(eval_df["fitness"].mean()),
            }
        )
        population = select_and_reproduce(population, eval_df, cfg, rng, generation=gen)
        if gen == n_gens:
            best_final_agent = Agent(float(best["a_L"]), float(best["a_R"]), float(best["lambda"])).clip(cfg)

    # Final robust evaluation
    eval_rng = np.random.default_rng(lineage_seed + 10_000_000)
    final_fitness, final_rt, final_rts = evaluate_agent(
        best_final_agent, eval_rng, cfg, m_L, m_R, cfg.final_eval_trials, env_mode=env_mode, mirror_env=mirror_env
    )
    final_series = pd.Series(
        {
            "stage": stage_name,
            "lineage_seed": lineage_seed,
            "a_L": best_final_agent.a_L,
            "a_R": best_final_agent.a_R,
            "lambda": best_final_agent.lambd if cfg.allow_evolvable_lambda else cfg.fixed_lambda,
            "Delta_a": best_final_agent.a_R - best_final_agent.a_L,
            "abs_Delta_a": abs(best_final_agent.a_R - best_final_agent.a_L),
            "polarity": stable_sign(best_final_agent.a_R - best_final_agent.a_L),
            "fitness_final": final_fitness,
            "mean_rt_final": final_rt,
            "sd_rt_final": float(np.std(final_rts, ddof=1)) if len(final_rts) > 1 else 0.0,
        }
    )
    return pd.DataFrame(gen_rows), best_final_agent, final_series


def run_stage_block(
    stage_name: str,
    lineages: int,
    n_gens: int,
    cfg: EvoConfig,
    m_L: float,
    m_R: float,
    env_mode: str,
    seed_base: int,
    outdir: Path,
    initial_populations: Optional[Dict[int, List[Agent]]] = None,
    mirror_env: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, Agent]]:
    all_gen: List[pd.DataFrame] = []
    finals: List[pd.Series] = []
    best_agents: Dict[int, Agent] = {}
    for i in range(lineages):
        lineage_seed = seed_base + i
        init_pop = None if initial_populations is None else initial_populations[lineage_seed]
        gen_df, best_agent, final_series = run_lineage(
            lineage_seed=lineage_seed,
            cfg=cfg,
            n_gens=n_gens,
            m_L=m_L,
            m_R=m_R,
            env_mode=env_mode,
            stage_name=stage_name,
            initial_population=init_pop,
            mirror_env=mirror_env,
        )
        all_gen.append(gen_df)
        finals.append(final_series)
        best_agents[lineage_seed] = best_agent

    gen_df = pd.concat(all_gen, ignore_index=True)
    final_df = pd.DataFrame(finals)
    gen_df.to_csv(outdir / "generation_traces" / f"{stage_name.lower()}_generation_trace.csv", index=False)
    final_df.to_csv(outdir / "tables" / f"{stage_name.lower()}_final_lineages.csv", index=False)
    return gen_df, final_df, best_agents


# -----------------------------
# Analysis utilities
# -----------------------------
def summarize_stage(final_df: pd.DataFrame, stage_name: str) -> Dict[str, float]:
    arr = final_df["abs_Delta_a"].to_numpy()
    rt = final_df["mean_rt_final"].to_numpy()
    return {
        "stage": stage_name,
        "n": int(len(final_df)),
        "right_wider": int((final_df["polarity"] > 0).sum()),
        "left_wider": int((final_df["polarity"] < 0).sum()),
        "median_abs_Delta_a": float(np.median(arr)),
        "iqr_lo": float(np.quantile(arr, 0.25)),
        "iqr_hi": float(np.quantile(arr, 0.75)),
        "min_abs_Delta_a": float(arr.min()),
        "max_abs_Delta_a": float(arr.max()),
        "mean_rt": float(rt.mean()),
        "sd_rt": float(rt.std(ddof=1)) if len(rt) > 1 else 0.0,
    }


def summarize_generation(gen_df: pd.DataFrame, generation: int, label: str) -> Dict[str, float]:
    sub = gen_df.loc[gen_df["generation"] == generation].copy()
    arr = sub["best_abs_Delta_a"].to_numpy()
    rt = sub["best_mean_rt"].to_numpy()
    return {
        "stage": label,
        "n": int(len(sub)),
        "diverged_ge_threshold": int((sub["best_abs_Delta_a"] >= sub["best_abs_Delta_a"].iloc[0]*0 + 0.3).sum()),
        "right_wider": int((sub["best_polarity"] > 0).sum()),
        "left_wider": int((sub["best_polarity"] < 0).sum()),
        "median_abs_Delta_a": float(np.median(arr)),
        "iqr_lo": float(np.quantile(arr, 0.25)),
        "iqr_hi": float(np.quantile(arr, 0.75)),
        "min_abs_Delta_a": float(arr.min()),
        "max_abs_Delta_a": float(arr.max()),
        "mean_rt": float(rt.mean()),
        "sd_rt": float(rt.std(ddof=1)) if len(rt) > 1 else 0.0,
    }


def exact_binomial_summary(right_count: int, total: int, label: str) -> Dict[str, float]:
    res = binomtest(int(right_count), int(total), p=0.5, alternative="two-sided")
    return {
        "comparison": label,
        "right_count": int(right_count),
        "total": int(total),
        "p_value": float(res.pvalue),
    }


def wilcoxon_summary(x: np.ndarray, y: np.ndarray, label: str) -> Dict[str, float]:
    res = wilcoxon(x, y, alternative="two-sided", zero_method="wilcox")
    return {
        "comparison": label,
        "n": int(len(x)),
        "statistic": float(res.statistic),
        "p_value": float(res.pvalue),
        "median_x": float(np.median(x)),
        "median_y": float(np.median(y)),
    }


def paired_wilcoxon_by_seed(df_x: pd.DataFrame, df_y: pd.DataFrame, value_col: str, label: str) -> Dict[str, float]:
    merged = (
        df_x[["lineage_seed", value_col]]
        .rename(columns={value_col: f"{value_col}_x"})
        .merge(
            df_y[["lineage_seed", value_col]].rename(columns={value_col: f"{value_col}_y"}),
            on="lineage_seed",
            how="inner",
        )
        .sort_values("lineage_seed")
        .reset_index(drop=True)
    )
    if merged.empty:
        raise ValueError(f"No matched lineage_seed values for paired comparison: {label}")
    x = merged[f"{value_col}_x"].to_numpy()
    y = merged[f"{value_col}_y"].to_numpy()
    res = wilcoxon(y, x, alternative="two-sided", zero_method="wilcox")
    return {
        "comparison": label,
        "n": int(len(merged)),
        "statistic": float(res.statistic),
        "p_value": float(res.pvalue),
        "median_x": float(np.median(x)),
        "median_y": float(np.median(y)),
    }


def compute_reversal_summary(gen_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    for seed, sub in gen_df.groupby("lineage_seed"):
        sub = sub.sort_values("generation")
        locked_sign = None
        reversal = False
        for _, row in sub.iterrows():
            s = stable_sign(float(row["best_Delta_a"]))
            mag = float(row["best_abs_Delta_a"])
            if locked_sign is None and mag >= threshold and s != 0:
                locked_sign = s
            elif locked_sign is not None and s != 0 and s != locked_sign:
                reversal = True
                break
        rows.append(
            {
                "lineage_seed": seed,
                "locked_sign": locked_sign,
                "reversal_observed": reversal,
                "final_sign": stable_sign(float(sub.iloc[-1]["best_Delta_a"])),
                "final_abs_Delta_a": float(sub.iloc[-1]["best_abs_Delta_a"]),
            }
        )
    return pd.DataFrame(rows)


# -----------------------------
# Supplementary analyses
# -----------------------------
def run_symmetry_controls(stage1_final: pd.DataFrame, cfg: EvoConfig, outdir: Path) -> pd.DataFrame:
    rows = []
    for _, row in stage1_final.iterrows():
        seed = int(row["lineage_seed"])
        agent = Agent(float(row["a_L"]), float(row["a_R"]), float(row["lambda"]))
        swapped = agent.swapped()
        rng1 = np.random.default_rng(seed + 200_000)
        rng2 = np.random.default_rng(seed + 200_000)
        rng3 = np.random.default_rng(seed + 300_000)
        fit1, rt1, _ = evaluate_agent(agent, rng1, cfg, 1.0, 1.0, cfg.symmetry_control_eval_trials, env_mode="structured", mirror_env=False)
        fit2, rt2, _ = evaluate_agent(swapped, rng2, cfg, 1.0, 1.0, cfg.symmetry_control_eval_trials, env_mode="structured", mirror_env=False)
        fit3, rt3, _ = evaluate_agent(agent, rng3, cfg, 1.0, 1.0, cfg.symmetry_control_eval_trials, env_mode="structured", mirror_env=True)
        rows.append(
            {
                "lineage_seed": seed,
                "orig_a_L": agent.a_L,
                "orig_a_R": agent.a_R,
                "orig_mean_rt": rt1,
                "swap_mean_rt": rt2,
                "mirror_mean_rt": rt3,
                "abs_rt_diff_swap": abs(rt1 - rt2),
                "abs_rt_diff_mirror": abs(rt1 - rt3),
                "orig_Delta_a": agent.a_R - agent.a_L,
                "swap_Delta_a": swapped.a_R - swapped.a_L,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "tables" / "symmetry_controls.csv", index=False)
    return df


def run_bias_sweep(cfg: EvoConfig, outdir: Path) -> pd.DataFrame:
    rows = []
    seed_base = cfg.master_seed + 40_000
    for idx, delta in enumerate(cfg.bias_sweep_deltas):
        m_L = 1.0 - delta / 2.0
        m_R = 1.0 + delta / 2.0
        _, final_df, _ = run_stage_block(
            stage_name=f"BiasSweep_{delta:+.3f}",
            lineages=cfg.bias_sweep_lineages,
            n_gens=cfg.stage_gens,
            cfg=cfg,
            m_L=m_L,
            m_R=m_R,
            env_mode="structured",
            seed_base=seed_base + idx * 1000,
            outdir=outdir,
        )
        rows.append(
            {
                "delta": float(delta),
                "m_L": float(m_L),
                "m_R": float(m_R),
                "n": int(len(final_df)),
                "right_wider": int((final_df["polarity"] > 0).sum()),
                "left_wider": int((final_df["polarity"] < 0).sum()),
                "p_right": float((final_df["polarity"] > 0).mean()),
                "median_abs_Delta_a": float(final_df["abs_Delta_a"].median()),
                "iqr_lo": float(final_df["abs_Delta_a"].quantile(0.25)),
                "iqr_hi": float(final_df["abs_Delta_a"].quantile(0.75)),
                "mean_rt": float(final_df["mean_rt_final"].mean()),
                "sd_rt": float(final_df["mean_rt_final"].std(ddof=1)) if len(final_df) > 1 else 0.0,
                "binom_p": float(binomtest(int((final_df["polarity"] > 0).sum()), int(len(final_df)), p=0.5).pvalue),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "tables" / "bias_sweep_summary.csv", index=False)
    return df


def run_environmental_comparison(cfg: EvoConfig, outdir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed_base_sp = cfg.master_seed + 60_000
    seed_base_pl = cfg.master_seed + 60_000   # paired seeds
    _, spatial_final, _ = run_stage_block(
        stage_name="EnvSpatial",
        lineages=cfg.env_compare_lineages,
        n_gens=cfg.stage_gens,
        cfg=cfg,
        m_L=1.0,
        m_R=1.0,
        env_mode="structured",
        seed_base=seed_base_sp,
        outdir=outdir,
    )
    _, planar_final, _ = run_stage_block(
        stage_name="EnvPlanar",
        lineages=cfg.env_compare_lineages,
        n_gens=cfg.stage_gens,
        cfg=cfg,
        m_L=1.0,
        m_R=1.0,
        env_mode="planar",
        seed_base=seed_base_pl,
        outdir=outdir,
    )
    paired = spatial_final[["lineage_seed", "abs_Delta_a", "mean_rt_final"]].merge(
        planar_final[["lineage_seed", "abs_Delta_a", "mean_rt_final"]],
        on="lineage_seed",
        suffixes=("_spatial", "_planar"),
    )
    paired["delta_abs_Delta_a"] = paired["abs_Delta_a_spatial"] - paired["abs_Delta_a_planar"]
    paired["delta_mean_rt"] = paired["mean_rt_final_spatial"] - paired["mean_rt_final_planar"]
    paired.to_csv(outdir / "tables" / "environmental_comparison_paired.csv", index=False)
    return spatial_final, planar_final, paired


def run_fitness_valley(cfg: EvoConfig, stage2_final: pd.DataFrame, stage2r_final: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    # use strongest polarized examples from positive and negative seeds
    reps = []
    if len(stage2_final) > 0:
        reps.append(("Stage2_rep", stage2_final.sort_values("abs_Delta_a", ascending=False).iloc[0]))
    if len(stage2r_final) > 0:
        reps.append(("Stage2R_rep", stage2r_final.sort_values("abs_Delta_a", ascending=False).iloc[0]))
    rows = []
    t_grid = np.linspace(0.0, 1.0, cfg.fitness_valley_points)
    for label, row in reps:
        agent = Agent(float(row["a_L"]), float(row["a_R"]), float(row["lambda"]))
        swap = agent.swapped()
        for t in t_grid:
            interp = Agent(
                (1 - t) * agent.a_L + t * swap.a_L,
                (1 - t) * agent.a_R + t * swap.a_R,
                (1 - t) * agent.lambd + t * swap.lambd if cfg.allow_evolvable_lambda else cfg.fixed_lambda,
            ).clip(cfg)
            rng = np.random.default_rng(cfg.master_seed + 80_000 + int(round(t * 1000)))
            fit, rt, _ = evaluate_agent(interp, rng, cfg, 1.0, 1.0, cfg.fitness_valley_trials, env_mode="structured", mirror_env=False)
            rows.append(
                {
                    "path_label": label,
                    "t": float(t),
                    "a_L": interp.a_L,
                    "a_R": interp.a_R,
                    "abs_Delta_a": abs(interp.a_R - interp.a_L),
                    "fitness": fit,
                    "mean_rt": rt,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "tables" / "fitness_valley.csv", index=False)
    return df


def run_robustness_analyses(cfg: EvoConfig, outdir: Path) -> pd.DataFrame:
    rows = []
    variants = [
        ("baseline", {"pop_size": cfg.pop_size, "trials_per_agent": cfg.trials_per_agent, "mutation_sigma": cfg.mutation_sigma}),
        ("pop40", {"pop_size": 40, "trials_per_agent": cfg.trials_per_agent, "mutation_sigma": cfg.mutation_sigma}),
        ("trials16", {"pop_size": cfg.pop_size, "trials_per_agent": 16, "mutation_sigma": cfg.mutation_sigma}),
        ("mut03", {"pop_size": cfg.pop_size, "trials_per_agent": cfg.trials_per_agent, "mutation_sigma": 0.03}),
        ("mut07", {"pop_size": cfg.pop_size, "trials_per_agent": cfg.trials_per_agent, "mutation_sigma": 0.07}),
    ]
    seed_base = cfg.master_seed + 90_000
    for idx, (label, params) in enumerate(variants):
        cfg_v = EvoConfig(**asdict(cfg))
        for k, v in params.items():
            setattr(cfg_v, k, v)
        _, final_df, _ = run_stage_block(
            stage_name=f"Robust_{label}",
            lineages=cfg.robustness_lineages,
            n_gens=cfg.stage_gens,
            cfg=cfg_v,
            m_L=1.0,
            m_R=1.0,
            env_mode="structured",
            seed_base=seed_base + idx * 1000,
            outdir=outdir,
        )
        rows.append(
            {
                "label": label,
                "pop_size": cfg_v.pop_size,
                "trials_per_agent": cfg_v.trials_per_agent,
                "mutation_sigma": cfg_v.mutation_sigma,
                "n": int(len(final_df)),
                "right_wider": int((final_df["polarity"] > 0).sum()),
                "left_wider": int((final_df["polarity"] < 0).sum()),
                "median_abs_Delta_a": float(final_df["abs_Delta_a"].median()),
                "iqr_lo": float(final_df["abs_Delta_a"].quantile(0.25)),
                "iqr_hi": float(final_df["abs_Delta_a"].quantile(0.75)),
                "mean_rt": float(final_df["mean_rt_final"].mean()),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "tables" / "robustness_summary.csv", index=False)
    return df


# -----------------------------
# Plotting
# -----------------------------
def save_scatter(df: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.scatter(df["a_L"], df["a_R"], alpha=0.85)
    lo = min(df["a_L"].min(), df["a_R"].min()) - 0.1
    hi = max(df["a_L"].max(), df["a_R"].max()) + 0.1
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("a_L")
    ax.set_ylabel("a_R")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_bias_sweep_plot(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    ax.plot(df["delta"], df["p_right"], marker="o")
    ax.axhline(0.5, linestyle="--")
    ax.axvline(0.0, linestyle="--")
    ax.set_xlabel("Developmental gain asymmetry (delta)")
    ax.set_ylabel("Proportion right-wider")
    ax.set_title("Bias sweep: polarity selection curve")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_fitness_valley_plot(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    for label, sub in df.groupby("path_label"):
        ax.plot(sub["t"], sub["fitness"], marker="o", label=label)
    ax.axvline(0.5, linestyle="--")
    ax.set_xlabel("Interpolation position (t)")
    ax.set_ylabel("Fitness")
    ax.set_title("Fitness valley between polarized states")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_environment_boxplot(spatial_final: pd.DataFrame, planar_final: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    ax.boxplot([spatial_final["abs_Delta_a"].to_numpy(), planar_final["abs_Delta_a"].to_numpy()], tick_labels=["Spatial", "Planar"])
    ax.set_ylabel("|Delta_a|")
    ax.set_title("Environmental necessity analysis")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_robustness_plot(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    ax.plot(range(len(df)), df["median_abs_Delta_a"], marker="o")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["label"], rotation=30, ha="right")
    ax.set_ylabel("Median |Delta_a|")
    ax.set_title("Robustness across evolutionary settings")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_symmetry_control_plot(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.scatter(df["orig_mean_rt"], df["swap_mean_rt"], alpha=0.8, label="swap")
    lo = min(df["orig_mean_rt"].min(), df["swap_mean_rt"].min()) - 1e-4
    hi = max(df["orig_mean_rt"].max(), df["swap_mean_rt"].max()) + 1e-4
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_xlabel("Original mean RT")
    ax.set_ylabel("Swapped-agent mean RT")
    ax.set_title("Symmetry control: original vs swapped evaluations")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


# -----------------------------
# Main orchestration
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="Run a reduced-size smoke test.")
    parser.add_argument("--outdir", type=str, default=None, help="Optional output directory.")
    args = parser.parse_args()

    cfg = EvoConfig()
    if args.smoke_test:
        cfg.apply_smoke_test()

    outdir = make_outdir(args.outdir, smoke=cfg.smoke_test)

    # Save configuration
    write_text(outdir / "logs" / "run_config.json", json.dumps(asdict(cfg), indent=2))

    # Main manuscript analyses, all under lambda fixed = 0 for clean symmetry
    stage1_gen, stage1_final, stage1_best = run_stage_block(
        stage_name="Stage1",
        lineages=cfg.stage1_lineages,
        n_gens=cfg.stage_gens,
        cfg=cfg,
        m_L=1.0,
        m_R=1.0,
        env_mode="structured",
        seed_base=cfg.master_seed + 1_000,
        outdir=outdir,
    )
    stage2_gen, stage2_final, stage2_best = run_stage_block(
        stage_name="Stage2",
        lineages=cfg.stage2_lineages,
        n_gens=cfg.stage_gens,
        cfg=cfg,
        m_L=0.99,
        m_R=1.01,
        env_mode="structured",
        seed_base=cfg.master_seed + 2_000,
        outdir=outdir,
    )
    stage2r_gen, stage2r_final, stage2r_best = run_stage_block(
        stage_name="Stage2R",
        lineages=cfg.stage2r_lineages,
        n_gens=cfg.stage_gens,
        cfg=cfg,
        m_L=1.01,
        m_R=0.99,
        env_mode="structured",
        seed_base=cfg.master_seed + 3_000,
        outdir=outdir,
    )

    # Stage 3 from both positive- and negative-bias seeds
    stage3_pos_init = {
        seed: reseed_population_from_template(agent, cfg, np.random.default_rng(seed + 55_555))
        for seed, agent in stage2_best.items()
    }
    stage3_pos_gen, stage3_pos_final, _ = run_stage_block(
        stage_name="Stage3_PosSeed",
        lineages=cfg.stage2_lineages,
        n_gens=cfg.stage3_gens,
        cfg=cfg,
        m_L=1.0,
        m_R=1.0,
        env_mode="structured",
        seed_base=cfg.master_seed + 2_000,
        outdir=outdir,
        initial_populations=stage3_pos_init,
    )

    stage3_neg_init = {
        seed: reseed_population_from_template(agent, cfg, np.random.default_rng(seed + 66_666))
        for seed, agent in stage2r_best.items()
    }
    stage3_neg_gen, stage3_neg_final, _ = run_stage_block(
        stage_name="Stage3_NegSeed",
        lineages=cfg.stage2r_lineages,
        n_gens=cfg.stage3_gens,
        cfg=cfg,
        m_L=1.0,
        m_R=1.0,
        env_mode="structured",
        seed_base=cfg.master_seed + 3_000,
        outdir=outdir,
        initial_populations=stage3_neg_init,
    )

    # Summaries
    summaries = [
        summarize_stage(stage1_final, "Stage1_final"),
        summarize_generation(stage1_gen, min(20, cfg.stage_gens), f"Stage1_generation{min(20, cfg.stage_gens)}"),
        summarize_stage(stage2_final, "Stage2_final"),
        summarize_stage(stage2r_final, "Stage2R_final"),
        summarize_stage(stage3_pos_final, "Stage3_PosSeed_final"),
        summarize_stage(stage3_neg_final, "Stage3_NegSeed_final"),
    ]
    pd.DataFrame(summaries).to_csv(outdir / "tables" / "results_summary.csv", index=False)

    # Main table
    table1 = pd.DataFrame([
        {
            "Stage": "Stage 1",
            "delta": 0.00,
            "n": len(stage1_final),
            "Right-wider": int((stage1_final["polarity"] > 0).sum()),
            "Left-wider": int((stage1_final["polarity"] < 0).sum()),
            "Median |Delta_a|": float(stage1_final["abs_Delta_a"].median()),
            "IQR low": float(stage1_final["abs_Delta_a"].quantile(0.25)),
            "IQR high": float(stage1_final["abs_Delta_a"].quantile(0.75)),
            "Mean RT": float(stage1_final["mean_rt_final"].mean()),
            "SD RT": float(stage1_final["mean_rt_final"].std(ddof=1)) if len(stage1_final) > 1 else 0.0,
        },
        {
            "Stage": "Stage 2",
            "delta": 0.02,
            "n": len(stage2_final),
            "Right-wider": int((stage2_final["polarity"] > 0).sum()),
            "Left-wider": int((stage2_final["polarity"] < 0).sum()),
            "Median |Delta_a|": float(stage2_final["abs_Delta_a"].median()),
            "IQR low": float(stage2_final["abs_Delta_a"].quantile(0.25)),
            "IQR high": float(stage2_final["abs_Delta_a"].quantile(0.75)),
            "Mean RT": float(stage2_final["mean_rt_final"].mean()),
            "SD RT": float(stage2_final["mean_rt_final"].std(ddof=1)) if len(stage2_final) > 1 else 0.0,
        },
        {
            "Stage": "Stage 2R",
            "delta": -0.02,
            "n": len(stage2r_final),
            "Right-wider": int((stage2r_final["polarity"] > 0).sum()),
            "Left-wider": int((stage2r_final["polarity"] < 0).sum()),
            "Median |Delta_a|": float(stage2r_final["abs_Delta_a"].median()),
            "IQR low": float(stage2r_final["abs_Delta_a"].quantile(0.25)),
            "IQR high": float(stage2r_final["abs_Delta_a"].quantile(0.75)),
            "Mean RT": float(stage2r_final["mean_rt_final"].mean()),
            "SD RT": float(stage2r_final["mean_rt_final"].std(ddof=1)) if len(stage2r_final) > 1 else 0.0,
        },
        {
            "Stage": "Stage 3 (from +bias seeds)",
            "delta": 0.00,
            "n": len(stage3_pos_final),
            "Right-wider": int((stage3_pos_final["polarity"] > 0).sum()),
            "Left-wider": int((stage3_pos_final["polarity"] < 0).sum()),
            "Median |Delta_a|": float(stage3_pos_final["abs_Delta_a"].median()),
            "IQR low": float(stage3_pos_final["abs_Delta_a"].quantile(0.25)),
            "IQR high": float(stage3_pos_final["abs_Delta_a"].quantile(0.75)),
            "Mean RT": float(stage3_pos_final["mean_rt_final"].mean()),
            "SD RT": float(stage3_pos_final["mean_rt_final"].std(ddof=1)) if len(stage3_pos_final) > 1 else 0.0,
        },
        {
            "Stage": "Stage 3 (from -bias seeds)",
            "delta": 0.00,
            "n": len(stage3_neg_final),
            "Right-wider": int((stage3_neg_final["polarity"] > 0).sum()),
            "Left-wider": int((stage3_neg_final["polarity"] < 0).sum()),
            "Median |Delta_a|": float(stage3_neg_final["abs_Delta_a"].median()),
            "IQR low": float(stage3_neg_final["abs_Delta_a"].quantile(0.25)),
            "IQR high": float(stage3_neg_final["abs_Delta_a"].quantile(0.75)),
            "Mean RT": float(stage3_neg_final["mean_rt_final"].mean()),
            "SD RT": float(stage3_neg_final["mean_rt_final"].std(ddof=1)) if len(stage3_neg_final) > 1 else 0.0,
        },
    ])
    table1.to_csv(outdir / "tables" / "table1_stage_outcomes.csv", index=False)

    # Statistics
    stats_rows = [
        exact_binomial_summary(int((stage1_final["polarity"] > 0).sum()), len(stage1_final), "Stage 1 polarity vs chance"),
        exact_binomial_summary(int((stage2_final["polarity"] > 0).sum()), len(stage2_final), "Stage 2 polarity vs chance"),
        exact_binomial_summary(int((stage2r_final["polarity"] > 0).sum()), len(stage2r_final), "Stage 2R polarity vs chance"),
        paired_wilcoxon_by_seed(stage2_final, stage3_pos_final, "abs_Delta_a", "Stage3_PosSeed vs Stage2: |Delta_a|"),
        paired_wilcoxon_by_seed(stage2_final, stage3_pos_final, "mean_rt_final", "Stage3_PosSeed vs Stage2: mean RT"),
        paired_wilcoxon_by_seed(stage2r_final, stage3_neg_final, "abs_Delta_a", "Stage3_NegSeed vs Stage2R: |Delta_a|"),
        paired_wilcoxon_by_seed(stage2r_final, stage3_neg_final, "mean_rt_final", "Stage3_NegSeed vs Stage2R: mean RT"),
    ]
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(outdir / "tables" / "statistical_tests.csv", index=False)

    paired_stage3_vs_parent = pd.DataFrame([
        paired_wilcoxon_by_seed(stage2_final, stage3_pos_final, "abs_Delta_a", "Stage3_PosSeed vs Stage2: |Delta_a|"),
        paired_wilcoxon_by_seed(stage2_final, stage3_pos_final, "mean_rt_final", "Stage3_PosSeed vs Stage2: mean RT"),
        paired_wilcoxon_by_seed(stage2r_final, stage3_neg_final, "abs_Delta_a", "Stage3_NegSeed vs Stage2R: |Delta_a|"),
        paired_wilcoxon_by_seed(stage2r_final, stage3_neg_final, "mean_rt_final", "Stage3_NegSeed vs Stage2R: mean RT"),
    ])
    paired_stage3_vs_parent.to_csv(outdir / "tables" / "paired_stage3_vs_parent_tests.csv", index=False)

    # Reversal summaries
    stage3_pos_rev = compute_reversal_summary(stage3_pos_gen, cfg.divergence_threshold)
    stage3_pos_rev.to_csv(outdir / "tables" / "stage3_pos_reversal_summary.csv", index=False)
    stage3_neg_rev = compute_reversal_summary(stage3_neg_gen, cfg.divergence_threshold)
    stage3_neg_rev.to_csv(outdir / "tables" / "stage3_neg_reversal_summary.csv", index=False)

    # Supplementary analyses
    symmetry_df = run_symmetry_controls(stage1_final, cfg, outdir)
    bias_sweep_df = run_bias_sweep(cfg, outdir)
    spatial_final, planar_final, env_paired = run_environmental_comparison(cfg, outdir)
    fitness_valley_df = run_fitness_valley(cfg, stage2_final, stage2r_final, outdir)
    robustness_df = run_robustness_analyses(cfg, outdir)

    # Figures
    save_scatter(stage1_final, outdir / "figures" / "figure1_stage1_scatter.png", "Figure 1. Spontaneous symmetry breaking under symmetric development")
    save_scatter(stage2_final, outdir / "figures" / "figure2_stage2_scatter.png", "Figure 2. Directional symmetry breaking under positive developmental bias")
    save_scatter(stage3_pos_final, outdir / "figures" / "figure3_stage3_pos_scatter.png", "Figure 3. Stability of specialization after restoring symmetry (+bias seeds)")
    save_scatter(stage2r_final, outdir / "figures" / "supplement_stage2r_scatter.png", "Supplementary. Directional symmetry breaking under reversed bias")
    save_scatter(stage3_neg_final, outdir / "figures" / "supplement_stage3_neg_scatter.png", "Supplementary. Stability after restoring symmetry (-bias seeds)")
    save_bias_sweep_plot(bias_sweep_df, outdir / "figures" / "figure4_bias_sweep.png")
    save_fitness_valley_plot(fitness_valley_df, outdir / "figures" / "figure5_fitness_valley.png")
    save_environment_boxplot(spatial_final, planar_final, outdir / "figures" / "figure6_environmental_necessity.png")
    save_robustness_plot(robustness_df, outdir / "figures" / "supplement_robustness.png")
    save_symmetry_control_plot(symmetry_df, outdir / "figures" / "supplement_symmetry_controls.png")

    # Simple run log
    log_lines = []
    log_lines.append(f"Smoke test: {cfg.smoke_test}")
    log_lines.append(f"Output directory: {outdir}")
    log_lines.append("")
    log_lines.append("Main stage summaries")
    for row in summaries:
        log_lines.append(json.dumps(row, ensure_ascii=False))
    log_lines.append("")
    log_lines.append("Statistical tests")
    for _, row in stats_df.iterrows():
        log_lines.append(json.dumps({k: row[k] for k in stats_df.columns}, ensure_ascii=False, default=float))
    log_lines.append("")
    log_lines.append("Bias sweep summary")
    log_lines.append(bias_sweep_df.to_string(index=False))
    log_lines.append("")
    log_lines.append("Environmental paired comparison summary")
    log_lines.append(env_paired[["delta_abs_Delta_a", "delta_mean_rt"]].describe().to_string())
    log_lines.append("")
    log_lines.append(f"Stage3 (+seed) reversals: {int(stage3_pos_rev['reversal_observed'].sum())}/{len(stage3_pos_rev)}")
    log_lines.append(f"Stage3 (-seed) reversals: {int(stage3_neg_rev['reversal_observed'].sum())}/{len(stage3_neg_rev)}")
    write_text(outdir / "logs" / "run_summary.txt", "\n".join(log_lines))

    print("Done.")
    print(f"Outputs written to: {outdir}")


if __name__ == "__main__":
    main()
