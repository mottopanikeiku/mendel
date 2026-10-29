"""Diagnose the neural PPO same-variant null-control rejections.

Observed null-control outcomes and raw root scores are MEAS. Test sizes at a
planned seed count resample committed base roots that the null controls did not
use; that is not new training. The block-effect detection probability is SIM.
``--check`` regenerates the JSON, report and figure in memory, compares them
with the committed files and checks the claims the report makes.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
import warnings

import numpy as np
from scipy import stats

from seed_power.experiment import make_jobs
from seed_power.stats import welch_test

if __package__:
    from .calibrate import compare_json, svg_difference
    from .neural_analyze import load_records, raw_path, validate_records
else:
    from calibrate import compare_json, svg_difference
    from neural_analyze import load_records, raw_path, validate_records

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261030
ALPHA = 0.05
TESTS = ("welch", "student", "permutation", "mann_whitney", "yuen")
SIZE_TESTS = ("welch", "student", "mann_whitney", "yuen")
TEST_LABELS = {
    "welch": "Welch t (the committed test)",
    "student": "Student pooled t",
    "permutation": "Permutation of the mean gap",
    "mann_whitney": "Mann-Whitney U",
    "yuen": "Yuen 20% trimmed means",
}
PERMUTATIONS = 4000
REPLICATES = 4000
FAMILIES = 10000
SIMULATIONS = 1000
SIZE_GRID = (2, 3, 5, 8, 16, 32, 64, 128, 256, 512)
BOUNDARY = {"CartPole-v1": 500.0, "Acrobot-v1": -500.0}
DESIGN = "configs/neural_design.json"
PLAN = "results/neural/pilot_plan.json"
PILOT_RAW = "results/neural/pilot/runs.jsonl.gz"
CONFIRMATION_RAW = "results/neural/confirmation/runs.jsonl.gz"
ENVIRONMENT = "results/neural/confirmation/environment.json"
OUTPUT = "results/neural/null_diagnostics.json"
REPORT = "results/neural/null_diagnostics.md"
FIGURE = "figures/null_controls.svg"
QUOTING = ("README.md", "docs/NEURAL_PROTOCOL.md")
QUOTED = re.compile(r"\d+/\d+|\d+\.\d+")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stream(*key):
    return np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=key))


def tail_pvalue(null, observed):
    """(1 + #{null >= observed}) / (1 + B); near-equal values count, allowing for summation order."""
    hits = int((np.asarray(null) >= observed - 1e-9 * max(1.0, abs(observed))).sum())
    return (1 + hits) / (1 + len(null))


# Tests on one pair of samples.

def permutation_pvalue(a, b, rng, permutations=PERMUTATIONS):
    """Two-sided Monte Carlo permutation p-value for the mean gap."""
    pooled = np.concatenate([a, b])
    order = rng.permuted(np.tile(np.arange(pooled.size), (permutations, 1)), axis=1)
    shuffled = pooled[order]
    gaps = np.abs(shuffled[:, :a.size].mean(axis=1) - shuffled[:, a.size:].mean(axis=1))
    return tail_pvalue(gaps, abs(a.mean() - b.mean()))


def pair_pvalues(a, b, rng):
    """p-values of every test; an undefined p-value (SciPy warns and returns nan) is None."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        values = {
            "welch": welch_test(a, b)["pvalue"],
            "student": float(stats.ttest_ind(a, b, equal_var=True).pvalue),
            "permutation": permutation_pvalue(a, b, rng),
            "mann_whitney": float(stats.mannwhitneyu(a, b, alternative="two-sided",
                                                     method="asymptotic").pvalue),
            "yuen": float(stats.ttest_ind(a, b, equal_var=False, trim=0.2).pvalue),
        }
    return {key: (value if math.isfinite(value) else None) for key, value in values.items()}


# Vectorized sizes on resampled roots.

def welch_pvalues(a, b):
    """Row-wise stats.welch_test p-values for equal arm sizes."""
    n = a.shape[1]
    va, vb = a.var(axis=1, ddof=1), b.var(axis=1, ddof=1)
    va[np.all(a == a[:, :1], axis=1)] = 0.0
    vb[np.all(b == b[:, :1], axis=1)] = 0.0
    gap = a.mean(axis=1) - b.mean(axis=1)
    both = (va == 0) & (vb == 0)
    component = np.where(both, 1.0, (va + vb) / n)
    df = component**2 * (n - 1) / ((va / n) ** 2 + (vb / n) ** 2 + both)
    pvalue = 2.0 * stats.t.sf(np.abs(gap) / np.sqrt(component), df)
    return np.where(both, np.where(gap == 0, 1.0, 0.0), pvalue)


def size_rejections(pool, n, replicates, rng, chunk=1000):
    """Rejections per test when both arms draw n roots with replacement from one pool."""
    counts = dict.fromkeys(SIZE_TESTS, 0)
    for start in range(0, replicates, chunk):
        rows = min(chunk, replicates - start)
        draws = pool[rng.integers(0, pool.size, size=(rows, 2 * n))]
        a, b = draws[:, :n], draws[:, n:]
        # Resampled arms can be constant; SciPy then warns and may return nan, counted as p=1.
        with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pvalues = {
                "welch": welch_pvalues(a, b),
                "student": stats.ttest_ind(a, b, axis=1, equal_var=True).pvalue,
                "mann_whitney": stats.mannwhitneyu(a, b, axis=1, alternative="two-sided",
                                                   method="asymptotic").pvalue,
                "yuen": stats.ttest_ind(a, b, axis=1, equal_var=False, trim=0.2).pvalue,
            }
        for key, values in pvalues.items():
            counts[key] += int((np.nan_to_num(values, nan=1.0) < ALPHA).sum())
    return counts


def resampled_controls(pool, seeds_per_arm, replicates, rng, chunk=1000):
    """Welch rejection counts and sums of z^2 when every control is redrawn from one pool.

    Each replicate redraws all controls independently, both arms with
    replacement at the control's planned count. This calibrates the observed
    count and dispersion against the empirical score distribution instead of
    exact 5% size and uniform p-values.
    """
    counts, squares = np.zeros(replicates, int), np.zeros(replicates)
    for n in seeds_per_arm:
        for start in range(0, replicates, chunk):
            rows = min(chunk, replicates - start)
            draws = pool[rng.integers(0, pool.size, size=(rows, 2 * n))]
            with np.errstate(invalid="ignore", divide="ignore"):
                pvalues = welch_pvalues(draws[:, :n], draws[:, n:])
            counts[start:start + rows] += pvalues < ALPHA
            squares[start:start + rows] += stats.norm.isf(pvalues / 2) ** 2
    return counts, squares



def poisson_binomial_tail(probabilities, observed):
    """P(at least ``observed`` successes) for independent Bernoulli probabilities."""
    distribution = np.zeros(len(probabilities) + 1)
    distribution[0] = 1.0
    for p in probabilities:
        distribution[1:] = distribution[1:] * (1 - p) + distribution[:-1] * p
        distribution[0] *= 1 - p
    return float(distribution[observed:].sum())


# Groups of roots: batches, production commits and seed blocks.

def anova_parts(values, labels):
    """One-way ANOVA mean squares and the effective group size for labels 0..k-1."""
    counts = np.bincount(labels).astype(float)
    means = np.bincount(labels, weights=values) / counts
    groups, roots = counts.size, values.size
    between = float((counts * (means - values.mean()) ** 2).sum()) / (groups - 1)
    within = float(((values - means[labels]) ** 2).sum()) / (roots - groups)
    size = (roots - float((counts**2).sum()) / roots) / (groups - 1)
    return between, within, size


def anova_f(values, labels):
    between, within, _ = anova_parts(values, labels)
    return between / within


def group_test(groups, rng, permutations=PERMUTATIONS):
    """ANOVA F with a permutation p-value over random regroupings, and Kruskal-Wallis."""
    values = np.concatenate([np.asarray(group, float) for group in groups])
    labels = np.repeat(np.arange(len(groups)), [len(group) for group in groups])
    observed = anova_f(values, labels)
    permuted = np.array([anova_f(rng.permutation(values), labels)
                         for _ in range(permutations)])
    summary = {"groups": len(groups), "roots": int(values.size), "anova_f": observed,
               "permutation_pvalue": tail_pvalue(permuted, observed),
               "kruskal_pvalue": float(stats.kruskal(*groups).pvalue)}
    return summary, values, labels, permuted


def normal_rejections(seeds_per_arm, share):
    """Expected Welch rejections when each arm's roots share a N(0, share*sigma^2) shift.

    The mean gap then has variance 2 sigma^2 (1/n + share), so z has variance
    1 + n*share. This is a normal approximation that ignores t degrees of freedom.
    """
    critical = stats.norm.isf(ALPHA / 2)
    return float(sum(2 * stats.norm.sf(critical / math.sqrt(1 + n * share))
                     for n in seeds_per_arm))


def detection_probability(values, labels, permuted, share, within, rng,
                          simulations=SIMULATIONS):
    """SIM: how often the block test gives p < ALPHA when blocks share Gaussian shifts.

    Roots are drawn with replacement from the held-out block roots; every block
    adds one N(0, share * within) shift. The p-value uses the observed
    permutation distribution of F, as for the observed test.
    """
    counts = np.bincount(labels)
    scale = math.sqrt(share * within)
    hits = 0
    for _ in range(simulations):
        shifts = np.repeat(rng.normal(0.0, scale, counts.size), counts)
        simulated = values[rng.integers(0, values.size, values.size)] + shifts
        hits += tail_pvalue(permuted, anova_f(simulated, labels)) < ALPHA
    return hits / simulations


# Assembly.

def load(root):
    design = json.loads((root / DESIGN).read_text())
    plans = json.loads((root / PLAN).read_text())
    pilot = load_records(root / PILOT_RAW)
    confirmation = load_records(root / CONFIRMATION_RAW)
    validate_records(design, "pilot", pilot)
    validate_records(design, "confirmation", confirmation, plans)
    batches = json.loads((root / ENVIRONMENT).read_text())["batches"]
    jobs = list(make_jobs(design, "confirmation", plans))
    if len(jobs) != len(batches) or any(
            batch["index"] != index or batch["task"] != job["task"]
            or batch["variant"] != job["variant"] or batch["useful_seeds"] != len(job["specs"])
            for index, (job, batch) in enumerate(zip(jobs, batches))):
        raise ValueError("recorded confirmation batches differ from the committed job order")
    batch_of = {spec["seed"]: index for index, job in enumerate(jobs) for spec in job["specs"]}
    producers = [batch["producer_code_commit"] for batch in batches]
    return design, plans, pilot, confirmation, batch_of, producers


def describe(values, boundary):
    values = np.asarray(values, float)
    return {
        "roots": int(values.size), "mean": float(values.mean()), "sd": float(values.std(ddof=1)),
        "skewness": float(stats.skew(values)), "excess_kurtosis": float(stats.kurtosis(values)),
        "boundary_fraction": float(np.mean(values == boundary)),
        "quantiles": dict(zip(("5", "25", "50", "75", "95"),
                              np.quantile(values, [.05, .25, .5, .75, .95]).tolist())),
    }


def diagnose(task_index, task, design, plans, pilot, confirmation, batch_of, producers):
    records = [record for record in confirmation if record["task"] == task]
    by_arm = defaultdict(list)
    for record in records:
        by_arm[record["comparison_index"], record["repetition"], record["arm"]].append(
            record["score"])
    null_indices = {index for index, pair in enumerate(design["comparisons"] * len(design["tasks"]))
                    if pair["null_control"]}

    # Every test on every executed confirmation cell.
    test_rng = stream(0, task_index)
    rows = []
    for cell in plans["cells"]:
        if cell["task"] != task or cell["status"] != "planned":
            continue
        a = by_arm[cell["comparison_index"], cell["repetition"], 0]
        b = by_arm[cell["comparison_index"], cell["repetition"], 1]
        rows.append({"null_control": cell["null_control"],
                     "comparison_index": cell["comparison_index"],
                     "repetition": cell["repetition"],
                     "seeds_per_arm": cell["required_seeds_per_arm"],
                     "pvalues": pair_pvalues(a, b, test_rng)})
    nulls = [row for row in rows if row["null_control"]]
    seeds_per_arm = [row["seeds_per_arm"] for row in nulls]
    welch = np.array([row["pvalues"]["welch"] for row in nulls])
    rejected = int((welch < ALPHA).sum())
    squares = float((stats.norm.isf(welch / 2) ** 2).sum())
    by_test = {}
    for key in TESTS:
        null_values = [row["pvalues"][key] for row in nulls]
        others = [row["pvalues"][key] for row in rows if not row["null_control"]]
        by_test[key] = {
            "null_rejections": sum(value is not None and value < ALPHA for value in null_values),
            "null_undefined": sum(value is None for value in null_values),
            "nonnull_detections": sum(value is not None and value < ALPHA for value in others),
            "nonnull_undefined": sum(value is None for value in others),
            "nonnull_plans": len(others),
        }

    # Sizes from base roots the null controls did not use.
    base = [record for record in records if record["variant"] == "base"]
    in_null = [record for record in base if record["comparison_index"] in null_indices]
    elsewhere = [record for record in base if record["comparison_index"] not in null_indices]
    pilot_base = [record["score"] for record in pilot
                  if record["task"] == task and record["variant"] == "base"]
    held_out = np.asarray(pilot_base + [record["score"] for record in elsewhere])
    size_rng = stream(1, task_index)
    sizes_at_plans = {}
    for n in sorted(set(seeds_per_arm)):
        counts = size_rejections(held_out, n, REPLICATES, size_rng)
        sizes_at_plans[n] = {key: value / REPLICATES for key, value in counts.items()}
    expected = {key: sum(sizes_at_plans[n][key] for n in seeds_per_arm) for key in SIZE_TESTS}
    null_counts, null_squares = resampled_controls(held_out, seeds_per_arm, FAMILIES,
                                                   stream(7, task_index))
    count_tail = tail_pvalue(null_counts, rejected)
    dispersion_tail = tail_pvalue(null_squares, squares)
    grid_rng = stream(2, task_index)
    size_grid = []
    for n in SIZE_GRID:
        counts = size_rejections(held_out, n, REPLICATES, grid_rng)
        size_grid.append({"seeds_per_arm": n,
                          **{key: value / REPLICATES for key, value in counts.items()}})

    # Execution artifacts.
    batches = defaultdict(list)
    by_producer = defaultdict(list)
    for record in base:
        batches[batch_of[record["seed"]]].append(record["score"])
        by_producer[producers[batch_of[record["seed"]]]].append(record["score"])
    batch_test, *_ = group_test([group for group in batches.values() if len(group) >= 2],
                                stream(3, task_index))
    producer_test = (group_test(list(by_producer.values()), stream(4, task_index))[0]
                     if len(by_producer) > 1 else None)
    arm_a = [record["score"] for record in in_null if record["arm"] == 0]
    arm_b = [record["score"] for record in in_null if record["arm"] == 1]

    # Seed blocks: the base roots of one arm of one comparison repetition.
    blocks = defaultdict(list)
    for record in elsewhere:
        blocks[record["comparison_index"], record["repetition"], record["arm"]].append(
            record["score"])
    block_test, values, labels, permuted = group_test(
        [group for group in blocks.values() if len(group) >= 2], stream(5, task_index))
    between, within, size = anova_parts(values, labels)
    estimated = (between - within) / (size * within)
    implied = max(0.0, (squares - len(nulls)) / sum(seeds_per_arm))

    return {
        "null_controls": {
            "executed": len(nulls), "welch_rejections": rejected,
            "binomial_tail": float(stats.binom.sf(rejected - 1, len(nulls), ALPHA)),
            "ks_uniform_reference_pvalue": float(stats.kstest(welch, "uniform").pvalue),
            "dispersion_chi2": squares, "dispersion_df": len(nulls),
            "dispersion_chi2_reference_pvalue": float(stats.chi2.sf(squares, len(nulls))),
            "resampled_count_tail": count_tail,
            "resampled_count_mc_se": math.sqrt(count_tail * (1 - count_tail) / FAMILIES),
            "resampled_dispersion_pvalue": dispersion_tail,
            "resampled_dispersion_mc_se": math.sqrt(dispersion_tail * (1 - dispersion_tail)
                                                    / FAMILIES),
            "resampled_mean_dispersion": float(null_squares.mean()),
        },
        "cells": [{"comparison_index": row["comparison_index"], "repetition": row["repetition"],
                   "seeds_per_arm": row["seeds_per_arm"],
                   "welch_pvalue": row["pvalues"]["welch"],
                   "welch_size_at_this_count": sizes_at_plans[row["seeds_per_arm"]]["welch"]}
                  for row in sorted(nulls, key=lambda row: row["seeds_per_arm"])],
        "tests": by_test,
        "nonnull_plans_with_at_most_3_seeds_per_arm": {
            "plans": sum(row["seeds_per_arm"] <= 3 for row in rows if not row["null_control"]),
            "detections": {key: sum(
                row["pvalues"][key] is not None and row["pvalues"][key] < ALPHA
                for row in rows if not row["null_control"] and row["seeds_per_arm"] <= 3)
                for key in TESTS},
        },
        "planned_count_size": {
            "held_out_base_roots": int(held_out.size),
            "expected_rejections": expected,
            "welch_poisson_binomial_tail": poisson_binomial_tail(
                [sizes_at_plans[n]["welch"] for n in seeds_per_arm], rejected),
        },
        "size_by_seeds_per_arm": size_grid,
        "artifact_checks": {
            "batches": batch_test,
            "producer_commits": len(by_producer),
            "producers": producer_test,
            "null_arm_means": [float(np.mean(arm_a)), float(np.mean(arm_b))],
            "null_arm_welch_pvalue": welch_test(arm_a, arm_b)["pvalue"],
            "null_arm_ks_pvalue": float(stats.ks_2samp(arm_a, arm_b).pvalue),
            "pilot_vs_confirmation_ks_pvalue": float(
                stats.ks_2samp(pilot_base, [record["score"] for record in base]).pvalue),
            "null_cells_vs_other_base_ks_pvalue": float(stats.ks_2samp(
                [record["score"] for record in in_null],
                [record["score"] for record in elsewhere]).pvalue),
        },
        "block_effect": {
            "held_out_blocks": block_test,
            "estimated_share": estimated,
            "implied_share": implied,
            "expected_welch_rejections": {
                "resampled_without_block_effect": expected["welch"],
                "normal_at_estimated_share": normal_rejections(seeds_per_arm, max(estimated, 0.0)),
                "normal_at_implied_share": normal_rejections(seeds_per_arm, implied),
            },
            "detection_probability_at_implied_share": (
                detection_probability(values, labels, permuted, implied, within,
                                      stream(6, task_index)) if implied > 0 else None),
        },
        "distributions": {
            variant: describe([record["score"] for record in pilot + confirmation
                               if record["task"] == task and record["variant"] == variant],
                              BOUNDARY[task])
            for variant in design["variants"]
        },
    }


def build(root):
    design, plans, pilot, confirmation, batch_of, producers = load(root)
    result = {
        "label": ("Committed null-control outcomes and root scores are MEAS. Sizes at planned "
                  "counts resample held-out committed base roots with replacement; they are not "
                  "new training. The block-effect detection probability is SIM."),
        "seed": SEED, "alpha": ALPHA, "permutations": PERMUTATIONS, "replicates": REPLICATES,
        "simulations": SIMULATIONS, "resampled_families": FAMILIES,
        "tests": {key: TEST_LABELS[key] for key in TESTS},
        "small_sample_limits": {
            "permutation_min_pvalue": [2 / math.comb(2 * n, n) for n in (2, 3)],
            "mann_whitney_constant_arms_of_three_pvalue": float(stats.mannwhitneyu(
                np.zeros(3), np.ones(3), alternative="two-sided", method="asymptotic").pvalue),
        },
        "streams": ("NumPy PCG64 from SeedSequence(seed, spawn_key=(stream, task index)). "
                    "0 permutation tests, 1 sizes at planned counts, 2 size grid, "
                    "3 batch regroupings, 4 producer regroupings, 5 block regroupings, "
                    "6 block-effect simulation, 7 all controls redrawn together."),
        "sources": {path: sha256(raw_path(root / path))
                    for path in (DESIGN, PLAN, PILOT_RAW, CONFIRMATION_RAW, ENVIRONMENT)},
        "per_task": {task: diagnose(index, task, design, plans, pilot, confirmation, batch_of,
                                    producers)
                     for index, task in enumerate(design["tasks"])},
    }
    entries = list(result["per_task"].values())
    executed = sum(entry["null_controls"]["executed"] for entry in entries)
    rejected = sum(entry["null_controls"]["welch_rejections"] for entry in entries)
    result["family"] = {
        "tasks": len(entries), "executed": executed, "welch_rejections": rejected,
        "binomial_tail": float(stats.binom.sf(rejected - 1, executed, ALPHA)),
        "bonferroni_binomial_tail": min(1.0, len(entries) * min(
            entry["null_controls"]["binomial_tail"] for entry in entries)),
        "bonferroni_dispersion_chi2_reference_pvalue": min(1.0, len(entries) * min(
            entry["null_controls"]["dispersion_chi2_reference_pvalue"] for entry in entries)),
        "bonferroni_resampled_count_tail": min(1.0, len(entries) * min(
            entry["null_controls"]["resampled_count_tail"] for entry in entries)),
        "bonferroni_resampled_dispersion_pvalue": min(1.0, len(entries) * min(
            entry["null_controls"]["resampled_dispersion_pvalue"] for entry in entries)),
    }
    return result, pilot, confirmation


# Report.

def p(value):
    return f"{value:.3f}" if value >= 0.001 else f"{value:.1e}"


def percent(value):
    return f"{100 * value:.1f}%"


def render_report(result):
    tasks = list(result["per_task"])
    cart, acro = (result["per_task"][task] for task in ("CartPole-v1", "Acrobot-v1"))
    family = result["family"]
    nc, size = cart["null_controls"], cart["planned_count_size"]
    tests, block = cart["tests"], cart["block_effect"]
    held = block["held_out_blocks"]
    small = cart["nonnull_plans_with_at_most_3_seeds_per_arm"]
    limits = result["small_sample_limits"]
    largest_se = max(entry["null_controls"][key] for entry in result["per_task"].values()
                     for key in ("resampled_count_mc_se", "resampled_dispersion_mc_se"))
    detected = block["detection_probability_at_implied_share"]
    detection_half_width = 100 * 1.96 * math.sqrt(detected * (1 - detected)
                                                  / result["simulations"])
    artifacts = cart["artifact_checks"]
    expected = size["expected_rejections"]
    lines = [
        "# Null-control diagnostics for the neural PPO confirmation",
        "",
        "`scripts/null_diagnostics.py` writes this report, "
        "[null_diagnostics.json](null_diagnostics.json) and the "
        "[figure](../../figures/null_controls.svg) from the committed raw roots; `--check` "
        "regenerates all three and compares them with the committed files. Observed outcomes "
        "and root scores are measurements. Test sizes at the planned counts come from "
        "resampling committed base roots that the null controls did not use, which is not new "
        "training. The detection probability for a block effect is a simulation (SIM). I chose "
        "these checks after I saw the CartPole excess, so every p-value here is exploratory and "
        "unadjusted unless I say otherwise.",
        "",
        "## Summary",
        "",
        f"CartPole-v1 rejected {nc['welch_rejections']}/{nc['executed']} base-versus-base "
        f"controls with the committed Welch test. If every test had exactly 5% size, "
        f"{nc['welch_rejections']} or more rejections would have probability "
        f"{p(nc['binomial_tail'])}. The {nc['executed']} z-statistics are also over-dispersed: "
        f"their squares sum to {nc['dispersion_chi2']:.3f}. As an approximate reference that "
        "assumes independent, exactly uniform p-values, a chi-square with "
        f"{nc['dispersion_df']} degrees of freedom gives p = "
        f"{p(nc['dispersion_chi2_reference_pvalue'])}. Welch p-values on these skewed scores are "
        "only approximately uniform, so I also calibrated both statistics empirically: "
        f"redrawing all {nc['executed']} controls from the held-out base roots "
        f"{result['resampled_families']} times gives P(sum of z² ≥ "
        f"{nc['dispersion_chi2']:.3f}) = {p(nc['resampled_dispersion_pvalue'])} "
        f"(Monte Carlo SE {nc['resampled_dispersion_mc_se']:.4f}) and P({nc['welch_rejections']} "
        f"or more rejections) = {p(nc['resampled_count_tail'])} (SE "
        f"{nc['resampled_count_mc_se']:.4f}). Acrobot-v1 rejected "
        f"{acro['null_controls']['welch_rejections']}/{acro['null_controls']['executed']} "
        f"(tail {p(acro['null_controls']['binomial_tail'])}; squares "
        f"{acro['null_controls']['dispersion_chi2']:.1f}, chi-square reference p = "
        f"{p(acro['null_controls']['dispersion_chi2_reference_pvalue'])}, resampled p = "
        f"{p(acro['null_controls']['resampled_dispersion_pvalue'])}). Both tasks together "
        f"rejected {family['welch_rejections']}/{family['executed']} (tail "
        f"{p(family['binomial_tail'])}). Adjusted only for looking at two tasks (Bonferroni), "
        "not for the other exploratory checks here, the CartPole tail is "
        f"{p(family['bonferroni_binomial_tail'])} at exact 5% size and "
        f"{p(family['bonferroni_resampled_count_tail'])} resampled, and its dispersion p-value "
        f"is {p(family['bonferroni_dispersion_chi2_reference_pvalue'])} against the chi-square "
        f"reference and {p(family['bonferroni_resampled_dispersion_pvalue'])} resampled.",
        "",
        "These diagnostics did not identify an explanation for the CartPole excess:",
        "",
        f"- **Other tests.** On the same {nc['executed']} controls, Student's t rejected "
        f"{tests['student']['null_rejections']}, a permutation test of the mean gap "
        f"{tests['permutation']['null_rejections']}, Mann-Whitney "
        f"{tests['mann_whitney']['null_rejections']} and Yuen's 20% trimmed test "
        f"{tests['yuen']['null_rejections']}. Elevated rejection counts also appear under tests "
        "that do not use the t approximation.",
        f"- **Test size at the planned counts.** Drawing both arms from "
        f"{size['held_out_base_roots']} held-out base roots at each control's planned count, "
        f"Welch should reject {expected['welch']:.2f} of the {nc['executed']} (Student "
        f"{expected['student']:.2f}, Mann-Whitney {expected['mann_whitney']:.2f}, Yuen "
        f"{expected['yuen']:.2f}). With those sizes, {nc['welch_rejections']} or more Welch "
        f"rejections has probability {p(size['welch_poisson_binomial_tail'])}. Resampled IID "
        "draws from this empirical pool gave sizes near nominal at these counts.",
        f"- **Execution.** These checks detected no difference between "
        f"{artifacts['batches']['groups']} training batches (permutation p = "
        f"{p(artifacts['batches']['permutation_pvalue'])}), between arm 0 and arm 1 of the "
        f"controls (Welch p = {p(artifacts['null_arm_welch_pvalue'])}), between pilot and "
        f"confirmation roots (KS p = {p(artifacts['pilot_vs_confirmation_ks_pvalue'])}), or "
        f"between control roots and other base roots (KS p = "
        f"{p(artifacts['null_cells_vs_other_base_ks_pvalue'])}). They cannot rule out small "
        "differences. Every CartPole base-variant batch came from the same transport commit.",
        f"- **Seed blocks.** The over-dispersion is what a small random shift shared by all "
        f"roots of one arm would produce. A shift variance of {block['implied_share']:.4f} times "
        f"the root variance would explain it on average. In the {held['groups']} base blocks "
        f"outside the controls ({held['roots']} roots), a one-way ANOVA gives F = "
        f"{held['anova_f']:.3f} (permutation p = {p(held['permutation_pvalue'])}, "
        f"Kruskal-Wallis p = {p(held['kruskal_pvalue'])}) and an estimated share of "
        f"{block['estimated_share']:.4f}. A share of {block['implied_share']:.4f} would have "
        f"given p < 0.05 in {percent(block['detection_probability_at_implied_share'])} of "
        f"{result['simulations']} simulations (95% Monte Carlo half-width "
        f"{detection_half_width:.1f} points). At the estimated share, a normal approximation "
        f"puts the expected Welch rejections at "
        f"{block['expected_welch_rejections']['normal_at_estimated_share']:.2f} instead of "
        f"{expected['welch']:.2f}. I know of no mechanism that gives consecutive root seeds a "
        "shared shift, and the batch check, which also groups consecutive seeds, detected no "
        "difference.",
        "",
        "So these diagnostics did not identify an explanation. It may be chance: adjusted only "
        "for the two tasks, the resampled tail is "
        f"{p(family['bonferroni_resampled_count_tail'])} and "
        "the resampled dispersion p-value is "
        f"{p(family['bonferroni_resampled_dispersion_pvalue'])}. It may also be a small "
        "seed-block effect, which these data neither confirm nor rule out. The Acrobot-v1 "
        "checks gave no unusual result.",
        "",
        "## Null controls by task",
        "",
        "| Task | Executed | Welch rejections | P(at least this many), exact 5% size "
        "| P(at least this many), resampled | Sum of z² | Chi-square reference p "
        "| Resampled p | KS p against uniform (reference) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task in tasks:
        entry = result["per_task"][task]["null_controls"]
        lines.append(
            f"| {task} | {entry['executed']} | {entry['welch_rejections']} | "
            f"{p(entry['binomial_tail'])} | {p(entry['resampled_count_tail'])} | "
            f"{entry['dispersion_chi2']:.3f} | {p(entry['dispersion_chi2_reference_pvalue'])} | "
            f"{p(entry['resampled_dispersion_pvalue'])} | "
            f"{p(entry['ks_uniform_reference_pvalue'])} |")
    lines += [
        "",
        "Each z-statistic is the normal quantile of a two-sided Welch p-value. The chi-square "
        f"({nc['dispersion_df']} and {acro['null_controls']['dispersion_df']} degrees of "
        "freedom) and KS columns are approximate references: they assume independent, exactly "
        "uniform p-values, which Welch on these skewed scores only approximates. The resampled "
        f"columns redraw all controls together {result['resampled_families']} times, both arms "
        "with replacement from the held-out base roots at each control's planned count, and "
        "assume roots are independent draws from that pool. Their Monte Carlo SE is at most "
        f"{largest_se:.4f}. "
        "The resampled sum of z² averages "
        f"{nc['resampled_mean_dispersion']:.1f} for CartPole-v1 and "
        f"{acro['null_controls']['resampled_mean_dispersion']:.1f} for Acrobot-v1, close to "
        "the chi-square means.",
        "",
        "## Every test on every executed cell",
        "",
        "| Task | Test | Null rejections | Different-variant detections |",
        "|---|---|---:|---:|",
    ]
    for task in tasks:
        for key in TESTS:
            entry = result["per_task"][task]["tests"][key]
            executed = result["per_task"][task]["null_controls"]["executed"]
            lines.append(
                f"| {task} | {TEST_LABELS[key]} | {entry['null_rejections']}/{executed} | "
                f"{entry['nonnull_detections']}/{entry['nonnull_plans']} |")
    lines += [
        "",
        "No test was undefined on any cell. The permutation test uses "
        f"{result['permutations']} random relabelings. With 2 or 3 seeds per arm its exact "
        f"p-value is at least {limits['permutation_min_pvalue'][0]:.3f} or "
        f"{limits['permutation_min_pvalue'][1]:.3f}, so it cannot reach p < 0.05. The "
        "asymptotic, tie-corrected Mann-Whitney p-value I use can, but only with 3 seeds per "
        "arm when each arm is constant and the two differ (p = "
        f"{p(limits['mann_whitney_constant_arms_of_three_pvalue'])}). "
        f"{small['plans']} of the {tests['welch']['nonnull_plans']} CartPole different-variant "
        "plans used 2 or 3 seeds. On those plans Welch detected "
        f"{small['detections']['welch']}, the permutation test "
        f"{small['detections']['permutation']} and Mann-Whitney "
        f"{small['detections']['mann_whitney']}; on the other "
        f"{tests['welch']['nonnull_plans'] - small['plans']} they detected "
        f"{tests['welch']['nonnull_detections'] - small['detections']['welch']}, "
        f"{tests['permutation']['nonnull_detections'] - small['detections']['permutation']} "
        f"and {tests['mann_whitney']['nonnull_detections'] - small['detections']['mann_whitney']}.",
        "",
        "## Test size from held-out roots",
        "",
        "| Task | Held-out base roots | Expected Welch rejections | Student | Mann-Whitney "
        "| Yuen | P(at least the observed Welch count) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for task in tasks:
        entry = result["per_task"][task]["planned_count_size"]
        rejections = entry["expected_rejections"]
        lines.append(
            f"| {task} | {entry['held_out_base_roots']} | {rejections['welch']:.2f} | "
            f"{rejections['student']:.2f} | {rejections['mann_whitney']:.2f} | "
            f"{rejections['yuen']:.2f} | {p(entry['welch_poisson_binomial_tail'])} |")
    lines += [
        "",
        f"The held-out roots are all pilot base roots plus confirmation base roots outside the "
        f"null controls. For each control, both arms draw its planned count with replacement "
        f"from that pool, {result['replicates']} times; the expected count sums the resulting "
        "sizes over the controls. The last column treats those sizes as exact. The permutation "
        "test's size is at most 5% by construction, so I do not resample it.",
        "",
        "Size at fixed counts (rejection rate with both arms from the held-out pool):",
        "",
        "| Seeds per arm | " + " | ".join(
            f"{task.split('-')[0]} {name}" for task in tasks
            for name in ("Welch", "Student", "Mann-Whitney", "Yuen")) + " |",
        "|---:|" + "---:|" * (len(tasks) * len(SIZE_TESTS)),
    ]
    grids = [result["per_task"][task]["size_by_seeds_per_arm"] for task in tasks]
    for rows in zip(*grids):
        lines.append(f"| {rows[0]['seeds_per_arm']} | " + " | ".join(
            percent(row[key]) for row in rows for key in SIZE_TESTS) + " |")
    lines += [
        "",
        f"Each rate uses {result['replicates']} draws, so its Monte Carlo standard error near "
        f"5% is about {percent(math.sqrt(ALPHA * (1 - ALPHA) / result['replicates']))}.",
        "",
        "## Execution checks",
        "",
        "| Check | " + " | ".join(tasks) + " |",
        "|---|" + "---:|" * len(tasks),
    ]
    checks = [result["per_task"][task]["artifact_checks"] for task in tasks]
    lines += [
        "| Training batches with at least two base roots | " + " | ".join(
            str(entry["batches"]["groups"]) for entry in checks) + " |",
        "| Batch ANOVA F, permutation p | " + " | ".join(
            f"{entry['batches']['anova_f']:.3f}, {p(entry['batches']['permutation_pvalue'])}"
            for entry in checks) + " |",
        "| Batch Kruskal-Wallis p | " + " | ".join(
            p(entry["batches"]["kruskal_pvalue"]) for entry in checks) + " |",
        "| Transport commits that produced base batches | " + " | ".join(
            str(entry["producer_commits"]) for entry in checks) + " |",
        "| Between those commits: permutation p | " + " | ".join(
            "one commit" if entry["producers"] is None
            else p(entry["producers"]["permutation_pvalue"]) for entry in checks) + " |",
        "| Null controls, arm 0 vs arm 1 means | " + " | ".join(
            f"{entry['null_arm_means'][0]:.1f} vs {entry['null_arm_means'][1]:.1f}"
            for entry in checks) + " |",
        "| Arm 0 vs arm 1: Welch p, KS p | " + " | ".join(
            f"{p(entry['null_arm_welch_pvalue'])}, {p(entry['null_arm_ks_pvalue'])}"
            for entry in checks) + " |",
        "| Pilot vs confirmation base roots: KS p | " + " | ".join(
            p(entry["pilot_vs_confirmation_ks_pvalue"]) for entry in checks) + " |",
        "| Null-control vs other base roots: KS p | " + " | ".join(
            p(entry["null_cells_vs_other_base_ks_pvalue"]) for entry in checks) + " |",
        "",
        "A training batch is one vmapped job of up to 64 consecutive seeds of one task and "
        "variant, matched to the recorded batches in the confirmation environment file. "
        f"Permutation p-values regroup the roots at random {result['permutations']} times.",
        "",
        "## Seed-block effects",
        "",
        "| | " + " | ".join(tasks) + " |",
        "|---|" + "---:|" * len(tasks),
    ]
    blocks = [result["per_task"][task]["block_effect"] for task in tasks]
    lines += [
        "| Held-out base blocks (roots) | " + " | ".join(
            f"{entry['held_out_blocks']['groups']} ({entry['held_out_blocks']['roots']})"
            for entry in blocks) + " |",
        "| Block ANOVA F, permutation p | " + " | ".join(
            f"{entry['held_out_blocks']['anova_f']:.3f}, "
            f"{p(entry['held_out_blocks']['permutation_pvalue'])}" for entry in blocks) + " |",
        "| Block Kruskal-Wallis p | " + " | ".join(
            p(entry["held_out_blocks"]["kruskal_pvalue"]) for entry in blocks) + " |",
        "| Estimated share of root variance | " + " | ".join(
            f"{entry['estimated_share']:.4f}" for entry in blocks) + " |",
        "| Share implied by the null-control dispersion | " + " | ".join(
            f"{entry['implied_share']:.4f}" for entry in blocks) + " |",
        "| SIM: block test detects the implied share | " + " | ".join(
            "no excess" if entry["detection_probability_at_implied_share"] is None
            else percent(entry["detection_probability_at_implied_share"])
            for entry in blocks) + " |",
        "| Expected Welch rejections: resampled, no block effect | " + " | ".join(
            f"{entry['expected_welch_rejections']['resampled_without_block_effect']:.2f}"
            for entry in blocks) + " |",
        "| Normal approximation at the estimated share | " + " | ".join(
            f"{entry['expected_welch_rejections']['normal_at_estimated_share']:.2f}"
            for entry in blocks) + " |",
        "| Normal approximation at the implied share | " + " | ".join(
            f"{entry['expected_welch_rejections']['normal_at_implied_share']:.2f}"
            for entry in blocks) + " |",
        "",
        "A block is the base roots of one arm of one comparison repetition: consecutive seeds "
        "that were planned, trained and tested together. If each block shares a shift with "
        "variance share × sigma², a control's z-statistic has variance 1 + n × share, so the "
        "implied share is (sum of z² − controls) / (sum of planned counts). The held-out blocks "
        "are the base arms of the different-variant comparisons, with at least two roots; the "
        "estimated share is the one-way random-effects moment estimate, truncated at zero for "
        "the expected rejections. The simulation draws the held-out roots with replacement, "
        "adds one Gaussian shift per block at the implied share and applies the same "
        "permutation test.",
        "",
        "## Score distributions",
        "",
        "| Task | Variant | Roots | Mean | SD | Skewness | At the 500-step bound |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for task in tasks:
        for variant, entry in result["per_task"][task]["distributions"].items():
            lines.append(
                f"| {task} | {variant} | {entry['roots']} | {entry['mean']:.1f} | "
                f"{entry['sd']:.1f} | {entry['skewness']:.2f} | "
                f"{percent(entry['boundary_fraction'])} |")
    lines += [
        "",
        "Roots pool the pilot and the confirmation. CartPole's bound is the 500-step maximum; "
        "Acrobot's is the -500 return of a policy that never reaches the goal.",
        "",
    ]
    return "\n".join(lines)


def check_claims(result):
    """The qualitative statements in the report."""
    cart, acro = (result["per_task"][task] for task in ("CartPole-v1", "Acrobot-v1"))
    assert cart["null_controls"]["binomial_tail"] < ALPHA, "CartPole excess"
    for key in ("dispersion_chi2_reference_pvalue", "resampled_dispersion_pvalue",
                "resampled_count_tail"):
        assert cart["null_controls"][key] < ALPHA, f"CartPole excess: {key}"
        assert acro["null_controls"][key] > ALPHA, f"Acrobot: {key}"
    for entry in (cart, acro):
        controls = entry["null_controls"]
        assert abs(controls["resampled_mean_dispersion"] - controls["dispersion_df"]) < 1.0
    assert result["family"]["bonferroni_resampled_count_tail"] > ALPHA, "chance plausible"
    assert result["family"]["bonferroni_binomial_tail"] > ALPHA, "chance remains plausible"
    expected = cart["planned_count_size"]["expected_rejections"]["welch"]
    for key in ("student", "permutation", "mann_whitney"):
        assert cart["tests"][key]["null_rejections"] >= cart["null_controls"]["welch_rejections"]
    assert cart["tests"]["yuen"]["null_rejections"] > expected, "Yuen still above expectation"
    assert cart["planned_count_size"]["welch_poisson_binomial_tail"] < ALPHA, "size not the cause"
    block = cart["block_effect"]
    assert block["held_out_blocks"]["permutation_pvalue"] > ALPHA, "block effect not confirmed"
    assert block["detection_probability_at_implied_share"] < 0.95, "block effect not ruled out"
    assert expected < block["expected_welch_rejections"]["normal_at_estimated_share"]
    for entry in (cart, acro):
        executed = entry["null_controls"]["executed"]
        for key, value in entry["planned_count_size"]["expected_rejections"].items():
            assert abs(value / executed - ALPHA) < 0.01, f"size near nominal: {key}"
        checks = entry["artifact_checks"]
        pvalues = [checks["batches"]["permutation_pvalue"], checks["batches"]["kruskal_pvalue"],
                   checks["null_arm_welch_pvalue"], checks["null_arm_ks_pvalue"],
                   checks["pilot_vs_confirmation_ks_pvalue"],
                   checks["null_cells_vs_other_base_ks_pvalue"]]
        if checks["producers"] is not None:
            pvalues.append(checks["producers"]["permutation_pvalue"])
        assert min(pvalues) > ALPHA, "no execution artifact"
        for key in TESTS:
            assert entry["tests"][key]["null_undefined"] == 0
            assert entry["tests"][key]["nonnull_undefined"] == 0
        small = entry["nonnull_plans_with_at_most_3_seeds_per_arm"]
        assert small["detections"]["permutation"] == 0, "permutation cannot reject 2-3 seeds"
    limits = result["small_sample_limits"]
    assert min(limits["permutation_min_pvalue"]) > ALPHA
    assert limits["mann_whitney_constant_arms_of_three_pvalue"] < ALPHA
    assert cart["artifact_checks"]["producer_commits"] == 1, "one CartPole transport commit"
    acro_blocks = acro["block_effect"]["held_out_blocks"]
    assert acro["null_controls"]["binomial_tail"] > ALPHA
    assert acro["null_controls"]["dispersion_chi2_reference_pvalue"] > ALPHA
    assert acro_blocks["permutation_pvalue"] > ALPHA, "Acrobot shows nothing unusual"


def check_quotes(root, report):
    """Every decimal or fraction on a line that cites the report must appear in it."""
    for name in QUOTING:
        lines = [line for line in (root / name).read_text().splitlines()
                 if "null_diagnostics.md" in line]
        assert lines, f"{name} does not cite the null-control report"
        for line in lines:
            for number in QUOTED.findall(line):
                assert number in report, f"{name}: {number} is not in {REPORT}"


def make_figure(result, pilot, confirmation, path):
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "svg.fonttype": "none", "svg.hashsalt": "seed-power"})
    tasks = list(result["per_task"])
    colors = {"CartPole-v1": "#286a9b", "Acrobot-v1": "#b34d30"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, task in zip(axes[0], tasks, strict=True):
        scores = [record["score"] for record in pilot + confirmation
                  if record["task"] == task and record["variant"] == "base"]
        summary = result["per_task"][task]["distributions"]["base"]
        ax.hist(scores, bins=100, color=colors[task])
        ax.set(title=f"{task} base roots: {summary['roots']} scores, "
                     f"{100 * summary['boundary_fraction']:.1f}% at {BOUNDARY[task]:g}",
               xlabel="Root score (mean of 16 evaluation episodes)", ylabel="Roots")
    ax = axes[1, 0]
    for task in tasks:
        cells = result["per_task"][task]["cells"]
        ax.scatter([cell["seeds_per_arm"] for cell in cells],
                   [cell["welch_pvalue"] for cell in cells], s=18, color=colors[task],
                   label=task)
    ax.axhline(ALPHA, color="#555555", linestyle="--", linewidth=.8)
    ax.set(xscale="log", yscale="log", ylim=(1e-3, 1.1), xlabel="Planned seeds per arm",
           ylabel="Welch p-value", title="Each executed base-versus-base confirmation")
    ax.legend(loc="lower right")
    ax = axes[1, 1]
    styles = {"welch": "-", "student": "--", "mann_whitney": ":", "yuen": "-."}
    for task in tasks:
        grid = result["per_task"][task]["size_by_seeds_per_arm"]
        for key in SIZE_TESTS:
            ax.plot([row["seeds_per_arm"] for row in grid], [row[key] for row in grid],
                    color=colors[task], linestyle=styles[key], marker="o", markersize=2.5,
                    label=f"{task}, {TEST_LABELS[key]}")
    ax.axhline(ALPHA, color="#555555", linestyle="--", linewidth=.8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(SIZE_GRID, [str(n) for n in SIZE_GRID])
    ax.minorticks_off()
    ax.set(ylim=(0, 0.12), xlabel="Seeds per arm",
           ylabel="Rejection rate with identical arms",
           title=f"Test size on resampled held-out base roots ({REPLICATES} draws per point)")
    ax.legend(fontsize=6, ncol=2, loc="upper right")
    fig.suptitle("Neural PPO null controls: base score distributions, observed p-values, and "
                 "test sizes from resampled roots (not new training)")
    fig.savefig(path, metadata={"Date": None})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="regenerate in memory and compare with the committed files")
    args = parser.parse_args()
    result, pilot, confirmation = build(ROOT)
    document = json.loads(json.dumps(result, allow_nan=False))
    check_claims(document)
    report = render_report(document)
    if args.check:
        compare_json(document, json.loads((ROOT / OUTPUT).read_text()), OUTPUT)
        assert report == (ROOT / REPORT).read_text(), f"{REPORT} is stale"
        check_quotes(ROOT, report)
        with tempfile.TemporaryDirectory() as scratch:
            fresh = Path(scratch) / "figure.svg"
            make_figure(document, pilot, confirmation, fresh)
            difference = svg_difference(fresh.read_text(), (ROOT / FIGURE).read_text())
            assert difference is None, f"{FIGURE} is stale: {difference}"
        print("Regenerated null_diagnostics.json, null_diagnostics.md and the figure; all match "
              "and every checked claim holds.")
        return
    (ROOT / OUTPUT).write_text(json.dumps(document, indent=2) + "\n")
    (ROOT / REPORT).write_text(report)
    make_figure(document, pilot, confirmation, ROOT / FIGURE)
    print(report.split("## Null controls by task")[0])


if __name__ == "__main__":
    main()
