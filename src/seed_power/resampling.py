"""Planning rules on committed PPO root scores: resampling, not new training.

Each committed root is one independently trained policy score. I split every
task/variant pool into two disjoint halves several times; each half is a
finite population with an exactly known mean gap and variance. Within a
population, every replicate draws a pilot and then a confirmation without
replacement, so no root is in both. Replicates reuse roots across each other,
so they measure planning behaviour on these score distributions, not new
independent evidence. Half-sample variation across populations estimates how
much the finite pool itself moves a rate (a delete-half jackknife).
"""

from __future__ import annotations

import math

import numpy as np

from seed_power.calibration import (
    CAP, OVER_CAP, PLANNED, RULES, TARGET, ALPHA, UNDEFINED, plan, welch_rejections,
)
from seed_power.stats import plan_seeds

POOL_LIMITED = 3


def half_splits(size, splits, rng):
    """Disjoint halves (first, second) of range(size) for each random split."""
    halves = []
    for _ in range(splits):
        order = rng.permutation(size)
        halves.append((np.sort(order[: size // 2]), np.sort(order[size // 2:])))
    return halves


def _moments(values, counts):
    """Means and unbiased variances of each row's first ``counts`` entries.

    Two passes avoid cancellation, and constant samples get exactly zero
    variance, matching stats.welch_test.
    """
    mask = np.arange(values.shape[1]) < counts[:, None]
    means = np.where(mask, values, 0.0).sum(axis=1) / counts
    deviations = np.where(mask, values - means[:, None], 0.0)
    variances = (deviations**2).sum(axis=1) / (counts - 1)
    low = np.where(mask, values, np.inf).min(axis=1)
    high = np.where(mask, values, -np.inf).max(axis=1)
    variances[low == high] = 0.0
    return means, variances


def _permutation_prefixes(rng, size, replicates, width):
    """Each row: the first ``width`` positions of an independent permutation."""
    order = np.tile(np.arange(size, dtype=np.int32), (replicates, 1))
    return rng.permuted(order, axis=1)[:, :width]


def draw_roots(size_a, size_b, same_pool, longest, replicates, rng):
    """Root indices for each replicate's arms, in draw order, never reused within a row.

    Positions [0, pilot) are the pilot and the next ones the confirmation, so
    a row never repeats a root. With ``same_pool`` both arms index one pool and
    interleave a single permutation, so the arms are disjoint too.
    """
    if same_pool:
        width = min(size_a // 2, longest)
        order = _permutation_prefixes(rng, size_a, replicates, 2 * width)
        return order[:, 0::2], order[:, 1::2]
    width = min(size_a, size_b, longest)
    return (_permutation_prefixes(rng, size_a, replicates, width),
            _permutation_prefixes(rng, size_b, replicates, width))


def resample_population(scores_a, scores_b, same_pool, shift, effect, pilot_sizes, replicates,
                        rng, cap=CAP):
    """Run every rule on one finite population; return per-replicate outcomes.

    With ``same_pool`` both arms draw disjoint roots from ``scores_a`` and arm a
    adds ``shift``, so the true gap is exactly ``shift``. Otherwise the arms
    are separate pools. A plan larger than the roots left after the pilot is
    POOL_LIMITED and not run; with CAP it is never silently shortened.
    """
    pool_b = scores_a if same_pool else scores_b
    index_a, index_b = draw_roots(len(scores_a), len(pool_b), same_pool,
                                  max(pilot_sizes) + cap, replicates, rng)
    values_a, values_b = scores_a[index_a] + shift, pool_b[index_b]
    width = index_a.shape[1]
    truth = shift if same_pool else float(np.mean(scores_a) - np.mean(scores_b))
    variance_a, variance_b = np.var(scores_a), np.var(pool_b)
    outcomes = {}
    for pilot in pilot_sizes:
        sizes = np.full(replicates, pilot)
        mean_a, var_a = _moments(values_a[:, :pilot], sizes)
        mean_b, var_b = _moments(values_b[:, :pilot], sizes)
        room = width - pilot
        for rule in RULES:
            if rule == "known_variance":
                count = plan_seeds(effect, variance_a, variance_b, TARGET, ALPHA)
                planned = count is not None and count <= cap
                seeds = np.full(replicates, count if planned else 0)
                status = np.full(replicates, PLANNED if planned else OVER_CAP)
            else:
                planning = mean_a - mean_b if rule == "observed_gap" else effect
                seeds, status = plan(rule, planning, var_a, var_b, pilot, cap)
            status = np.where((status == PLANNED) & (seeds > room), POOL_LIMITED, status)
            executed = np.flatnonzero(status == PLANNED)
            rejected = np.zeros(replicates, bool)
            wrong = np.zeros(replicates, bool)
            if executed.size:
                n = seeds[executed]
                stop = pilot + int(n.max())
                ca, va = _moments(values_a[executed, pilot:stop], n)
                cb, vb = _moments(values_b[executed, pilot:stop], n)
                reject, gap = welch_rejections(ca, va, cb, vb, n)
                rejected[executed] = reject
                if truth != 0:
                    wrong[executed] = reject & (np.sign(gap) != np.sign(truth))
            outcomes[pilot, rule] = {
                "status": status, "seeds": np.where(status == PLANNED, seeds, 0),
                "rejected": rejected, "wrong": wrong, "truth": truth,
            }
    return outcomes


def summarize_populations(results, pilot):
    """Pool per-population outcomes for one rule and pilot size.

    Each population's rate has a replicate-level Monte Carlo SE. Their spread
    across the 2*splits half populations, less that Monte Carlo noise, is the
    delete-half jackknife estimate of finite-pool variance.
    """
    rates, variances, eligibility, costs, wrong, counts = [], [], [], [], [], []
    statuses = {UNDEFINED: 0, OVER_CAP: 0, POOL_LIMITED: 0, PLANNED: 0}
    for outcome in results:
        rejected = outcome["rejected"].astype(float)
        executed = outcome["status"] == PLANNED
        rates.append(rejected.mean())
        variances.append(rejected.var(ddof=1) / rejected.size)
        eligibility.append(executed.mean())
        costs.append((2 * pilot + 2 * outcome["seeds"]).mean())
        wrong.append(outcome["wrong"].mean())
        counts.extend(outcome["seeds"][executed].tolist())
        for key in statuses:
            statuses[key] += int((outcome["status"] == key).sum())
    rates, variances = np.asarray(rates), np.asarray(variances)
    detections = sum(int(outcome["rejected"].sum()) for outcome in results)
    worlds = rates.size
    spread = float(((rates - rates.mean()) ** 2).mean())
    monte_carlo = float(variances.mean())
    finite_pool = max(0.0, spread - monte_carlo * (worlds - 1) / worlds)
    return {
        "populations": worlds,
        "replicates": sum(outcome["rejected"].size for outcome in results),
        "undefined_plans": statuses[UNDEFINED], "over_cap_plans": statuses[OVER_CAP],
        "pool_limited_plans": statuses[POOL_LIMITED], "executed_plans": statuses[PLANNED],
        "eligibility": float(np.mean(eligibility)),
        "workflow_rate": float(rates.mean()),
        "workflow_mc_se": math.sqrt(monte_carlo / worlds),
        "workflow_finite_pool_se": math.sqrt(finite_pool),
        "workflow_total_se": math.sqrt(finite_pool + monte_carlo / worlds),
        "conditional_rate": (detections / statuses[PLANNED]) if statuses[PLANNED] else None,
        "wrong_sign_rate": float(np.mean(wrong)),
        "mean_total_seeds": float(np.mean(costs)),
        "executed_seeds_median": float(np.median(counts)) if counts else None,
    }
