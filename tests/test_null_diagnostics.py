"""Null-control diagnostics: vectorized tests, permutation p-values, block tests, report."""

from itertools import product
import json
import math

import numpy as np
import pytest
from scipy import stats

from scripts import null_diagnostics as diagnostics
from seed_power.stats import welch_test


def test_vectorized_welch_matches_welch_test_including_constant_arms():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, size=(40, 6))
    b = rng.normal(0.5, 2, size=(40, 6))
    a[0], b[0] = 2.0, 2.0
    a[1], b[1] = 2.0, 3.0
    a[2] = 1.0
    a[3], b[3] = rng.integers(0, 3, 6), rng.integers(0, 3, 6)
    pvalues = diagnostics.welch_pvalues(a, b)
    for row_a, row_b, pvalue in zip(a, b, pvalues):
        assert pvalue == pytest.approx(welch_test(row_a, row_b)["pvalue"], rel=1e-10, abs=1e-14)


def test_permutation_pvalue_counts_ties_and_matches_exact_enumeration():
    rng = np.random.default_rng(2)
    assert diagnostics.permutation_pvalue(np.ones(4), np.ones(4), rng) == 1.0
    a, b = np.array([0.0, 0.0, 1.0]), np.array([5.0, 6.0, 7.0])
    pooled = np.concatenate([a, b])
    observed = abs(a.mean() - b.mean())
    exact = np.mean([abs(pooled[list(s)].mean() - np.delete(pooled, list(s)).mean()) >= observed
                     for s in product(range(6), repeat=3) if len(set(s)) == 3])
    estimate = diagnostics.permutation_pvalue(a, b, rng, permutations=20_000)
    assert estimate == pytest.approx(exact, abs=4.5 * math.sqrt(exact * (1 - exact) / 20_000))


def test_permutation_test_keeps_its_level_under_exchangeable_nulls():
    rng = np.random.default_rng(3)
    trials = 300
    rejections = sum(
        diagnostics.permutation_pvalue(rng.exponential(size=5), rng.exponential(size=5), rng,
                                       permutations=199) < 0.05
        for _ in range(trials))
    assert rejections <= stats.binom.ppf(0.999, trials, 0.05)


def test_poisson_binomial_tail_matches_brute_force_and_binomial():
    probabilities = [0.05, 0.2, 0.5, 0.01, 0.3]
    for observed in range(len(probabilities) + 1):
        brute = sum(
            math.prod(p if hit else 1 - p for p, hit in zip(probabilities, outcome))
            for outcome in product((0, 1), repeat=len(probabilities)) if sum(outcome) >= observed)
        assert diagnostics.poisson_binomial_tail(probabilities, observed) == pytest.approx(brute)
    assert diagnostics.poisson_binomial_tail([0.05] * 36, 5) == pytest.approx(
        stats.binom.sf(4, 36, 0.05))


def test_resampled_sizes_are_near_nominal_for_a_gaussian_pool():
    pool = np.random.default_rng(4).normal(size=20_000)
    counts = diagnostics.size_rejections(pool, 20, 8000, np.random.default_rng(5))
    for key, count in counts.items():
        assert abs(count / 8000 - 0.05) <= 4.5 * math.sqrt(0.05 * 0.95 / 8000), key


def test_pair_pvalues_reports_undefined_tests_as_none():
    pvalues = diagnostics.pair_pvalues(np.full(3, 2.0), np.full(3, 2.0), np.random.default_rng(6))
    assert pvalues["welch"] == 1.0 and pvalues["permutation"] == 1.0
    assert pvalues["student"] is None and pvalues["yuen"] is None


def test_anova_matches_scipy_and_estimates_a_known_block_share():
    rng = np.random.default_rng(7)
    groups = [rng.normal(size=size) for size in (2, 3, 9, 40)]
    values = np.concatenate(groups)
    labels = np.repeat(np.arange(4), [2, 3, 9, 40])
    assert diagnostics.anova_f(values, labels) == pytest.approx(
        stats.f_oneway(*groups).statistic, rel=1e-12)
    sizes = rng.integers(5, 80, size=600)
    labels = np.repeat(np.arange(sizes.size), sizes)
    values = rng.normal(size=labels.size) + rng.normal(0, math.sqrt(0.02), sizes.size)[labels]
    between, within, size = diagnostics.anova_parts(values, labels)
    assert (between - within) / (size * within) == pytest.approx(0.02, abs=0.006)


def test_block_test_keeps_its_level_and_detects_block_shifts():
    rng = np.random.default_rng(8)
    null = [diagnostics.group_test([rng.exponential(size=6) for _ in range(12)], rng,
                                   permutations=199)[0]["permutation_pvalue"]
            for _ in range(200)]
    assert sum(value < 0.05 for value in null) <= stats.binom.ppf(0.999, 200, 0.05)
    shifted = [rng.exponential(size=30) + shift for shift in (0.0, 0.0, 1.0, 1.0)]
    summary, values, labels, permuted = diagnostics.group_test(shifted, rng, permutations=199)
    assert summary["permutation_pvalue"] == 1 / 200
    assert summary["groups"] == 4 and summary["roots"] == 120
    _, within, _ = diagnostics.anova_parts(values, labels)
    assert diagnostics.detection_probability(values, labels, permuted, 25.0, within, rng,
                                             simulations=50) == 1.0


def test_normal_rejections_are_nominal_without_a_block_effect_and_grow_with_it():
    counts = [10, 40, 200]
    assert diagnostics.normal_rejections(counts, 0.0) == pytest.approx(0.15)
    with_share = diagnostics.normal_rejections(counts, 0.005)
    assert 0.15 < with_share < diagnostics.normal_rejections(counts, 0.01)
    # Planned counts 10, 40 and 200 at share 0.005 give z variances 1.05, 1.2 and 2.
    assert with_share == pytest.approx(sum(
        2 * stats.norm.sf(stats.norm.isf(0.025) / math.sqrt(variance))
        for variance in (1.05, 1.2, 2.0)), rel=1e-12)


def test_committed_report_renders_from_committed_json_and_its_claims_hold():
    committed = json.loads((diagnostics.ROOT / diagnostics.OUTPUT).read_text())
    report = diagnostics.render_report(committed)
    assert report == (diagnostics.ROOT / diagnostics.REPORT).read_text()
    diagnostics.check_claims(committed)
    diagnostics.check_quotes(diagnostics.ROOT, report)


def test_quoted_numbers_must_appear_in_the_report(tmp_path):
    report = (diagnostics.ROOT / diagnostics.REPORT).read_text()
    for name in diagnostics.QUOTING:
        source = diagnostics.ROOT / name
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(source.read_text())
    diagnostics.check_quotes(tmp_path, report)
    readme = tmp_path / "README.md"
    cited = next(line for line in readme.read_text().splitlines()
                 if "null_diagnostics.md" in line)
    readme.write_text(readme.read_text().replace(cited, cited + " It was 0.999 of 99/100."))
    with pytest.raises(AssertionError, match="0.999"):
        diagnostics.check_quotes(tmp_path, report)
