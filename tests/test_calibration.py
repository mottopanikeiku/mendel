"""Known-truth calibration: planners, exact references, samplers and resampling draws."""

from fractions import Fraction

import math

import numpy as np
import pytest
from scipy.stats import chi2, nct, norm, t

from scripts import calibrate
from seed_power import calibration as sim
from seed_power.resampling import (
    _moments, draw_roots, half_splits, summarize_populations,
)
from seed_power.stats import plan_seeds, plan_seeds_with_pilot_uncertainty, welch_test


def random_pilots(seed, size):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.01, 3.0, size), rng.uniform(0.05, 4.0, size), rng.uniform(0.05, 4.0, size)


@pytest.mark.parametrize("rule", ["fixed_effect", "observed_gap", "known_variance"])
def test_vectorized_plans_match_the_scalar_planner(rule):
    effects, variance_a, variance_b = random_pilots(1, 300)
    seeds, status = sim.plan(rule, -effects, variance_a, variance_b, 8)
    for effect, a, b, n, state in zip(effects, variance_a, variance_b, seeds, status):
        expected = plan_seeds(effect, a, b)
        if expected <= sim.CAP:
            assert (state, n) == (sim.PLANNED, expected)
        else:
            assert (state, n) == (sim.OVER_CAP, 0)


def test_zero_effect_and_zero_variance_plans_are_undefined():
    seeds, status = sim.plan("fixed_effect", [0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 2.0], 8)
    assert status.tolist() == [sim.UNDEFINED, sim.UNDEFINED, sim.PLANNED]
    assert seeds[:2].tolist() == [0, 0]


def test_ucl_rule_plans_with_browne_scaled_variances():
    effects, variance_a, variance_b = random_pilots(2, 200)
    for pilot in (8, 32):
        seeds, status = sim.plan("variance_ucl", effects, variance_a, variance_b, pilot)
        for effect, a, b, n, state in zip(effects, variance_a, variance_b, seeds, status):
            share = a / (a + b)
            df = (pilot - 1) / (share**2 + (1 - share) ** 2)
            factor = df / chi2.ppf(0.2, df)
            expected = plan_seeds(effect, a * factor, b * factor)
            assert (n if state == sim.PLANNED else sim.CAP + 1) == min(expected, sim.CAP + 1)


def julious_owen(effect, variance_a, variance_b, pilot, cap=sim.CAP):
    """Direct scalar search of Julious and Owen's noncentral-t count with Welch df."""
    share = variance_a / (variance_a + variance_b)
    factor = 1 / (share**2 + (1 - share) ** 2)
    for n in range(2, cap + 1):
        quantile = nct.ppf(0.8, (pilot - 1) * factor, t.isf(0.025, (n - 1) * factor))
        if n * effect**2 >= quantile**2 * (variance_a + variance_b):
            return n
    return cap + 1


def test_noncentral_t_rule_matches_a_direct_julious_owen_search():
    effects, variance_a, variance_b = random_pilots(3, 60)
    seeds, status = sim.plan("variance_nct", effects, variance_a, variance_b, 8)
    for effect, a, b, n, state in zip(effects, variance_a, variance_b, seeds, status):
        planned = n if state == sim.PLANNED else sim.CAP + 1
        assert planned == julious_owen(effect, a, b, 8)
        assert planned == min(plan_seeds_with_pilot_uncertainty(effect, a, b, 8), sim.CAP + 1)


@pytest.mark.parametrize("total,ucl,nct_factor", [
    (20, 1.400, 1.099), (24, 1.349, 1.080), (30, 1.297, 1.062), (40, 1.244, 1.045),
    (50, 1.211, 1.036), (70, 1.172, 1.025), (100, 1.139, 1.017), (200, 1.093, 1.009),
])
def test_inflation_factors_match_published_tables(total, ucl, nct_factor):
    # Whitehead et al. (2016), Tables 1 and 2: total pilot size, 2*(m-1) pooled df,
    # 80% UCL and 80%-power noncentral-t inflation over the normal formula.
    assert round(float(sim.ucl_factor(total // 2, 0.5)), 3) == ucl
    z = norm.isf(0.025) + norm.isf(0.2)
    assert round((nct.ppf(0.8, total - 2, norm.isf(0.025)) / z) ** 2, 3) == nct_factor


@pytest.mark.parametrize("pilot_df", [7, 14, 62])
def test_noncentral_t_count_is_expected_power_over_variance_uncertainty(pilot_df):
    # With sigma^2 = s^2 * k / chi2_k, average one-tailed power Phi(x*sigma'/sigma - c)
    # equals the noncentral-t CDF Julious and Owen invert.
    critical = t.isf(0.025, 40)
    values, weights = sim.chi2_nodes(pilot_df)
    for x in (1.0, 2.5, 3.5):
        expected_power = norm.cdf(x * np.sqrt(values / pilot_df) - critical) @ weights
        assert expected_power == pytest.approx(nct.cdf(x, pilot_df, critical), abs=1e-9)


@pytest.mark.parametrize("n", [2, 3, 5, 16, 64, 300])
@pytest.mark.parametrize("gap", [0.0, 0.3, 1.0])
def test_pooled_quadrature_reproduces_noncentral_t_power(n, gap):
    exact, _ = sim.exact_welch(gap, 1.0, 1.0, n, pooled=True)
    critical = t.isf(0.025, 2 * n - 2)
    expected = sim.two_sided_nct(critical, 2 * n - 2, gap / math.sqrt(2 / n))
    # 48 probit-Hermite nodes: about 1e-8 at n=2, rounding error from n=5 on.
    assert exact == pytest.approx(float(expected), abs=1e-7)


@pytest.mark.parametrize("n,gap", [(3, 1.5), (12, 0.0), (12, 1.2)])
def test_exact_welch_matches_simulated_gaussian_welch(n, gap):
    rng = np.random.default_rng(4)
    trials = 40_000
    sizes = np.full(trials, n)
    mean_a, var_a = sim.Normal(gap, 2.0).sample_moments(rng, sizes)
    mean_b, var_b = sim.Normal(0.0, 1.0).sample_moments(rng, sizes)
    reject, difference = sim.welch_rejections(mean_a, var_a, mean_b, var_b, sizes)
    exact, wrong = sim.exact_welch(gap, 4.0, 1.0, n)
    assert abs(reject.mean() - exact) <= 4.5 * math.sqrt(exact * (1 - exact) / trials)
    if gap:
        rate = (reject & (difference < 0)).mean()
        assert abs(rate - wrong) <= 4.5 * math.sqrt(wrong * (1 - wrong) / trials) + 1e-4


def test_vectorized_welch_matches_welch_test_including_constant_arms():
    rng = np.random.default_rng(5)
    samples = [(rng.normal(0, 1, 6), rng.normal(0.5, 2, 6)) for _ in range(200)]
    samples += [(np.full(4, 2.0), np.full(4, 2.0)), (np.full(4, 2.0), np.full(4, 3.0)),
                (np.full(5, 1.0), rng.normal(0, 1, 5)), (rng.integers(0, 3, 7) * 1.0,
                                                         rng.integers(0, 3, 7) * 1.0)]
    for a, b in samples:
        expected = welch_test(a, b)
        variance_a = 0.0 if np.all(a == a[0]) else a.var(ddof=1)
        variance_b = 0.0 if np.all(b == b[0]) else b.var(ddof=1)
        reject, gap = sim.welch_rejections(np.array([a.mean()]), np.array([variance_a]),
                                           np.array([b.mean()]), np.array([variance_b]),
                                           np.array([a.size]))
        assert bool(reject[0]) == (expected["pvalue"] < 0.05)
        assert gap[0] == pytest.approx(expected["mean_difference"], abs=1e-12)


@pytest.mark.parametrize("rule", sim.PILOT_RULES)
def test_exact_plan_distribution_matches_simulated_gaussian_pilots(rule):
    pilot, effect, truth, pilots = 8, 0.5, 0.25, 6_000
    pmf, over = sim.exact_plan_distribution(rule, pilot, 1.0, effect, truth)
    assert pmf.sum() + over == pytest.approx(1.0, abs=1e-12)
    rng = np.random.default_rng(6)
    sizes = np.full(pilots, pilot)
    mean_a, var_a = sim.Normal(truth, 1.0).sample_moments(rng, sizes)
    mean_b, var_b = sim.Normal(0.0, 1.0).sample_moments(rng, sizes)
    planning = mean_a - mean_b if rule == "observed_gap" else effect
    seeds, status = sim.plan(rule, planning, var_a, var_b, pilot)
    seeds = np.where(status == sim.PLANNED, seeds, sim.CAP + 1)
    for count in (20, 50, 80, 150, 300, sim.CAP):
        probability = pmf[: count + 1].sum()
        se = math.sqrt(max(probability * (1 - probability), 1e-12) / pilots)
        assert abs((seeds <= count).mean() - probability) <= 4.5 * se + 1e-9, count


def test_grid_moments_are_unbiased_and_constant_samples_have_zero_variance():
    grid = sim.FAILURE_CONTROL
    rng = np.random.default_rng(7)
    means, variances = grid.sample_moments(rng, np.full(40_000, 5))
    assert abs(means.mean() - grid.mean) <= 4.5 * grid.sd / math.sqrt(5 * 40_000)
    assert abs(variances.mean() - grid.variance) <= 4.5 * variances.std() / math.sqrt(40_000)
    point = sim._point(0.3)
    means, variances = point.sample_moments(rng, np.full(10, 4))
    assert np.all(variances == 0.0)
    reject, _ = sim.welch_rejections(means, variances, means, variances, np.full(10, 4))
    assert not reject.any()


def test_populations_have_exact_truths_and_meaningful_effects():
    for family, label, arm_a, arm_b, effect in sim.scenarios():
        fraction = dict(sim.TRUTHS)[label]
        assert arm_a.mean - arm_b.mean == pytest.approx(fraction * effect, abs=1e-12)
        if family != sim.SHAPE_NULL:
            assert effect == pytest.approx(0.5 * sim.FAMILIES[family][1].sd)
    assert sim.CEILING_CONTROL.pmf[-1] == pytest.approx(0.10)
    assert sim.FAILURE_CONTROL.pmf[0] == pytest.approx(0.26)


def test_simulation_is_reproducible_and_accounts_for_every_plan():
    arm_a, arm_b = sim.ceiling_treatment(0.05), sim.CEILING_CONTROL
    first = sim.simulate(arm_a, arm_b, 0.08, 8, 200, 2, 11, (9, 9))
    second = sim.simulate(arm_a, arm_b, 0.08, 8, 200, 2, 11, (9, 9))
    assert first == second
    for row in first:
        assert row["undefined_plans"] + row["over_cap_plans"] + row["executed_plans"] == 200


def test_half_splits_are_disjoint_and_complete():
    for first, second in half_splits(101, 3, np.random.default_rng(8)):
        assert not set(first) & set(second)
        assert sorted(np.concatenate([first, second])) == list(range(101))


@pytest.mark.parametrize("same_pool", [True, False])
def test_pilot_and_confirmation_roots_are_never_shared(same_pool):
    index_a, index_b = draw_roots(300, 250, same_pool, 140, 50, np.random.default_rng(9))
    for row_a, row_b in zip(index_a, index_b):
        assert len(set(row_a)) == row_a.size and len(set(row_b)) == row_b.size
        if same_pool:
            assert not set(row_a) & set(row_b)
    assert index_a.shape == index_b.shape == (50, 140)


def test_ragged_moments_match_numpy():
    rng = np.random.default_rng(10)
    values = rng.normal(size=(30, 12))
    values[0] = 4.0
    counts = rng.integers(2, 13, 30)
    counts[0] = 7
    means, variances = _moments(values, counts)
    for row, n, mean, variance in zip(values, counts, means, variances):
        assert mean == pytest.approx(row[:n].mean(), abs=1e-12)
        assert variance == pytest.approx(row[:n].var(ddof=1), abs=1e-12)
    assert variances[0] == 0.0


def test_half_sample_summary_separates_monte_carlo_and_finite_pool_error():
    def outcome(rate, size=400):
        rejected = np.arange(size) < round(rate * size)
        return {"status": np.full(size, 1), "seeds": np.full(size, 10), "rejected": rejected,
                "wrong": np.zeros(size, bool), "truth": 1.0}
    summary = summarize_populations([outcome(0.5), outcome(0.7)], 8)
    monte_carlo = (0.25 * 400 / 399 + 0.21 * 400 / 399) / 400 / 2
    assert summary["workflow_rate"] == pytest.approx(0.6)
    assert summary["workflow_mc_se"] == pytest.approx(math.sqrt(monte_carlo / 2))
    finite = 0.01 - monte_carlo / 2
    assert summary["workflow_finite_pool_se"] == pytest.approx(math.sqrt(finite))
    assert summary["conditional_rate"] == pytest.approx(0.6)


def workflow_fixture(powers):
    rows = []
    for rule, (power, seeds) in powers.items():
        for pilot in sim.PILOT_SIZES:
            for family in sim.FAMILIES:
                rows.append({"rule": rule, "pilot_seeds": pilot, "truth_label": "meaningful",
                             "population": family, "workflow_rate": power,
                             "mean_total_seeds": seeds})
    return rows


def test_recommendation_is_the_cheapest_rule_that_holds_everywhere():
    rows = workflow_fixture({"observed_gap": (0.5, 100), "fixed_effect": (0.77, 150),
                             "variance_nct": (0.79, 170), "variance_ucl": (0.9, 230)})
    assert calibrate.recommend(rows)["rule"] == "variance_nct"
    rows = workflow_fixture({"observed_gap": (0.5, 100), "fixed_effect": (0.6, 150),
                             "variance_nct": (0.7, 170), "variance_ucl": (0.75, 230)})
    recommendation = calibrate.recommend(rows)
    assert (recommendation["rule"], recommendation["holds_everywhere"]) == ("variance_ucl", False)


def test_grid_moments_are_exact_rationals_rounded_once():
    rng = np.random.default_rng(12)
    counts = rng.multinomial(np.full(50, 16), sim.CEILING_CONTROL.pmf)
    means, variances = sim.CEILING_CONTROL.moments(counts)
    points = sim.CEILING_CONTROL.points
    for row, mean, variance in zip(counts, means, variances):
        values = [Fraction(int(j), 100) for j, c in zip(points, row) for _ in range(int(c))]
        exact_mean = sum(values) / len(values)
        exact_variance = sum((v - exact_mean) ** 2 for v in values) / (len(values) - 1)
        assert mean == float(exact_mean) and variance == float(exact_variance)


def test_bounded_null_pilot_ties_are_exact_zero_gaps_on_every_platform():
    # workflow.csv row 125: bounded null, 16-seed pilots. A float dot product found 8
    # zero gaps on x86-64 and 7 on arm64; integer counts find all 12.
    index, (family, label, arm_a, arm_b, _) = 6, list(sim.scenarios())[6]
    assert (family, label) == ("bounded", "null")
    rng = sim.stream(calibrate.SEED, (0, index, 1, 0))
    sizes = np.full(2000, 16)
    mean_a, var_a = arm_a.sample_moments(rng, sizes)
    mean_b, var_b = arm_b.sample_moments(rng, sizes)
    ties = np.flatnonzero(mean_a - mean_b == 0)
    assert ties.size == 12 and {591, 953, 1427, 1685} <= set(ties.tolist())
    _, status = sim.plan("observed_gap", mean_a - mean_b, var_a, var_b, 16)
    assert (status == sim.UNDEFINED).sum() == 12


def test_declared_nulls_report_no_wrong_sign_errors():
    # The shape null has equal means only to rounding; a rounding-sized gap is not a direction.
    family, label, arm_a, arm_b, effect = list(sim.scenarios())[-1]
    assert (family, label) == (sim.SHAPE_NULL, "null")
    assert abs(arm_a.mean - arm_b.mean) < 1e-12
    for row in sim.simulate(arm_a, arm_b, effect, 8, 300, 2, 11, (9, 8), null=True):
        assert row["wrong_sign_rate"] is None and row["wrong_sign_mc_se"] is None
