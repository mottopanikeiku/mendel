"""Known-truth calibration of seed-planning rules: SIM, no training.

A study runs ``pilot`` independent seeds per arm, applies a planning rule,
then runs that many fresh seeds per arm and applies the two-sided Welch test
at ALPHA. The rules differ only in how they turn the pilot into a count:

- ``observed_gap``: the pilot's absolute mean gap and unbiased variances, the
  rule behind the committed fresh-seed experiments.
- ``fixed_effect``: a meaningful effect fixed before the pilot, with the
  pilot variances plugged in.
- ``variance_ucl``: the meaningful effect, with both pilot variances scaled to
  Browne's (1995) one-sided 80% upper confidence limit on their sum.
- ``variance_nct``: the meaningful effect and Julious and Owen's (2006)
  noncentral-t count, which averages one-tailed power over the pilot
  variance's chi-square uncertainty.
- ``known_variance``: an oracle comparator using the true variances.

The first four see only pilot data and the prespecified effect. Pilot
variances use Satterthwaite degrees of freedom for their sum, which equal the
pooled 2*(pilot-1) when the two pilot variances are equal. Counts above CAP
are not run and count as no detection. Pilots, plans and confirmations are
exact draws from known populations; Monte Carlo errors cluster at the pilot.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
import math

import numpy as np
from scipy.special import roots_hermitenorm, roots_jacobi
from scipy.stats import beta, chi2, nct, norm, t

from seed_power.stats import plan_seeds

ALPHA = 0.05
TARGET = 0.8
CAP = 512
UCL_LEVEL = 0.8
RULES = ("observed_gap", "fixed_effect", "variance_ucl", "variance_nct", "known_variance")
PILOT_RULES = RULES[:-1]
PILOT_SIZES = (8, 16, 32, 64)
TRUTHS = (("null", 0.0), ("half", 0.5), ("meaningful", 1.0))
# Quadrature nodes for the exact Gaussian references. calibrate.py reports each
# reference's numerical error as its change when this doubles.
NODES = 48
UNDEFINED, PLANNED, OVER_CAP = 0, 1, 2


@dataclass(frozen=True, eq=False)
class Normal:
    """Gaussian seed scores; sample moments are drawn from their exact laws."""

    mean: float
    sd: float

    @property
    def variance(self) -> float:
        return self.sd * self.sd

    def sample_moments(self, rng, n):
        n = np.asarray(n, dtype=np.int64)
        means = rng.normal(self.mean, self.sd / np.sqrt(n))
        variances = self.variance * rng.chisquare(n - 1) / (n - 1)
        return means, variances


@dataclass(frozen=True, eq=False)
class Grid:
    """Scores j/GRID_DENOMINATOR on a finite grid; multinomial counts give exact moments."""

    support: np.ndarray
    pmf: np.ndarray

    def __post_init__(self):
        points = self.support * GRID_DENOMINATOR
        if (self.support.shape != self.pmf.shape or self.pmf.min() < 0
                or not np.allclose(points, np.rint(points), rtol=0, atol=1e-9)
                or not math.isclose(float(self.pmf.sum()), 1.0, rel_tol=0, abs_tol=1e-12)):
            raise ValueError("grid needs probabilities on multiples of 1/GRID_DENOMINATOR")

    @property
    def points(self) -> np.ndarray:
        return np.rint(self.support * GRID_DENOMINATOR).astype(np.int64)

    @property
    def mean(self) -> float:
        return float(np.sum(self.pmf * self.support))

    @property
    def variance(self) -> float:
        return float(np.sum(self.pmf * (self.support - self.mean) ** 2))

    @property
    def sd(self) -> float:
        return math.sqrt(self.variance)

    def sample_moments(self, rng, n):
        return self.moments(rng.multinomial(np.asarray(n, dtype=np.int64), self.pmf))

    def moments(self, counts):
        """Sample means and unbiased variances from integer counts, each rounded once.

        Integer sums make equal samples give bitwise-equal means on every
        platform, so an exactly zero pilot gap or a constant sample never
        depends on floating-point summation order (BLAS dot products differ
        between x86-64 and arm64). Constant samples get exactly zero variance.
        """
        counts = np.asarray(counts, dtype=np.int64)
        n = counts.sum(axis=1)
        total = counts @ self.points
        squares = counts @ (self.points * self.points)
        means = total / (n * GRID_DENOMINATOR)
        variances = (n * squares - total * total) / (n * (n - 1) * GRID_DENOMINATOR**2)
        return means, variances


def mixture(first: Grid, second: Grid, weight: float) -> Grid:
    """(1-weight)*first + weight*second on a shared support."""
    if not np.array_equal(first.support, second.support) or not 0 <= weight <= 1:
        raise ValueError("mixtures need a shared support and a weight in [0, 1]")
    return Grid(first.support, (1 - weight) * first.pmf + weight * second.pmf)


GRID_DENOMINATOR = 100
SUPPORT = np.arange(GRID_DENOMINATOR + 1) / GRID_DENOMINATOR


def _normalized(weights):
    weights = np.asarray(weights, dtype=float)
    return weights / weights.sum()


def _point(value):
    return Grid(SUPPORT, (SUPPORT == value).astype(float))


def _bell(center, sd, floor=0.0):
    weights = np.exp(-0.5 * ((SUPPORT - center) / sd) ** 2) * (SUPPORT > floor)
    return Grid(SUPPORT, _normalized(weights))


def _bell_with_mean(target, sd):
    """Discretized bell whose grid mean equals ``target`` to rounding error."""
    low, high = target - 3 * sd, target + 3 * sd
    for _ in range(200):
        middle = (low + high) / 2
        if _bell(middle, sd, -1.0).mean < target:
            low = middle
        else:
            high = middle
    return _bell((low + high) / 2, sd, -1.0)


# A ceiling-heavy bounded score and a failure/success bimodal score, loosely
# shaped like normalized CartPole and Acrobot returns. They are stylized SIM
# populations, not fits to the committed PPO scores.
CEILING_BODY = Grid(SUPPORT, _normalized(beta.pdf(SUPPORT, 6.0, 2.5) * (SUPPORT < 1.0)))
CEILING_CONTROL = mixture(CEILING_BODY, _point(1.0), 0.10)
SUCCESS_MODE = _bell(0.80, 0.06, floor=0.5)
FAILURE_CONTROL = mixture(SUCCESS_MODE, _point(0.0), 0.26)


def ceiling_treatment(effect):
    """Move a fraction of control seeds to the ceiling, raising the mean by ``effect``."""
    return mixture(CEILING_CONTROL, _point(1.0), effect / (1.0 - CEILING_CONTROL.mean))


def failure_treatment(effect):
    """Turn a fraction of control seeds into successes, raising the mean by ``effect``."""
    weight = effect / (SUCCESS_MODE.mean - FAILURE_CONTROL.mean)
    return mixture(FAILURE_CONTROL, SUCCESS_MODE, weight)


# Arm a is the treatment and arm b the control; the true gap is mean(a)-mean(b).
# The meaningful effect is half the control SD in every family.
FAMILIES = {
    "gaussian": (lambda effect: Normal(effect, 1.0), Normal(0.0, 1.0)),
    "gaussian_unequal": (lambda effect: Normal(effect, 2.0), Normal(0.0, 1.0)),
    "bounded": (ceiling_treatment, CEILING_CONTROL),
    "multimodal": (failure_treatment, FAILURE_CONTROL),
}
SHAPE_NULL = "multimodal_vs_unimodal"


def meaningful_effect(family: str) -> float:
    control = FAMILIES[family][1]
    return 0.5 * control.sd


def scenarios():
    """Yield (population, truth label, arm a, arm b, meaningful effect)."""
    for family, (treatment, control) in FAMILIES.items():
        effect = meaningful_effect(family)
        for label, fraction in TRUTHS:
            yield family, label, treatment(fraction * effect), control, effect
    # Equal means, different shapes: Welch's null when arms differ in skew and spread.
    yield (SHAPE_NULL, "null", _bell_with_mean(FAILURE_CONTROL.mean, 0.10), FAILURE_CONTROL,
           meaningful_effect("multimodal"))


# Planning: vectorized versions of seed_power.stats, on standardized effects.

def _df_factor(share):
    return 1.0 / (share * share + (1.0 - share) ** 2)


def two_sided_nct(critical, df, nc):
    """P(|T| > critical) for noncentral t with nc >= 0, elementwise.

    P(T < -c) <= P(Z < -nc), so that tail is skipped wherever it cannot change
    the float sum. Where SciPy still returns NaN, integrate the defining
    normal/chi-square representation instead, as stats.power_at_n does.
    """
    critical, df, nc = np.broadcast_arrays(*(np.asarray(v, float) for v in (critical, df, nc)))
    with np.errstate(invalid="ignore"):
        power = np.array(nct.sf(critical, df, nc), dtype=float)
        needed = norm.cdf(-nc) > power * 2.0**-54
        power[needed] += nct.cdf(-critical[needed], df[needed], nc[needed])
    failed = ~np.isfinite(power)
    if failed.any():
        c, k, m = critical[failed, None], df[failed, None], nc[failed, None]
        values, weights = chi2_nodes(k)
        threshold = c * np.sqrt(values / k)
        power[failed] = (norm.sf(threshold - m) + norm.cdf(-threshold - m)) @ weights
    return power


def planning_power(d, share, n, alpha=ALPHA):
    """stats.power_at_n for effect d, variances share and 1-share, n seeds per arm."""
    d, share, n = np.broadcast_arrays(np.asarray(d, float), np.asarray(share, float),
                                      np.asarray(n, float))
    df = (n - 1.0) * _df_factor(share)
    return np.clip(two_sided_nct(t.isf(alpha / 2.0, df), df, d * np.sqrt(n)), 0.0, 1.0)


def _smallest_count(criterion, size, cap):
    """Smallest n in [2, cap] with a monotone criterion; cap+1 when none qualifies."""
    low, high = np.ones(size, np.int64), np.full(size, cap + 1, np.int64)
    while True:
        active = np.flatnonzero(high - low > 1)
        if active.size == 0:
            return high
        middle = (low[active] + high[active]) // 2
        passed = criterion(active, middle)
        high[active[passed]] = middle[passed]
        low[active[~passed]] = middle[~passed]


def ucl_factor(pilot, share, level=UCL_LEVEL):
    """Browne's one-sided upper confidence limit for summed pilot variances, as a factor."""
    df = (pilot - 1.0) * _df_factor(np.asarray(share, float))
    return df / chi2.ppf(1.0 - level, df)


def plan(rule, effect, variance_a, variance_b, pilot, cap=CAP, alpha=ALPHA, target=TARGET):
    """Return (seeds per arm, status) arrays; seeds are 0 unless status is PLANNED.

    ``effect`` is the planning gap: the absolute pilot gap for observed_gap and
    the meaningful effect otherwise. Zero effect or zero total variance is
    UNDEFINED, as in stats.plan_seeds. Counts above ``cap`` are OVER_CAP.
    """
    effect = np.abs(np.asarray(effect, float))
    variance_a = np.asarray(variance_a, float)
    variance_b = np.asarray(variance_b, float)
    effect, variance_a, variance_b = np.broadcast_arrays(effect, variance_a, variance_b)
    total = variance_a + variance_b
    defined = (effect > 0) & (total > 0)
    safe_total = np.where(defined, total, 1.0)
    d = np.where(defined, effect, 1.0) / np.sqrt(safe_total)
    share = np.where(defined, variance_a, 0.5) / safe_total
    if rule in ("observed_gap", "fixed_effect", "known_variance"):
        scale = np.ones_like(d)
    elif rule == "variance_ucl":
        scale = ucl_factor(pilot, share)
    elif rule == "variance_nct":
        scale = None
    else:
        raise ValueError(f"unknown rule {rule}")
    d, share = d.ravel(), share.ravel()
    if scale is None:
        pilot_df = (pilot - 1.0) * _df_factor(share)
        factor = _df_factor(share)

        def criterion(index, n):
            critical = t.isf(alpha / 2.0, (n - 1.0) * factor[index])
            return nct.cdf(d[index] * np.sqrt(n), pilot_df[index], critical) >= target
    else:
        standardized = d / np.sqrt(np.ravel(scale))

        def criterion(index, n):
            return planning_power(standardized[index], share[index], n, alpha) >= target

    seeds = _smallest_count(criterion, d.size, cap).reshape(defined.shape)
    status = np.where(~defined, UNDEFINED, np.where(seeds > cap, OVER_CAP, PLANNED))
    return np.where(status == PLANNED, seeds, 0), status


def welch_rejections(mean_a, variance_a, mean_b, variance_b, n, alpha=ALPHA):
    """Vectorized stats.welch_test decision for n seeds per arm; returns (reject, gap)."""
    gap = mean_a - mean_b
    component = (variance_a + variance_b) / n
    both_constant = (variance_a == 0) & (variance_b == 0)
    safe = np.where(both_constant, 1.0, component)
    df = safe**2 * (n - 1) / ((variance_a / n) ** 2 + (variance_b / n) ** 2 + both_constant)
    pvalue = 2.0 * t.sf(np.abs(gap) / np.sqrt(safe), df)
    # Two constant arms: distinct constants reject (p=0), identical ones never do (p=1).
    return np.where(both_constant, gap != 0, pvalue < alpha), gap


# Exact Gaussian references: deterministic quadrature, no Monte Carlo error but a
# small numerical error, which calibrate.py reports separately.

def chi2_nodes(df, nodes=NODES):
    """Chi-square quadrature: probabilists' Gauss-Hermite nodes mapped through probits.

    A chi-square variable is F^-1(Phi(Z)) for standard normal Z, so smooth
    bounded integrands converge spectrally. Each tail uses its own quantile
    function, so no node rounds to probability 0 or 1.
    """
    z, w = roots_hermitenorm(nodes)
    values = np.where(z < 0, chi2.ppf(norm.cdf(z), df), chi2.isf(norm.sf(z), df))
    return values, w / math.sqrt(2.0 * math.pi)


def exact_welch(gap, variance_a, variance_b, n, alpha=ALPHA, nodes=NODES, pooled=False):
    """Exact two-sided Welch rejection and wrong-sign probabilities for Gaussian arms.

    Sample means and variances are independent: the gap is normal and each
    variance a scaled chi-square with n-1 df. Product quadrature over the two
    chi-square variables integrates the conditional normal rejection
    probability, including the data-dependent Welch df. ``pooled`` swaps in the
    pooled-t critical value with 2n-2 df, whose exact power is noncentral t.
    """
    q, w = chi2_nodes(n - 1, nodes)
    # With many nodes, the outermost chi-square quantiles underflow to 0 or overflow.
    # Their total weight is negligible, and they would make the Welch df 0/0.
    representable = (q > 0) & np.isfinite(q)
    if w[~representable].sum() > 1e-15:
        raise FloatingPointError("dropped quadrature nodes carry non-negligible weight")
    q, w = q[representable] / (n - 1), w[representable]
    sample_a, sample_b = np.meshgrid(variance_a * q, variance_b * q, indexing="ij")
    weight = np.outer(w, w)
    se = np.sqrt((sample_a + sample_b) / n)
    # Normalize before squaring so tiny sampled variances do not underflow to 0/0.
    largest = np.maximum(sample_a, sample_b)
    ratio_a, ratio_b = sample_a / largest, sample_b / largest
    df = 2.0 * n - 2.0 if pooled else (n - 1) * (ratio_a + ratio_b) ** 2 / (
        ratio_a**2 + ratio_b**2)
    threshold = t.isf(alpha / 2.0, df) * se
    spread = math.sqrt((variance_a + variance_b) / n)
    upper = float((weight * norm.sf((threshold - gap) / spread)).sum())
    lower = float((weight * norm.cdf((-threshold - gap) / spread)).sum())
    if not math.isfinite(upper + lower):
        raise FloatingPointError("Welch quadrature produced a non-finite probability")
    wrong = None if gap == 0 else (lower if gap > 0 else upper)
    return upper + lower, wrong


def exact_welch_table(gap, variance_a, variance_b, cap=CAP, alpha=ALPHA, nodes=NODES):
    """Exact (rejection, wrong-sign) arrays indexed by n = 0..cap; n < 2 are unused."""
    rejection, wrong = np.zeros(cap + 1), np.zeros(cap + 1)
    for n in range(2, cap + 1):
        rejection[n], wrong_n = exact_welch(gap, variance_a, variance_b, n, alpha, nodes)
        wrong[n] = 0.0 if wrong_n is None else wrong_n
    return rejection, wrong


@cache
def _share_nodes(pilot, nodes):
    """Gauss-Jacobi nodes and weights for the variance share r ~ Beta(m/2, m/2)."""
    m = pilot - 1
    x, w = roots_jacobi(nodes, m / 2 - 1, m / 2 - 1)
    return (x + 1) / 2, w / w.sum()


@cache
def _gap_thresholds(pilot, nodes, cap, alpha=ALPHA, target=TARGET):
    """Smallest standardized effect d whose plain plan is at most each count.

    Plans depend on the pilot only through d = effect/sqrt(total variance) and
    the variance share. Rows index counts 2..cap, columns share nodes.
    """
    share, counts = np.meshgrid(_share_nodes(pilot, nodes)[0], np.arange(2, cap + 1))
    # nc = (critical + 1) * 4 already exceeds 80% power at every df >= 1.
    critical = t.isf(alpha / 2.0, (counts - 1.0) * _df_factor(share))
    low, high = np.zeros_like(share), 4.0 * (critical + 1.0) / np.sqrt(counts)
    if not (planning_power(high, share, counts, alpha) >= target).all():
        raise FloatingPointError("threshold bracket does not reach the target power")
    for _ in range(60):
        middle = (low + high) / 2.0
        reached = planning_power(middle, share, counts, alpha) >= target
        high, low = np.where(reached, middle, high), np.where(reached, low, middle)
    return high


def _thresholds(rule, pilot, nodes, cap, alpha=ALPHA, target=TARGET):
    """Rule-specific thresholds: n_hat <= n exactly when d >= threshold(n, r)."""
    share = _share_nodes(pilot, nodes)[0]
    if rule == "variance_nct":
        share, counts = np.meshgrid(share, np.arange(2, cap + 1))
        factor = _df_factor(share)
        quantile = nct.ppf(target, (pilot - 1.0) * factor,
                           t.isf(alpha / 2.0, (counts - 1.0) * factor))
        return quantile / np.sqrt(counts)
    threshold = _gap_thresholds(pilot, nodes, cap, alpha, target)
    if rule == "variance_ucl":
        return threshold * np.sqrt(ucl_factor(pilot, share))
    if rule in ("observed_gap", "fixed_effect"):
        return threshold
    raise ValueError(f"no threshold form for {rule}")


def exact_plan_distribution(rule, pilot, sd, effect, truth, cap=CAP, nodes=NODES):
    """P(plan = n) for n = 0..cap and P(over cap), equal-variance Gaussian arms.

    With equal true variances, the pooled pilot variance (2m df, m = pilot-1)
    and the pilot gap g ~ N(truth, 2 sd^2/pilot) are independent of the share
    r ~ Beta(m/2, m/2) of the summed variance S. The plan is at most n exactly
    when d >= threshold(n, r). For a fixed effect, d = effect/sqrt(S) and that
    event is a chi-square CDF. For the observed gap, sqrt(pilot)*d is the
    absolute pooled t statistic, noncentral t with 2m df. Gauss-Jacobi nodes
    integrate r.
    """
    m = pilot - 1
    share_weight = _share_nodes(pilot, nodes)[1]
    threshold = _thresholds(rule, pilot, nodes, cap)
    if rule == "observed_gap":
        noncentrality = abs(truth) / (sd * math.sqrt(2.0 / pilot))
        conditional = two_sided_nct(threshold * math.sqrt(pilot), 2 * m, noncentrality)
    else:
        conditional = chi2.cdf(m * effect**2 / (sd * sd * threshold**2), 2 * m)
    cdf = conditional @ share_weight
    if not np.isfinite(cdf).all():
        raise FloatingPointError("SciPy could not evaluate the exact plan distribution")
    pmf = np.zeros(cap + 1)
    # Clip last-bit negative differences of a monotone CDF.
    pmf[2:] = np.maximum(np.diff(np.concatenate(([0.0], cdf))), 0.0)
    return pmf, 1.0 - cdf[-1]


# Monte Carlo.

def _cluster_mean(values):
    values = np.asarray(values, float)
    return float(values.mean()), float(values.std(ddof=1) / math.sqrt(values.size))


def _ratio(numerator, denominator):
    """Ratio of cluster means with a delta-method SE; None without denominator."""
    total = float(denominator.sum())
    if total == 0:
        return None, None
    ratio = float(numerator.sum()) / total
    residual = numerator - ratio * denominator
    se = float(residual.std(ddof=1) * math.sqrt(residual.size) / total)
    return ratio, se


def stream(root, key):
    """Generator for an explicit SeedSequence spawn key, independent of call order."""
    return np.random.default_rng(np.random.SeedSequence(root, spawn_key=tuple(key)))


def simulate(arm_a, arm_b, effect, pilot, outer, inner, root, key, exact=None,
             null=False):
    """Simulate every rule on shared pilots; confirmation streams differ by rule.

    Spawn key ``key + (0,)`` draws pilots and ``key + (1 + i,)`` confirms rule i.
    Each pilot gets ``inner`` independent confirmations of its planned count.
    ``exact`` optionally maps n to exact Gaussian rejection probabilities. A
    declared ``null`` has no direction, so wrong-sign fields stay empty even
    when the two means differ by rounding.
    """
    pilot_rng = stream(root, (*key, 0))
    sizes = np.full(outer, pilot)
    mean_a, variance_a = arm_a.sample_moments(pilot_rng, sizes)
    mean_b, variance_b = arm_b.sample_moments(pilot_rng, sizes)
    truth = arm_a.mean - arm_b.mean
    rows = []
    for index, rule in enumerate(RULES):
        if rule == "known_variance":
            count = plan_seeds(effect, arm_a.variance, arm_b.variance, TARGET, ALPHA)
            planned = count is not None and count <= CAP
            seeds = np.full(outer, count if planned else 0)
            status = np.full(outer, PLANNED if planned else OVER_CAP)
        else:
            planning_effect = mean_a - mean_b if rule == "observed_gap" else effect
            seeds, status = plan(rule, planning_effect, variance_a, variance_b, pilot)
        executed = status == PLANNED
        rng = stream(root, (*key, 1 + index))
        rejected = np.zeros(outer)
        wrong = np.zeros(outer)
        if executed.any():
            repeated = np.repeat(seeds[executed], inner)
            ca, va = arm_a.sample_moments(rng, repeated)
            cb, vb = arm_b.sample_moments(rng, repeated)
            reject, gap = welch_rejections(ca, va, cb, vb, repeated)
            rejected[executed] = reject.reshape(-1, inner).mean(axis=1)
            if not null:
                opposite = reject & (np.sign(gap) != np.sign(truth))
                wrong[executed] = opposite.reshape(-1, inner).mean(axis=1)
        cost = 2 * pilot + 2 * seeds
        rate, rate_se = _cluster_mean(rejected)
        eligibility, eligibility_se = _cluster_mean(executed)
        conditional, conditional_se = _ratio(rejected, executed.astype(float))
        wrong_rate, wrong_se = _cluster_mean(wrong)
        mean_cost, cost_se = _cluster_mean(cost)
        quartiles = (np.quantile(seeds[executed], [0.25, 0.5, 0.75]).tolist()
                     if executed.any() else [None] * 3)
        row = {
            "rule": rule, "pilots": outer, "confirmations_per_pilot": inner,
            "undefined_plans": int((status == UNDEFINED).sum()),
            "over_cap_plans": int((status == OVER_CAP).sum()),
            "executed_plans": int(executed.sum()),
            "eligibility": eligibility, "eligibility_mc_se": eligibility_se,
            "workflow_rate": rate, "workflow_mc_se": rate_se,
            "conditional_rate": conditional, "conditional_mc_se": conditional_se,
            "wrong_sign_rate": None if null else wrong_rate,
            "wrong_sign_mc_se": None if null else wrong_se,
            "mean_total_seeds": mean_cost, "mean_total_seeds_mc_se": cost_se,
            "executed_seeds_q25": quartiles[0], "executed_seeds_median": quartiles[1],
            "executed_seeds_q75": quartiles[2],
            "conditional_exact_rate": None, "conditional_exact_mc_se": None,
        }
        if exact is not None:
            # Exact power of each pilot's own plan: checks the confirmation simulator.
            power = np.where(executed, exact[seeds], 0.0)
            row["conditional_exact_rate"] = float(power.mean())
            row["conditional_exact_mc_se"] = float(
                math.sqrt((power * (1 - power)).sum() / inner) / outer)
        rows.append(row)
    return rows
