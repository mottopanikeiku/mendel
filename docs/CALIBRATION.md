# Comparing seed-planning rules

Fixing the target effect and allowing for uncertainty in the pilot variance kept power near 80%. Planning from the pilot's observed gap often fell short. I compared five rules in simulation, checked the Gaussian cases by numerical integration, and repeated the comparison by resampling PPO scores. The [report](../results/calibration/report.md) contains the results.

## Study design

Each study draws a pilot, chooses a seed count, then tests two new groups with a two-sided Welch test at 5%. Every rule targets 80% power. Pilot sizes are 8, 16, 32 and 64 seeds per arm. Total cost includes both arms and the pilot.

Plans above 512 seeds per arm are not run and count as misses. A zero pilot gap or zero total pilot variance gives no plan and also counts as a miss. The report gives both power across all pilots and detection frequency among plans that ran.

## Planning rules

- `observed_gap`: use the pilot's absolute mean gap and unbiased variances. This was the rule used in the PPO and affine training experiments.
- `fixed_effect`: choose the meaningful effect before the pilot and plug in the pilot variances.
- `variance_ucl`: use the fixed effect and scale both pilot variances to the one-sided 80% upper confidence limit on their sum. This is Browne's method [1, 3].
- `variance_nct`: use the fixed effect and Julious and Owen's noncentral-t count [2, 3]. Choose the smallest n satisfying n·effect² ≥ q²·(s²_a + s²_b), where q is the 80% quantile of a noncentral t with the pilot's degrees of freedom and noncentrality equal to the planned test's two-sided 5% critical value.
- `known_variance`: use the fixed effect and the population variances as a reference.

The variance-uncertainty methods were derived for one pooled variance. I use Satterthwaite degrees of freedom for the summed pilot variance:

(pilot − 1)(s²_a + s²_b)² / (s⁴_a + s⁴_b).

For equal pilot variances this reduces to 2·(pilot − 1), and the tests reproduce the published inflation tables [3]. The planned test uses Welch degrees of freedom. The noncentral-t rule averages one-tailed power over chi-square uncertainty in the pilot variance.

## Score populations

Arm a is the treatment; arm b is the control. The meaningful effect is half the control SD. Each family has three true gaps: zero, half the meaningful effect and the meaningful effect.

| Population | Control | Treatment |
|---|---|---|
| Gaussian | N(0, 1) | N(effect, 1) |
| Gaussian, unequal variance | N(0, 1) | N(effect, 2²) |
| Bounded with a ceiling | Discretized Beta(6, 2.5), with 10% mass at 1 | Move a fraction of scores to the ceiling |
| Failure/success mixture | 26% mass at 0 and a discretized bell near 0.8 | Turn a fraction of failures into successes |

The two bounded populations use a 101-point grid in [0, 1], loosely resembling the ceiling and failure mass in control-task returns. A fifth, null-only comparison matches the mixture's mean with a unimodal bell to test equal means with different shapes. [configuration.json](../results/calibration/configuration.json) stores the probability masses.

For Gaussian scores I draw sample means and variances from their normal and scaled chi-square distributions. For grid scores I draw multinomial counts and compute moments from them. Constant samples have exactly zero variance, matching `stats.welch_test`.

## Simulation and exact comparisons

Each population, gap and pilot-size combination uses 2,000 pilots. All rules share those pilots; each resulting plan gets eight separate confirmation draws. Standard errors treat the pilot as the cluster, with a delta-method error for the conditional detection rate. Fixed-count reference rows use 20,000 independent trials.

For Gaussian arms, the sample gap is normal and independent of the two sample variances. I integrate the conditional rejection probability over both chi-square variances, including the sample-dependent Welch degrees of freedom. This also gives wrong-sign probabilities.

The integration uses 48 probabilists' Gauss–Hermite nodes mapped through probits. Doubling to 96 nodes estimates numerical error separately from Monte Carlo error. That error reaches 3.3e-5 at two seeds per arm and stays below 1e-9 from three seeds onward. Tests compare the estimate with adaptive integration. Using a pooled-t critical value reproduces noncentral-t power to 1e-7 at two seeds and to rounding error from five seeds onward.

For equal population variances I also integrate each rule's entire plan distribution. A plan depends on d = effect/√(s²_a + s²_b) and the variance share r = s²_a/(s²_a + s²_b). The share follows Beta((pilot − 1)/2, (pilot − 1)/2), independently of the summed variance and pilot gap. A fixed-effect plan's count threshold is a chi-square CDF; an observed-gap plan's threshold is an absolute noncentral-t probability. Gauss–Jacobi integration over r gives power, eligibility, mean cost and wrong-sign rates without simulation noise.

## Resampling PPO scores

I pool pilot and confirmation scores by task and variant, excluding development runs. Each score comes from one trained policy evaluated on the task's fixed reset set. Four random splits divide each pool into disjoint halves, giving eight finite populations per comparison.

For each population, 500 replicates draw a pilot and a confirmation from one permutation without replacement. Same-pool comparisons interleave the two arms within that permutation, so arms and phases never share a policy in a replicate.

There are three comparison types:

- `null`: base PPO against itself, with a true gap of zero.
- `shift`: add half the base SD to one base arm.
- `variant`: compare each PPO variant with base, using the half-population's mean gap as the target effect.

A plan that needs more policies than remain after the pilot is not run. The variation across half-populations, after subtracting Monte Carlo noise, gives a delete-half jackknife estimate of finite-pool variance. Total standard errors combine these sources. The report also compares resampled observed-gap detection rates with the training experiment's rates.

## Choosing the recommended rule

I select the lowest-cost rule whose simulated power reaches 78% in all four populations at all four pilot sizes. The threshold allows two percentage points below the 80% target. If no rule meets it, the script reports the rule with the highest minimum power. Resampling results do not affect this selection.

The threshold applies to estimated power. The report identifies cases whose 95% Monte Carlo intervals cross it; this includes the direct pilot-variance rule with 32-seed pilots and unequal Gaussian variances.

## Recompute and check

```sh
uv sync --locked --python 3.12
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
uv run python scripts/calibrate.py
uv run python scripts/calibrate.py --check
```

The generator writes the report, three CSV tables, configuration and `figures/known_truth.svg`. The seed is 20261029; configuration includes the NumPy PCG64 stream definitions. `--check` regenerates these outputs in memory and checks:

- Fixed-count and full-study Gaussian simulations agree with the integration results within 4.5 Monte Carlo standard errors plus numerical error.
- Every pilot and resampling replicate is accounted for as undefined, over the cap, limited by the score pool, or run.
- The README table matches the calculated table.
- Saved numeric outputs match to relative 1e-12; numerical-error columns use absolute 1e-10. Report text matches exactly. SVG geometry allows relative 1e-6 drift, while text and styles match exactly; clip-path IDs are normalized.

CI runs these checks on x86-64 and arm64.

## Limitations

- The simulation covers a few score shapes and one target effect size. Smaller effects, heavier tails or other shapes may behave differently.
- All fixed-effect rules depend on the chosen effect. An overoptimistic target leaves the study underpowered.
- Satterthwaite degrees of freedom extend methods originally derived for a pooled variance.
- Resampling reuses policies across replicates and conditions on fixed evaluation resets. It does not replace training new policies to test the recommended rule.
- Monte Carlo intervals are pointwise across the parameter combinations.

## References

1. Richard H. Browne (1995). [On the use of a pilot sample for sample size determination](https://doi.org/10.1002/sim.4780141709).
2. Steven A. Julious and Roger J. Owen (2006). [Sample size calculations for clinical studies allowing for uncertainty about the variance](https://pubmed.ncbi.nlm.nih.gov/17080926).
3. Amy L. Whitehead et al. (2016). [Estimating the sample size for a pilot randomised trial to minimise the overall trial sample size](https://pmc.ncbi.nlm.nih.gov/articles/PMC4876429/). Section 2.1 describes both methods; Tables 1 and 2 give the inflation factors used in the tests.
