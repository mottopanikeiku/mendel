# Known-truth calibration of seed-planning rules

My fresh-seed experiments showed that an 80% pilot plan built from the observed pilot gap detected far less often than 80%. They could not say why, because I never knew the true gap. This benchmark fixes that. I simulate populations whose means and variances I know, run each planning rule from pilot to confirmation, and measure how often it detects a real effect, how often it rejects a true null, and how many seeds it spends. Every simulated number is labelled **SIM**. I then check the chosen rule on the committed PPO root scores by resampling them, which is not new training.

The generated [report](../results/calibration/report.md) starts with the recommendation and holds every number. This page describes the method.

## Reproduce

```sh
uv sync --locked --python 3.12
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 uv run python scripts/calibrate.py
uv run python scripts/calibrate.py --check
```

The first command needs no training, downloads or cloud compute. It writes `results/calibration/{configuration.json,reference.csv,workflow.csv,resampling.csv,report.md}` and `figures/known_truth.svg`. The second regenerates everything in memory, compares it with the committed files (floats to a relative 1e-12, except the `*_numerical_error` columns, which are differences of nearly equal floats and are compared to an absolute 1e-10; everything else exactly, the report byte for byte, and the SVG with numbers in drawn-geometry attributes (path data, positions, sizes, transforms and the view box) within a relative 1e-6 and all text, styles and other characters identical, because matplotlib's last coordinate digit and its hashed clip-path ids differ between x86-64 and arm64; ids are renumbered before comparing), checks that the README table matches, and checks the claims listed under [checks](#what---check-asserts). CI runs it on x86-64 and arm64. The root seed is 20261029; `configuration.json` records the NumPy PCG64 spawn-key recipe for every stream. I make no timing claims.

## The study each rule plans

A study runs `pilot` independent seeds per arm, applies a planning rule, then runs that many fresh seeds per arm and applies the two-sided Welch test at 5%. Every rule targets 80% power. Plans above 512 seeds per arm, the cap in my fresh-seed experiments, are not run and count as misses. A zero pilot gap or zero pilot variance has no plan and also counts as a miss. Pilot sizes are 8, 16, 32 and 64 seeds per arm. Total seed cost counts both arms, including the pilot.

## Rules

All rules use the repository's noncentral-t planner (`stats.plan_seeds`) or published variants of it. The first four see only the pilot and an effect fixed in advance.

- `observed_gap`: the pilot's absolute mean gap and unbiased variances. This is the rule behind my fresh-seed experiments.
- `fixed_effect`: the meaningful effect, with the pilot variances plugged in.
- `variance_ucl`: the meaningful effect, with both pilot variances scaled up to the one-sided 80% upper confidence limit on their sum. This is Browne's (1995) method at the level he recommends [1, 3].
- `variance_nct`: the meaningful effect and Julious and Owen's (2006) noncentral-t count [2, 3]. It picks the smallest n with n·effect² ≥ q²·(s²_a + s²_b), where q is the 80% quantile of a noncentral t with the pilot's degrees of freedom and noncentrality equal to the two-sided 5% critical value. This averages one-tailed power over the chi-square uncertainty of the pilot variance.
- `known_variance`: an oracle that plugs in the true variances. It shows what perfect variance knowledge buys under each population.

The published methods assume one pooled variance. I give the summed pilot variance Satterthwaite degrees of freedom, (pilot − 1)(s²_a + s²_b)²/(s⁴_a + s⁴_b). That equals the pooled 2·(pilot − 1) when the two pilot variances are equal, so the tests reproduce the published inflation tables exactly [3]. The main test keeps the planner's Welch degrees of freedom.

## Populations

Arm a is the treatment and arm b the control; the true gap is mean(a) − mean(b). In every family the meaningful effect is half the control SD, and each family has three truths: no difference, half the meaningful effect and the meaningful effect.

- **Gaussian:** N(effect, 1) against N(0, 1).
- **Gaussian, unequal:** N(effect, 2²) against N(0, 1).
- **Bounded with a ceiling:** scores on a 101-point grid in [0, 1]. The control has 10% mass at the ceiling and a discretized Beta(6, 2.5) body. The treatment moves a fraction of seeds to the ceiling.
- **Failure/success bimodal:** the control has 26% mass at 0 and a discretized bell near 0.8. The treatment turns a fraction of failures into successes.
- **Equal means, different shapes:** a unimodal grid bell against the bimodal control, with equal means. This null tests the Welch test when arms differ in skew and spread, as PPO variants can.

The two grid families are loosely shaped like normalized CartPole and Acrobot scores, which have a 500-step ceiling and a failure mass. They are stylized populations, not fits. `configuration.json` stores every probability mass function.

Sampling is exact. For Gaussian arms I draw each sample mean and variance from their exact normal and scaled chi-square laws. For grid arms I draw multinomial counts and compute moments from them; a constant sample has exactly zero variance, as in `stats.welch_test`.

## Exact references

For Gaussian arms I compute exact Welch rejection and wrong-sign probabilities at any n. The sample gap is normal and independent of the two sample variances, so I integrate the conditional normal rejection probability over both chi-square variances, including the data-dependent Welch degrees of freedom. I use 48 probabilists' Gauss–Hermite nodes mapped through probits. From 3 seeds per arm the quadrature error is below 1e-9. At 2 seeds per arm each sample variance has one degree of freedom, convergence is slower, and the error reaches 3.3e-5. I report this numerical error separately from Monte Carlo error, as each exact value's change when the nodes double to 96, and tests check that this estimate bounds the error against an independent adaptive integration. With the pooled-t critical value, the same quadrature reproduces noncentral-t power to 1e-7 at n = 2 and to rounding error from n = 5.

For equal-variance Gaussian arms I also compute each rule's whole plan distribution exactly. A plan depends on the pilot only through d = effect/√(s²_a + s²_b) and the variance share r = s²_a/(s²_a + s²_b). With equal true variances, r is Beta((pilot − 1)/2, (pilot − 1)/2) and independent of the summed variance and the pilot gap. A plan is at most n exactly when d reaches a threshold that depends on n and r. For a fixed effect that event is a chi-square CDF. For the observed gap, √pilot·d is the absolute pooled t statistic, which is noncentral t. Gauss–Jacobi nodes integrate r. Combining plan probabilities with exact Welch power gives exact workflow power, eligibility, mean cost and wrong-sign rate with no Monte Carlo error.

## Monte Carlo design

Each cell draws 2,000 pilots. Every rule plans from the same pilots, and each plan gets 8 independent confirmations from its own stream. I report workflow power (detections among all pilots), conditional power (among plans that ran), eligibility, wrong-sign rejections and mean total seeds. Standard errors treat the pilot as the cluster, with a delta-method SE for the conditional ratio. Fixed-count reference rows use 20,000 independent trials. Intervals are pointwise, not simultaneous across the grid.

## Resampling committed PPO roots

Each committed pilot or confirmation root is one independently trained PPO policy score, conditional on the fixed evaluation reset sets. I pool both phases per task and variant. Development roots are excluded. For each of four random splits, I divide every pool into two disjoint halves, and each half becomes a finite population with an exactly known mean gap and variance. That gives eight populations per comparison. Within a population, each of 500 replicates draws a pilot and then a confirmation from one permutation without replacement, so no root is in both. In same-pool comparisons, the two arms interleave one permutation and never share a root.

Per task there are three kinds of comparison. `null` is base against base, with a true gap of exactly zero. `shift` adds half the base SD to one base arm, a known effect on real score shapes. The four committed variant pairs are planned at their own population gap, as if I had chosen the true gap as the meaningful effect. A plan bigger than the roots left after the pilot is pool-limited and not run.

The spread of a rate across the eight half populations, minus its Monte Carlo noise, is a delete-half jackknife estimate of finite-pool variance. The reported total SE combines both. Replicates reuse roots, so this shows how the rules behave on these PPO score distributions. It is not fresh training and adds no independent seeds. The report also compares resampled observed-gap detection rates with the committed fresh-seed rates.

## How the recommendation is chosen

Code chooses the rule from the SIM tables only. A rule holds at a pilot size if its power at the meaningful effect is at least 78% (80% minus a 2-point tolerance) in all four SIM populations. Among rules that hold at every pilot size, the one with the lowest mean total seed cost is recommended. If none holds everywhere, the report names the rule with the best worst case and says so. The resampling results do not feed into the choice. "Holds" classifies simulated point estimates, each with Monte Carlo error. It does not show that a rule's true power is at least 78%, and the report lists the calls that fall within a 95% Monte Carlo half-width of the line.

## What `--check` asserts

- Fixed-count Monte Carlo rejection rates are within 4.5 binomial SEs, plus the exact value's numerical error, of exact Gaussian Welch power.
- For equal-variance Gaussian cells, simulated workflow power, eligibility and mean cost are within 4.5 SEs, plus numerical error, of their exact values.
- For all Gaussian cells, simulated power is within 4.5 SEs, plus numerical error, of the exact power of each pilot's own plan.
- Every pilot and every resampling replicate is accounted for as undefined, over the cap, pool-limited or run.
- The README's calibration table matches the report, and every percentage the README quotes on a line about the simulation or resampling appears in the report.

## Limits

- The populations are stylized and few. The meaningful effect is always half the control SD, so plans are moderate in size; much smaller effects, heavier tails or other shapes can behave differently.
- Every rule needs a meaningful effect fixed in advance. None protects against choosing that effect badly, and none finds a true effect that is smaller than planned.
- The variance-uncertainty rules adapt methods derived for a pooled variance and a normal or t test. Satterthwaite degrees of freedom for unequal pilot variances are my choice.
- Resampling reuses the committed roots, conditions on the fixed evaluation resets and covers two classic-control tasks with one small PPO architecture. It is not a prospective test of the recommended rule on newly trained seeds.

## References

1. Richard H. Browne. 1995. *On the use of a pilot sample for sample size determination*. Statistics in Medicine 14(17):1933–1940. [DOI 10.1002/sim.4780141709](https://doi.org/10.1002/sim.4780141709).
2. Steven A. Julious and Roger J. Owen. 2006. *Sample size calculations for clinical studies allowing for uncertainty about the variance*. Pharmaceutical Statistics 5(1):29–37. [PubMed 17080926](https://pubmed.ncbi.nlm.nih.gov/17080926).
3. Amy L. Whitehead, Steven A. Julious, Cindy L. Cooper and Michael J. Campbell. 2016. *Estimating the sample size for a pilot randomised trial to minimise the overall trial sample size for the external pilot and main trial for a continuous outcome variable*. Statistical Methods in Medical Research 25(3):1057–1073. [Open access](https://pmc.ncbi.nlm.nih.gov/articles/PMC4876429/). Section 2.1 describes both methods; Browne recommends the 80% limit. Tables 1 and 2 give the inflation factors my tests reproduce.
