# Null-control diagnostics for the neural PPO confirmation

`scripts/null_diagnostics.py` writes this report, [null_diagnostics.json](null_diagnostics.json) and the [figure](../../figures/null_controls.svg) from the committed raw roots; `--check` regenerates all three and compares them with the committed files. Observed outcomes and root scores are measurements. Test sizes at the planned counts come from resampling committed base roots that the null controls did not use, which is not new training. The detection probability for a block effect is a simulation (SIM). I chose these checks after I saw the CartPole excess, so every p-value here is exploratory and unadjusted unless I say otherwise.

## Summary

CartPole-v1 rejected 5/36 base-versus-base controls with the committed Welch test. If every test had exactly 5% size, 5 or more rejections would have probability 0.032. The 36 z-statistics are also over-dispersed: their squares sum to 55.112. As an approximate reference that assumes independent, exactly uniform p-values, a chi-square with 36 degrees of freedom gives p = 0.022. Welch p-values on these skewed scores are only approximately uniform, so I also calibrated both statistics empirically: redrawing all 36 controls from the held-out base roots 10000 times gives P(sum of z² ≥ 55.112) = 0.021 (Monte Carlo SE 0.0014) and P(5 or more rejections) = 0.033 (SE 0.0018). Acrobot-v1 rejected 2/38 (tail 0.573; squares 34.7, chi-square reference p = 0.625, resampled p = 0.615). Both tasks together rejected 7/74 (tail 0.077). Adjusted only for looking at two tasks (Bonferroni), not for the other exploratory checks here, the CartPole tail is 0.065 at exact 5% size and 0.066 resampled, and its dispersion p-value is 0.043 against the chi-square reference and 0.042 resampled.

These diagnostics did not identify an explanation for the CartPole excess:

- **Other tests.** On the same 36 controls, Student's t rejected 5, a permutation test of the mean gap 6, Mann-Whitney 5 and Yuen's 20% trimmed test 3. Elevated rejection counts also appear under tests that do not use the t approximation.
- **Test size at the planned counts.** Drawing both arms from 11091 held-out base roots at each control's planned count, Welch should reject 1.80 of the 36 (Student 1.81, Mann-Whitney 1.76, Yuen 1.84). With those sizes, 5 or more Welch rejections has probability 0.033. Resampled IID draws from this empirical pool gave sizes near nominal at these counts.
- **Execution.** These checks detected no difference between 247 training batches (permutation p = 0.323), between arm 0 and arm 1 of the controls (Welch p = 0.403), between pilot and confirmation roots (KS p = 0.975), or between control roots and other base roots (KS p = 0.714). They cannot rule out small differences. Every CartPole base-variant batch came from the same transport commit.
- **Seed blocks.** The over-dispersion is what a small random shift shared by all roots of one arm would produce. A shift variance of 0.0054 times the root variance would explain it on average. In the 175 base blocks outside the controls (8691 roots), a one-way ANOVA gives F = 1.170 (permutation p = 0.060, Kruskal-Wallis p = 0.093) and an estimated share of 0.0035. A share of 0.0054 would have given p < 0.05 in 68.5% of 1000 simulations (95% Monte Carlo half-width 2.9 points). At the estimated share, a normal approximation puts the expected Welch rejections at 3.24 instead of 1.80. I know of no mechanism that gives consecutive root seeds a shared shift, and the batch check, which also groups consecutive seeds, detected no difference.

So these diagnostics did not identify an explanation. It may be chance: adjusted only for the two tasks, the resampled tail is 0.066 and the resampled dispersion p-value is 0.042. It may also be a small seed-block effect, which these data neither confirm nor rule out. The Acrobot-v1 checks gave no unusual result.

## Null controls by task

| Task | Executed | Welch rejections | P(at least this many), exact 5% size | P(at least this many), resampled | Sum of z² | Chi-square reference p | Resampled p | KS p against uniform (reference) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CartPole-v1 | 36 | 5 | 0.032 | 0.033 | 55.112 | 0.022 | 0.021 | 0.105 |
| Acrobot-v1 | 38 | 2 | 0.573 | 0.569 | 34.661 | 0.625 | 0.615 | 0.485 |

Each z-statistic is the normal quantile of a two-sided Welch p-value. The chi-square (36 and 38 degrees of freedom) and KS columns are approximate references: they assume independent, exactly uniform p-values, which Welch on these skewed scores only approximates. The resampled columns redraw all controls together 10000 times, both arms with replacement from the held-out base roots at each control's planned count, and assume roots are independent draws from that pool. Their Monte Carlo SE is at most 0.0050. The resampled sum of z² averages 36.0 for CartPole-v1 and 37.9 for Acrobot-v1, close to the chi-square means.

## Every test on every executed cell

| Task | Test | Null rejections | Different-variant detections |
|---|---|---:|---:|
| CartPole-v1 | Welch t (the committed test) | 5/36 | 109/175 |
| CartPole-v1 | Student pooled t | 5/36 | 115/175 |
| CartPole-v1 | Permutation of the mean gap | 6/36 | 49/175 |
| CartPole-v1 | Mann-Whitney U | 5/36 | 39/175 |
| CartPole-v1 | Yuen 20% trimmed means | 3/36 | 95/175 |
| Acrobot-v1 | Welch t (the committed test) | 2/38 | 30/132 |
| Acrobot-v1 | Student pooled t | 2/38 | 30/132 |
| Acrobot-v1 | Permutation of the mean gap | 2/38 | 31/132 |
| Acrobot-v1 | Mann-Whitney U | 1/38 | 36/132 |
| Acrobot-v1 | Yuen 20% trimmed means | 2/38 | 30/132 |

No test was undefined on any cell. The permutation test uses 4000 random relabelings. With 2 or 3 seeds per arm its exact p-value is at least 0.333 or 0.100, so it cannot reach p < 0.05. The asymptotic, tie-corrected Mann-Whitney p-value I use can, but only with 3 seeds per arm when each arm is constant and the two differ (p = 0.047). 69 of the 175 CartPole different-variant plans used 2 or 3 seeds. On those plans Welch detected 60, the permutation test 0 and Mann-Whitney 0; on the other 106 they detected 49, 49 and 39.

## Test size from held-out roots

| Task | Held-out base roots | Expected Welch rejections | Student | Mann-Whitney | Yuen | P(at least the observed Welch count) |
|---|---:|---:|---:|---:|---:|---:|
| CartPole-v1 | 11091 | 1.80 | 1.81 | 1.76 | 1.84 | 0.033 |
| Acrobot-v1 | 19340 | 1.91 | 1.92 | 1.85 | 1.91 | 0.577 |

The held-out roots are all pilot base roots plus confirmation base roots outside the null controls. For each control, both arms draw its planned count with replacement from that pool, 4000 times; the expected count sums the resulting sizes over the controls. The last column treats those sizes as exact. The permutation test's size is at most 5% by construction, so I do not resample it.

Size at fixed counts (rejection rate with both arms from the held-out pool):

| Seeds per arm | CartPole Welch | CartPole Student | CartPole Mann-Whitney | CartPole Yuen | Acrobot Welch | Acrobot Student | Acrobot Mann-Whitney | Acrobot Yuen |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 2.9% | 6.2% | 0.0% | 2.9% | 6.7% | 10.2% | 0.0% | 6.7% |
| 3 | 4.1% | 6.0% | 0.0% | 4.1% | 3.2% | 4.5% | 0.0% | 3.2% |
| 5 | 4.2% | 5.3% | 3.2% | 4.8% | 3.9% | 5.9% | 3.6% | 5.8% |
| 8 | 4.5% | 4.7% | 3.5% | 4.6% | 4.9% | 5.4% | 4.2% | 3.9% |
| 16 | 4.4% | 4.5% | 4.3% | 4.9% | 5.2% | 5.3% | 5.2% | 5.1% |
| 32 | 4.5% | 4.6% | 5.0% | 4.9% | 5.5% | 5.5% | 5.2% | 6.0% |
| 64 | 5.3% | 5.3% | 5.0% | 5.0% | 5.6% | 5.6% | 5.4% | 5.8% |
| 128 | 4.5% | 4.5% | 4.7% | 4.8% | 5.1% | 5.1% | 5.1% | 5.1% |
| 256 | 5.2% | 5.2% | 4.8% | 4.8% | 4.9% | 4.9% | 4.8% | 4.9% |
| 512 | 4.5% | 4.5% | 4.6% | 4.6% | 4.7% | 4.7% | 5.2% | 4.7% |

Each rate uses 4000 draws, so its Monte Carlo standard error near 5% is about 0.3%.

## Execution checks

| Check | CartPole-v1 | Acrobot-v1 |
|---|---:|---:|
| Training batches with at least two base roots | 247 | 380 |
| Batch ANOVA F, permutation p | 1.040, 0.323 | 0.849, 0.984 |
| Batch Kruskal-Wallis p | 0.353 | 0.968 |
| Transport commits that produced base batches | 1 | 2 |
| Between those commits: permutation p | one commit | 0.151 |
| Null controls, arm 0 vs arm 1 means | 400.2 vs 398.6 | -247.3 vs -245.7 |
| Arm 0 vs arm 1: Welch p, KS p | 0.403, 0.770 | 0.693, 0.294 |
| Pilot vs confirmation base roots: KS p | 0.975 | 0.723 |
| Null-control vs other base roots: KS p | 0.714 | 0.137 |

A training batch is one vmapped job of up to 64 consecutive seeds of one task and variant, matched to the recorded batches in the confirmation environment file. Permutation p-values regroup the roots at random 4000 times.

## Seed-block effects

| | CartPole-v1 | Acrobot-v1 |
|---|---:|---:|
| Held-out base blocks (roots) | 175 (8691) | 132 (16940) |
| Block ANOVA F, permutation p | 1.170, 0.060 | 0.920, 0.740 |
| Block Kruskal-Wallis p | 0.093 | 0.588 |
| Estimated share of root variance | 0.0035 | -0.0006 |
| Share implied by the null-control dispersion | 0.0054 | 0.0000 |
| SIM: block test detects the implied share | 68.5% | no excess |
| Expected Welch rejections: resampled, no block effect | 1.80 | 1.91 |
| Normal approximation at the estimated share | 3.24 | 1.90 |
| Normal approximation at the implied share | 3.94 | 1.90 |

A block is the base roots of one arm of one comparison repetition: consecutive seeds that were planned, trained and tested together. If each block shares a shift with variance share × sigma², a control's z-statistic has variance 1 + n × share, so the implied share is (sum of z² − controls) / (sum of planned counts). The held-out blocks are the base arms of the different-variant comparisons, with at least two roots; the estimated share is the one-way random-effects moment estimate, truncated at zero for the expected rejections. The simulation draws the held-out roots with replacement, adds one Gaussian shift per block at the implied share and applies the same permutation test.

## Score distributions

| Task | Variant | Roots | Mean | SD | Skewness | At the 500-step bound |
|---|---|---:|---:|---:|---:|---:|
| CartPole-v1 | base | 18199 | 399.5 | 79.3 | -0.52 | 10.7% |
| CartPole-v1 | lr_fast | 4037 | 365.1 | 115.6 | -0.35 | 12.1% |
| CartPole-v1 | lr_slow | 555 | 124.8 | 29.3 | 1.31 | 0.0% |
| CartPole-v1 | fewer_epochs | 572 | 156.5 | 43.2 | 1.07 | 0.0% |
| CartPole-v1 | no_entropy | 5127 | 398.0 | 80.9 | -0.51 | 11.3% |
| Acrobot-v1 | base | 26714 | -249.4 | 179.2 | -0.49 | 26.0% |
| Acrobot-v1 | lr_fast | 4482 | -210.2 | 164.1 | -0.91 | 15.7% |
| Acrobot-v1 | lr_slow | 4591 | -265.1 | 178.2 | -0.35 | 28.2% |
| Acrobot-v1 | fewer_epochs | 4700 | -216.0 | 166.7 | -0.87 | 18.0% |
| Acrobot-v1 | no_entropy | 4767 | -246.0 | 179.3 | -0.52 | 25.9% |

Roots pool the pilot and the confirmation. CartPole's bound is the 500-step maximum; Acrobot's is the -500 return of a policy that never reaches the goal.
