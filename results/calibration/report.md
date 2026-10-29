# Seed-planning calibration report

`uv run python scripts/calibrate.py` writes this file, the tables beside it and `figures/known_truth.svg`. Rows marked SIM are simulations with known truth. Rows marked resampling reuse committed PPO root scores; they train nothing and are not new independent evidence.

## Recommendation

I chose the rule from the SIM tables alone. The committed-root resampling below is a check on that choice and did not change it.

1. Fix the smallest effect you care about before running the pilot, and plan from that effect, not from the pilot's mean gap. Planning from the observed pilot gap gave 36.3–80.5% power when the true effect equalled the meaningful one, and put 25.0–68.0% of null pilots above 512 seeds per arm (SIM).
2. Then plan with `variance_nct` (Fixed effect, noncentral-t (Julious-Owen)). It was the cheapest rule whose power stayed at or above 78.0% in all four SIM populations at all four pilot sizes. Its power ranged 79.3–83.3% over all populations and pilot sizes. Among the other fixed-effect rules, `fixed_effect` held at pilot sizes 32, 64 and used 0.95 times the seeds on average; `variance_ucl` held at every pilot size and used 1.16 times the seeds on average. "Held" compares simulated point estimates with 78.0%; it does not show that a rule's true power is above that line. This call is within a 95% Monte Carlo half-width of the line: `fixed_effect` at 32 pilot seeds, Gaussian, treatment SD x2: 78.0% ± 0.7. The closest call for `variance_nct` is 3.2 Monte Carlo SEs above the line.
3. If the true effect is only half the one you planned for, no rule rescues the study: the fixed-effect rules gave 28.4–41.3% power (SIM).
4. The Welch test, not the planning rule, decides the false-positive rate. With identical null arms the fixed-effect rules rejected 4.7–5.4%. When the two arms had equal means but different shapes (one bimodal with a failure mode, one unimodal), they rejected 5.3–6.8% (SIM, nominal 5%).
5. On resampled committed PPO roots, `variance_nct` gave 80.5–82.6% power when I shifted one base arm by its planned effect, and rejected 4.5–5.3% of base-versus-base nulls, across both tasks and all pilot sizes. This is resampling of existing roots, not fresh training.

Power at the meaningful effect, lowest to highest over the four SIM populations, with mean total seeds (both arms, including the pilot) in brackets:

| Pilot seeds per arm | Fixed effect, pilot variance | Fixed effect, noncentral-t (Julious-Owen) | Fixed effect, 80% variance UCL |
|---:|---:|---:|---:|
| 8 | 75.5–77.0% (184) | 79.6–81.7% (212) | 85.1–89.4% (271) |
| 16 | 77.1–80.4% (199) | 79.3–83.3% (211) | 86.2–90.0% (254) |
| 32 | 78.0–80.6% (231) | 79.7–82.6% (236) | 84.8–88.0% (264) |
| 64 | 78.9–81.8% (296) | 79.8–82.5% (298) | 83.8–87.3% (318) |

The known-variance oracle gave 79.6–83.3% at the same effect, so even perfect variance knowledge does not give exactly 80% outside the Gaussian model. `uv run python scripts/plan.py --effect E --sd-a A --sd-b B --pilot-seeds N` applies the recommended rule to your own pilot SDs.

## Power at the meaningful effect (SIM)

Workflow power counts plans above the cap as misses. Ranges run over the four populations; per-population values with Monte Carlo errors are in `workflow.csv`.

| Planning rule | 8-seed pilot: power | total seeds | 32-seed pilot: power | total seeds |
|---|---:|---:|---:|---:|
| Observed pilot gap | 36.3–62.1% | 131–174 | 44.6–75.3% | 173–273 |
| Fixed effect, pilot variance | 75.5–77.0% | 96–334 | 78.0–80.6% | 144–380 |
| Fixed effect, 80% variance UCL | 85.1–89.4% | 149–491 | 84.8–88.0% | 161–448 |
| Fixed effect, noncentral-t (Julious-Owen) | 79.6–81.7% | 113–390 | 79.7–82.6% | 147–391 |
| Fixed effect, true variance (oracle) | 79.8–83.3% | 96–334 | 79.6–82.3% | 144–382 |

| Population | Pilot | `observed_gap` | `fixed_effect` | `variance_ucl` | `variance_nct` | `known_variance` |
|---|---:|---:|---:|---:|---:|---:|
| Gaussian, equal SD | 8 | 50.9% ± 1.7 (154) | 75.6% ± 0.9 (144) | 88.9% ± 0.7 (209) | 80.4% ± 0.8 (162) | 80.1% ± 0.6 (144) |
| Gaussian, equal SD | 16 | 55.9% ± 1.7 (180) | 78.4% ± 0.7 (162) | 87.3% ± 0.6 (199) | 80.8% ± 0.7 (169) | 79.7% ± 0.6 (160) |
| Gaussian, equal SD | 32 | 66.1% ± 1.5 (232) | 79.1% ± 0.7 (192) | 84.8% ± 0.6 (215) | 79.7% ± 0.7 (195) | 80.1% ± 0.6 (192) |
| Gaussian, equal SD | 64 | 74.2% ± 1.2 (297) | 80.1% ± 0.6 (257) | 83.9% ± 0.6 (272) | 80.7% ± 0.6 (258) | 80.8% ± 0.6 (256) |
| Gaussian, treatment SD x2 | 8 | 36.3% ± 1.6 (174) | 75.5% ± 0.9 (334) | 85.1% ± 1.0 (491) | 79.6% ± 0.9 (390) | 80.0% ± 0.6 (334) |
| Gaussian, treatment SD x2 | 16 | 40.5% ± 1.6 (215) | 77.1% ± 0.8 (346) | 86.2% ± 0.7 (455) | 79.3% ± 0.8 (371) | 79.9% ± 0.6 (350) |
| Gaussian, treatment SD x2 | 32 | 44.6% ± 1.6 (273) | 78.0% ± 0.7 (380) | 85.5% ± 0.6 (448) | 79.8% ± 0.7 (391) | 80.4% ± 0.6 (382) |
| Gaussian, treatment SD x2 | 64 | 52.9% ± 1.6 (377) | 79.2% ± 0.7 (446) | 84.1% ± 0.6 (491) | 79.8% ± 0.7 (451) | 80.0% ± 0.6 (446) |
| Bounded with a ceiling | 8 | 49.5% ± 1.7 (160) | 77.0% ± 0.8 (161) | 89.4% ± 0.6 (234) | 81.3% ± 0.8 (181) | 79.8% ± 0.6 (160) |
| Bounded with a ceiling | 16 | 55.4% ± 1.6 (190) | 78.5% ± 0.7 (176) | 87.0% ± 0.6 (218) | 80.4% ± 0.7 (185) | 80.1% ± 0.6 (176) |
| Bounded with a ceiling | 32 | 64.6% ± 1.5 (242) | 79.4% ± 0.7 (208) | 85.1% ± 0.6 (234) | 80.5% ± 0.7 (212) | 79.6% ± 0.6 (208) |
| Bounded with a ceiling | 64 | 71.2% ± 1.3 (309) | 78.9% ± 0.6 (271) | 83.8% ± 0.6 (288) | 80.1% ± 0.6 (273) | 80.0% ± 0.6 (272) |
| Failure/success bimodal | 8 | 62.1% ± 1.6 (131) | 75.9% ± 1.2 (96) | 88.9% ± 1.1 (149) | 81.7% ± 1.1 (113) | 83.3% ± 0.6 (96) |
| Failure/success bimodal | 16 | 69.0% ± 1.4 (152) | 80.4% ± 0.8 (113) | 90.0% ± 0.7 (142) | 83.3% ± 0.8 (120) | 83.0% ± 0.6 (112) |
| Failure/success bimodal | 32 | 75.3% ± 1.2 (173) | 80.6% ± 0.7 (144) | 88.0% ± 0.6 (161) | 82.6% ± 0.7 (147) | 82.3% ± 0.6 (144) |
| Failure/success bimodal | 64 | 80.5% ± 0.9 (229) | 81.8% ± 0.6 (208) | 87.3% ± 0.6 (220) | 82.5% ± 0.6 (209) | 82.3% ± 0.6 (208) |

Each cell is workflow power ± a 95% Monte Carlo half-width, then mean total seeds.

## Null rejection (SIM)

| Population | Pilot | `observed_gap` | `fixed_effect` | `variance_ucl` | `variance_nct` | `known_variance` |
|---|---:|---:|---:|---:|---:|---:|
| Gaussian, equal SD | 8 | 3.7% ± 0.3 | 5.1% ± 0.3 | 5.1% ± 0.3 | 5.2% ± 0.3 | 5.1% ± 0.3 |
| Gaussian, equal SD | 16 | 3.2% ± 0.3 | 5.1% ± 0.3 | 5.1% ± 0.3 | 5.0% ± 0.3 | 4.7% ± 0.3 |
| Gaussian, equal SD | 32 | 2.3% ± 0.3 | 4.7% ± 0.3 | 4.8% ± 0.3 | 4.8% ± 0.3 | 4.5% ± 0.3 |
| Gaussian, equal SD | 64 | 1.6% ± 0.2 | 4.8% ± 0.3 | 5.0% ± 0.3 | 5.2% ± 0.3 | 5.1% ± 0.3 |
| Gaussian, treatment SD x2 | 8 | 3.7% ± 0.3 | 5.2% ± 0.3 | 4.7% ± 0.3 | 5.2% ± 0.3 | 5.0% ± 0.3 |
| Gaussian, treatment SD x2 | 16 | 3.3% ± 0.3 | 4.7% ± 0.3 | 4.9% ± 0.3 | 4.8% ± 0.3 | 4.9% ± 0.3 |
| Gaussian, treatment SD x2 | 32 | 2.5% ± 0.3 | 5.1% ± 0.3 | 5.0% ± 0.3 | 4.8% ± 0.3 | 5.3% ± 0.3 |
| Gaussian, treatment SD x2 | 64 | 1.6% ± 0.2 | 4.7% ± 0.3 | 5.0% ± 0.4 | 5.1% ± 0.3 | 5.2% ± 0.3 |
| Bounded with a ceiling | 8 | 3.6% ± 0.3 | 4.9% ± 0.3 | 5.1% ± 0.3 | 5.0% ± 0.3 | 4.9% ± 0.3 |
| Bounded with a ceiling | 16 | 2.6% ± 0.3 | 4.9% ± 0.3 | 5.1% ± 0.3 | 5.1% ± 0.3 | 4.7% ± 0.3 |
| Bounded with a ceiling | 32 | 2.5% ± 0.3 | 5.0% ± 0.3 | 5.1% ± 0.3 | 4.9% ± 0.3 | 4.9% ± 0.3 |
| Bounded with a ceiling | 64 | 1.5% ± 0.2 | 5.0% ± 0.3 | 5.2% ± 0.3 | 5.3% ± 0.3 | 5.2% ± 0.3 |
| Failure/success bimodal | 8 | 3.6% ± 0.3 | 5.1% ± 0.3 | 4.7% ± 0.3 | 4.9% ± 0.3 | 5.2% ± 0.3 |
| Failure/success bimodal | 16 | 3.0% ± 0.3 | 5.3% ± 0.4 | 5.0% ± 0.3 | 5.1% ± 0.3 | 5.2% ± 0.3 |
| Failure/success bimodal | 32 | 2.4% ± 0.3 | 5.4% ± 0.3 | 5.0% ± 0.3 | 5.0% ± 0.3 | 4.9% ± 0.3 |
| Failure/success bimodal | 64 | 1.5% ± 0.2 | 4.9% ± 0.3 | 4.8% ± 0.3 | 4.8% ± 0.3 | 4.9% ± 0.3 |
| Equal means, bimodal vs unimodal | 8 | 5.2% ± 0.4 | 6.8% ± 0.4 | 6.1% ± 0.4 | 6.4% ± 0.4 | 5.3% ± 0.4 |
| Equal means, bimodal vs unimodal | 16 | 3.4% ± 0.3 | 5.7% ± 0.4 | 5.8% ± 0.4 | 5.7% ± 0.4 | 5.7% ± 0.4 |
| Equal means, bimodal vs unimodal | 32 | 2.4% ± 0.3 | 5.3% ± 0.3 | 5.4% ± 0.3 | 5.7% ± 0.4 | 5.1% ± 0.3 |
| Equal means, bimodal vs unimodal | 64 | 1.7% ± 0.2 | 5.7% ± 0.4 | 5.3% ± 0.3 | 5.6% ± 0.4 | 5.4% ± 0.3 |

The observed-gap rule rejects less often under the null mainly because many of its null plans exceed the cap and are not run.

## Exact Gaussian checks

For equal-variance Gaussian arms I compute every rule's plan distribution, power, eligibility, cost and wrong-sign rate by quadrature, with no Monte Carlo error (`exact_*` columns in `workflow.csv`). The largest simulated-minus-exact workflow power was 2.73 Monte Carlo SEs across 60 rows. For all Gaussian rows, including unequal variances, the simulated rate agrees with the exact power of each pilot's own plan within 2.59 SEs (120 rows). Fixed-count Welch rejection matched exact quadrature within 2.87 binomial SEs (60 rows). The planner's noncentral-t formula differed from exact Welch power by at most 3.26 points below 8 seeds per arm and 0.51 points from 8 seeds on.

The quadrature has numerical error, which I report separately from Monte Carlo error (`*_numerical_error` columns). I estimate it as the change in each exact value when the nodes double from 48 to 96; tests check that this estimate bounds the error against an independent adaptive integration. For fixed-count Welch rejection it is at most 3.3e-05 at 2 seeds per arm, where each sample variance has one degree of freedom, and at most 5.3e-10 from 3 seeds on. For workflow power and eligibility it is at most 3.0e-12, and for mean total seeds 1.4e-11 seeds. The smallest Monte Carlo SE of workflow power in those rows is 0.0011. Every check above allows 4.5 Monte Carlo SEs plus the numerical error.

## Committed PPO roots, resampled

Each task/variant pool of committed pilot and confirmation roots is split 4 times into disjoint halves. Each half is a finite population with an exactly known gap. Every replicate draws a pilot and then a confirmation without replacement, so no root is in both. `null` compares base with base; `shift` adds half the base SD to one arm's roots, a known effect on real score shapes; the other rows are the committed variant pairs, planned at their own population gap. SE combines Monte Carlo and finite-pool (delete-half jackknife) error.

| Task | Comparison | Gap | Pilot | `observed_gap` | `fixed_effect` | `variance_ucl` | `variance_nct` | `known_variance` |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CartPole-v1 | base vs base | 0.0 | 8 | 3.1% ± 0.5 (71.5% run) | 4.8% ± 0.7 (100.0% run) | 5.0% ± 0.7 (100.0% run) | 4.6% ± 0.6 (100.0% run) | 4.9% ± 0.7 (100.0% run) |
| CartPole-v1 | base vs base | 0.0 | 32 | 2.4% ± 0.5 (49.9% run) | 5.5% ± 1.4 (100.0% run) | 5.5% ± 1.8 (100.0% run) | 5.3% ± 1.5 (100.0% run) | 5.3% ± 0.7 (100.0% run) |
| CartPole-v1 | base + shift vs base | 39.7 | 8 | 50.2% ± 1.6 (82.7% run) | 76.2% ± 2.0 (100.0% run) | 90.4% ± 0.9 (100.0% run) | 81.3% ± 1.2 (100.0% run) | 80.0% ± 3.3 (100.0% run) |
| CartPole-v1 | base + shift vs base | 39.7 | 32 | 65.6% ± 1.5 (90.4% run) | 80.3% ± 3.3 (100.0% run) | 86.1% ± 1.1 (100.0% run) | 81.5% ± 2.6 (100.0% run) | 80.1% ± 1.3 (100.0% run) |
| CartPole-v1 | base vs lr_fast | 34.3 | 8 | 39.7% ± 3.7 (78.1% run) | 78.0% ± 3.3 (100.0% run) | 91.5% ± 2.9 (100.0% run) | 83.6% ± 3.4 (100.0% run) | 80.4% ± 1.2 (100.0% run) |
| CartPole-v1 | base vs lr_fast | 34.3 | 32 | 49.4% ± 3.5 (78.3% run) | 79.6% ± 1.2 (100.0% run) | 86.1% ± 1.1 (100.0% run) | 80.4% ± 1.2 (100.0% run) | 79.4% ± 1.3 (100.0% run) |
| CartPole-v1 | base vs lr_slow | 274.6 | 8 | 84.2% ± 1.1 (100.0% run) | 84.2% ± 1.1 (100.0% run) | 91.8% ± 0.9 (100.0% run) | 85.6% ± 1.1 (100.0% run) | 82.4% ± 1.2 (100.0% run) |
| CartPole-v1 | base vs lr_slow | 274.6 | 32 | 83.6% ± 1.5 (100.0% run) | 82.7% ± 2.2 (100.0% run) | 86.5% ± 1.1 (100.0% run) | 82.0% ± 2.2 (100.0% run) | 81.7% ± 2.6 (100.0% run) |
| CartPole-v1 | base vs fewer_epochs | 243.0 | 8 | 81.7% ± 1.2 (100.0% run) | 81.9% ± 1.2 (100.0% run) | 90.3% ± 0.9 (100.0% run) | 83.4% ± 1.2 (100.0% run) | 76.2% ± 1.3 (100.0% run) |
| CartPole-v1 | base vs fewer_epochs | 243.0 | 32 | 81.0% ± 1.2 (100.0% run) | 79.8% ± 1.2 (100.0% run) | 86.2% ± 1.1 (100.0% run) | 78.5% ± 1.3 (100.0% run) | 76.0% ± 1.3 (100.0% run) |
| CartPole-v1 | base vs no_entropy | 1.5 | 8 | 3.8% ± 1.0 (72.6% run) | 0.0% ± 0.0 (0.0% run) | 0.0% ± 0.0 (0.0% run) | 0.0% ± 0.0 (0.0% run) | 0.0% ± 0.0 (0.0% run) |
| CartPole-v1 | base vs no_entropy | 1.5 | 32 | 2.0% ± 0.4 (48.4% run) | 0.0% ± 0.0 (0.0% run) | 0.0% ± 0.0 (0.0% run) | 0.0% ± 0.0 (0.0% run) | 0.0% ± 0.0 (0.0% run) |
| Acrobot-v1 | base vs base | 0.0 | 8 | 4.1% ± 0.6 (72.5% run) | 4.9% ± 0.7 (100.0% run) | 4.9% ± 0.7 (100.0% run) | 4.9% ± 0.7 (100.0% run) | 4.9% ± 0.7 (100.0% run) |
| Acrobot-v1 | base vs base | 0.0 | 32 | 2.1% ± 0.4 (47.3% run) | 5.0% ± 0.7 (100.0% run) | 5.1% ± 0.7 (100.0% run) | 4.8% ± 0.7 (100.0% run) | 5.1% ± 0.7 (100.0% run) |
| Acrobot-v1 | base + shift vs base | 89.6 | 8 | 50.5% ± 3.1 (83.1% run) | 78.3% ± 3.0 (100.0% run) | 91.4% ± 0.9 (100.0% run) | 82.6% ± 1.2 (100.0% run) | 79.5% ± 1.3 (100.0% run) |
| Acrobot-v1 | base + shift vs base | 89.6 | 32 | 66.7% ± 1.5 (91.3% run) | 81.0% ± 1.2 (100.0% run) | 85.9% ± 1.1 (100.0% run) | 81.6% ± 1.2 (100.0% run) | 80.2% ± 1.2 (100.0% run) |
| Acrobot-v1 | base vs lr_fast | -39.3 | 8 | 25.4% ± 7.0 (76.4% run) | 77.6% ± 6.1 (98.6% run) | 58.3% ± 46.3 (65.8% run) | 78.2% ± 18.1 (93.8% run) | 81.2% ± 1.2 (100.0% run) |
| Acrobot-v1 | base vs lr_fast | -39.3 | 32 | 29.9% ± 6.8 (63.6% run) | 81.1% ± 1.2 (100.0% run) | 84.2% ± 8.2 (97.6% run) | 82.0% ± 1.2 (100.0% run) | 81.7% ± 1.2 (100.0% run) |
| Acrobot-v1 | base vs lr_slow | 15.7 | 8 | 7.9% ± 2.1 (73.5% run) | 0.1% ± 0.4 (0.2% run) | 0.0% ± 0.0 (0.0% run) | 0.1% ± 0.2 (0.1% run) | 0.0% ± 0.0 (0.0% run) |
| Acrobot-v1 | base vs lr_slow | 15.7 | 32 | 7.4% ± 4.5 (50.6% run) | 0.0% ± 0.0 (0.0% run) | 0.0% ± 0.0 (0.0% run) | 0.0% ± 0.0 (0.0% run) | 0.0% ± 0.0 (0.0% run) |
| Acrobot-v1 | base vs fewer_epochs | -33.4 | 8 | 20.9% ± 3.5 (73.2% run) | 59.2% ± 23.3 (78.1% run) | 14.7% ± 12.9 (18.9% run) | 44.1% ± 25.8 (56.8% run) | 80.4% ± 1.7 (100.0% run) |
| Acrobot-v1 | base vs fewer_epochs | -33.4 | 32 | 24.3% ± 3.4 (60.6% run) | 75.5% ± 23.7 (93.8% run) | 47.7% ± 40.6 (56.7% run) | 74.0% ± 28.1 (90.7% run) | 80.5% ± 2.5 (100.0% run) |
| Acrobot-v1 | base vs no_entropy | -3.4 | 8 | 3.1% ± 0.5 (72.2% run) | 0.0% ± 0.0 (0.0% run) | 0.0% ± 0.0 (0.0% run) | 0.0% ± 0.0 (0.0% run) | 0.0% ± 0.0 (0.0% run) |
| Acrobot-v1 | base vs no_entropy | -3.4 | 32 | 2.2% ± 1.1 (48.4% run) | 0.0% ± 0.0 (0.0% run) | 0.0% ± 0.0 (0.0% run) | 0.0% ± 0.0 (0.0% run) | 0.0% ± 0.0 (0.0% run) |

Each cell is workflow rejection ± a 95% half-width, then the fraction of plans run. Gaps are mean(a) − mean(b) averaged over the half populations. Variant pairs are planned at each half population's own gap, which moves with finite-pool noise. Near the 512-seed cap that decides whether a plan runs at all, so those rows carry large finite-pool errors, and rules that inflate the variance push more plans over the cap. The fixed-effect rules ran under 1% of plans for CartPole-v1 base vs no_entropy, Acrobot-v1 base vs lr_slow, Acrobot-v1 base vs no_entropy: under the noncentral-t planning model, the finite-population gaps of these pairs need far more than 512 seeds per arm.

## Fresh training versus resampling

The committed fresh-seed experiment planned from 8-seed pilots with the observed gap. Resampling the same rule from the committed roots should give similar detection rates if the roots are a fair stand-in for new training. Fresh rates are MEAS; resampled rates are not new evidence.

| Task | Comparison | Fresh: detected/run, Wilson 95% (run/planned) | Resampled: detected among run (run) |
|---|---|---:|---:|
| Acrobot-v1 | base vs base | 2/38 = 5.3%, 1.5–17.3% (38/50) | 5.6% (72.5%) |
| Acrobot-v1 | base vs fewer_epochs | 14/32 = 43.8%, 28.2–60.7% (32/50) | 28.6% (73.2%) |
| Acrobot-v1 | base vs lr_fast | 14/39 = 35.9%, 22.7–51.6% (39/50) | 33.2% (76.4%) |
| Acrobot-v1 | base vs lr_slow | 1/34 = 2.9%, 0.5–14.9% (34/50) | 10.8% (73.5%) |
| Acrobot-v1 | base vs no_entropy | 1/27 = 3.7%, 0.7–18.3% (27/50) | 4.3% (72.2%) |
| CartPole-v1 | base vs base | 5/36 = 13.9%, 6.1–28.7% (36/50) | 4.3% (71.5%) |
| CartPole-v1 | base vs fewer_epochs | 41/50 = 82.0%, 69.2–90.2% (50/50) | 81.8% (100.0%) |
| CartPole-v1 | base vs lr_fast | 20/40 = 50.0%, 35.2–64.8% (40/50) | 50.8% (78.1%) |
| CartPole-v1 | base vs lr_slow | 47/50 = 94.0%, 83.8–97.9% (50/50) | 84.2% (100.0%) |
| CartPole-v1 | base vs no_entropy | 1/35 = 2.9%, 0.5–14.5% (35/50) | 5.2% (72.6%) |

The resampled rate falls inside the fresh Wilson interval for 9 of 10 comparisons; it falls outside for CartPole-v1 base vs base. Separate 95% intervals are not a joint test.

## Limits

- SIM populations are stylized: two Gaussian, one bounded with a ceiling mass and one failure/success mixture, each with a meaningful effect of half the control SD. Other shapes, heavier tails or smaller effects can behave differently.
- Every rule here plans for 80% power at one prespecified effect. None protects against choosing that effect badly.
- Resampling reuses the same committed roots across replicates and conditions on the fixed evaluation reset sets. It shows how the rules behave on these PPO score distributions; it is not fresh training and adds no independent seeds.
- Monte Carlo intervals are pointwise, not simultaneous across the grid.
