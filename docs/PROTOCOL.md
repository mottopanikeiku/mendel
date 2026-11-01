# Affine policy-search experiment

I compared ARS and CEM seed budgets on CartPole-v1 and Acrobot-v1. The [design](../configs/design.json) fixes six different-variant comparisons and one same-variant control per task, each repeated twelve times with disjoint pilots. Comparisons and hyperparameters stay fixed throughout the experiment.

## Policies and pilots

Policies use an affine map from observations to deterministic argmax actions. ARS uses 32 candidates, perturbation scale 0.5, learning rate 0.2 and the top eight directions. CEM uses 32 candidates, four elites, smoothing 0.5 and minimum standard deviation 0.05. Variants train for two, four or eight iterations.

Each arm has eight pilot training seeds. Each score averages sixteen complete evaluation episodes, with reset randomness separate from training and optimization. Evaluation episodes measure a single trained policy. Training transition counts vary with episode length.

## Choosing confirmation counts

For each pilot, I calculate the mean gap and two unbiased sample variances, then find the smallest equal per-arm count giving at least 80% modeled power at two-sided alpha 0.05. The noncentral-t planner uses Welch–Satterthwaite degrees of freedom fixed at the pilot variances. Under equal-variance Gaussian scores this gives exact pooled-t power; Welch's sample-based degrees of freedom make it approximate for the actual test.

A zero gap or zero total variance gives no plan. Counts above 256 per arm exceed the experiment's budget. The [pilot plan](../results/pilot_plan.json) includes those cases and their uncapped counts. The detection-rate denominator includes every executable different-variant plan, whether or not its pilot was significant.

The pilot plan was saved before confirmation. Each executable plan received exactly its requested number of new training seeds, with pilot scores excluded from the final test. The runner checks the plan and design against their saved revision.

## Analysis

The [summary](../results/summary.json) reports 67/119 detections (56.3%), using two-sided Welch tests at `p < 0.05`. It includes direction reversals, per-comparison rates, eligibility and Wilson 95% intervals, with same-variant controls reported separately. Individual tests are unadjusted because the outcome is marginal detection frequency.

Development runs measured cloud throughput on separate seeds. Score files include every evaluation return, training seed and configuration. Environment files include software, CPU allocation and numerical thread counts. [Running instructions](RUNNING.md) explain how to recompute the analysis or repeat training.

## Interpretation

- Different variants can have zero or small true mean gaps. The pooled detection rate mixes these unknown effects.
- The planner ignores pilot estimation uncertainty. Bounded, skewed and ceiling-heavy scores can also disagree with the Gaussian model; this experiment cannot separate those causes.
- The design compares short-budget affine policy-search variants. Training horizons vary, so the results do not rank equal-step learning efficiency.
