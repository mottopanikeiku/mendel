# PPO seed-planning experiment

I trained PPO on CartPole-v1 and Acrobot-v1 to test seed budgets derived from small pilots. The [design](../configs/neural_design.json) compares base PPO with a faster learning rate, a slower learning rate, fewer update epochs and no entropy coefficient. Base versus base provides a separate null control.

## Training and evaluation

Actor and critic each have two 32-unit tanh hidden layers. The implementation adapts [coldstart's JAX PPO](https://github.com/mottopanikeiku/coldstart/blob/b090dad8a3b53f937782676c6033ba3557cedfa1/control_clock/jax_ppo.py), under MIT, and follows PureJaxRL's JAX design, under Apache-2.0. [Attribution and licenses](PRIOR_WORK.md).

Every policy trains for 32,768 environment transitions on CartPole or 65,536 on Acrobot, then gets evaluated once. All variants on a task use the same budget. A compiled `vmap` keeps each training seed's parameters, Adam state, rollout randomness, advantage normalization and gradient clipping separate.

A score is the mean of sixteen deterministic-argmax episodes on a fixed task-specific reset list, shared across variants and phases. A mask stops counting rewards after the first episode ends, even though Gymnax resets automatically. Episodes end at termination or 500 steps. PPO does not bootstrap past termination or time-limit completion.

## From pilots to confirmation

Each comparison has 50 repetitions with disjoint eight-seed pilots per arm. I chose 50 from the candidates 30, 40 and 50 using L4 training throughput and the available allocation before running the pilots.

For each pilot, the planner uses the observed mean gap and unbiased variances to find the smallest equal per-arm count with modeled 80% power at two-sided alpha 0.05. It holds Welch degrees of freedom fixed at the pilot variances. Plans above 512 seeds per arm, zero gaps and zero total variances are ineligible.

The count plan and pilot scores were fixed before confirmation. Each executable plan received exactly its requested number of new training seeds. Pilot scores are excluded from the confirmation test. Batch padding is discarded before analysis.

The experiment contains 8,000 pilot policies and 65,744 confirmation policies, totaling 3.90 billion training transitions. The [pilot scores](../results/neural/pilot/runs.jsonl.gz), [confirmation scores](../results/neural/confirmation/runs.jsonl.gz) and [runtime details](../results/neural/confirmation/environment.json) are included.

## Results

Different-variant comparisons detected differences in 139/307 tests (45.3%; Wilson 95% interval 39.8–50.9%). Detection rates were 109/175 on CartPole and 30/132 on Acrobot. Twenty detections reversed the pilot's direction.

The [count plan](../results/neural/pilot_plan.json) also includes 93/400 different-variant comparisons above the 512-seed limit. Executable counts had median 37 seeds per arm and interquartile range 4–120.5. Individual comparison rates ranged from 2.9% to 94.0% ([figure](../figures/neural_comparisons.png)).

Same-variant controls rejected in 7/74 tests (9.5%; 4.7–18.3%), including 5/36 on CartPole (13.9%; 6.1–28.7%).

The [null-control diagnostics](../results/neural/null_diagnostics.md) did not identify a cause. Student, permutation, Mann–Whitney and Yuen tests rejected 5, 6, 5 and 3 of the 36 CartPole controls. Resampling held-out base-policy scores at the planned counts gave 1.80 expected Welch rejections; at exact 5% size, the chance of at least five rejections, adjusted for the two tasks, is 0.065. These checks were chosen after seeing the excess.

The analysis uses two-sided Welch detections at `p < 0.05`, with separate null controls, per-comparison rates and Wilson intervals. All 381 completed tests used ordinary Welch calculations. For constant groups, the implementation assigns p=1 to identical constants and p=0 to distinct constants, labels those cases and summarizes ordinary Welch tests separately. Tests are unadjusted because the outcome is marginal detection frequency across the chosen comparisons.

[Summary](../results/neural/summary.json) · [Every test](../results/neural/outcomes.csv)

## Saved batches

Each completed GPU batch is compressed and atomically saved to a Modal Volume before returning to the client. Cache keys include the training source and runtime image. Restarting exports saved batches and submits only missing work; up to five sequential clients share one deadline. The final JSONL is replaced only after every batch has been exported.

Each score record includes its training seed, configuration, sixteen episode returns, evaluation resets and comparison. Environment files include hardware, software, batch durations and plan hashes. [Running instructions](RUNNING.md) cover replication and the checks that prevent mixing training versions or plans.

## Interpretation

- These are detection frequencies across comparisons with different, unknown true gaps. The [simulation](CALIBRATION.md) measures power at specified effects.
- Results cover two classic-control tasks, one small architecture and fixed evaluation resets. The independent unit is a trained policy; its episodes are repeated measurements.
- The observed-gap planner ignores pilot estimation uncertainty and approximates Welch power. Bounded returns and near-zero true gaps can also lower detection rates.
- Affine and neural experiments differ in optimizers, horizons, reset handling, precision and comparisons. Their rate difference cannot isolate the effect of policy architecture.
