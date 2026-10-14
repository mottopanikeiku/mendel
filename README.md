# seed-power

How many independent training seeds do I need?

I planned neural PPO comparisons for **80% modeled power** from eight-seed pilots. On fresh seeds, **139/307 detected a difference (45.3%)**, with a descriptive Wilson 95% interval of **39.8–50.9%**. My earlier affine ARS/CEM experiment detected **67/119 (56.3%; 47.3–64.9%)**. I compare base PPO with faster/slower learning rates, fewer update epochs and no entropy coefficient.

| Fresh-seed detections | Affine ARS/CEM | Neural PPO |
|---|---:|---:|
| All executable different-variant plans | 67/119 (56.3%) | 139/307 (45.3%) |
| CartPole-v1 | 30/60 (50.0%) | 109/175 (62.3%) |
| Acrobot-v1 | 37/59 (62.7%) | 30/132 (22.7%) |

Same-variant controls rejected in **7/74 tests (9.5%; 4.7–18.3%)**.

## How many seeds?

I use this planner for a fixed mean gap and two standard deviations:

```sh
uv sync --locked --python 3.12
uv run python scripts/plan.py --effect 0.5 --sd-a 1 --sd-b 1
```

For an illustrative equal-variance Gaussian pooled-t model, two-sided alpha=0.05 and 80% power:

| Standardized mean gap | Seeds per arm |
|---|---:|
| 0.2 | 394 |
| 0.5 | 64 |
| 0.8 | 26 |
| 1.0 | 17 |

![Pilot-derived budgets: affine policy search and neural PPO](figures/neural_vs_linear.png)

[Neural summary](results/neural/summary.json) · [Every test](results/neural/outcomes.csv) · [Original summary](results/summary.json) · [Vector figure](figures/neural_vs_linear.svg)

## What I built

I adapt MIT [Control Clock PPO](https://github.com/mottopanikeiku/control-clock/blob/b090dad8a3b53f937782676c6033ba3557cedfa1/control_clock/jax_ppo.py), inspired by Apache-2.0 PureJaxRL. Actor and critic each have two 32-unit tanh hidden layers. I train independent roots in a compiled `vmap` ([implementation](src/seed_power/neural.py)).

I train every seed for exactly **32,768 CartPole** or **65,536 Acrobot** environment transitions, without qualification stopping. I average sixteen deterministic-argmax evaluation episodes on fixed task-specific reset seeds shared across variants and phases. I count only each first episode, ending at termination or 500 steps. I describe the comparisons and collection in the [neural protocol](docs/NEURAL_PROTOCOL.md) and [original protocol](docs/PROTOCOL.md).

## Reproduce

Recompute from committed scores without training or paid compute:

```sh
uv run python scripts/neural_analyze.py confirmation
uv run python scripts/neural_figures.py
uv run python scripts/analyze.py confirmation
uv run python scripts/public_analysis.py
```

[Cloud reproduction and tests](docs/RUNNING.md).

## Limits

- These are detection frequencies in two fixed classic-control designs, not true power at known effects or a general deep-RL benchmark. I did not run **93/400** different-variant plans above the **512-seed-per-arm** limit.
- The noncentral-t planner uses noisy pilot gaps and variances with fixed Welch degrees of freedom. Bounded, skewed returns may not fit its model, and different variants may have no true mean gap. I cannot attribute every missed detection to pilot noise or claim exact 5% calibration.
- Episodes measure each trained policy; they are not independent training replicates. Inference is conditional on the fixed reset sets and executable plans.
- The affine and neural experiments differ in optimizers, horizons, budgets, reset handling, precision and comparison mix. Their rate difference is not a causal effect of neural policies.

I use established seed-count guidance, not a new formula: [Henderson](https://arxiv.org/abs/1709.06560), [Colas](https://arxiv.org/abs/1806.08295), [Agarwal/rliable](https://arxiv.org/abs/2108.13264), [Patterson](https://arxiv.org/abs/2304.01315v1). [Attribution and licenses](docs/PRIOR_WORK.md). Code is MIT; third-party notices are retained.

Written with AI coding assistance.
