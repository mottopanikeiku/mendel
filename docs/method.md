# Method

## Planning rules

Each study draws a pilot of 8, 16, 32 or 64 seeds per arm, uses one of the rules below to pick a seed count for 80% power, and then tests two fresh groups with a two-sided Welch test at 5%. Plans above 512 seeds per arm are not run and count as misses, as does a pilot with zero gap or zero variance.

| Rule in the files | What it does |
|---|---|
| `observed_gap` | plugs in the pilot's absolute gap and variances (what the PPO and affine runs used) |
| `fixed_effect` | fixes the effect first, plugs in pilot variances |
| `variance_ucl` | fixed effect, with the pilot variances scaled up to a one-sided 80% upper confidence limit (Browne [1, 3]) |
| `variance_nct` | fixed effect, Julious and Owen's noncentral-t count [2, 3]: the smallest n with n·effect² ≥ q²(s²_a + s²_b), q the 80% quantile of a noncentral t |
| `known_variance` | fixed effect with the true variances, as a reference |

Both variance-uncertainty rules were derived for one pooled variance. Here the summed pilot variance gets Satterthwaite degrees of freedom, (pilot − 1)(s²_a + s²_b)² / (s⁴_a + s⁴_b). For equal variances that is 2(pilot − 1) and the tests reproduce the published inflation tables [3].

## Score populations

Arm a is the treatment, arm b the control, and the meaningful effect is half the control SD. Each population has true gaps of zero, half the effect and the full effect.

| Population | Control | Treatment |
|---|---|---|
| Gaussian | N(0, 1) | N(effect, 1) |
| Gaussian, unequal variance | N(0, 1) | N(effect, 2²) |
| Bounded with a ceiling | discretized Beta(6, 2.5) with 10% mass at 1 | move a fraction of scores to the ceiling |
| Failure/success mixture | 26% mass at 0 and a bell near 0.8 | turn a fraction of failures into successes |

The bounded populations live on a 101-point grid in [0, 1], loosely like the ceiling and failure mass in control-task returns. A fifth, null-only population has the mixture's mean with a unimodal shape, and the probability masses are in [`configuration.json`](../results/calibration/configuration.json).

## Checks on the simulation

- Every rule sees the same 2,000 pilots per population, gap and pilot size (seed 20261029), and each plan gets eight confirmation draws, with standard errors clustered by pilot.
- For Gaussian arms I integrate the rejection probability over both chi-square variances, with Gauss–Hermite nodes. Doubling the nodes shows numerical error below 1e-9 from three seeds per arm up, and the simulation agrees with the integral within 4.5 Monte Carlo SEs.
- For equal variances the whole plan distribution is integrated too (Gauss–Jacobi over the variance share), giving power, cost and wrong-sign rates without simulation noise.
- Resampling PPO scores pools pilot and confirmation scores by task and variant and takes four random half-splits with 500 replicates each, drawing pilot and confirmation from one permutation without replacement. The comparisons are base vs base (null), base shifted by half its SD, and each variant vs base.
- The recommended rule is the cheapest one reaching 78% power in all four populations at all four pilot sizes, and resampling plays no part in the choice. `variance_nct` wins, with its closest estimate 3.2 Monte Carlo SEs above the threshold, while `fixed_effect` at 32 pilot seeds with unequal Gaussian variances sits at 78.0% ± 0.7.

`uv run python scripts/calibrate.py --check` regenerates the report, tables and figure in memory and compares them with the saved ones (relative 1e-12 for numbers, exact text, 1e-6 geometry drift for SVG).

## PPO experiment

The PPO experiment compares base PPO against four variants (faster or slower learning rate, fewer update epochs, no entropy bonus) on CartPole-v1 and Acrobot-v1, plus base vs base as a null control, with the settings in [`configs/neural_design.json`](../configs/neural_design.json).

- Actor and critic have two 32-unit tanh layers. Training is one compiled JAX `vmap` over seeds, each with its own parameters, Adam state, rollout noise and gradient clipping.
- Every policy trains 32,768 transitions on CartPole or 65,536 on Acrobot, then is scored once: the mean of 16 deterministic episodes on a fixed reset list shared across variants and phases.
- Each comparison has 50 repetitions with disjoint 8-seed pilots per arm, and each pilot's observed gap and variances plan the number of new seeds for 80% power. I picked 50 repetitions from {30, 40, 50} using L4 throughput, before the pilots ran.
- The count plan and pilot scores were saved before confirmation, and the runner refuses to start from any other revision. Pilot scores are excluded from the confirmation test.
- In total the experiment trained 8,000 pilot policies and 65,744 confirmation policies on 3.9 billion training transitions.

Of the 307 different-variant tests, 139 detected a difference (45%: 109/175 on CartPole and 30/132 on Acrobot), and 20 of those reversed the pilot's sign. Another 93 of 400 plans needed more than 512 seeds per arm and were not run, while the executable plans had a median of 37 seeds per arm.

Same-variant controls rejected 7 of 74 times (9.5%), including 5 of 36 on CartPole. [`null_diagnostics.md`](../results/neural/null_diagnostics.md) tries Student, permutation, Mann–Whitney and Yuen tests (5, 6, 5 and 3 of 36 CartPole rejections) and resampling of held-out base scores (1.8 expected rejections at the planned counts). None of them explains the excess, and I chose these checks only after seeing it.

## Affine policy search

An earlier experiment compared ARS and CEM with affine policies on the same two tasks, running six variant comparisons and one control per task with twelve repetitions each, 8-seed pilots and plans from the observed gap. It detected a difference in 67 of 119 comparisons (56%), and its design is in [`configs/design.json`](../configs/design.json) with the scores in `results/confirmation`.

## Public RL scores

`scripts/public_analysis.py` takes 140 training runs from the MIT-licensed [coldstart](https://github.com/mottopanikeiku/coldstart) release (two tasks, five implementations) and compares all ten implementation pairs per task. For 19 of 20 pairs, the plug-in power at the observed gap was under 80% ([`public.json`](../results/public.json)). To detect effects of 0.2, 0.5, 0.8 and 1.0 SD you need about 394, 64, 26 and 17 seeds per arm under equal-variance Gaussian scores. If you need the input again, `scripts/import_public.py` rebuilds `data/public_scores.csv` from the pinned archive.

## Running things

```sh
uv sync --locked --python 3.12 --extra neural
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
uv run ruff check . && uv run python -m pytest -q
uv run python scripts/calibrate.py --check
uv run python scripts/null_diagnostics.py --check
```

These recompute the analyses from the saved scores without training anything:

```sh
uv run python scripts/analyze.py confirmation
uv run python scripts/neural_analyze.py confirmation
uv run python scripts/figures.py && uv run python scripts/neural_figures.py
```

These rebuild each pilot plan, compare counts and statuses exactly, and check file hashes. Last-bit float differences across CPU architectures are allowed, so regenerated PNGs and the last digits of `results/public*` can differ between machines.

Retraining uses Modal and costs money: about $0.03 for the affine run and $1.20 for PPO including failed attempts ([`cost.json`](../results/cost.json), [`neural/cost.json`](../results/neural/cost.json)). The tags `plan-committed` and `neural-plan-committed` mark the commits where the pilot plans were saved. The affine runner requires HEAD to be exactly that commit. The PPO runner accepts any later commit whose training code, design and plan are unchanged, and of those `neural-transport` is the one with resumable batches:

```sh
git switch --detach plan-committed
SEED_POWER_TIMEOUT_SECONDS=600 SEED_POWER_CONTAINERS=8 uv run modal run modal_app.py \
  --stage confirmation --output results/replication/confirmation --plan-commit plan-committed

git switch --detach neural-transport
SEED_POWER_TIMEOUT_SECONDS=3600 uv run modal run neural_modal.py \
  --stage confirmation --output results/replication/neural --plan-commit neural-plan-committed
```

The PPO runner saves each 64-policy batch to a Modal Volume as it finishes, keyed by design, plan, training source and runtime image, so an interrupted run resumes without retraining.

For a new experiment, choose comparisons, design and unused seed ranges first, run the `development`, `pilot` and `plan` stages, commit and push the plan, then run `confirmation` with that commit.

## References and attribution

1. Browne (1995). [On the use of a pilot sample for sample size determination](https://doi.org/10.1002/sim.4780141709). Statistics in Medicine.
2. Julious and Owen (2006). [Sample size calculations for clinical studies allowing for uncertainty about the variance](https://pubmed.ncbi.nlm.nih.gov/17080926).
3. Whitehead et al. (2016). [Estimating the sample size for a pilot randomised trial to minimise the overall trial sample size](https://pmc.ncbi.nlm.nih.gov/articles/PMC4876429/). Section 2.1 describes both methods, and its Tables 1 and 2 are what the tests reproduce.

Related RL work includes [Henderson et al. 2018](https://arxiv.org/abs/1709.06560), [Colas, Sigaud and Oudeyer 2018](https://arxiv.org/abs/1806.08295) (the pilot-then-more-runs workflow this repo repeats with fresh training runs), [Agarwal et al. 2021](https://arxiv.org/abs/2108.13264) and [rliable](https://github.com/google-research/rliable), [Patterson et al. 2024](https://jmlr.org/papers/v25/23-0183.html), and [Albers and Lakens 2018](https://doi.org/10.1016/j.jesp.2017.09.004) on bias from planning around pilot effects.

Affine ARS/CEM, the NumPy batch environments and the JAX PPO adapt code from [coldstart](https://github.com/mottopanikeiku/coldstart) (MIT). The PPO also follows [PureJaxRL](https://github.com/luchris429/purejaxrl)'s vmap/scan design (Apache-2.0, [license](../third_party/PureJaxRL-LICENSE)), although I swapped its stopping rule for fixed transition budgets and one evaluation per policy in Gymnax. Environment sources and the Gymnasium and Acrobot/RLPy notices are in [`ENVIRONMENTS.md`](ENVIRONMENTS.md).
