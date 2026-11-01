# Related work and attribution

Mendel compares established sample-size rules by running each from pilot to confirmation. The training experiments test budgets based on the observed pilot gap; the [simulation](CALIBRATION.md) compares that rule with fixed-effect alternatives.

## Seed planning and RL evaluation

- [Henderson et al., *Deep Reinforcement Learning that Matters* (2018)](https://arxiv.org/abs/1709.06560) document variability across seeds and use bootstrap power analysis with an artificial score lift.
- [Colas, Sigaud and Oudeyer, *How Many Random Seeds?* (2018)](https://arxiv.org/abs/1806.08295) give seed-count guidance, a pilot-to-additional-runs workflow, and examples of small-pilot and distributional failures. Mendel repeats that workflow across separate pilots and new training runs.
- [Agarwal et al., *Deep Reinforcement Learning at the Edge of the Statistical Precipice* (2021)](https://arxiv.org/abs/2108.13264) and [rliable](https://github.com/google-research/rliable) provide stratified bootstrap intervals, IQM and performance profiles for aggregate benchmarks. Mendel plans per-task mean-return comparisons.
- [Patterson et al., *Empirical Design in Reinforcement Learning* (2024)](https://jmlr.org/papers/v25/23-0183.html) discuss experimental units, hyperparameter selection and designer bias. Equal numeric seed labels alone do not establish pairing; evaluation episodes measure a policy rather than add training replicates.
- [Albers and Lakens, *When power analyses based on pilot data are biased* (2018)](https://doi.org/10.1016/j.jesp.2017.09.004) explain the bias from planning around pilot effect estimates.

## Variance uncertainty

[Browne (1995)](https://doi.org/10.1002/sim.4780141709) and [Julious and Owen (2006)](https://pubmed.ncbi.nlm.nih.gov/17080926) account for uncertainty in pilot variance estimates. Mendel adapts the 80% upper-confidence-limit and noncentral-t methods to Welch comparisons of training seeds. [Whitehead et al. (2016)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4876429/) describe both methods and give the inflation tables reproduced by the tests.

## Implementation sources

Affine ARS/CEM and the NumPy batch environments come from [coldstart](https://github.com/mottopanikeiku/coldstart/tree/ee9c303de8e7698ca25569fc585e173e3e172bb8), under MIT. The [environment documentation](ENVIRONMENTS.md) includes Gymnasium's MIT and Acrobot/RLPy's BSD notices.

The neural experiment adapts [coldstart's JAX PPO](https://github.com/mottopanikeiku/coldstart/blob/b090dad8a3b53f937782676c6033ba3557cedfa1/control_clock/jax_ppo.py), under MIT. It follows [PureJaxRL](https://github.com/luchris429/purejaxrl)'s JAX/vmap/scan design, under Apache-2.0 ([license](../third_party/PureJaxRL-LICENSE)). I replaced qualification stopping with fixed transition budgets and one evaluation per policy in Gymnax.

## Historical score analysis

The public-data analysis uses 140 training runs from the MIT coldstart release: two tasks and five implementations. ARS, CEM and batched PPO have 20 runs per task; adapted CleanRL PPO and Zoo SB3 PPO have five. The [source manifest](../data/public_sources.json) includes archive and raw-record hashes, selection rules and license information. The importer checks each final score against its 100 episode returns.

I compare all ten unordered implementation pairs per task using their actual, sometimes unequal, counts. In [public.json](../results/public.json), 19/20 pairs have less than 80% plug-in model power for their observed gaps. The [standardized-effect table](../results/public_standardized.csv) gives 394, 64, 26 and 17 seeds per arm for effects of 0.2, 0.5, 0.8 and 1.0 SD under the equal-variance Gaussian model.

```sh
uv run python scripts/import_public.py
uv run python scripts/public_analysis.py
```

## Limits of the historical data

- Scores come from the last evaluation under a stopping policy, with a shared reset set used for both stopping and evaluation. Training budgets vary.
- Plug-in power treats the observed gap and variances as population values. The 19/20 count describes these pairs, whose dependence rules out a binomial interval.
- No Atari scores are included. The rliable archive was unavailable when this analysis was collected; [issue 31](https://github.com/google-research/rliable/issues/31) describes the access problem.
