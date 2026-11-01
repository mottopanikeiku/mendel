# Running mendel

## Analyze the saved scores

Install the pinned Python environment:

```sh
uv sync --locked --python 3.12
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
```

These commands recompute the analyses and figures without training policies:

```sh
uv run python scripts/public_analysis.py
uv run python scripts/analyze.py confirmation
uv run python scripts/figures.py
uv run python scripts/neural_analyze.py confirmation
uv run python scripts/neural_figures.py
uv run python scripts/calibrate.py
uv run python scripts/null_diagnostics.py
```

The affine and PPO analyzers rebuild the pilot plans, compare counts and statuses exactly, and check file hashes. They allow last-bit floating-point differences across CPU architectures and keep saved plan values in the outputs. Regenerated PNGs and the last digits of `results/public*` can differ across machines.

The optional `scripts/import_public.py` downloads the pinned 0.7 MB coldstart archive and checks archive and score hashes before rebuilding `data/public_scores.csv`.

## Tests and result checks

```sh
uv run ruff check .
uv run python -m pytest -q
uv run python scripts/calibrate.py --check
uv run python scripts/null_diagnostics.py --check
```

The [calibration check](CALIBRATION.md) regenerates the simulation, resampling tables, report and figure in memory, then compares them with the saved outputs and README table. The [null-control check](../results/neural/null_diagnostics.md) recomputes five tests, resampling estimates and batch diagnostics; it also checks numbers quoted next to report links.

Neural implementation tests need the optional CPU JAX dependencies:

```sh
uv sync --locked --python 3.12 --extra neural
JAX_PLATFORMS=cpu uv run python -m pytest -q
```

CI installs this extra and runs on x86-64 and arm64. No GPU is needed for the tests.

## Repeat the affine training run

Training uses Modal and incurs cloud charges. The runner requires the exact plan revision and refuses to overwrite an existing result directory. Repeating these commands uses the original pseudorandom training seeds.

```sh
git switch --detach 55a5e58e9156069bd7ff33fa744cd1c27b3a0028
uv sync --locked --python 3.12
SEED_POWER_TIMEOUT_SECONDS=600 SEED_POWER_CONTAINERS=8 uv run modal run modal_app.py --stage confirmation --output results/replication/confirmation --plan-commit 55a5e58e9156069bd7ff33fa744cd1c27b3a0028
uv run python scripts/analyze.py confirmation --confirmation results/replication/confirmation/runs.jsonl --output results/replication
```

Each container requests one CPU core and 1 GiB RAM, with one numerical thread and at most eight concurrent containers. The [runtime file](../results/confirmation/environment.json) includes per-batch durations. Modal reported the CPU model as unknown.

## Repeat or resume PPO training

The transport revision below supports resumable batches while keeping the original training code, design and count plan:

```sh
git switch --detach 4681992
uv sync --locked --python 3.12
SEED_POWER_TIMEOUT_SECONDS=3600 uv run --python 3.12 modal run neural_modal.py --stage confirmation --output results/replication/neural --plan-commit 59cc9168e36e584d91decb3c98739c6754b34ef5
uv run python scripts/neural_analyze.py confirmation --confirmation results/replication/neural/runs.jsonl --output results/replication/neural
uv run python scripts/neural_figures.py --neural results/replication/neural/summary.json --output figures/replication
```

One container requests an L4, two CPU cores and 8 GiB RAM, processing 64 policies per batch. Completed batches are saved to the `seed-power-day-results` Modal Volume. The cache keys include the design, plan, training source and runtime image. Rerunning exports saved batches and submits missing ones, with up to five sequential clients sharing the deadline. An empty cache retrains the original seeds.

Batch persistence and export add cloud wall time beyond the training-only throughput estimate. [PPO experiment details](NEURAL_PROTOCOL.md).

## Cloud cost estimates

| Experiment | Estimated compute, including failed attempts | Rounded up |
|---|---:|---:|
| [Affine](../results/cost.json) | $0.0221 | $0.03 |
| [PPO](../results/neural/cost.json) | $1.1905 | $1.20 |

These estimates use allocated hardware and app wall time; they exclude recurring Volume storage.

## Collect a new experiment

Choose comparisons, design parameters and unused phase seed ranges before collecting scores. Use a new output directory for each phase. The pilot output path must match the analyzer's `--pilot` argument. The affine workflow is:

```sh
SEED_POWER_TIMEOUT_SECONDS=300 uv run modal run modal_app.py --stage development --output results/development
SEED_POWER_TIMEOUT_SECONDS=300 uv run modal run modal_app.py --stage pilot --output results/pilot
uv run python scripts/analyze.py plan
# Commit and push the pilot, design and results/pilot_plan.json before continuing.
SEED_POWER_TIMEOUT_SECONDS=600 uv run modal run modal_app.py --stage confirmation --output results/confirmation --plan-commit YOUR_PUSHED_PLAN_COMMIT
```

These experiment runners use the observed-gap planning rule studied in the training experiments. For the recommended fixed-effect rule, use the standalone planner with an effect chosen before the pilot:

```sh
uv run python scripts/plan.py --effect 0.5 --sd-a 1 --sd-b 1 --pilot-seeds 8
```

This gives 72 new seeds per arm, allowing for the uncertainty of SDs estimated from eight pilot seeds. The planner assumes the pilot and future runs share the same variance.
