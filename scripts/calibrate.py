"""Write the seed-planning calibration: SIM tables, committed-root resampling, report, figure.

SIM rows simulate known populations with seed_power.calibration. Resampling
rows draw disjoint pilots and confirmations from committed PPO root scores and
train nothing. ``--check`` regenerates everything in memory, compares it with
the committed files and checks the claims the tables support.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import tempfile

import numpy as np

from seed_power import calibration as sim
from seed_power.resampling import half_splits, resample_population, summarize_populations
from seed_power.stats import plan_seeds, wilson_interval

if __package__:
    from .neural_analyze import load_records, raw_path, validate_records
else:
    from neural_analyze import load_records, raw_path, validate_records

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261029
SIZES = {"outer": 2000, "inner": 8, "reference_trials": 20000, "splits": 4, "replicates": 500}
REFERENCE_COUNTS = (2, 3, 5, 8, 16, 32, 64, 128, 256, 512)
TOLERANCE = 0.02
DESIGN = "configs/neural_design.json"
PILOT_RAW = "results/neural/pilot/runs.jsonl.gz"
CONFIRMATION_RAW = "results/neural/confirmation/runs.jsonl.gz"
PLAN = "results/neural/pilot_plan.json"
FRESH = "results/neural/summary.json"
OUTPUT = "results/calibration"
FIGURE = "figures/known_truth.svg"
TABLES = ("reference", "workflow", "resampling")
LABELS = {
    "observed_gap": "Observed pilot gap",
    "fixed_effect": "Fixed effect, pilot variance",
    "variance_ucl": "Fixed effect, 80% variance UCL",
    "variance_nct": "Fixed effect, noncentral-t (Julious-Owen)",
    "known_variance": "Fixed effect, true variance (oracle)",
}
FAMILY_LABELS = {
    "gaussian": "Gaussian, equal SD",
    "gaussian_unequal": "Gaussian, treatment SD x2",
    "bounded": "Bounded with a ceiling",
    "multimodal": "Failure/success bimodal",
}
BOUNDARY = {"CartPole-v1": 500.0, "Acrobot-v1": -500.0}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# SIM.

def gaussian(arm):
    return isinstance(arm, sim.Normal)


def describe(arm):
    if gaussian(arm):
        return {"kind": "normal", "mean": arm.mean, "sd": arm.sd}
    return {"kind": "grid", "mean": arm.mean, "sd": arm.sd,
            "pmf": [float(value) for value in arm.pmf]}


def reference_rows(trials):
    """Fixed-count Welch rejection: Monte Carlo, noncentral-t formula, exact Gaussian."""
    rows = []
    for index, (family, label, arm_a, arm_b, _) in enumerate(sim.scenarios()):
        truth = arm_a.mean - arm_b.mean
        total = arm_a.variance + arm_b.variance
        for count_index, n in enumerate(REFERENCE_COUNTS):
            rng = sim.stream(SEED, (1, index, count_index))
            sizes = np.full(trials, n)
            mean_a, var_a = arm_a.sample_moments(rng, sizes)
            mean_b, var_b = arm_b.sample_moments(rng, sizes)
            reject, gap = sim.welch_rejections(mean_a, var_a, mean_b, var_b, sizes)
            rate = float(reject.mean())
            formula = float(sim.planning_power(abs(truth) / math.sqrt(total),
                                               arm_a.variance / total, n))
            exact = wrong = None
            if gaussian(arm_a) and gaussian(arm_b):
                exact, wrong = sim.exact_welch(truth, arm_a.variance, arm_b.variance, n)
            rows.append({
                "population": family, "truth_label": label, "truth_effect": truth,
                "seeds_per_arm": n, "trials": trials, "mc_rate": rate,
                "mc_se": math.sqrt(rate * (1 - rate) / trials),
                "mc_wrong_sign": (float((reject & (np.sign(gap) != np.sign(truth))).mean())
                                  if label != "null" else None),
                "noncentral_t_formula": formula if label != "null" else sim.ALPHA,
                "exact_rate": exact, "exact_wrong_sign": wrong,
            })
    return rows


def workflow_rows(outer, inner):
    """Every rule on every population, truth and pilot size, with exact Gaussian columns."""
    rows, tables = [], {}
    for index, (family, label, arm_a, arm_b, effect) in enumerate(sim.scenarios()):
        truth = arm_a.mean - arm_b.mean
        exact = None
        if gaussian(arm_a) and gaussian(arm_b):
            tables[index] = sim.exact_welch_table(truth, arm_a.variance, arm_b.variance)
            exact = tables[index][0]
        equal = exact is not None and arm_a.sd == arm_b.sd
        for pilot_index, pilot in enumerate(sim.PILOT_SIZES):
            simulated = sim.simulate(arm_a, arm_b, effect, pilot, outer, inner, SEED,
                                     (0, index, pilot_index), exact,
                                     null=label == "null")
            for row in simulated:
                extra = {"exact_workflow_rate": None, "exact_eligibility": None,
                         "exact_mean_total_seeds": None, "exact_wrong_sign_rate": None}
                if equal:
                    rejection, wrong = tables[index]
                    if row["rule"] == "known_variance":
                        pmf = np.zeros(sim.CAP + 1)
                        pmf[plan_seeds(effect, arm_a.variance, arm_b.variance)] = 1.0
                    else:
                        pmf, _ = sim.exact_plan_distribution(row["rule"], pilot, arm_a.sd,
                                                             effect, truth)
                    extra = {
                        "exact_workflow_rate": float(pmf @ rejection),
                        "exact_eligibility": float(pmf.sum()),
                        "exact_mean_total_seeds": float(2 * pilot + 2 * pmf @ np.arange(pmf.size)),
                        "exact_wrong_sign_rate": (float(pmf @ wrong) if label != "null"
                                                  else None),
                    }
                rows.append({"population": family, "truth_label": label, "truth_effect": truth,
                             "meaningful_effect": effect, "pilot_seeds": pilot, **row, **extra})
    return rows


def sim_configuration():
    populations = []
    for family, label, arm_a, arm_b, effect in sim.scenarios():
        count = plan_seeds(effect, arm_a.variance, arm_b.variance)
        total = arm_a.variance + arm_b.variance
        truth = arm_a.mean - arm_b.mean
        populations.append({
            "population": family, "truth_label": label, "truth_effect": truth,
            "meaningful_effect": effect, "arm_a": describe(arm_a), "arm_b": describe(arm_b),
            "known_variance_seeds_per_arm": count,
            "known_variance_formula_power_at_meaningful_effect": float(
                sim.planning_power(effect / math.sqrt(total), arm_a.variance / total, count)),
        })
    return populations


# Resampling committed roots.

def load_pools(root):
    """Validated committed PPO roots grouped by task and variant, ordered by seed."""
    design = json.loads((root / DESIGN).read_text())
    plans = json.loads((root / PLAN).read_text())
    pilot = load_records(root / PILOT_RAW)
    confirmation = load_records(root / CONFIRMATION_RAW)
    validate_records(design, "pilot", pilot)
    validate_records(design, "confirmation", confirmation, plans)
    seeds = [record["seed"] for record in pilot + confirmation]
    if len(seeds) != len(set(seeds)):
        raise ValueError("a root seed appears in both committed phases")
    pools = {}
    for record in sorted(pilot + confirmation, key=lambda record: record["seed"]):
        pools.setdefault((record["task"], record["variant"]), []).append(record["score"])
    return design, {key: np.asarray(values) for key, values in pools.items()}


def comparisons(design):
    """Per task: a base-vs-base null, a shifted base, then each committed variant pair."""
    for task in design["tasks"]:
        yield task, "base", "base", "null"
        yield task, "base", "base", "shift"
        for pair in design["comparisons"]:
            if not pair["null_control"]:
                yield task, pair["a"], pair["b"], "variant"


def resampling_rows(design, pools, splits, replicates):
    halves = {}
    variants = sorted({variant for _, variant in pools})
    for task_index, task in enumerate(design["tasks"]):
        for variant_index, variant in enumerate(variants):
            rng = sim.stream(SEED, (2, task_index, variant_index))
            halves[task, variant] = half_splits(len(pools[task, variant]), splits, rng)
    rows = []
    for comparison_index, (task, a, b, kind) in enumerate(comparisons(design)):
        task_index = design["tasks"].index(task)
        outcomes, truths, effects = [], [], []
        for split in range(splits):
            for half in (0, 1):
                arm_a = pools[task, a][halves[task, a][split][half]]
                arm_b = pools[task, b][halves[task, b][split][half]]
                control_sd = float(np.std(arm_b))
                if kind == "variant":
                    shift, effect = 0.0, abs(float(arm_a.mean() - arm_b.mean()))
                else:
                    effect = 0.5 * control_sd
                    shift = effect if kind == "shift" else 0.0
                rng = sim.stream(SEED, (3, task_index, comparison_index, split, half))
                outcomes.append(resample_population(arm_a, arm_b, kind != "variant", shift,
                                                    effect, sim.PILOT_SIZES, replicates, rng))
                truths.append(outcomes[-1][sim.PILOT_SIZES[0], sim.RULES[0]]["truth"])
                effects.append(effect)
        for pilot in sim.PILOT_SIZES:
            for rule in sim.RULES:
                summary = summarize_populations([outcome[pilot, rule] for outcome in outcomes],
                                                pilot)
                if kind == "null":
                    summary["wrong_sign_rate"] = None
                rows.append({"task": task, "a": a, "b": b, "kind": kind,
                             "truth_mean": float(np.mean(truths)),
                             "truth_min": float(np.min(truths)), "truth_max": float(np.max(truths)),
                             "meaningful_effect_mean": float(np.mean(effects)),
                             "pilot_seeds": pilot, "rule": rule, **summary})
    return rows


def pool_summary(pools):
    return [{"task": task, "variant": variant, "roots": int(values.size),
             "mean": float(values.mean()), "sd": float(values.std(ddof=1)),
             "boundary_fraction": float(np.mean(values == BOUNDARY[task]))}
            for (task, variant), values in sorted(pools.items())]


# Recommendation and report.

def recommend(workflow):
    """Cheapest pilot-only rule that keeps every SIM population near the target.

    A rule holds at a pilot size if its workflow power at the meaningful effect
    is at least TARGET - TOLERANCE in every Gaussian, bounded and bimodal
    population. Among rules that hold at every pilot size, the lowest mean
    total seed cost wins. If none does, the rule with the best worst case wins
    and ``holds_everywhere`` is False.
    """
    by_rule = {}
    for rule in sim.PILOT_RULES:
        by_rule[rule] = []
        for pilot in sim.PILOT_SIZES:
            rows = select(workflow, rule=rule, pilot_seeds=pilot, truth_label="meaningful")
            powers = [row["workflow_rate"] for row in rows]
            by_rule[rule].append({
                "pilot_seeds": pilot, "min_power": min(powers), "max_power": max(powers),
                "mean_total_seeds": float(np.mean([row["mean_total_seeds"] for row in rows])),
                "holds": min(powers) >= sim.TARGET - TOLERANCE,
            })
    cost = {rule: np.mean([entry["mean_total_seeds"] for entry in entries])
            for rule, entries in by_rule.items()}
    holding = [rule for rule, entries in by_rule.items() if all(e["holds"] for e in entries)]
    if holding:
        chosen = min(holding, key=cost.get)
    else:
        chosen = max(by_rule, key=lambda rule: min(e["min_power"] for e in by_rule[rule]))
    return {"rule": chosen, "holds_everywhere": bool(holding), "by_rule": by_rule}


def percent(value):
    return f"{100 * value:.1f}%"


def span(values):
    low, high = min(values), max(values)
    return percent(low) if f"{100 * low:.1f}" == f"{100 * high:.1f}" else (
        f"{100 * low:.1f}–{100 * high:.1f}%")


def select(rows, **conditions):
    return [row for row in rows if all(row[key] == value for key, value in conditions.items())]


def headline_table(workflow):
    """Markdown table shared by the report and the README."""
    lines = ["| Planning rule | 8-seed pilot: power | total seeds | "
             "32-seed pilot: power | total seeds |",
             "|---|---:|---:|---:|---:|"]
    for rule in sim.RULES:
        cells = [LABELS[rule]]
        for pilot in (8, 32):
            rows = select(workflow, rule=rule, pilot_seeds=pilot, truth_label="meaningful")
            cells.append(span([row["workflow_rate"] for row in rows]))
            costs = [row["mean_total_seeds"] for row in rows]
            cells.append(f"{min(costs):.0f}–{max(costs):.0f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_report(configuration, reference, workflow, resampling, recommendation, fresh):
    chosen = recommendation["rule"]
    by_rule = recommendation["by_rule"]
    meaningful = select(workflow, truth_label="meaningful")
    observed = select(meaningful, rule="observed_gap")
    oracle = select(meaningful, rule="known_variance")
    fixed_rules = ("fixed_effect", "variance_nct", "variance_ucl")
    half = [row for row in select(workflow, truth_label="half") if row["rule"] in fixed_rules]
    shape = [row for row in select(workflow, population=sim.SHAPE_NULL)
             if row["rule"] in fixed_rules]
    nulls = [row for row in select(workflow, truth_label="null")
             if row["population"] != sim.SHAPE_NULL and row["rule"] in fixed_rules]
    observed_null = select(workflow, truth_label="null", rule="observed_gap")
    average = {rule: np.mean([entry["mean_total_seeds"] for entry in entries])
               for rule, entries in by_rule.items()}
    others = []
    for rule in fixed_rules:
        if rule == chosen:
            continue
        held = [str(entry["pilot_seeds"]) for entry in by_rule[rule] if entry["holds"]]
        where = (f"held at pilot sizes {', '.join(held)}" if held and len(held) < len(by_rule[rule])
                 else "held at every pilot size" if held else "held at no pilot size")
        others.append(f"`{rule}` {where} and used {average[rule] / average[chosen]:.2f} times "
                      "the seeds on average")
    real = select(resampling, rule=chosen)
    shifted = select(real, kind="shift")
    real_nulls = select(real, kind="null")
    lines = [
        "# Seed-planning calibration report",
        "",
        "`uv run python scripts/calibrate.py` writes this file, the tables beside it and "
        "`figures/known_truth.svg`. Rows marked SIM are simulations with known truth. Rows "
        "marked resampling reuse committed PPO root scores; they train nothing and are not new "
        "independent evidence.",
        "",
        "## Recommendation",
        "",
        "I chose the rule from the SIM tables alone. The committed-root resampling below "
        "is a check on that choice and did not change it.",
        "",
        "1. Fix the smallest effect you care about before running the pilot, and plan from "
        "that effect, not from the pilot's mean gap. Planning from the observed pilot gap "
        f"gave {span([row['workflow_rate'] for row in observed])} power when the true effect "
        "equalled the meaningful one, and put "
        f"{span([1 - row['eligibility'] for row in observed_null])} of null pilots above "
        f"{sim.CAP} seeds per arm (SIM).",
        f"2. Then plan with `{chosen}` ({LABELS[chosen]}). "
        + (f"It was the cheapest rule whose power stayed at or above "
           f"{percent(sim.TARGET - TOLERANCE)} in all four SIM populations at all four pilot "
           "sizes. " if recommendation["holds_everywhere"] else
           f"No rule stayed at or above {percent(sim.TARGET - TOLERANCE)} everywhere; this "
           "one had the best worst case. ")
        + "Its power ranged "
        + span([value for entry in by_rule[chosen]
                for value in (entry["min_power"], entry["max_power"])])
        + " over all populations and pilot sizes. "
        + "Among the other fixed-effect rules, " + "; ".join(others) + ".",
        "3. If the true effect is only half the one you planned for, no rule rescues the "
        f"study: the fixed-effect rules gave {span([row['workflow_rate'] for row in half])} "
        "power (SIM).",
        "4. The Welch test, not the planning rule, decides the false-positive rate. With "
        "identical null arms the fixed-effect rules rejected "
        f"{span([row['workflow_rate'] for row in nulls])}. When the two arms had equal means "
        "but different shapes (one bimodal with a failure mode, one unimodal), they rejected "
        f"{span([row['workflow_rate'] for row in shape])} (SIM, nominal 5%).",
        f"5. On resampled committed PPO roots, `{chosen}` gave "
        f"{span([row['workflow_rate'] for row in shifted])} power when I shifted one base arm "
        "by its planned effect, and rejected "
        f"{span([row['workflow_rate'] for row in real_nulls])} of base-versus-base nulls, "
        "across both tasks and all pilot sizes. This is resampling of existing roots, not "
        "fresh training.",
        "",
        "Power at the meaningful effect, lowest to highest over the four SIM populations, "
        "with mean total seeds (both arms, including the pilot) in brackets:",
        "",
        "| Pilot seeds per arm | " + " | ".join(LABELS[rule] for rule in fixed_rules) + " |",
        "|---:|" + "---:|" * len(fixed_rules),
    ]
    for index, pilot in enumerate(sim.PILOT_SIZES):
        cells = []
        for rule in fixed_rules:
            entry = by_rule[rule][index]
            cells.append(f"{span([entry['min_power'], entry['max_power']])} "
                         f"({entry['mean_total_seeds']:.0f})")
        lines.append(f"| {pilot} | " + " | ".join(cells) + " |")
    lines += [
        "",
        f"The known-variance oracle gave {span([row['workflow_rate'] for row in oracle])} at "
        "the same effect, so even perfect variance knowledge does not give exactly 80% "
        "outside the Gaussian model."
        + (" `uv run python scripts/plan.py --effect E --sd-a A --sd-b B --pilot-seeds N` "
           "applies the recommended rule to your own pilot SDs." if chosen == "variance_nct"
           else ""),
        "",
        "## Power at the meaningful effect (SIM)",
        "",
        "Workflow power counts plans above the cap as misses. Ranges run over the four "
        "populations; per-population values with Monte Carlo errors are in `workflow.csv`.",
        "",
        headline_table(workflow),
        "",
    ]
    lines += ["| Population | Pilot | " + " | ".join(f"`{rule}`" for rule in sim.RULES) + " |",
              "|---|---:|" + "---:|" * len(sim.RULES)]
    for family in sim.FAMILIES:
        for pilot in sim.PILOT_SIZES:
            cells = []
            for rule in sim.RULES:
                row = select(meaningful, population=family, pilot_seeds=pilot, rule=rule)[0]
                cells.append(f"{percent(row['workflow_rate'])} ± "
                             f"{100 * 1.96 * row['workflow_mc_se']:.1f} "
                             f"({row['mean_total_seeds']:.0f})")
            lines.append(f"| {FAMILY_LABELS[family]} | {pilot} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Each cell is workflow power ± a 95% Monte Carlo half-width, then mean total seeds.",
        "",
        "## Null rejection (SIM)",
        "",
        "| Population | Pilot | " + " | ".join(f"`{rule}`" for rule in sim.RULES) + " |",
        "|---|---:|" + "---:|" * len(sim.RULES),
    ]
    for family in list(sim.FAMILIES) + [sim.SHAPE_NULL]:
        for pilot in sim.PILOT_SIZES:
            cells = []
            for rule in sim.RULES:
                row = select(workflow, population=family, truth_label="null", pilot_seeds=pilot,
                             rule=rule)[0]
                cells.append(f"{percent(row['workflow_rate'])} ± "
                             f"{100 * 1.96 * row['workflow_mc_se']:.1f}")
            name = FAMILY_LABELS.get(family, "Equal means, bimodal vs unimodal")
            lines.append(f"| {name} | {pilot} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "The observed-gap rule rejects less often under the null mainly because many of its "
        "null plans exceed the cap and are not run.",
        "",
        "## Exact Gaussian checks",
        "",
    ]
    exact_rows = [row for row in workflow if row["exact_workflow_rate"] is not None]
    worst = max(abs(row["workflow_rate"] - row["exact_workflow_rate"]) / row["workflow_mc_se"]
                for row in exact_rows if row["workflow_mc_se"] > 0)
    conditional = [row for row in workflow if row["conditional_exact_rate"] is not None
                   and row["conditional_exact_mc_se"] > 0]
    worst_conditional = max(abs(row["workflow_rate"] - row["conditional_exact_rate"])
                            / row["conditional_exact_mc_se"] for row in conditional)
    fixed = [row for row in reference if row["exact_rate"] is not None]
    worst_fixed = max(abs(row["mc_rate"] - row["exact_rate"]) / math.sqrt(
        row["exact_rate"] * (1 - row["exact_rate"]) / row["trials"]) for row in fixed)
    gaps = {small: max(abs(row["noncentral_t_formula"] - row["exact_rate"]) for row in fixed
                       if row["truth_label"] != "null" and (row["seeds_per_arm"] < 8) == small)
            for small in (True, False)}
    lines += [
        "For equal-variance Gaussian arms I compute every rule's plan distribution, power, "
        "eligibility, cost and wrong-sign rate by quadrature, with no Monte Carlo error "
        "(`exact_*` columns in `workflow.csv`). The largest simulated-minus-exact workflow "
        f"power was {worst:.2f} Monte Carlo SEs across {len(exact_rows)} rows. For all Gaussian "
        "rows, including unequal variances, the simulated rate agrees with the exact power "
        f"of each pilot's own plan within {worst_conditional:.2f} SEs ({len(conditional)} "
        "rows). Fixed-count Welch rejection matched exact quadrature within "
        f"{worst_fixed:.2f} binomial SEs ({len(fixed)} rows). The planner's noncentral-t "
        "formula differed from exact Welch power by at most "
        f"{100 * gaps[True]:.2f} points below 8 seeds per arm and {100 * gaps[False]:.2f} "
        "points from 8 seeds on.",
        "",
        "## Committed PPO roots, resampled",
        "",
        "Each task/variant pool of committed pilot and confirmation roots is split "
        f"{configuration['sizes']['splits']} times into disjoint halves. Each half is a finite "
        "population with an exactly known gap. Every replicate draws a pilot and then a "
        "confirmation without replacement, so no root is in both. `null` compares base with "
        "base; `shift` adds half the base SD to one arm's roots, a known effect on real score "
        "shapes; the other rows are the committed variant pairs, planned at their own "
        "population gap. SE combines Monte Carlo and finite-pool (delete-half jackknife) "
        "error.",
        "",
        "| Task | Comparison | Gap | Pilot | " + " | ".join(f"`{rule}`" for rule in sim.RULES)
        + " |",
        "|---|---|---:|---:|" + "---:|" * len(sim.RULES),
    ]
    for task in dict.fromkeys(row["task"] for row in resampling):
        for row in select(resampling, task=task, pilot_seeds=8, rule="observed_gap"):
            for pilot in (8, 32):
                cells = []
                for rule in sim.RULES:
                    match = select(resampling, task=task, a=row["a"], b=row["b"],
                                   kind=row["kind"], pilot_seeds=pilot, rule=rule)[0]
                    cells.append(f"{percent(match['workflow_rate'])} ± "
                                 f"{100 * 1.96 * match['workflow_total_se']:.1f} "
                                 f"({percent(match['eligibility'])} run)")
                name = {"null": "base vs base", "shift": "base + shift vs base"}.get(
                    row["kind"], f"{row['a']} vs {row['b']}")
                lines.append(f"| {task} | {name} | {row['truth_mean']:.1f} | {pilot} | "
                             + " | ".join(cells) + " |")
    rare = [f"{row['task']} {row['a']} vs {row['b']}"
            for row in select(resampling, kind="variant", pilot_seeds=8, rule="fixed_effect")
            if all(match["eligibility"] < 0.01 for match in select(
                resampling, task=row["task"], a=row["a"], b=row["b"], kind="variant")
                if match["rule"] in ("fixed_effect", "variance_nct", "variance_ucl"))]
    lines += [
        "",
        "Each cell is workflow rejection ± a 95% half-width, then the fraction of plans run. "
        "Gaps are mean(a) − mean(b) averaged over the half populations. Variant pairs are "
        "planned at each half population's own gap, which moves with finite-pool noise. Near "
        f"the {sim.CAP}-seed cap that decides whether a plan runs at all, so those rows carry "
        "large finite-pool errors, and rules that inflate the variance push more plans over "
        "the cap. The fixed-effect rules ran under 1% of plans for "
        f"{', '.join(rare)}: under the noncentral-t planning model, the finite-population "
        f"gaps of these pairs need far more than {sim.CAP} seeds per arm.",
        "",
        "## Fresh training versus resampling",
        "",
        "The committed fresh-seed experiment planned from 8-seed pilots with the observed gap. "
        "Resampling the same rule from the committed roots should give similar detection "
        "rates if the roots are a fair stand-in for new training. Fresh rates are MEAS; "
        "resampled rates are not new evidence.",
        "",
        "| Task | Comparison | Fresh: detected/run, Wilson 95% (run/planned) "
        "| Resampled: detected among run (run) |",
        "|---|---|---:|---:|",
    ]
    outside = []
    for row in fresh:
        low, high = wilson_interval(row["fresh_detections"], row["fresh_plans"])
        if not low <= row["resampled_conditional_rate"] <= high:
            outside.append(f"{row['task']} {row['a']} vs {row['b']}")
        lines.append(f"| {row['task']} | {row['a']} vs {row['b']} | "
                     f"{row['fresh_detections']}/{row['fresh_plans']} = "
                     f"{percent(row['fresh_detections'] / row['fresh_plans'])}, "
                     f"{span([low, high])} ({row['fresh_plans']}/{row['fresh_attempted']}) | "
                     f"{percent(row['resampled_conditional_rate'])} "
                     f"({percent(row['resampled_eligibility'])}) |")
    lines += [
        "",
        f"The resampled rate falls inside the fresh Wilson interval for "
        f"{len(fresh) - len(outside)} of {len(fresh)} comparisons"
        + (f"; it falls outside for {', '.join(outside)}." if outside else ".")
        + " Separate 95% intervals are not a joint test.",
        "",
        "## Limits",
        "",
        "- SIM populations are stylized: two Gaussian, one bounded with a ceiling mass and one "
        "failure/success mixture, each with a meaningful effect of half the control SD. Other "
        "shapes, heavier tails or smaller effects can behave differently.",
        "- Every rule here plans for 80% power at one prespecified effect. None protects "
        "against choosing that effect badly.",
        "- Resampling reuses the same committed roots across replicates and conditions on the "
        "fixed evaluation reset sets. It shows how the rules behave on these PPO score "
        "distributions; it is not fresh training and adds no independent seeds.",
        "- Monte Carlo intervals are pointwise, not simultaneous across the grid.",
    ]
    return "\n".join(lines) + "\n"


def fresh_comparison(resampling, root):
    summary = json.loads((root / FRESH).read_text())
    rows = []
    for group in summary["per_comparison"]:
        kind = "null" if group["null_control"] else "variant"
        match = select(resampling, task=group["task"], a=group["a"], b=group["b"], kind=kind,
                       pilot_seeds=8, rule="observed_gap")[0]
        rows.append({"task": group["task"], "a": group["a"], "b": group["b"],
                     "fresh_detections": group["detections"], "fresh_plans": group["plans"],
                     "fresh_attempted": group["attempted_plans"],
                     "resampled_conditional_rate": match["conditional_rate"],
                     "resampled_eligibility": match["eligibility"]})
    return rows


def make_figure(workflow, path):
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "svg.fonttype": "none", "svg.hashsalt": "seed-power"})
    colors = {"observed_gap": "#8b8f97", "fixed_effect": "#286a9b", "variance_ucl": "#b34d30",
              "variance_nct": "#36774b", "known_variance": "#162b3e"}
    fig, axes = plt.subplots(2, len(sim.FAMILIES), figsize=(14, 7), constrained_layout=True)
    for column, family in enumerate(sim.FAMILIES):
        top, bottom = axes[0, column], axes[1, column]
        for rule in sim.RULES:
            rows = select(workflow, population=family, truth_label="meaningful", rule=rule)
            x = [row["mean_total_seeds"] for row in rows]
            y = [row["workflow_rate"] for row in rows]
            top.errorbar(x, y, yerr=[1.96 * row["workflow_mc_se"] for row in rows],
                         color=colors[rule], marker="o", markersize=3, label=LABELS[rule])
            for row in rows:
                top.annotate(str(row["pilot_seeds"]), (row["mean_total_seeds"],
                             row["workflow_rate"]), fontsize=6, color=colors[rule],
                             xytext=(3, -8), textcoords="offset points")
            nulls = select(workflow, population=family, truth_label="null", rule=rule)
            bottom.plot(sim.PILOT_SIZES, [row["workflow_rate"] for row in nulls],
                        color=colors[rule], marker="o", markersize=3)
        top.axhline(sim.TARGET, color="#555555", linestyle="--", linewidth=.8)
        top.set(title=FAMILY_LABELS[family], ylim=(0.3, 1.0),
                xlabel="Mean total seeds, both arms, including the pilot")
        bottom.axhline(sim.ALPHA, color="#555555", linestyle="--", linewidth=.8)
        bottom.set_xscale("log", base=2)
        bottom.set_xticks(sim.PILOT_SIZES, [str(pilot) for pilot in sim.PILOT_SIZES])
        bottom.minorticks_off()
        bottom.set(ylim=(0, 0.1), xlabel="Pilot seeds per arm")
        if family == "multimodal":
            shape = select(workflow, population=sim.SHAPE_NULL, rule="variance_nct")
            bottom.plot(sim.PILOT_SIZES, [row["workflow_rate"] for row in shape],
                        color=colors["variance_nct"], linestyle=":", marker="x",
                        label="Noncentral-t, equal means but\nbimodal vs unimodal arms")
            bottom.legend(loc="upper right", fontsize=7)
        for ax in (top, bottom):
            ax.grid(alpha=.2)
            ax.set_axisbelow(True)
    axes[0, 0].set_ylabel("Power at the meaningful effect\n(labels: pilot seeds per arm)")
    axes[1, 0].set_ylabel("Rejection rate with no true difference")
    axes[0, -1].legend(loc="lower right", fontsize=7)
    fig.suptitle("Simulated seed-planning rules with known truth (SIM): power against total "
                 "seeds, and null rejection.\nWelch test at 5%, 80% power target, plans above "
                 f"{sim.CAP} seeds per arm not run. Error bars are 95% Monte Carlo intervals.")
    fig.savefig(path, metadata={"Date": None})
    plt.close(fig)


# Files.

def cell(value):
    return "" if value is None else repr(value) if isinstance(value, float) else str(value)


def csv_text(rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows({key: cell(value) for key, value in row.items()} for row in rows)
    return stream.getvalue()


def build(root, sizes):
    design, pools = load_pools(root)
    reference = reference_rows(sizes["reference_trials"])
    workflow = workflow_rows(sizes["outer"], sizes["inner"])
    resampling = resampling_rows(design, pools, sizes["splits"], sizes["replicates"])
    recommendation = recommend(workflow)
    fresh = fresh_comparison(resampling, root)
    configuration = {
        "label": ("SIM: reference.csv and workflow.csv simulate known populations. "
                  "resampling.csv reuses committed PPO root scores and trains nothing."),
        "seed": SEED, "sizes": sizes, "alpha": sim.ALPHA, "target_power": sim.TARGET,
        "maximum_seeds_per_arm": sim.CAP, "ucl_level": sim.UCL_LEVEL,
        "pilot_sizes": list(sim.PILOT_SIZES), "rules": list(sim.RULES),
        "recommendation_tolerance": TOLERANCE,
        "streams": ("NumPy PCG64 from SeedSequence(seed, spawn_key=key). Workflow: "
                    "(0, scenario, pilot index, 0) pilots and (..., 1 + rule index) "
                    "confirmations. Fixed-count reference: (1, scenario, count index). "
                    "Half splits: (2, task index, sorted variant index). Resampling: "
                    "(3, task index, comparison index, split, half)."),
        "support": [float(value) for value in sim.SUPPORT],
        "populations": sim_configuration(),
        "committed_roots": {
            "sources": {path: sha256(raw_path(root / path))
                        for path in (DESIGN, PLAN, PILOT_RAW, CONFIRMATION_RAW, FRESH)},
            "pools": pool_summary(pools),
        },
        "recommendation": recommendation,
    }
    report = render_report(configuration, reference, workflow, resampling, recommendation, fresh)
    return {"configuration": configuration, "reference": reference, "workflow": workflow,
            "resampling": resampling, "report": report}


def write(result, output, figure):
    output.mkdir(parents=True, exist_ok=True)
    (output / "configuration.json").write_text(
        json.dumps(result["configuration"], indent=2, allow_nan=False) + "\n")
    for name in TABLES:
        (output / f"{name}.csv").write_text(csv_text(result[name]))
    (output / "report.md").write_text(result["report"])
    figure.parent.mkdir(parents=True, exist_ok=True)
    make_figure(result["workflow"], figure)


def same(fresh, committed, where):
    if fresh == committed:
        return
    try:
        close = math.isclose(float(fresh), float(committed), rel_tol=1e-12, abs_tol=1e-15)
    except ValueError:
        close = False
    if not close:
        raise AssertionError(f"{where}: regenerated {fresh!r} != committed {committed!r}")


def compare_json(fresh, committed, where):
    if isinstance(committed, dict):
        assert isinstance(fresh, dict) and fresh.keys() == committed.keys(), where
        for key in committed:
            compare_json(fresh[key], committed[key], f"{where}.{key}")
    elif isinstance(committed, list):
        assert isinstance(fresh, list) and len(fresh) == len(committed), where
        for index, (left, right) in enumerate(zip(fresh, committed)):
            compare_json(left, right, f"{where}[{index}]")
    elif isinstance(committed, float) or isinstance(fresh, float):
        same(fresh, committed, where)
    else:
        assert type(fresh) is type(committed) and fresh == committed, where


def within(value, reference, se):
    """Within 4.5 Monte Carlo SEs, plus float rounding of the exact reference."""
    return abs(value - reference) <= 4.5 * se + 1e-12


def check_claims(result):
    """The claims the report relies on, each within stated Monte Carlo tolerance."""
    for row in result["reference"]:
        if row["exact_rate"] is not None:
            exact = row["exact_rate"]
            assert within(row["mc_rate"], exact,
                          math.sqrt(exact * (1 - exact) / row["trials"])), row
    for row in result["workflow"]:
        accounted = row["undefined_plans"] + row["over_cap_plans"] + row["executed_plans"]
        assert accounted == row["pilots"], row
        tests = row["pilots"] * row["confirmations_per_pilot"]
        if row["exact_workflow_rate"] is not None:
            for key, se_key, floor in (
                    ("workflow_rate", "workflow_mc_se", tests),
                    ("eligibility", "eligibility_mc_se", row["pilots"])):
                exact = row[f"exact_{key}"]
                se = max(row[se_key], math.sqrt(max(exact * (1 - exact), 0.0) / floor))
                assert within(row[key], exact, se), (key, row)
            assert within(row["mean_total_seeds"], row["exact_mean_total_seeds"],
                          row["mean_total_seeds_mc_se"]), row
        if row["conditional_exact_rate"] is not None:
            assert within(row["workflow_rate"], row["conditional_exact_rate"],
                          row["conditional_exact_mc_se"]), row
    for row in result["resampling"]:
        accounted = (row["undefined_plans"] + row["over_cap_plans"]
                     + row["pool_limited_plans"] + row["executed_plans"])
        assert accounted == row["replicates"], row


NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:e[-+]?\d+)?")


def same_svg(fresh, committed):
    """Identical SVG text, with drawn coordinates allowed to differ in the sixth figure.

    The tables already match to a relative 1e-12. Matplotlib's last printed
    coordinate digit can still differ between x86-64 and arm64, so every
    number in the SVG may differ by a relative 1e-6 and everything else must match.
    """
    return svg_difference(fresh, committed) is None


def svg_difference(fresh, committed):
    """None if same_svg holds, otherwise the first differing text or number."""
    texts = NUMBER.split(fresh), NUMBER.split(committed)
    if texts[0] != texts[1]:
        index = next((i for i, (a, b) in enumerate(zip(*texts)) if a != b), min(map(len, texts)))
        return f"text {index}: {texts[0][index:index + 2]!r} vs {texts[1][index:index + 2]!r}"
    for a, b in zip(NUMBER.findall(fresh), NUMBER.findall(committed), strict=True):
        if not math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-6):
            return f"number {a} vs {b}"
    return None


def check(root, output, figure):
    configuration = json.loads((output / "configuration.json").read_text())
    result = build(root, configuration["sizes"])
    compare_json(json.loads(json.dumps(result["configuration"], allow_nan=False)),
                 configuration, "configuration.json")
    for name in TABLES:
        committed = list(csv.DictReader(io.StringIO((output / f"{name}.csv").read_text())))
        fresh = list(csv.DictReader(io.StringIO(csv_text(result[name]))))
        assert len(fresh) == len(committed), f"{name}.csv row count"
        for index, (left, right) in enumerate(zip(fresh, committed)):
            assert list(left) == list(right), f"{name}.csv[{index}] columns"
            for key in right:
                same(left[key], right[key], f"{name}.csv[{index}].{key}")
    assert result["report"] == (output / "report.md").read_text(), "report.md is stale"
    readme = (root / "README.md").read_text()
    assert headline_table(result["workflow"]) in readme, "README calibration table is stale"
    for line in readme.splitlines():
        if "simulat" in line.lower() or "resampl" in line.lower():
            for number in re.findall(r"\d+(?:\.\d+)?(?:–\d+(?:\.\d+)?)?%", line):
                assert number in result["report"], f"README number {number} is not in the report"
    with tempfile.TemporaryDirectory() as scratch:
        regenerated = Path(scratch) / figure.name
        make_figure(result["workflow"], regenerated)
        difference = svg_difference(regenerated.read_text(), figure.read_text())
        assert difference is None, f"{figure} is stale: {difference}"
    check_claims(result)
    print(f"Regenerated {', '.join(f'{name}.csv' for name in TABLES)}, configuration.json, "
          "report.md and the figure; all match and every checked claim holds.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="regenerate in memory and compare with committed files")
    parser.add_argument("--output", type=Path, default=ROOT / OUTPUT)
    parser.add_argument("--figure", type=Path, default=ROOT / FIGURE)
    for key, value in SIZES.items():
        parser.add_argument(f"--{key.replace('_', '-')}", type=int, default=value)
    args = parser.parse_args()
    if args.check:
        check(ROOT, args.output, args.figure)
        return
    sizes = {key: getattr(args, key) for key in SIZES}
    result = build(ROOT, sizes)
    check_claims(result)
    write(result, args.output, args.figure)
    print(result["report"].split("## Power at the meaningful effect")[0])


if __name__ == "__main__":
    main()
