"""Print an equal-allocation independent-training-seed budget as JSON."""

from __future__ import annotations

import argparse
import json
import math

from seed_power.stats import plan_seeds, plan_seeds_with_pilot_uncertainty, power_at_n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--effect", type=float, required=True,
                        help="Smallest raw mean gap worth detecting, fixed in advance; sign ignored")
    parser.add_argument("--sd-a", type=float, required=True, help="Seed-level SD in arm A")
    parser.add_argument("--sd-b", type=float, required=True, help="Seed-level SD in arm B")
    parser.add_argument("--pilot-seeds", type=int,
                        help="Pilot seeds per arm behind the SDs; adds the Julious-Owen "
                             "allowance for their uncertainty, the rule docs/CALIBRATION.md "
                             "recommends")
    parser.add_argument("--power", type=float, default=0.8, help="Target power (default: 0.8)")
    parser.add_argument("--alpha", type=float, default=0.05, help="Two-sided size (default: 0.05)")
    args = parser.parse_args()
    variances = []
    for name, sd in (("sd-a", args.sd_a), ("sd-b", args.sd_b)):
        if not math.isfinite(sd) or sd < 0.0:
            parser.error(f"--{name} must be finite and nonnegative")
        variance = sd * sd
        if not math.isfinite(variance) or (sd > 0.0 and variance == 0.0):
            parser.error(f"--{name} squared must be representable as a finite positive variance")
        variances.append(variance)
    try:
        plug_in = plan_seeds(args.effect, *variances, target=args.power, alpha=args.alpha)
        if args.pilot_seeds is None:
            n = plug_in
        else:
            n = plan_seeds_with_pilot_uncertainty(args.effect, *variances, args.pilot_seeds,
                                                  target=args.power, alpha=args.alpha)
        modeled_power = (
            power_at_n(args.effect, *variances, n=n, alpha=args.alpha) if n is not None else None
        )
    except (ValueError, FloatingPointError) as error:
        parser.error(str(error))
    reason = None
    if n is None:
        reason = "zero_effect" if args.effect == 0.0 else "zero_total_variance"
    if args.pilot_seeds is None:
        uncertainty = "SDs treated as known, without pilot uncertainty."
    else:
        uncertainty = (f"SDs estimated from {args.pilot_seeds} pilot seeds per arm; Julious-Owen "
                       "noncentral-t allowance for their uncertainty, Satterthwaite pilot df.")
    result = {
        "n": n,
        "n_per_arm": n,
        "total_training_seeds": 2 * n if n is not None else None,
        "plug_in_n_per_arm": plug_in,
        "pilot_seeds": args.pilot_seeds,
        "effect_absolute": abs(args.effect),
        "sd_a": args.sd_a,
        "sd_b": args.sd_b,
        "target_power": args.power,
        "alpha": args.alpha,
        "modeled_power": modeled_power,
        "unplannable_reason": reason,
        "assumptions": [
            "Two-sided independent Welch comparison, not paired training seeds.",
            "Equal allocation; n and n_per_arm both count training seeds in each arm.",
            "Noncentral-t approximation with variance-based Welch-Satterthwaite degrees of freedom.",
            "Approximately normal independent seed-level returns; unequal-variance power is approximate.",
            "The effect is fixed in advance; planning from a pilot's observed gap is not calibrated.",
            uncertainty,
            "modeled_power plugs the SDs into the noncentral-t formula at n.",
            "No budget cap or substituted standard deviation; zero effect or total variance is unplannable.",
            "Nominal modeled power is not guaranteed prospective detection frequency.",
        ],
    }
    print(json.dumps(result, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
