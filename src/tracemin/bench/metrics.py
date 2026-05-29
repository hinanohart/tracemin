"""Run the failure-injection suite and emit measured metrics.

Every figure here is produced by running the real engine against synthetic cases
whose ground truth is known by construction. The headline honesty result is the
**false-reproducer rate with vs without the failure signature**: dropping the
signature lets the engine "reproduce" a *different* failure (the decoy), which the
signature-gated engine refuses.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
from pathlib import Path
from statistics import fmean

from tracemin import __version__
from tracemin.bench import inject
from tracemin.engine import minimize
from tracemin.oracle import ExitCodeOracle
from tracemin.replay import Verdict, run_trial
from tracemin.signature import failure_signature

BENCH_SCHEMA = "tracemin-bench/1"


def _recovers(minimal_ids: tuple[str, ...], ground_truth: frozenset[str]) -> bool:
    return set(ground_truth) <= set(minimal_ids)


def _is_one_minimal(case: inject.Case, minimal_ids: tuple[str, ...]) -> bool:
    traj = case.trajectory
    oracle = ExitCodeOracle()
    by_id = traj.by_id()
    order_of = {a.id: a.order for a in traj.atoms}
    removable = set(traj.removable_ids)
    minimal = set(minimal_ids)
    ref = run_trial(case.replay_fn, oracle, failure_signature, list(traj.atoms)).signature
    for atom_id in minimal & removable:
        survivors = traj.survivors_after_removing(removable - (minimal - {atom_id}))
        if survivors is None:
            continue
        atoms = [by_id[i] for i in sorted(survivors, key=lambda i: order_of[i])]
        res = run_trial(case.replay_fn, oracle, failure_signature, atoms)
        if res.verdict is Verdict.FAIL and (ref is None or res.signature == ref):
            return False  # removing this atom still reproduces the same failure
    return True


def run_suite(
    *, seed: int = 0, n_per_family: int = 20, stoch_k: int = 40, stoch_size: int = 12
) -> dict[str, object]:
    rng = random.Random(seed)
    oracle = ExitCodeOracle()

    rec_p: list[float] = []
    rec_r: list[float] = []
    reduction: list[float] = []
    calls: list[int] = []
    one_minimal = 0
    total = 0
    for gen in inject.DETERMINISTIC_FAMILIES:
        for _ in range(n_per_family):
            case = gen(rng)
            res = minimize(case.trajectory, case.replay_fn, oracle, double_check=False)
            minimal = set(res.minimal_ids)
            gt = set(case.ground_truth)
            inter = len(minimal & gt)
            rec_p.append(inter / len(minimal) if minimal else 0.0)
            rec_r.append(inter / len(gt) if gt else 1.0)
            reduction.append(1.0 - len(minimal) / len(case.trajectory.atoms))
            calls.append(int(res.stats["replay_calls"]))
            total += 1
            if _is_one_minimal(case, res.minimal_ids):
                one_minimal += 1

    # decoy: false-reproducer rate with the signature vs without it
    fr_sig = fr_nosig = n_decoy = 0
    for _ in range(n_per_family):
        case = inject.decoy(rng)
        n_decoy += 1
        with_sig = minimize(case.trajectory, case.replay_fn, oracle, double_check=False)
        no_sig = minimize(
            case.trajectory, case.replay_fn, oracle, signature_fn=None, double_check=False
        )
        if not _recovers(with_sig.minimal_ids, case.ground_truth):
            fr_sig += 1
        if not _recovers(no_sig.minimal_ids, case.ground_truth):
            fr_nosig += 1

    # controlled-stochastic pass^k on the recovered minimal set
    sc = inject.stochastic(rng, size=stoch_size, p=0.9)
    res = minimize(sc.trajectory, sc.replay_fn, oracle, double_check=False)
    by_id = sc.trajectory.by_id()
    order_of = {a.id: a.order for a in sc.trajectory.atoms}
    atoms = [by_id[i] for i in sorted(set(res.minimal_ids), key=lambda i: order_of[i])]
    c = 0
    for _ in range(stoch_k):
        if run_trial(sc.replay_fn, oracle, failure_signature, atoms).verdict is Verdict.FAIL:
            c += 1

    return {
        "recovery_precision": round(fmean(rec_p), 4),
        "recovery_recall": round(fmean(rec_r), 4),
        "reduction_ratio": round(fmean(reduction), 4),
        "replay_calls_mean": round(fmean(calls), 2),
        "minimality_verify_rate": round(one_minimal / total, 4),
        "false_reproducer_rate_no_sig": round(fr_nosig / n_decoy, 4),
        "false_reproducer_rate_with_sig": round(fr_sig / n_decoy, 4),
        "passk_under_noise_c_over_k": f"{c}/{stoch_k}",
        "n_deterministic_cases": total,
        "n_decoy_cases": n_decoy,
        "n_per_family": n_per_family,
        "seed": seed,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Run the tracemin synthetic failure-injection benchmark."
    )
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--date", default="unknown", help="date stamp for reproducible metadata")
    args = ap.parse_args(argv)

    metrics = run_suite(seed=args.seed, n_per_family=args.n)
    payload = {
        "schema": BENCH_SCHEMA,
        "version": __version__,
        "metadata": {
            "date": args.date,
            "python": platform.python_version(),
            "os": platform.system(),
            "machine": platform.machine(),
            "seed": args.seed,
            "n_per_family": args.n,
            "source": "synthetic-failure-injection (ground truth known by construction)",
        },
        "metrics": metrics,
    }
    blob = json.dumps(payload, indent=2)
    if args.out:
        Path(args.out).write_text(blob, encoding="utf-8")
    print(blob)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
