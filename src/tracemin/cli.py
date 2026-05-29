"""tracemin command-line interface: ``reduce`` / ``doctor`` / ``replay`` / ``diff``.

``doctor`` and ``diff`` work fully offline. ``reduce`` and ``replay --verify`` need a
replay engine (the ``hf`` adapter) and therefore an ``HF_TOKEN``; without one they
exit with a clear message rather than pretending to run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tracemin import __version__
from tracemin.artifact import build_artifact, render_bug_report, validate_mode
from tracemin.atoms import Trajectory
from tracemin.cache import ReplayBudget
from tracemin.doctor import run_doctor
from tracemin.engine import minimize
from tracemin.oracle import (
    AnswerMismatchOracle,
    ExceptionOracle,
    ExitCodeOracle,
    NotRegexOracle,
    Oracle,
    RegexOracle,
)


def build_oracle(spec: str) -> Oracle:
    """Parse an oracle spec like ``exit-nonzero`` / ``regex:KeyError`` / ``exception:KeyError``."""
    if spec in ("exit-nonzero", "exit"):
        return ExitCodeOracle()
    kind, sep, arg = spec.partition(":")
    if sep:
        if kind == "regex":
            return RegexOracle(arg)
        if kind in ("not-regex", "notregex"):
            return NotRegexOracle(arg)
        if kind == "exception":
            return ExceptionOracle(arg)
        if kind == "answer":
            return AnswerMismatchOracle(arg)
    raise ValueError(
        f"unknown oracle spec {spec!r}; use exit-nonzero | regex:PAT | not-regex:PAT | "
        "exception:TYPE | answer:TEXT"
    )


def load_input(path: str, adapter: str) -> Trajectory:
    from tracemin.adapters import claude, openhands

    if adapter == "openhands":
        return openhands.load_trajectory(path)
    if adapter == "claude":
        return claude.load_transcript(path)
    if adapter == "json":
        return openhands.load_trajectory(json.loads(Path(path).read_text(encoding="utf-8")))
    raise ValueError(f"unknown adapter {adapter!r}; use openhands | claude | json")


def cmd_doctor(args: argparse.Namespace) -> int:
    print(run_doctor().render())
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    a = json.loads(Path(args.a).read_text(encoding="utf-8"))
    b = json.loads(Path(args.b).read_text(encoding="utf-8"))
    sa = set(a.get("minimal_atoms", []))
    sb = set(b.get("minimal_atoms", []))
    print(f"only in {args.a}: {sorted(sa - sb)}")
    print(f"only in {args.b}: {sorted(sb - sa)}")
    print(f"shared: {sorted(sa & sb)}")
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    data = json.loads(Path(args.repro).read_text(encoding="utf-8"))
    try:
        validate_mode(data)
    except Exception as exc:  # noqa: BLE001
        print(f"artifact failed mode-consistency validation: {exc}", file=sys.stderr)
        return 1
    print(f"schema: {data.get('schema')}")
    print(f"certified: {data.get('certified')} ({data.get('certified_reason')})")
    print(f"replay_mode: {data.get('replay_mode')}")
    print(f"minimal_atoms: {data.get('minimal_atoms')}")
    print(f"reference_signature: {data.get('reference_signature')}")
    return 0


def cmd_reduce(args: argparse.Namespace) -> int:
    from tracemin.adapters.hf import HFReplay, has_live_token

    if not args.model:
        print("reduce requires --model (the hf replay engine)", file=sys.stderr)
        return 2
    if not has_live_token():
        print(
            "HF_TOKEN absent — live replay unavailable (see `tracemin doctor`). "
            "Set HF_TOKEN to run reduce against a live endpoint.",
            file=sys.stderr,
        )
        return 2

    trajectory = load_input(args.input, args.adapter)
    oracle = build_oracle(args.oracle)
    replay = HFReplay(args.model)
    budget = ReplayBudget(max_calls=args.budget_calls, max_usd=args.budget_usd)
    result = minimize(trajectory, replay, oracle, budget=budget)

    confidence_method = "illustrative"
    if args.k > 1:
        from tracemin.stochastic import certify

        result = certify(result, trajectory, replay, oracle, k=args.k)
        confidence_method = f"measured(k={args.k})"

    art = build_artifact(
        result,
        oracle_spec=args.oracle,
        replay_capable=True,
        confidence_method=confidence_method,
        k=args.k,
    )
    blob = art.to_json()
    if args.out:
        Path(args.out).write_text(blob, encoding="utf-8")
    print(blob)
    if args.report:
        print(render_bug_report(result, oracle_spec=args.oracle), file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tracemin", description="Minimal-reproducer delta-debugger for failed LLM-agent runs."
    )
    p.add_argument("--version", action="version", version=f"tracemin {__version__}")
    sub = p.add_subparsers(dest="command")

    d = sub.add_parser("doctor", help="report environment status (LIVE/MOCK/MISSING)")
    d.set_defaults(func=cmd_doctor)

    r = sub.add_parser("reduce", help="reduce a failing trajectory to a minimal reproducer")
    r.add_argument("input", help="path to the trajectory")
    r.add_argument("--adapter", default="json", choices=["openhands", "claude", "json"])
    r.add_argument("--model", default=None, help="hf model id for the replay engine")
    r.add_argument("--oracle", default="exit-nonzero", help="oracle spec (see docs)")
    r.add_argument(
        "--k",
        type=int,
        default=1,
        help="trials per candidate; k>1 enables [stochastic] certification",
    )
    r.add_argument("--budget-calls", type=int, default=None)
    r.add_argument("--budget-usd", type=float, default=None)
    r.add_argument("--out", default=None, help="write the artifact JSON here")
    r.add_argument("--report", action="store_true", help="also print a bug report to stderr")
    r.set_defaults(func=cmd_reduce)

    rp = sub.add_parser("replay", help="validate and summarize a saved repro artifact")
    rp.add_argument("repro", help="path to a tracemin-repro/1 artifact")
    rp.set_defaults(func=cmd_replay)

    df = sub.add_parser("diff", help="compare the minimal atoms of two artifacts")
    df.add_argument("a")
    df.add_argument("b")
    df.set_defaults(func=cmd_diff)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    func = args.func
    return int(func(args))


if __name__ == "__main__":
    raise SystemExit(main())
