"""tracemin — re-execution-verified, dependency-aware delta-debugging for failed LLM-agent runs.

Given a failed agent run and a user-supplied ``replay_fn``, tracemin shrinks the
trajectory's context (messages / tool definitions / retrieved files / instructions)
to a *1-minimal* subset by re-executing each candidate and keeping only those that
still reproduce the *same* failure. The result is a minimal reproducer — not a
root-cause explanation.

The public API below is import-light: it pulls in only the core dependency
(``huggingface_hub`` is loaded lazily, only when the ``hf`` adapter is used).
Statistical certification lives in the optional ``[stochastic]`` extra; the
single-shot core never claims certification.
"""

from __future__ import annotations

__version__ = "0.1.0a1"

from tracemin.artifact import Artifact, build_artifact
from tracemin.atoms import Atom, AtomKind, Trajectory
from tracemin.engine import MinimizeResult, PreflightError, Witness, minimize
from tracemin.oracle import (
    AnswerMismatchOracle,
    ExceptionOracle,
    ExitCodeOracle,
    NotRegexOracle,
    Oracle,
    PredicateOracle,
    RegexOracle,
    all_of,
    any_of,
)
from tracemin.replay import RawOutput, ReplayError, ReplayResult, Verdict, run_trial
from tracemin.signature import Signature, failure_signature

__all__ = [
    "AnswerMismatchOracle",
    "Artifact",
    "Atom",
    "AtomKind",
    "ExceptionOracle",
    "ExitCodeOracle",
    "MinimizeResult",
    "NotRegexOracle",
    "Oracle",
    "PredicateOracle",
    "PreflightError",
    "RawOutput",
    "RegexOracle",
    "ReplayError",
    "ReplayResult",
    "Signature",
    "Trajectory",
    "Verdict",
    "Witness",
    "__version__",
    "all_of",
    "any_of",
    "build_artifact",
    "failure_signature",
    "minimize",
    "run_trial",
]
