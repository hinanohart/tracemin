"""The unified replay contract: how a candidate subset is re-executed and judged.

Three responsibilities are deliberately separated so that adapters, oracles and
signatures evolve independently:

    replay_fn(subset)        -> RawOutput     # how to re-run the candidate
    oracle.classify(output)  -> Verdict       # FAIL / PASS / ERROR
    failure_signature(output)-> Signature     # identity key of the failure

``run_trial`` composes them into a single :class:`ReplayResult` and centralizes
the three-valued logic, including the rule that a transport/infrastructure error
yields :attr:`Verdict.ERROR` (which the engine excludes from both accept and
reject — it is neither evidence of necessity nor of sufficiency).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

from tracemin.atoms import Atom


class Verdict(Enum):
    """The three possible outcomes of classifying a replayed output."""

    FAIL = "FAIL"
    PASS = "PASS"
    ERROR = "ERROR"


class ReplayError(RuntimeError):
    """Raised by a ``replay_fn`` to signal a transport/infrastructure failure.

    This is *not* the bug under investigation (network outage, quota, 5xx). The
    engine maps it to :attr:`Verdict.ERROR` and retries / aborts rather than
    treating it as PASS or FAIL.
    """


@dataclass
class RawOutput:
    """The observable result of re-executing a candidate subset.

    ``transport_error`` is an alternative, non-raising channel for infrastructure
    errors; when set, classification short-circuits to :attr:`Verdict.ERROR`.
    """

    text: str = ""
    exit_code: int | None = None
    exception_type: str | None = None
    exception_message: str | None = None
    stdout: str = ""
    stderr: str = ""
    transport_error: str | None = None
    extra: dict[str, object] = field(default_factory=dict)


@runtime_checkable
class SignatureLike(Protocol):
    """Anything carrying a stable string ``key`` identifying a failure."""

    @property
    def key(self) -> str: ...


@runtime_checkable
class ReplayFn(Protocol):
    """Re-execute a candidate context subset and return its raw output."""

    def __call__(self, subset: Sequence[Atom]) -> RawOutput: ...


OracleFn = Callable[[RawOutput], Verdict]
SignatureFn = Callable[[RawOutput], "SignatureLike | None"]


@dataclass(frozen=True)
class ReplayResult:
    """Outcome of a single trial: a verdict plus the failure's identity key."""

    verdict: Verdict
    output: RawOutput
    signature: str | None = None

    @property
    def is_fail(self) -> bool:
        return self.verdict is Verdict.FAIL

    @property
    def is_error(self) -> bool:
        return self.verdict is Verdict.ERROR


def run_trial(
    replay_fn: ReplayFn,
    oracle: OracleFn,
    signature_fn: SignatureFn | None,
    subset: Sequence[Atom],
) -> ReplayResult:
    """Run one trial and apply the three-valued classification rule.

    A transport error (raised :class:`ReplayError` or ``output.transport_error``)
    becomes :attr:`Verdict.ERROR`. A signature is computed only for FAIL outputs;
    if ``signature_fn`` is ``None`` the signature is left ``None`` (which the engine
    treats as "any FAIL matches").
    """
    try:
        output = replay_fn(subset)
    except ReplayError as exc:
        return ReplayResult(Verdict.ERROR, RawOutput(transport_error=str(exc)))

    if output.transport_error is not None:
        return ReplayResult(Verdict.ERROR, output)

    verdict = oracle(output)
    if verdict is Verdict.ERROR:
        return ReplayResult(Verdict.ERROR, output)

    signature: str | None = None
    if verdict is Verdict.FAIL and signature_fn is not None:
        sig = signature_fn(output)
        signature = sig.key if sig is not None else None
    return ReplayResult(verdict, output, signature)
