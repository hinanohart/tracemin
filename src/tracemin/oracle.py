"""Oracles: classify a replayed :class:`RawOutput` into FAIL / PASS / ERROR.

An oracle encodes "what counts as the failure". The builtins cover the common
cases (exit code, exception, regex presence/absence, exact-answer mismatch,
arbitrary predicate) and compose with :func:`all_of` / :func:`any_of`.

Every oracle is callable, so an instance can be passed directly wherever an
``OracleFn`` (``Callable[[RawOutput], Verdict]``) is expected.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

from tracemin.replay import RawOutput, Verdict

_TARGETS = ("text", "stdout", "stderr")


@runtime_checkable
class Oracle(Protocol):
    """Structural type for anything that classifies an output into a verdict."""

    def classify(self, output: RawOutput) -> Verdict: ...


def _target_text(output: RawOutput, target: str) -> str:
    if target not in _TARGETS:
        raise ValueError(f"unknown target {target!r}; use one of {_TARGETS}")
    return str(getattr(output, target) or "")


def _normalize_answer(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


class BaseOracle:
    """Base for builtin oracles: defines ``__call__`` in terms of ``classify``."""

    def classify(self, output: RawOutput) -> Verdict:  # pragma: no cover - abstract
        raise NotImplementedError

    def __call__(self, output: RawOutput) -> Verdict:
        return self.classify(output)


class ExitCodeOracle(BaseOracle):
    """FAIL when the process exit code indicates failure.

    With ``fail_on`` set, FAIL iff the code is in that set. Otherwise FAIL iff the
    code is non-zero (when ``fail_nonzero``). A ``None`` exit code is PASS (no signal).
    """

    def __init__(self, fail_on: Sequence[int] | None = None, *, fail_nonzero: bool = True) -> None:
        self.fail_on = frozenset(fail_on) if fail_on is not None else None
        self.fail_nonzero = fail_nonzero

    def classify(self, output: RawOutput) -> Verdict:
        code = output.exit_code
        if code is None:
            return Verdict.PASS
        if self.fail_on is not None:
            return Verdict.FAIL if code in self.fail_on else Verdict.PASS
        if self.fail_nonzero and code != 0:
            return Verdict.FAIL
        return Verdict.PASS


class ExceptionOracle(BaseOracle):
    """FAIL when an exception is present and (optionally) matches type/message regexes."""

    def __init__(self, type_pattern: str | None = None, message_pattern: str | None = None) -> None:
        self.type_re = re.compile(type_pattern) if type_pattern else None
        self.msg_re = re.compile(message_pattern) if message_pattern else None

    def classify(self, output: RawOutput) -> Verdict:
        if not output.exception_type:
            return Verdict.PASS
        if self.type_re and not self.type_re.search(output.exception_type):
            return Verdict.PASS
        if self.msg_re and not self.msg_re.search(output.exception_message or ""):
            return Verdict.PASS
        return Verdict.FAIL


class RegexOracle(BaseOracle):
    """FAIL when ``pattern`` is found in the chosen target stream."""

    def __init__(self, pattern: str, target: str = "text") -> None:
        self.re = re.compile(pattern)
        self.target = target

    def classify(self, output: RawOutput) -> Verdict:
        return Verdict.FAIL if self.re.search(_target_text(output, self.target)) else Verdict.PASS


class NotRegexOracle(BaseOracle):
    """FAIL when ``pattern`` is *absent* (e.g. an expected answer never appears)."""

    def __init__(self, pattern: str, target: str = "text") -> None:
        self.re = re.compile(pattern)
        self.target = target

    def classify(self, output: RawOutput) -> Verdict:
        return Verdict.PASS if self.re.search(_target_text(output, self.target)) else Verdict.FAIL


class AnswerMismatchOracle(BaseOracle):
    """FAIL when the (whitespace-normalized) text does not equal an expected answer."""

    def __init__(self, expected: str, *, target: str = "text") -> None:
        self.expected = _normalize_answer(expected)
        self.target = target

    def classify(self, output: RawOutput) -> Verdict:
        actual = _normalize_answer(_target_text(output, self.target))
        return Verdict.PASS if actual == self.expected else Verdict.FAIL


class PredicateOracle(BaseOracle):
    """Wrap a user predicate. It may return a :class:`Verdict` or a bool (True=FAIL)."""

    def __init__(self, fn: Callable[[RawOutput], bool | Verdict]) -> None:
        self.fn = fn

    def classify(self, output: RawOutput) -> Verdict:
        result = self.fn(output)
        if isinstance(result, Verdict):
            return result
        return Verdict.FAIL if result else Verdict.PASS


class _Combinator(BaseOracle):
    def __init__(self, *oracles: Callable[[RawOutput], Verdict], all_required: bool) -> None:
        if not oracles:
            raise ValueError("a combinator needs at least one oracle")
        self.oracles = oracles
        self.all_required = all_required

    def classify(self, output: RawOutput) -> Verdict:
        verdicts = [o(output) for o in self.oracles]
        if any(v is Verdict.ERROR for v in verdicts):
            return Verdict.ERROR
        fails = [v is Verdict.FAIL for v in verdicts]
        is_fail = all(fails) if self.all_required else any(fails)
        return Verdict.FAIL if is_fail else Verdict.PASS


def all_of(*oracles: Callable[[RawOutput], Verdict]) -> _Combinator:
    """FAIL only when every sub-oracle says FAIL (ERROR propagates)."""
    return _Combinator(*oracles, all_required=True)


def any_of(*oracles: Callable[[RawOutput], Verdict]) -> _Combinator:
    """FAIL when any sub-oracle says FAIL (ERROR propagates)."""
    return _Combinator(*oracles, all_required=False)
