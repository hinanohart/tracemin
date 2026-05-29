"""Trial memoization and a fail-closed replay budget.

The cache is intentionally *metadata-only*: it stores a verdict, a normalized
signature string and the accept decision keyed by a hash of the surviving atom
ids. It never stores raw atom payloads or model output, so it cannot become a
side channel for secrets.

The budget is *fail-closed*: it checks the limit before charging, so a trial that
would exceed the cap raises :class:`BudgetExceeded` instead of being run. The
engine catches this and returns the best reproducer found so far, marked as
``budget-truncated``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass

from tracemin.replay import Verdict


class BudgetExceeded(RuntimeError):
    """Raised when a replay would exceed the configured call/cost budget."""


@dataclass
class ReplayBudget:
    """A cap on replay calls and/or estimated USD cost. Checked before charging."""

    max_calls: int | None = None
    max_usd: float | None = None
    calls: int = 0
    usd: float = 0.0

    def charge(self, usd: float = 0.0) -> None:
        if self.max_calls is not None and self.calls + 1 > self.max_calls:
            raise BudgetExceeded(f"call budget exhausted ({self.max_calls} calls)")
        if self.max_usd is not None and self.usd + usd > self.max_usd:
            raise BudgetExceeded(f"cost budget exhausted (${self.max_usd:.4f})")
        self.calls += 1
        self.usd += usd


@dataclass(frozen=True)
class CachedTrial:
    """A memoized trial outcome (no raw content)."""

    verdict: Verdict
    signature: str | None
    accepted: bool


class SubsetCache:
    """Maps a canonical hash of surviving atom ids to a :class:`CachedTrial`."""

    def __init__(self) -> None:
        self._store: dict[str, CachedTrial] = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(survivor_ids: Iterable[str]) -> str:
        joined = ",".join(sorted(survivor_ids))
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]

    def get(self, key: str) -> CachedTrial | None:
        hit = self._store.get(key)
        if hit is None:
            self.misses += 1
        else:
            self.hits += 1
        return hit

    def put(self, key: str, trial: CachedTrial) -> None:
        self._store.setdefault(key, trial)
