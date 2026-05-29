"""The reduction engine: dependency-aware ddmin with closure-removal repair.

Given a failing trajectory and a replay/oracle/signature contract, :func:`minimize`
returns a *1-minimal* subset of context atoms that still reproduces the **same**
failure (matched by signature). Every candidate tested is well-formed by
construction: when ddmin proposes removing a set Δ, closure removal expands Δ to
also drop any atom left dangling, so the remainder always satisfies WF(S). The
reported minimality is therefore **wf-constrained** 1-minimality.

Honesty rules baked into the engine:
  * **Three-valued.** A transport/infrastructure ERROR is excluded from both
    accept and reject; it is retried, and if persistent the removal is treated as
    inconclusive (not a reduction). ERROR never collapses to PASS.
  * **Pre-flight sanity.** The full input must reproduce the failure first; if it
    does not, we abort (non-deterministic replay or wrong oracle) rather than
    "minimizing" nothing.
  * **Flakiness double-check.** With ``double_check`` (default on), an interesting
    result must reproduce twice consecutively to count — a cheap guard against a
    single flaky FAIL. This is a flakiness guard, *not* a statistical claim.
  * **Single-shot core is never certified.** The k=1 core sets ``certified=False``;
    statistical certification lives in the optional ``[stochastic]`` extra.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from tracemin.atoms import Atom, Trajectory
from tracemin.cache import BudgetExceeded, CachedTrial, ReplayBudget, SubsetCache
from tracemin.oracle import Oracle
from tracemin.replay import OracleFn, ReplayFn, ReplayResult, SignatureFn, Verdict, run_trial
from tracemin.signature import failure_signature


class PreflightError(RuntimeError):
    """The full input did not reproduce the failure — cannot establish a baseline."""


@dataclass(frozen=True)
class Witness:
    """Per-atom necessity record for the reported minimal set."""

    atom_id: str
    necessity: str  # "single-shot" | "proven" | "unproven" | "pinned"


@dataclass
class MinimizeResult:
    """The outcome of a reduction run (consumed by the artifact builder)."""

    minimal_ids: tuple[str, ...]
    minimal_atoms: tuple[Atom, ...]
    reference_signature: str | None
    verdict_on_minimal: Verdict
    certified: bool
    certified_reason: str | None
    replay_mode: str
    witnesses: tuple[Witness, ...]
    reverified_post_hoc: bool
    stats: dict[str, int] = field(default_factory=dict)


def _as_oracle_fn(oracle: Oracle | OracleFn) -> OracleFn:
    classify = getattr(oracle, "classify", None)
    return classify if callable(classify) else oracle  # type: ignore[return-value]


def minimize(
    trajectory: Trajectory,
    replay_fn: ReplayFn,
    oracle: Oracle | OracleFn,
    *,
    signature_fn: SignatureFn | None = failure_signature,
    double_check: bool = True,
    budget: ReplayBudget | None = None,
    error_retries: int = 2,
    cost_per_call: float = 0.0,
) -> MinimizeResult:
    """Reduce ``trajectory`` to a wf-constrained 1-minimal reproducer (single-shot core)."""
    oracle_fn = _as_oracle_fn(oracle)
    by_id = trajectory.by_id()
    all_removable = list(trajectory.removable_ids)
    removable_set = frozenset(all_removable)
    order_of = {a.id: a.order for a in trajectory.atoms}
    cache = SubsetCache()
    active_budget: ReplayBudget = budget or ReplayBudget()
    stats = {"replay_calls": 0, "errors": 0}

    def _order(ids: Iterable[str]) -> list[str]:
        return sorted(ids, key=lambda i: order_of[i])

    def _atoms_for(survivor_ids: Iterable[str]) -> list[Atom]:
        return [by_id[i] for i in _order(survivor_ids)]

    def _run(survivor_ids: frozenset[str]) -> ReplayResult:
        """One trial with budget charge + bounded ERROR retry."""
        result: ReplayResult | None = None
        for _ in range(error_retries + 1):
            active_budget.charge(cost_per_call)
            stats["replay_calls"] += 1
            result = run_trial(replay_fn, oracle_fn, signature_fn, _atoms_for(survivor_ids))
            if result.verdict is not Verdict.ERROR:
                return result
            stats["errors"] += 1
        assert result is not None  # error_retries >= 0 guarantees >= 1 attempt
        return result  # persistent ERROR

    # --- pre-flight: the full input must reproduce, establishing the reference --
    full_ids = frozenset(trajectory.ids)
    pre = _run(full_ids)
    if pre.verdict is not Verdict.FAIL:
        raise PreflightError(
            f"full input did not reproduce the failure (verdict={pre.verdict.name}); "
            "the replay_fn may be non-deterministic or the oracle may be wrong"
        )
    reference_signature: str | None = pre.signature
    if double_check:
        confirm = _run(full_ids)
        if not (confirm.verdict is Verdict.FAIL and confirm.signature == reference_signature):
            raise PreflightError("full input failure is flaky (did not reproduce on confirmation)")

    def _accept(result: ReplayResult) -> bool:
        if result.verdict is not Verdict.FAIL:
            return False
        if reference_signature is None:
            return True
        return result.signature == reference_signature

    def probe(keep: frozenset[str]) -> tuple[bool, frozenset[str]]:
        """Test keeping ``keep`` removable atoms; return (interesting, effective_kept)."""
        delta = removable_set - keep
        survivors = trajectory.survivors_after_removing(delta)
        if survivors is None:  # removal would orphan a pinned atom — not allowed
            return (False, keep)
        eff = frozenset(survivors) & removable_set
        key = SubsetCache.key(survivors)
        cached = cache.get(key)
        if cached is not None:
            return (cached.accepted, eff)
        result = _run(survivors)
        accepted = _accept(result)
        if accepted and double_check:
            confirm = _run(survivors)
            accepted = _accept(confirm)
        cache.put(key, CachedTrial(result.verdict, result.signature, accepted))
        return (accepted, eff)

    # --- dependency-aware ddmin (Zeller 2002), adopting closure-repaired survivors
    elements = _order(all_removable)
    truncated = False
    try:
        n = 2
        while len(elements) > 1:
            chunk_size = max(1, len(elements) // n)
            chunks = [
                frozenset(elements[i : i + chunk_size]) for i in range(0, len(elements), chunk_size)
            ]
            reduced = False
            # 1) reduce to a single interesting subset
            for chunk in chunks:
                ok, eff = probe(chunk)
                if ok:
                    elements = _order(eff)
                    n = 2
                    reduced = True
                    break
            if reduced:
                continue
            # 2) reduce to an interesting complement
            current = frozenset(elements)
            for chunk in chunks:
                comp = current - chunk
                if comp == current:
                    continue
                ok, eff = probe(comp)
                if ok:
                    elements = _order(eff)
                    n = max(n - 1, 2)
                    reduced = True
                    break
            if reduced:
                continue
            # 3) increase granularity, or stop
            if n >= len(elements):
                break
            n = min(len(elements), 2 * n)
    except BudgetExceeded:
        truncated = True

    # --- assemble the minimal set (pinned always kept) + post-hoc re-verification
    minimal_removable = frozenset(elements)
    delta = removable_set - minimal_removable
    survivors = trajectory.survivors_after_removing(delta)
    if survivors is None:  # defensive: fall back to full removable kept
        survivors = full_ids
    try:
        final = _run(survivors)
        verdict_on_minimal = final.verdict
        reverified = verdict_on_minimal is Verdict.FAIL
    except BudgetExceeded:
        # Budget was spent during exploration; we cannot afford the post-hoc
        # re-execution. Report honestly: not re-verified, truncated.
        truncated = True
        verdict_on_minimal = Verdict.ERROR
        reverified = False

    minimal_ids = tuple(_order(survivors))
    witnesses = tuple(
        Witness(i, "pinned" if by_id[i].pinned else "single-shot") for i in minimal_ids
    )
    certified_reason = "budget-truncated" if truncated else "single-shot-unverified"
    stats["cache_hits"] = cache.hits
    stats["minimal_size"] = len(minimal_ids)
    stats["original_size"] = len(trajectory.atoms)

    return MinimizeResult(
        minimal_ids=minimal_ids,
        minimal_atoms=tuple(_atoms_for(survivors)),
        reference_signature=reference_signature,
        verdict_on_minimal=verdict_on_minimal,
        certified=False,  # k=1 core is never certified
        certified_reason=certified_reason,
        replay_mode="single-shot",
        witnesses=witnesses,
        reverified_post_hoc=reverified,
        stats=stats,
    )
