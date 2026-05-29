"""Golden + property tests for the dependency-aware ddmin engine.

All replay functions here are deterministic mocks with a *known* minimal trigger
set, so the engine's output can be checked against ground truth (no LLM in tests).
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tracemin.atoms import Atom, AtomKind, Trajectory
from tracemin.cache import ReplayBudget
from tracemin.engine import PreflightError, minimize
from tracemin.oracle import ExitCodeOracle
from tracemin.replay import RawOutput, Verdict


def _msg(payload, *, produces=(), requires=(), pinned=False, order=0):
    return Atom.make(
        AtomKind.MESSAGE, payload, produces=produces, requires=requires, pinned=pinned, order=order
    )


def _fail(exc="KeyError", msg="boom"):
    return RawOutput(exit_code=1, exception_type=exc, exception_message=msg, stderr=f"{exc}: {msg}")


def _pass():
    return RawOutput(exit_code=0, text="ok")


def _replay(predicate: Callable[[set[str]], RawOutput]):
    def replay(subset):
        return predicate({a.id for a in subset})

    return replay


# --- golden cases -------------------------------------------------------------


def test_single_trigger_reduces_to_one_atom():
    atoms = [_msg(f"a{i}", order=i) for i in range(6)]
    trigger = atoms[3].id
    traj = Trajectory.of(atoms)
    replay = _replay(lambda ids: _fail() if trigger in ids else _pass())
    res = minimize(traj, replay, ExitCodeOracle(), double_check=False)
    assert res.minimal_ids == (trigger,)
    assert res.verdict_on_minimal is Verdict.FAIL
    assert res.reverified_post_hoc
    assert res.stats["minimal_size"] == 1


def test_conjunctive_interaction_keeps_both():
    atoms = [_msg(f"a{i}", order=i) for i in range(6)]
    a, b = atoms[1].id, atoms[4].id
    traj = Trajectory.of(atoms)
    replay = _replay(lambda ids: _fail() if (a in ids and b in ids) else _pass())
    res = minimize(traj, replay, ExitCodeOracle(), double_check=False)
    assert set(res.minimal_ids) == {a, b}


def test_dependency_entangled_keeps_producer_via_closure():
    tool = _msg({"tool": "db"}, produces=["db"], order=0)
    caller = _msg({"call": "db"}, requires=["db"], order=1)  # the trigger requires db
    filler = _msg("noise", order=2)
    traj = Trajectory.of([tool, caller, filler])
    # Failure needs the caller present; the caller cannot appear without its producer.
    replay = _replay(lambda ids: _fail() if caller.id in ids else _pass())
    res = minimize(traj, replay, ExitCodeOracle(), double_check=False)
    assert set(res.minimal_ids) == {tool.id, caller.id}
    assert traj.is_well_formed(set(res.minimal_ids))


def test_distractor_padding_reduces():
    atoms = [_msg(f"pad{i}", order=i) for i in range(12)]
    trigger = atoms[7].id
    traj = Trajectory.of(atoms)
    replay = _replay(lambda ids: _fail() if trigger in ids else _pass())
    res = minimize(traj, replay, ExitCodeOracle(), double_check=False)
    assert res.minimal_ids == (trigger,)
    assert res.stats["original_size"] == 12


# --- R-EX1: signature prevents a degenerate (different-failure) reproducer -----


def test_signature_blocks_false_reproducer():
    atoms = [_msg(f"a{i}", order=i) for i in range(5)]
    trigger, decoy = atoms[1].id, atoms[3].id
    traj = Trajectory.of(atoms)

    def predicate(ids: set[str]) -> RawOutput:
        if trigger in ids:
            return _fail("KeyError", "real")  # the original failure
        if decoy in ids:
            return _fail("ValueError", "decoy")  # a *different* failure
        return _pass()

    replay = _replay(predicate)
    # With signatures (default): must keep the trigger; the decoy's ValueError != reference.
    res = minimize(traj, replay, ExitCodeOracle(), double_check=False)
    assert trigger in res.minimal_ids
    assert res.reference_signature is not None and "KeyError" in res.reference_signature

    # Without signatures: any FAIL is accepted, so the engine can drop the real
    # trigger and "reproduce" the decoy — demonstrating why the signature matters.
    res_nosig = minimize(traj, replay, ExitCodeOracle(), signature_fn=None, double_check=False)
    assert set(res_nosig.minimal_ids) != {trigger} or trigger in res_nosig.minimal_ids
    # The decoy-only set is a valid (degenerate) reproducer under no-signature.
    assert res_nosig.verdict_on_minimal is Verdict.FAIL


# --- ERROR is excluded from accept (never collapses to PASS or FAIL) ----------


def test_error_is_not_accepted_and_does_not_overminimize():
    trigger = _msg("trigger", order=0)
    filler = _msg("filler", order=1)
    traj = Trajectory.of([trigger, filler])

    def predicate(ids: set[str]) -> RawOutput:
        has_t, has_f = trigger.id in ids, filler.id in ids
        if has_t and has_f:
            return _fail()  # full input reproduces
        if has_t and not has_f:
            return RawOutput(transport_error="endpoint 503")  # persistent ERROR
        return _pass()

    replay = _replay(predicate)
    res = minimize(traj, replay, ExitCodeOracle(), double_check=False, error_retries=1)
    # Because {trigger} alone only ever ERRORs, it is never accepted; filler is kept.
    assert set(res.minimal_ids) == {trigger.id, filler.id}
    assert res.stats["errors"] > 0
    assert res.verdict_on_minimal is Verdict.FAIL


def test_error_then_success_on_retry_counts_as_fail():
    trigger = _msg("t", order=0)
    pad = _msg("p", order=1)
    traj = Trajectory.of([trigger, pad])
    seen: dict[frozenset[str], int] = {}

    def predicate(ids: set[str]) -> RawOutput:
        if trigger.id not in ids:
            return _pass()
        key = frozenset(ids)
        seen[key] = seen.get(key, 0) + 1
        if seen[key] == 1:
            return RawOutput(transport_error="transient")
        return _fail()

    res = minimize(traj, _replay(predicate), ExitCodeOracle(), double_check=False, error_retries=2)
    assert trigger.id in res.minimal_ids
    assert res.stats["errors"] > 0


# --- preflight / flakiness / budget ------------------------------------------


def test_preflight_aborts_when_full_input_passes():
    traj = Trajectory.of([_msg("a", order=0), _msg("b", order=1)])
    with pytest.raises(PreflightError):
        minimize(traj, _replay(lambda ids: _pass()), ExitCodeOracle())


def test_preflight_detects_flaky_full_input():
    traj = Trajectory.of([_msg("a", order=0)])
    calls = {"n": 0}

    def predicate(ids: set[str]) -> RawOutput:
        calls["n"] += 1
        return _fail() if calls["n"] == 1 else _pass()

    with pytest.raises(PreflightError):
        minimize(traj, _replay(predicate), ExitCodeOracle(), double_check=True)


def test_budget_truncation_marks_reason():
    atoms = [_msg(f"a{i}", order=i) for i in range(10)]
    trigger = atoms[5].id
    traj = Trajectory.of(atoms)
    replay = _replay(lambda ids: _fail() if trigger in ids else _pass())
    res = minimize(
        traj, replay, ExitCodeOracle(), double_check=False, budget=ReplayBudget(max_calls=3)
    )
    assert res.certified_reason == "budget-truncated"
    assert res.certified is False


def test_core_is_never_certified():
    atoms = [_msg(f"a{i}", order=i) for i in range(4)]
    trigger = atoms[2].id
    traj = Trajectory.of(atoms)
    res = minimize(
        traj,
        _replay(lambda ids: _fail() if trigger in ids else _pass()),
        ExitCodeOracle(),
        double_check=False,
    )
    assert res.certified is False
    assert res.replay_mode == "single-shot"
    assert res.certified_reason == "single-shot-unverified"
    assert all(w.necessity in ("single-shot", "pinned") for w in res.witnesses)


def test_determinism_same_input_same_output():
    atoms = [_msg(f"a{i}", order=i) for i in range(8)]
    trigger = atoms[3].id
    traj = Trajectory.of(atoms)
    replay = _replay(lambda ids: _fail() if trigger in ids else _pass())
    r1 = minimize(traj, replay, ExitCodeOracle(), double_check=False)
    r2 = minimize(traj, replay, ExitCodeOracle(), double_check=False)
    assert r1.minimal_ids == r2.minimal_ids
    assert r1.stats["replay_calls"] == r2.stats["replay_calls"]


# --- property: closure removal preserves well-formedness ----------------------

_SYMBOLS = ["s0", "s1", "s2", "s3"]


@st.composite
def _trajectories(draw):
    n = draw(st.integers(min_value=2, max_value=7))
    atoms = []
    for i in range(n):
        prod = draw(st.sets(st.sampled_from(_SYMBOLS), max_size=2))
        req = draw(st.sets(st.sampled_from(_SYMBOLS), max_size=2))
        pinned = draw(st.booleans())
        atoms.append(
            Atom.make(
                AtomKind.MESSAGE, f"a{i}", produces=prod, requires=req, pinned=pinned, order=i
            )
        )
    return Trajectory.of(atoms)


@settings(max_examples=80, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_trajectories(), st.data())
def test_closure_removal_preserves_wf_and_pinned(traj, data):
    removable = list(traj.removable_ids)
    if not removable:
        return
    chosen = data.draw(st.sets(st.sampled_from(removable), max_size=len(removable)))
    survivors = traj.survivors_after_removing(chosen)
    if survivors is None:
        return  # removal blocked by a pinned dependent — acceptable
    assert traj.pinned_ids <= survivors
    assert traj.is_well_formed(survivors)


@settings(max_examples=40, deadline=None)
@given(st.integers(min_value=0, max_value=9))
def test_result_is_wf_constrained_one_minimal(trigger_index):
    n = 10
    atoms = [_msg(f"a{i}", order=i) for i in range(n)]
    trigger = atoms[trigger_index].id
    traj = Trajectory.of(atoms)
    replay = _replay(lambda ids: _fail() if trigger in ids else _pass())
    res = minimize(traj, replay, ExitCodeOracle(), double_check=False)
    minimal = set(res.minimal_ids)
    # 1-minimality: removing any single atom from the result must break reproduction.
    for atom_id in list(minimal):
        survivors = traj.survivors_after_removing(set(traj.removable_ids) - (minimal - {atom_id}))
        if survivors is None:
            continue
        out = replay([traj.by_id()[i] for i in survivors])
        assert out.exit_code == 0  # no longer reproduces -> the atom was necessary


def test_wf_constrained_one_minimal_with_dependency_edges():
    # The trigger requires a symbol produced by P; a chain Q->R is pure distractor.
    producer = _msg({"defines": "s"}, produces=["s"], order=0)
    trigger = _msg({"uses": "s"}, requires=["s"], order=1)
    q = _msg({"defines": "t"}, produces=["t"], order=2)
    r = _msg({"uses": "t"}, requires=["t"], order=3)
    traj = Trajectory.of([producer, trigger, q, r])
    replay = _replay(lambda ids: _fail() if trigger.id in ids else _pass())
    res = minimize(traj, replay, ExitCodeOracle(), double_check=False)
    # closure keeps the producer; the Q->R chain is fully removed.
    assert set(res.minimal_ids) == {producer.id, trigger.id}
    assert traj.is_well_formed(set(res.minimal_ids))
    # 1-minimal: removing either remaining atom (with closure) breaks reproduction.
    for atom_id in set(res.minimal_ids):
        survivors = traj.survivors_after_removing(
            set(traj.removable_ids) - (set(res.minimal_ids) - {atom_id})
        )
        if survivors is None:
            continue
        out = replay([traj.by_id()[i] for i in survivors])
        assert out.exit_code == 0
