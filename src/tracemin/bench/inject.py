"""Failure-injection case generators with known ground-truth minimal sets.

Every case carries the exact set of atom ids that *must* be recovered. Trigger
families:
  * single-atom          — fails iff one specific atom is present.
  * conjunctive (AND)     — fails iff two specific atoms are both present.
  * dependency-entangled  — the trigger requires a symbol a producer atom emits,
                            so closure removal must keep the producer too.
  * distractor-padding    — one trigger amid many irrelevant atoms.
  * decoy                 — removing the real trigger yields a *different* failure
                            (different signature); used to measure false reproducers.
  * stochastic            — fails with a known probability when the trigger is present.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field

from tracemin.atoms import Atom, AtomKind, Trajectory
from tracemin.replay import RawOutput, ReplayFn


@dataclass
class Case:
    trajectory: Trajectory
    replay_fn: ReplayFn
    ground_truth: frozenset[str]
    family: str
    is_decoy: bool = False
    p_fail: float = 1.0
    rng_seed: int = 0
    meta: dict[str, object] = field(default_factory=dict)


def _fail(exc: str = "KeyError", msg: str = "boom") -> RawOutput:
    return RawOutput(exit_code=1, exception_type=exc, exception_message=msg, stderr=f"{exc}: {msg}")


def _ok() -> RawOutput:
    return RawOutput(exit_code=0, text="ok")


def _msg(
    payload: object, *, order: int, produces: Sequence[str] = (), requires: Sequence[str] = ()
) -> Atom:
    return Atom.make(AtomKind.MESSAGE, payload, produces=produces, requires=requires, order=order)


def single_atom(rng: random.Random, size: int = 6) -> Case:
    atoms = [_msg(f"a{i}", order=i) for i in range(size)]
    trigger = atoms[rng.randrange(size)].id
    traj = Trajectory.of(atoms)

    def replay(subset: Sequence[Atom]) -> RawOutput:
        return _fail() if trigger in {a.id for a in subset} else _ok()

    return Case(traj, replay, frozenset({trigger}), "single-atom")


def conjunctive(rng: random.Random, size: int = 6) -> Case:
    atoms = [_msg(f"a{i}", order=i) for i in range(size)]
    i, j = rng.sample(range(size), 2)
    x, y = atoms[i].id, atoms[j].id
    traj = Trajectory.of(atoms)

    def replay(subset: Sequence[Atom]) -> RawOutput:
        ids = {a.id for a in subset}
        return _fail() if (x in ids and y in ids) else _ok()

    return Case(traj, replay, frozenset({x, y}), "conjunctive")


def dependency_entangled(rng: random.Random, size: int = 6) -> Case:
    sym = "dep_sym"
    producer = _msg({"defines": sym}, order=0, produces=[sym])
    caller = _msg({"uses": sym}, order=1, requires=[sym])  # trigger
    fillers = [_msg(f"f{i}", order=i + 2) for i in range(max(0, size - 2))]
    traj = Trajectory.of([producer, caller, *fillers])

    def replay(subset: Sequence[Atom]) -> RawOutput:
        return _fail() if caller.id in {a.id for a in subset} else _ok()

    return Case(traj, replay, frozenset({producer.id, caller.id}), "dependency-entangled")


def distractor_padding(rng: random.Random, size: int = 14) -> Case:
    atoms = [_msg(f"pad{i}", order=i) for i in range(size)]
    trigger = atoms[rng.randrange(size)].id
    traj = Trajectory.of(atoms)

    def replay(subset: Sequence[Atom]) -> RawOutput:
        return _fail() if trigger in {a.id for a in subset} else _ok()

    return Case(traj, replay, frozenset({trigger}), "distractor-padding")


def decoy(rng: random.Random, size: int = 6) -> Case:
    atoms = [_msg(f"a{i}", order=i) for i in range(size)]
    idx = rng.sample(range(size), 2)
    trigger, decoy_atom = atoms[idx[0]].id, atoms[idx[1]].id
    traj = Trajectory.of(atoms)

    def replay(subset: Sequence[Atom]) -> RawOutput:
        ids = {a.id for a in subset}
        if trigger in ids:
            return _fail("KeyError", "real")  # the original failure
        if decoy_atom in ids:
            return _fail("ValueError", "decoy")  # a *different* failure
        return _ok()

    return Case(traj, replay, frozenset({trigger}), "decoy", is_decoy=True)


def stochastic(rng: random.Random, size: int = 6, p: float = 0.9) -> Case:
    atoms = [_msg(f"a{i}", order=i) for i in range(size)]
    trigger = atoms[rng.randrange(size)].id
    traj = Trajectory.of(atoms)
    local = random.Random(rng.random())

    def replay(subset: Sequence[Atom]) -> RawOutput:
        if trigger in {a.id for a in subset}:
            return _fail() if local.random() < p else _ok()
        return _ok()

    return Case(traj, replay, frozenset({trigger}), "stochastic", p_fail=p)


DETERMINISTIC_FAMILIES = (single_atom, conjunctive, dependency_entangled, distractor_padding)
