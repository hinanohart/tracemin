from __future__ import annotations

import random

from tracemin.artifact import build_artifact
from tracemin.atoms import Atom, AtomKind, Trajectory
from tracemin.engine import MinimizeResult, Witness, minimize
from tracemin.oracle import ExitCodeOracle
from tracemin.replay import RawOutput, Verdict
from tracemin.signature import failure_signature
from tracemin.stochastic import beta_posterior, certify, intervals_disjoint, pass_pow_k


def test_pass_pow_k_native_values():
    assert pass_pow_k(10, 10, 3) == 1.0
    assert pass_pow_k(10, 0, 3) == 0.0
    assert abs(pass_pow_k(10, 5, 2, estimator="plugin") - 0.25) < 1e-9
    # unbiased: C(5,2)/C(10,2) = 10/45
    assert abs(pass_pow_k(10, 5, 2) - (10 / 45)) < 1e-9


def test_beta_posterior_and_intervals():
    post = beta_posterior(8, 10)
    lo, hi = post.credible_interval(0.95)
    assert 0.0 <= lo < hi <= 1.0
    assert 0.0 < post.expected_pow_k(2) < 1.0
    assert intervals_disjoint((0.8, 0.99), (0.0, 0.1))
    assert not intervals_disjoint((0.4, 0.7), (0.6, 0.9))


class _StochReplay:
    """A seeded stochastic policy: fails with p_high when the trigger is present."""

    def __init__(self, trigger_id, *, seed=0, p_high=0.95, p_low=0.03):
        self.trigger = trigger_id
        self.rng = random.Random(seed)
        self.p_high = p_high
        self.p_low = p_low

    def __call__(self, subset):
        ids = {a.id for a in subset}
        p = self.p_high if self.trigger in ids else self.p_low
        if self.rng.random() < p:
            return RawOutput(exit_code=1, exception_type="KeyError", exception_message="boom")
        return RawOutput(exit_code=0)


def test_certify_proven_yields_certified_stochastic_artifact():
    atoms = [Atom.make(AtomKind.MESSAGE, f"a{i}", order=i) for i in range(4)]
    trigger = atoms[1].id
    traj = Trajectory.of(atoms)
    # Establish the single-shot minimal set deterministically first.
    det = _StochReplay(trigger, seed=1, p_high=1.0, p_low=0.0)
    core = minimize(traj, det, ExitCodeOracle(), double_check=False)
    assert core.minimal_ids == (trigger,)
    assert core.certified is False  # core never certifies

    stoch = _StochReplay(trigger, seed=2, p_high=0.95, p_low=0.03)
    certified = certify(core, traj, stoch, ExitCodeOracle(), k=60)
    assert certified.replay_mode == "stochastic"
    assert certified.certified is True
    assert all(w.necessity in ("proven", "pinned") for w in certified.witnesses)

    art = build_artifact(
        certified,
        oracle_spec="exit-nonzero",
        replay_capable=True,
        confidence_method="measured(k=60)",
        k=60,
    )
    assert art.data["certified"] is True
    assert art.data["replay_mode"] == "stochastic"


def test_certify_marks_irrelevant_atom_unproven():
    trigger = Atom.make(AtomKind.MESSAGE, "trigger", order=0)
    extra = Atom.make(AtomKind.MESSAGE, "irrelevant", order=1)
    traj = Trajectory.of([trigger, extra])
    ref = failure_signature(RawOutput(exception_type="KeyError", exception_message="boom"))
    assert ref is not None
    # Pretend the core (wrongly) kept both atoms; certify must expose the extra as unproven.
    bogus = MinimizeResult(
        minimal_ids=(trigger.id, extra.id),
        minimal_atoms=(trigger, extra),
        reference_signature=ref.key,
        verdict_on_minimal=Verdict.FAIL,
        certified=False,
        certified_reason="single-shot-unverified",
        replay_mode="single-shot",
        witnesses=(Witness(trigger.id, "single-shot"), Witness(extra.id, "single-shot")),
        reverified_post_hoc=True,
        stats={"replay_calls": 0},
    )
    stoch = _StochReplay(trigger.id, seed=3, p_high=0.95, p_low=0.03)
    out = certify(bogus, traj, stoch, ExitCodeOracle(), k=60)
    necessity = {w.atom_id: w.necessity for w in out.witnesses}
    assert necessity[trigger.id] == "proven"
    assert necessity[extra.id] == "unproven"
    assert out.certified is False
