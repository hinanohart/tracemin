from __future__ import annotations

import json

import pytest

from tracemin.artifact import SCHEMA, ModeConsistencyError, build_artifact, validate_mode
from tracemin.atoms import Atom, AtomKind, Trajectory
from tracemin.engine import minimize
from tracemin.oracle import ExitCodeOracle
from tracemin.replay import RawOutput


def _result():
    atoms = [Atom.make(AtomKind.MESSAGE, f"a{i}", order=i) for i in range(5)]
    trigger = atoms[2].id
    traj = Trajectory.of(atoms)

    def replay(subset):
        ids = {a.id for a in subset}
        return (
            RawOutput(exit_code=1, exception_type="KeyError")
            if trigger in ids
            else RawOutput(exit_code=0)
        )

    return minimize(traj, replay, ExitCodeOracle(), double_check=False), trigger


def test_build_artifact_has_canonical_schema():
    result, trigger = _result()
    art = build_artifact(result, oracle_spec="exit_code!=0", replay_capable=True)
    d = art.data
    assert d["schema"] == SCHEMA
    assert d["certified"] is False
    assert d["verified"] is True
    assert d["replay_mode"] == "single-shot"
    assert d["minimality_scope"] == "wf-constrained"
    assert d["minimal_atoms"] == [trigger]
    assert d["minimality"]["certificate"]["engine"] == "ddmin"
    assert d["minimality"]["certificate"]["reverified_post_hoc"] is True
    assert d["confidence"]["method"] == "illustrative"
    assert d["oracle"]["verdict_on_minimal"] == "FAIL"
    assert d["provenance"]["sanitized"] is True


def test_artifact_json_round_trips():
    result, _ = _result()
    art = build_artifact(result, oracle_spec="x", replay_capable=False)
    parsed = json.loads(art.to_json())
    assert parsed["schema"] == SCHEMA


def test_validate_mode_rejects_certified_single_shot():
    bad = {
        "replay_mode": "single-shot",
        "certified": True,
        "confidence": {"method": "illustrative"},
        "minimality": {"witnesses": []},
    }
    with pytest.raises(ModeConsistencyError):
        validate_mode(bad)


def test_validate_mode_rejects_certified_with_unproven_witness():
    bad = {
        "replay_mode": "stochastic",
        "certified": True,
        "confidence": {"method": "measured(k=5)"},
        "minimality": {"witnesses": [{"necessity": "unproven"}]},
    }
    with pytest.raises(ModeConsistencyError):
        validate_mode(bad)


def test_validate_mode_accepts_valid_stochastic_certified():
    good = {
        "replay_mode": "stochastic",
        "certified": True,
        "confidence": {"method": "measured(k=5)"},
        "minimality": {"witnesses": [{"necessity": "proven"}, {"necessity": "pinned"}]},
    }
    validate_mode(good)  # must not raise


def test_validate_mode_rejects_single_shot_non_illustrative():
    bad = {
        "replay_mode": "single-shot",
        "certified": False,
        "confidence": {"method": "measured(k=3)"},
        "minimality": {"witnesses": []},
    }
    with pytest.raises(ModeConsistencyError):
        validate_mode(bad)
