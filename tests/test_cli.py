from __future__ import annotations

import json

import pytest

from tracemin.atoms import Atom, AtomKind, Trajectory
from tracemin.cli import build_oracle, load_input, main
from tracemin.engine import minimize
from tracemin.oracle import ExceptionOracle, ExitCodeOracle, NotRegexOracle, RegexOracle
from tracemin.replay import RawOutput


def test_build_oracle_specs():
    assert isinstance(build_oracle("exit-nonzero"), ExitCodeOracle)
    assert isinstance(build_oracle("regex:KeyError"), RegexOracle)
    assert isinstance(build_oracle("not-regex:OK"), NotRegexOracle)
    assert isinstance(build_oracle("exception:ValueError"), ExceptionOracle)
    with pytest.raises(ValueError):
        build_oracle("bogus")


def test_load_input_json(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(
        json.dumps([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}])
    )
    traj = load_input(str(p), "json")
    assert len(traj.atoms) == 2


def test_main_no_command_prints_help_returns_zero(capsys):
    assert main([]) == 0
    assert "tracemin" in capsys.readouterr().out


def test_main_version_exits():
    with pytest.raises(SystemExit) as ei:
        main(["--version"])
    assert ei.value.code == 0


def test_main_doctor(capsys):
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "tracemin doctor" in out and "[" in out


def test_main_reduce_without_token_exits_two(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    p = tmp_path / "t.json"
    p.write_text(json.dumps([{"role": "user", "content": "x"}]))
    rc = main(["reduce", str(p), "--model", "some/model"])
    assert rc == 2
    assert "HF_TOKEN absent" in capsys.readouterr().err


def _write_artifact(tmp_path):
    atoms = [Atom.make(AtomKind.MESSAGE, f"a{i}", order=i) for i in range(4)]
    trigger = atoms[1].id
    traj = Trajectory.of(atoms)

    def replay(subset):
        ids = {a.id for a in subset}
        return (
            RawOutput(exit_code=1, exception_type="KeyError")
            if trigger in ids
            else RawOutput(exit_code=0)
        )

    from tracemin.artifact import build_artifact

    result = minimize(traj, replay, ExitCodeOracle(), double_check=False)
    art = build_artifact(result, oracle_spec="exit-nonzero", replay_capable=True)
    p = tmp_path / "repro.json"
    p.write_text(art.to_json())
    return p, trigger


def test_main_replay_validates_and_summarizes(tmp_path, capsys):
    p, trigger = _write_artifact(tmp_path)
    assert main(["replay", str(p)]) == 0
    out = capsys.readouterr().out
    assert "tracemin-repro/1" in out
    assert trigger in out


def test_main_replay_rejects_inconsistent_artifact(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps(
            {
                "schema": "tracemin-repro/1",
                "replay_mode": "single-shot",
                "certified": True,
                "confidence": {"method": "illustrative"},
                "minimality": {"witnesses": []},
                "minimal_atoms": [],
            }
        )
    )
    assert main(["replay", str(bad)]) == 1


def test_main_diff(tmp_path, capsys):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps({"minimal_atoms": ["x", "y"]}))
    b.write_text(json.dumps({"minimal_atoms": ["y", "z"]}))
    assert main(["diff", str(a), str(b)]) == 0
    out = capsys.readouterr().out
    assert "shared" in out and "y" in out


def test_reduce_cost_per_call_makes_usd_budget_bite(tmp_path, monkeypatch, capsys):
    # Regression: --budget-usd was a dead knob because cmd_reduce dropped
    # cost_per_call (every charge was $0.00, so the ceiling never fired). With
    # --cost-per-call wired through, a per-call price makes the USD budget bite.
    import tracemin.adapters.hf as hf

    monkeypatch.setattr(hf, "has_live_token", lambda: True)

    class _FakeReplay:
        replay_capable = True

        def __init__(self, model: str, *, scrub_replay: bool = False) -> None:
            self.model = model
            self.scrub_replay = scrub_replay

        def __call__(self, subset):
            return RawOutput(exit_code=1, exception_type="KeyError")

    monkeypatch.setattr(hf, "HFReplay", _FakeReplay)

    p = tmp_path / "t.json"
    p.write_text(json.dumps([{"role": "user", "content": c} for c in "abcdef"]))
    # max_usd 2.5 admits the two pre-flight calls ($1 each); the first ddmin
    # probe charge ($3 total) trips the ceiling -> budget-truncated.
    rc = main(
        [
            "reduce",
            str(p),
            "--model",
            "m",
            "--oracle",
            "exit-nonzero",
            "--budget-usd",
            "2.5",
            "--cost-per-call",
            "1.0",
        ]
    )
    assert rc == 0
    art = json.loads(capsys.readouterr().out)
    assert art["certified"] is False
    assert art["certified_reason"] == "budget-truncated"
