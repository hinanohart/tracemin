from __future__ import annotations

from types import SimpleNamespace

import pytest

from tracemin.adapters.hf import HFReplay, parse_response
from tracemin.atoms import Atom, AtomKind, Trajectory
from tracemin.engine import minimize
from tracemin.oracle import RegexOracle
from tracemin.replay import ReplayError


def _resp(content: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def test_parse_response_object_and_dict_shapes():
    assert parse_response(_resp("hello")).text == "hello"
    dict_resp = {"choices": [{"message": {"content": "world"}}]}
    assert parse_response(dict_resp).text == "world"
    assert parse_response(object()).text != ""  # defensive fallback never crashes


def test_hf_replay_builds_request_and_parses():
    captured = {}

    class FakeClient:
        def chat_completion(self, **kw):
            captured.update(kw)
            return _resp("the answer is 42")

    atoms = [
        Atom.make(AtomKind.INSTRUCTION, "be terse", order=0),
        Atom.make(AtomKind.MESSAGE, {"role": "user", "content": "what is 6*7?"}, order=1),
    ]
    out = HFReplay("some/model", client=FakeClient())(atoms)
    assert out.text == "the answer is 42"
    assert captured["temperature"] == 0.0
    assert captured["messages"][0] == {"role": "system", "content": "be terse"}
    assert captured["messages"][1]["role"] == "user"


def test_hf_replay_wraps_infra_error():
    class BoomClient:
        def chat_completion(self, **kw):
            raise ConnectionError("503")

    with pytest.raises(ReplayError):
        HFReplay("m", client=BoomClient())([Atom.make(AtomKind.MESSAGE, "hi", order=0)])


def test_full_pipeline_with_hf_substitution_no_network():
    # A fake model that "fails" (emits KeyError) only when the trigger message is present.
    trigger = Atom.make(AtomKind.MESSAGE, {"role": "user", "content": "TRIGGER_BUG"}, order=1)
    atoms = [
        Atom.make(AtomKind.MESSAGE, {"role": "user", "content": "hello"}, order=0),
        trigger,
        Atom.make(AtomKind.MESSAGE, {"role": "user", "content": "bye"}, order=2),
    ]
    traj = Trajectory.of(atoms)

    class FakeModel:
        def chat_completion(self, *, messages, **kw):
            blob = " ".join(str(m.get("content", "")) for m in messages)
            return _resp("Traceback KeyError: 'x'" if "TRIGGER_BUG" in blob else "all good")

    replay = HFReplay("fake/model", client=FakeModel())
    result = minimize(traj, replay, RegexOracle("KeyError"), double_check=False)
    assert result.minimal_ids == (trigger.id,)
    assert result.verdict_on_minimal.name == "FAIL"
