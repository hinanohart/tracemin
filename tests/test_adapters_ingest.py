from __future__ import annotations

import json

from tracemin.adapters import claude, openhands
from tracemin.adapters._render import render_chat
from tracemin.atoms import Atom, AtomKind


def test_render_chat_maps_kinds():
    atoms = [
        Atom.make(AtomKind.INSTRUCTION, "system rules", order=0),
        Atom.make(AtomKind.TOOL_DEF, {"type": "function", "function": {"name": "search"}}, order=1),
        Atom.make(AtomKind.RETRIEVED_FILE, {"path": "a.py", "content": "x=1"}, order=2),
        Atom.make(AtomKind.MESSAGE, {"role": "assistant", "content": "hi"}, order=3),
        Atom.make(AtomKind.MESSAGE, "plain string", order=4),
    ]
    messages, tools = render_chat(atoms)
    assert messages[0] == {"role": "system", "content": "system rules"}
    assert tools[0]["function"]["name"] == "search"
    assert "retrieved file: a.py" in messages[1]["content"]
    assert messages[2] == {"role": "assistant", "content": "hi"}
    assert messages[3] == {"role": "user", "content": "plain string"}


def test_openhands_v0_list_form():
    data = [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "do the thing"},
        {"role": "assistant", "action": "run", "args": json.dumps({"cmd": "ls"})},
    ]
    traj = openhands.load_trajectory(data)
    assert len(traj.atoms) == 3
    assert traj.atoms[0].kind is AtomKind.INSTRUCTION
    # string-encoded args were deserialized then re-serialized into content
    assert "ls" in str(traj.atoms[2].payload)


def test_openhands_v0_dict_with_history():
    data = {"history": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]}
    traj = openhands.load_trajectory(data)
    assert len(traj.atoms) == 2


def test_openhands_v1_directory(tmp_path):
    events = tmp_path / "events"
    events.mkdir()
    (events / "event-0.json").write_text(json.dumps({"role": "user", "content": "first"}))
    (events / "event-1.json").write_text(json.dumps({"role": "assistant", "content": "second"}))
    traj = openhands.load_trajectory(tmp_path)
    assert len(traj.atoms) == 2
    assert traj.atoms[0].order == 0


def test_claude_jsonl_blocks():
    records = [
        {
            "type": "user",
            "message": {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "hmm"},
                    {"type": "tool_use", "name": "grep", "input": {"q": "x"}},
                ],
            },
        },
        {"type": "user", "message": {"role": "user", "content": []}},  # empty -> skipped
    ]
    traj = claude.load_transcript(records)
    assert len(traj.atoms) == 2
    assert "hello" in str(traj.atoms[0].payload)
    assert "tool_use grep" in str(traj.atoms[1].payload)
    assert claude.replay_capable is False


def test_claude_jsonl_from_file(tmp_path):
    p = tmp_path / "transcript.jsonl"
    lines = [
        json.dumps({"type": "user", "message": {"role": "user", "content": "a"}}),
        json.dumps({"type": "assistant", "message": {"role": "assistant", "content": "b"}}),
    ]
    p.write_text("\n".join(lines))
    traj = claude.load_transcript(p)
    assert len(traj.atoms) == 2
