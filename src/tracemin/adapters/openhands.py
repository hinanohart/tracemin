"""OpenHands trajectory ingestion (replay via the hf engine).

Accepts the V0 single-file form (a ``trajectory.json`` that is a list of events, or
a dict with a ``history``/``events``/``trajectory`` list) and the V1 directory form
(``events/event-*.json``). String-encoded tool ``args`` are deserialized. The adapter
itself does not replay — pair the resulting :class:`Trajectory` with the ``hf`` engine.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

from tracemin.atoms import Atom, AtomKind, Trajectory

replay_capable = False  # ingest-only


def _load_events(source: object) -> list[dict[str, object]]:
    data: object
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_dir():
            event_dir = path / "events" if (path / "events").is_dir() else path
            out: list[dict[str, object]] = []
            for f in sorted(event_dir.glob("event-*.json")):
                parsed = json.loads(f.read_text(encoding="utf-8"))
                if isinstance(parsed, dict):
                    out.append(parsed)
            return out
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = source

    if isinstance(data, list):
        return [e for e in data if isinstance(e, dict)]
    if isinstance(data, dict):
        for key in ("history", "events", "trajectory", "steps"):
            value = data.get(key)
            if isinstance(value, list):
                return [e for e in value if isinstance(e, dict)]
        return [data]
    return []


def _coerce_content(value: object) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _event_to_atom(event: dict[str, object], order: int) -> Atom:
    role = event.get("role") or event.get("source") or "user"
    args = event.get("args")
    if isinstance(args, str):
        with contextlib.suppress(json.JSONDecodeError):
            args = json.loads(args)
    raw_content = event.get("content") or event.get("message") or event.get("text") or args
    kind = (
        AtomKind.INSTRUCTION
        if role == "system" or event.get("action") == "system"
        else AtomKind.MESSAGE
    )
    payload: dict[str, object] = {"role": str(role), "content": _coerce_content(raw_content)}
    action = event.get("action")
    if action is not None:
        payload["action"] = action
    return Atom.make(kind, payload, order=order)


def load_trajectory(source: object) -> Trajectory:
    """Load an OpenHands trajectory (file path, directory, or parsed list/dict)."""
    events = _load_events(source)
    atoms = [_event_to_atom(event, i) for i, event in enumerate(events)]
    return Trajectory.of(atoms)
