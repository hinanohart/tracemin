"""Claude Code JSONL transcript ingestion (reduction-only by default).

A Claude Code transcript is newline-delimited JSON, one record per line, each with
a ``message`` whose ``content`` is a list of blocks (``text`` / ``thinking`` /
``tool_use`` / ``tool_result``) linked by ``parentUuid``. Because re-driving such a
run depends on the local tool tree, this adapter is **reduction-only** by default:
it yields a :class:`Trajectory` for reduction, and verification requires attaching
an explicit replay engine (e.g. the ``hf`` engine).
"""

from __future__ import annotations

import json
from pathlib import Path

from tracemin.atoms import Atom, AtomKind, Trajectory

replay_capable = False  # reduction-only; attach an engine to verify


def _load_records(source: object) -> list[dict[str, object]]:
    if isinstance(source, (str, Path)):
        text = Path(source).read_text(encoding="utf-8")
        lines = [ln for ln in text.splitlines() if ln.strip()]
        records = [json.loads(ln) for ln in lines]
    elif isinstance(source, list):
        records = list(source)
    else:
        records = []
    return [r for r in records if isinstance(r, dict)]


def _blocks_to_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content) if content is not None else ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            btype = block.get("type")
            if btype == "text":
                parts.append(str(block.get("text", "")))
            elif btype == "thinking":
                parts.append(str(block.get("thinking", "")))
            elif btype == "tool_use":
                name = block.get("name", "tool")
                args = json.dumps(block.get("input", {}), ensure_ascii=False, sort_keys=True)
                parts.append(f"[tool_use {name} {args}]")
            elif btype == "tool_result":
                parts.append(f"[tool_result] {_blocks_to_text(block.get('content'))}")
    return "\n".join(p for p in parts if p)


def _record_to_atom(record: dict[str, object], order: int) -> Atom | None:
    message = record.get("message")
    if isinstance(message, dict):
        role = str(message.get("role", record.get("type", "user")))
        content = _blocks_to_text(message.get("content"))
    else:
        role = str(record.get("role", record.get("type", "user")))
        content = _blocks_to_text(record.get("content"))
    if not content:
        return None
    kind = AtomKind.INSTRUCTION if role == "system" else AtomKind.MESSAGE
    payload: dict[str, object] = {"role": role, "content": content}
    return Atom.make(kind, payload, order=order)


def load_transcript(source: object) -> Trajectory:
    """Load a Claude Code JSONL transcript (file path or parsed list of records)."""
    records = _load_records(source)
    atoms: list[Atom] = []
    order = 0
    for record in records:
        atom = _record_to_atom(record, order)
        if atom is not None:
            atoms.append(atom)
            order += 1
    return Trajectory.of(atoms)
