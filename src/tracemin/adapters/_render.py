"""Shared rendering: turn a sequence of atoms into chat messages + tool schemas.

Payload conventions (lenient — adapters normalize to these):
  * MESSAGE         payload is ``{"role", "content"}`` or a plain string (-> user).
  * INSTRUCTION     payload is text or ``{"content"}`` (-> system message).
  * TOOL_DEF        payload is a tool/function schema dict.
  * RETRIEVED_FILE  payload is ``{"path", "content"}`` (-> a user context message).
"""

from __future__ import annotations

from collections.abc import Sequence

from tracemin.atoms import Atom, AtomKind

Message = dict[str, object]
Tool = dict[str, object]


def _text_of(payload: object) -> str:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        content = payload.get("content")
        if isinstance(content, str):
            return content
        return str(content) if content is not None else str(payload)
    return str(payload)


def _as_message(payload: object, default_role: str = "user") -> Message:
    if isinstance(payload, dict) and "role" in payload:
        role = payload.get("role", default_role)
        return {"role": str(role), "content": _text_of(payload)}
    return {"role": default_role, "content": _text_of(payload)}


def _as_tool(payload: object) -> Tool:
    if isinstance(payload, dict):
        return dict(payload)
    return {"type": "function", "function": {"name": str(payload)}}


def _as_file_message(payload: object) -> Message:
    if isinstance(payload, dict):
        path = payload.get("path", "file")
        content = payload.get("content", "")
        return {"role": "user", "content": f"[retrieved file: {path}]\n{content}"}
    return {"role": "user", "content": f"[retrieved file]\n{payload}"}


def render_chat(atoms: Sequence[Atom]) -> tuple[list[Message], list[Tool]]:
    """Render atoms into ``(messages, tools)`` suitable for a chat completion call."""
    messages: list[Message] = []
    tools: list[Tool] = []
    for atom in atoms:
        if atom.kind is AtomKind.TOOL_DEF:
            tools.append(_as_tool(atom.payload))
        elif atom.kind is AtomKind.INSTRUCTION:
            messages.append({"role": "system", "content": _text_of(atom.payload)})
        elif atom.kind is AtomKind.RETRIEVED_FILE:
            messages.append(_as_file_message(atom.payload))
        else:  # MESSAGE
            messages.append(_as_message(atom.payload))
    return messages, tools
