"""Failure signatures: a normalized identity key for "the same failure".

Reduction must shrink toward the *same* failure, not merely *a* failure. A
degenerate reproducer — a subset that fails for an unrelated reason after we
removed the real trigger — is the central correctness hazard. The signature is
the guard: the engine accepts a candidate only when its verdict is FAIL *and* its
signature equals the full input's signature.

Signatures are normalized so that incidental churn (absolute paths, line numbers,
memory addresses, whitespace) does not split one logical failure into many keys —
which would otherwise stop reduction early (over-minimization).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from tracemin.replay import RawOutput

_UNIX_PATH = re.compile(r"(?:/[\w.+\-]+)+/([\w.+\-]+)")
_WIN_PATH = re.compile(r"[A-Za-z]:\\(?:[\w.+\-]+\\)*([\w.+\-]+)")
_LINE_NO = re.compile(r"\bline\s+\d+", re.IGNORECASE)
_COLON_NO = re.compile(r":\d+(?=[:\s)\]]|$)")
_HEX = re.compile(r"0x[0-9a-fA-F]+")
_WS = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Strip incidental, run-to-run-varying detail from a failure string."""
    if not text:
        return ""
    text = _WIN_PATH.sub(r"\1", text)
    text = _UNIX_PATH.sub(r"\1", text)
    text = _HEX.sub("0xADDR", text)
    text = _LINE_NO.sub("line N", text)
    text = _COLON_NO.sub(":N", text)
    text = _WS.sub(" ", text)
    return text.strip()


def _last_meaningful_line(text: str) -> str:
    for line in reversed(text.splitlines()):
        if line.strip():
            return line.strip()
    return ""


@dataclass(frozen=True)
class Signature:
    """A normalized failure identity, with a canonical string ``key``."""

    parts: dict[str, str] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return "|".join(f"{k}={self.parts[k]}" for k in sorted(self.parts))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Signature) and other.key == self.key

    def __hash__(self) -> int:
        return hash(self.key)


def failure_signature(output: RawOutput) -> Signature | None:
    """Default signature: prefer the exception identity, else exit code + stderr tail.

    Returns ``None`` when the output carries no failure-identifying detail; the
    engine then treats every FAIL verdict as matching (the documented fallback to
    "any FAIL" when no signature is available).
    """
    parts: dict[str, str] = {}
    if output.exception_type:
        parts["exc_type"] = output.exception_type.strip()
        parts["exc_msg"] = normalize_text(output.exception_message or "")
        return Signature(parts)

    if output.exit_code not in (None, 0):
        parts["exit_code"] = str(output.exit_code)
        tail = _last_meaningful_line(output.stderr) or _last_meaningful_line(output.text)
        if tail:
            parts["tail"] = normalize_text(tail)
        return Signature(parts)

    tail = _last_meaningful_line(output.stderr) or _last_meaningful_line(output.text)
    if tail:
        parts["tail"] = normalize_text(tail)
        return Signature(parts)

    return None
