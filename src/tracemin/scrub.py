"""The secret-scrubbing chokepoint for tracemin's own artifacts and output.

The text tracemin *produces* — the embedded atoms of a ``--script``, a bug
report, CLI output, logs, the artifact's oracle spec and signature — must pass
through :func:`scrub`. It is **default-ON** with a stdlib-only regex fallback,
so these outputs are never leaky even with no extra installed. With the
``[sieve]`` extra it upgrades to context-sieve's blocklist + entropy engine;
:func:`active_mode` reports which is live.

Important scope note: this chokepoint covers tracemin's *own* outputs, not the
trajectory content sent to a live replay endpoint. By design, a replay engine
re-prompts a model with the candidate atoms verbatim (replay fidelity), so raw
trajectory text — which may contain secrets ingested from a transcript — is sent
to the configured endpoint unredacted unless the caller opts into scrubbing
(e.g. ``HFReplay(..., scrub_replay=True)`` / ``reduce --scrub-replay``). Scrubbing
replay input can change model behavior and therefore the minimization verdict, so
it is opt-in rather than the default.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping

_REDACT = "«REDACTED:{label}»"

# High-confidence token shapes.
_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"sk-[A-Za-z0-9]{16,}"), "OPENAI_KEY"),
    (re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"), "GITHUB_TOKEN"),
    (re.compile(r"hf_[A-Za-z0-9]{16,}"), "HF_TOKEN"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "AWS_ACCESS_KEY"),
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{16,}"), "BEARER"),
    (re.compile(r"eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{6,}"), "JWT"),
]

# `name = value` / `name: value` style secret assignments.
_ASSIGN = re.compile(
    r"(?i)\b(api[_-]?key|secret|password|passwd|access[_-]?token|token)\b\s*[:=]\s*['\"]?([^\s'\"]{8,})"
)

# Candidate high-entropy blobs (base64/hex-ish) of meaningful length.
_BLOB = re.compile(r"[A-Za-z0-9+/=_\-]{24,}")
_ENTROPY_BITS = 4.0


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts: dict[str, int] = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _redact_high_entropy(text: str) -> str:
    def repl(m: re.Match[str]) -> str:
        tok = m.group(0)
        return (
            _REDACT.format(label="HIGH_ENTROPY") if shannon_entropy(tok) >= _ENTROPY_BITS else tok
        )

    return _BLOB.sub(repl, text)


def _builtin_scrub(text: str) -> str:
    for pat, label in _PATTERNS:
        text = pat.sub(_REDACT.format(label=label), text)
    text = _ASSIGN.sub(lambda m: f"{m.group(1)}={_REDACT.format(label='SECRET')}", text)
    return _redact_high_entropy(text)


def active_mode() -> str:
    """Which scrubber is live: ``"context-sieve"`` (the [sieve] extra) or ``"builtin"``."""
    try:
        import context_sieve  # noqa: F401
    except Exception:
        return "builtin"
    return "context-sieve"


def scrub(text: str) -> str:
    """The single outbound chokepoint for any text that may carry secrets."""
    if not text:
        return text
    if active_mode() == "context-sieve":
        try:
            from context_sieve import Sanitizer

            cleaned, _ = Sanitizer().sanitize(text)
            return str(cleaned)
        except Exception:
            pass  # never let an optional dependency make us leakier than builtin
    return _builtin_scrub(text)


def scrub_obj(obj: object) -> object:
    """Recursively scrub strings inside dict / list / tuple structures (keys kept)."""
    if isinstance(obj, str):
        return scrub(obj)
    if isinstance(obj, Mapping):
        return {k: scrub_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [scrub_obj(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(scrub_obj(v) for v in obj)
    return obj
