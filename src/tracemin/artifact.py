"""The canonical reproducer artifact (schema ``tracemin-repro/1``).

The artifact embeds its own minimality certificate and a ``minimality_scope`` of
``wf-constrained`` (so it is never confused with unconstrained 1-minimality).
A **mode-consistency validator** runs on every build, making it structurally
impossible to emit an artifact that claims certification it did not earn:

  * ``replay_mode == "single-shot"``  ⟹  ``certified == false`` and an
    ``illustrative`` confidence method.
  * ``certified == true``  ⟹  ``replay_mode == "stochastic"`` and every
    (non-pinned) witness is ``proven``.

All free text that could carry secrets (oracle spec, signature, embedded atom
payloads) is routed through :func:`tracemin.scrub.scrub`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass

from tracemin.atoms import Atom
from tracemin.engine import MinimizeResult
from tracemin.scrub import scrub, scrub_obj

SCHEMA = "tracemin-repro/1"


class ModeConsistencyError(ValueError):
    """Raised when an artifact's certification claim is inconsistent with its mode."""


def validate_mode(data: Mapping[str, object]) -> None:
    """Enforce the honesty invariants. Raises :class:`ModeConsistencyError`."""
    replay_mode = data["replay_mode"]
    certified = data["certified"]
    confidence = data["confidence"]
    method = confidence["method"] if isinstance(confidence, Mapping) else None

    if replay_mode == "single-shot":
        if certified is not False:
            raise ModeConsistencyError("single-shot artifact must have certified=false")
        if not (isinstance(method, str) and method.startswith("illustrative")):
            raise ModeConsistencyError("single-shot confidence.method must be 'illustrative'")

    if certified is True:
        if replay_mode != "stochastic":
            raise ModeConsistencyError("certified=true requires replay_mode='stochastic'")
        minimality = data["minimality"]
        witnesses = minimality["witnesses"] if isinstance(minimality, Mapping) else []
        for w in witnesses:
            if w["necessity"] not in ("proven", "pinned"):
                raise ModeConsistencyError(
                    "certified=true requires every non-pinned witness to be 'proven'"
                )


@dataclass
class Artifact:
    """A built artifact: the validated schema dict plus serialization helpers."""

    data: dict[str, object]

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.data, indent=indent, ensure_ascii=False, sort_keys=False)


def build_artifact(
    result: MinimizeResult,
    *,
    oracle_spec: str,
    replay_capable: bool,
    confidence_method: str = "illustrative",
    k: int = 1,
    tau: float | None = None,
) -> Artifact:
    """Assemble and validate a ``tracemin-repro/1`` artifact from a reduction result."""
    data: dict[str, object] = {
        "schema": SCHEMA,
        "verified": bool(result.reverified_post_hoc),
        "certified": bool(result.certified),
        "certified_reason": result.certified_reason,
        "replay_mode": result.replay_mode,
        "confidence": {"method": confidence_method, "k": k, "tau": tau},
        "reference_signature": scrub(result.reference_signature)
        if result.reference_signature
        else None,
        "minimal_atoms": list(result.minimal_ids),
        "minimality_scope": "wf-constrained",
        "minimality": {
            "certificate": {
                "engine": "ddmin",
                "reverified_post_hoc": bool(result.reverified_post_hoc),
            },
            "witnesses": [{"atom": w.atom_id, "necessity": w.necessity} for w in result.witnesses],
        },
        "oracle": {
            "spec": scrub(oracle_spec),
            "verdict_on_minimal": result.verdict_on_minimal.name,
        },
        "stats": {
            "replay_calls": int(result.stats.get("replay_calls", 0)),
            "cache_hits": int(result.stats.get("cache_hits", 0)),
        },
        "provenance": {"sanitized": True, "replay_capable": bool(replay_capable)},
    }
    validate_mode(data)
    return Artifact(data)


def render_bug_report(
    result: MinimizeResult,
    *,
    oracle_spec: str,
    atoms: tuple[Atom, ...] | None = None,
) -> str:
    """Render a human-readable minimal-reproducer report. Atom payloads are scrubbed."""
    atoms = atoms if atoms is not None else result.minimal_atoms
    note = (
        "reproduced (single-shot, not certified)"
        if not result.certified
        else "reproduced and certified (stochastic)"
    )
    lines = [
        "# tracemin minimal reproducer",
        "",
        f"- schema: `{SCHEMA}`",
        f"- status: {note}",
        "- minimality scope: wf-constrained 1-minimal",
        f"- reference signature: `{scrub(result.reference_signature or '∅')}`",
        f"- oracle: `{scrub(oracle_spec)}`",
        f"- verdict on minimal set: {result.verdict_on_minimal.name}",
        f"- replay calls: {result.stats.get('replay_calls', 0)}",
        "",
        "## Minimal context atoms",
        "",
    ]
    for atom in atoms:
        payload = scrub_obj(atom.payload)
        lines.append(f"- `{atom.id}` ({atom.kind.value}): {payload!r}")
    lines.append("")
    lines.append(
        "> This is a minimal reproducer — the smallest context that still triggers "
        "the same failure — not an explanation of the underlying cause."
    )
    return "\n".join(lines)
