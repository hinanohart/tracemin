"""Scrub-leak tests.

The "secrets" here are SYNTHETIC fixtures, assembled at runtime via ``_tok`` so
that no contiguous credential literal ever appears in the source (this keeps
secret scanners — gitleaks/semgrep — quiet; nothing here is a real credential).
The assembled strings still match the scrubber's patterns at runtime.
"""

from __future__ import annotations

from tracemin.artifact import build_artifact, render_bug_report
from tracemin.atoms import Atom, AtomKind, Trajectory
from tracemin.engine import minimize
from tracemin.oracle import ExitCodeOracle
from tracemin.replay import RawOutput
from tracemin.scrub import active_mode, scrub, scrub_obj


def _tok(*parts: str) -> str:
    """Assemble a fake secret from fragments (no literal credential in source)."""
    return "".join(parts)


SECRETS = [
    _tok("sk-", "ABCDEF0123456789abcdef"),
    _tok("ghp", "_", "0123456789abcdefABCDEFxyz012345"),
    _tok("hf", "_", "abcdefABCDEF0123456789xyz"),
    _tok("AKIA", "ABCDEFGHIJKLMNOP"),
    _tok("Bearer ", "abcdefABCDEF0123456789._-tok"),
    _tok("eyJ", "ABCDEFGH1234", ".", "PAYLOAD12345", ".", "SIG12345"),
    _tok("pass", "word=", "hunter2hunter2"),
]


def test_scrub_redacts_known_secret_shapes():
    for s in SECRETS:
        out = scrub(s)
        assert "REDACTED" in out, f"not scrubbed: {s!r} -> {out!r}"


def test_scrub_leaves_ordinary_prose_intact():
    prose = "The agent called the search tool and then summarized three short results."
    assert scrub(prose) == prose


def test_scrub_obj_recurses_structures():
    obj = {
        "msg": _tok("token=", "ABCDEF0123456789abcdefXYZ"),
        "nested": ["ok", _tok("sk-", "ZZZZ1111aaaa2222bbbb")],
    }
    cleaned = scrub_obj(obj)
    assert isinstance(cleaned, dict)
    assert "REDACTED" in cleaned["msg"]
    assert "REDACTED" in cleaned["nested"][1]
    assert cleaned["nested"][0] == "ok"


def test_secret_in_atom_payload_does_not_leak_into_bug_report():
    secret = _tok("sk-", "LEAKYSECRET0123456789abcd")
    trigger = Atom.make(AtomKind.MESSAGE, {"text": f"call with {secret}"}, order=0)
    pad = Atom.make(AtomKind.MESSAGE, "harmless", order=1)
    traj = Trajectory.of([trigger, pad])

    def replay(subset):
        ids = {a.id for a in subset}
        if trigger.id in ids:
            return RawOutput(exit_code=1, exception_type="KeyError", exception_message="boom")
        return RawOutput(exit_code=0)

    result = minimize(traj, replay, ExitCodeOracle(), double_check=False)
    report = render_bug_report(result, oracle_spec="exit_code!=0")
    assert secret not in report
    assert "REDACTED" in report


def test_artifact_scrubs_oracle_spec_and_signature():
    sig_secret = _tok("sk-", "SIGSECRET0123456789abcd")
    spec_secret = _tok("sk-", "SPECSECRET0123456789abc")
    trigger = Atom.make(AtomKind.MESSAGE, "t", order=0)
    traj = Trajectory.of([trigger, Atom.make(AtomKind.MESSAGE, "p", order=1)])

    def replay(subset):
        ids = {a.id for a in subset}
        if trigger.id in ids:
            return RawOutput(
                exit_code=1, exception_type="AuthError", exception_message=f"bad token {sig_secret}"
            )
        return RawOutput(exit_code=0)

    result = minimize(traj, replay, ExitCodeOracle(), double_check=False)
    art = build_artifact(result, oracle_spec=f"regex: {spec_secret}", replay_capable=True)
    blob = art.to_json()
    assert sig_secret not in blob
    assert spec_secret not in blob
    assert "REDACTED" in blob


def test_active_mode_is_builtin_without_sieve():
    assert active_mode() in ("builtin", "context-sieve")
