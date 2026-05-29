from __future__ import annotations

from tracemin.replay import RawOutput
from tracemin.signature import Signature, failure_signature, normalize_text


def test_normalize_strips_incidental_detail():
    raw = "Traceback (/home/u/proj/app.py, line 42): boom at 0xdeadBEEF"
    norm = normalize_text(raw)
    assert "0xADDR" in norm
    assert "line N" in norm
    assert "/home/u/proj" not in norm
    assert "app.py" in norm


def test_exception_signature_groups_same_logical_failure():
    a = RawOutput(exception_type="KeyError", exception_message="'x' at /tmp/app.py line 10")
    b = RawOutput(exception_type="KeyError", exception_message="'x' at /var/lib/app.py line 99")
    sig_a = failure_signature(a)
    sig_b = failure_signature(b)
    assert sig_a is not None and sig_b is not None
    # Same exception type + same normalized message -> same key (paths/lineno stripped).
    assert sig_a.key == sig_b.key
    assert sig_a == sig_b


def test_different_exceptions_differ():
    a = failure_signature(RawOutput(exception_type="KeyError", exception_message="x"))
    b = failure_signature(RawOutput(exception_type="ValueError", exception_message="x"))
    assert a is not None and b is not None
    assert a.key != b.key


def test_exit_code_signature_uses_stderr_tail():
    sig = failure_signature(RawOutput(exit_code=2, stderr="line1\nfatal: bad config\n"))
    assert sig is not None
    assert sig.parts["exit_code"] == "2"
    assert "fatal: bad config" in sig.parts["tail"]


def test_no_signal_returns_none():
    assert failure_signature(RawOutput(exit_code=0)) is None
    assert failure_signature(RawOutput()) is None


def test_signature_hashable_and_equal_by_key():
    s1 = Signature({"a": "1", "b": "2"})
    s2 = Signature({"b": "2", "a": "1"})
    assert s1 == s2
    assert len({s1, s2}) == 1
