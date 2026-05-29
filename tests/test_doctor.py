from __future__ import annotations

from tracemin.doctor import run_doctor


def test_doctor_reports_three_valued_status():
    report = run_doctor()
    statuses = {r.status for r in report.rows}
    assert statuses <= {"LIVE", "MOCK", "MISSING"}
    assert report.status_of("tracemin") == "LIVE"
    # core dependency always present, so the hf adapter is never MISSING here
    assert report.status_of("adapter:hf") in ("LIVE", "MOCK")
    assert report.status_of("scrub") == "LIVE"


def test_doctor_hf_is_mock_without_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    report = run_doctor()
    assert report.status_of("adapter:hf") == "MOCK"


def test_doctor_hf_is_live_with_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "x")  # presence only; value never used here
    report = run_doctor()
    assert report.status_of("adapter:hf") == "LIVE"


def test_doctor_render_has_no_bare_checkmark():
    text = run_doctor().render()
    assert "✓" not in text
    assert "[LIVE" in text or "[MOCK" in text
    # every row shows an explicit bracketed status
    for line in text.splitlines()[2:]:
        if line.strip():
            assert "[" in line and "]" in line
