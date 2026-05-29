from __future__ import annotations

from tracemin.bench.metrics import run_suite


def test_suite_recovers_known_ground_truth():
    m = run_suite(seed=0, n_per_family=10)
    assert m["recovery_recall"] == 1.0
    assert m["recovery_precision"] == 1.0
    assert m["minimality_verify_rate"] == 1.0
    assert 0.0 <= float(m["reduction_ratio"]) <= 1.0  # type: ignore[arg-type]


def test_signature_reduces_false_reproducers():
    # C4: the failure signature must drive the false-reproducer rate strictly down.
    m = run_suite(seed=0, n_per_family=20)
    assert m["false_reproducer_rate_with_sig"] == 0.0
    assert float(m["false_reproducer_rate_no_sig"]) > float(m["false_reproducer_rate_with_sig"])  # type: ignore[arg-type]
    # the difference is statistically significant (paired exact McNemar)
    assert float(m["false_reproducer_mcnemar_p"]) < 0.05  # type: ignore[arg-type]


def test_suite_is_deterministic_for_a_fixed_seed():
    a = run_suite(seed=0, n_per_family=8)
    b = run_suite(seed=0, n_per_family=8)
    assert a == b
