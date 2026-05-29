from __future__ import annotations

from tracemin.oracle import ExitCodeOracle
from tracemin.replay import RawOutput, ReplayError, Verdict, run_trial
from tracemin.signature import failure_signature


def _fail_output():
    return RawOutput(
        exit_code=1, stderr="KeyError: 'x'", exception_type="KeyError", exception_message="'x'"
    )


def test_replay_error_maps_to_error_verdict():
    def boom(subset):
        raise ReplayError("503 from inference endpoint")

    res = run_trial(boom, ExitCodeOracle(), failure_signature, [])
    assert res.verdict is Verdict.ERROR
    assert res.is_error and not res.is_fail


def test_transport_error_field_maps_to_error():
    res = run_trial(
        lambda s: RawOutput(transport_error="quota"), ExitCodeOracle(), failure_signature, []
    )
    assert res.verdict is Verdict.ERROR


def test_fail_sets_signature():
    res = run_trial(lambda s: _fail_output(), ExitCodeOracle(), failure_signature, [])
    assert res.verdict is Verdict.FAIL
    assert res.signature is not None and "exc_type=KeyError" in res.signature


def test_pass_leaves_signature_none():
    res = run_trial(lambda s: RawOutput(exit_code=0), ExitCodeOracle(), failure_signature, [])
    assert res.verdict is Verdict.PASS
    assert res.signature is None


def test_oracle_returning_error_propagates():
    res = run_trial(
        lambda s: RawOutput(exit_code=1), lambda o: Verdict.ERROR, failure_signature, []
    )
    assert res.verdict is Verdict.ERROR


def test_signature_fn_none_means_any_fail():
    res = run_trial(lambda s: _fail_output(), ExitCodeOracle(), None, [])
    assert res.verdict is Verdict.FAIL
    assert res.signature is None
