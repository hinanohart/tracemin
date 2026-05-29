from __future__ import annotations

from tracemin.oracle import (
    AnswerMismatchOracle,
    ExceptionOracle,
    ExitCodeOracle,
    NotRegexOracle,
    PredicateOracle,
    RegexOracle,
    all_of,
    any_of,
)
from tracemin.replay import RawOutput, Verdict


def test_exit_code_oracle():
    assert ExitCodeOracle().classify(RawOutput(exit_code=1)) is Verdict.FAIL
    assert ExitCodeOracle().classify(RawOutput(exit_code=0)) is Verdict.PASS
    assert ExitCodeOracle().classify(RawOutput(exit_code=None)) is Verdict.PASS
    assert ExitCodeOracle(fail_on=[42]).classify(RawOutput(exit_code=42)) is Verdict.FAIL
    assert ExitCodeOracle(fail_on=[42]).classify(RawOutput(exit_code=1)) is Verdict.PASS


def test_exception_oracle_type_and_message():
    out = RawOutput(exception_type="KeyError", exception_message="missing 'x'")
    assert ExceptionOracle().classify(out) is Verdict.FAIL
    assert ExceptionOracle("Key.*").classify(out) is Verdict.FAIL
    assert ExceptionOracle("ValueError").classify(out) is Verdict.PASS
    assert ExceptionOracle(message_pattern="missing").classify(out) is Verdict.FAIL
    assert ExceptionOracle(message_pattern="nope").classify(out) is Verdict.PASS
    assert ExceptionOracle().classify(RawOutput()) is Verdict.PASS


def test_regex_and_notregex():
    assert RegexOracle("boom", "stderr").classify(RawOutput(stderr="boom!")) is Verdict.FAIL
    assert RegexOracle("boom", "stderr").classify(RawOutput(stderr="ok")) is Verdict.PASS
    assert NotRegexOracle("ANSWER:42").classify(RawOutput(text="no answer")) is Verdict.FAIL
    assert NotRegexOracle("ANSWER:42").classify(RawOutput(text="ANSWER:42")) is Verdict.PASS


def test_answer_mismatch_normalizes_whitespace():
    o = AnswerMismatchOracle("the   answer\nis 4")
    assert o.classify(RawOutput(text="the answer is 4")) is Verdict.PASS
    assert o.classify(RawOutput(text="the answer is 5")) is Verdict.FAIL


def test_predicate_oracle_bool_and_verdict():
    assert PredicateOracle(lambda o: True).classify(RawOutput()) is Verdict.FAIL
    assert PredicateOracle(lambda o: False).classify(RawOutput()) is Verdict.PASS
    assert PredicateOracle(lambda o: Verdict.ERROR).classify(RawOutput()) is Verdict.ERROR


def test_combinators_and_error_propagation():
    fail = ExitCodeOracle()
    passing = RegexOracle("never-matches")
    out = RawOutput(exit_code=1)
    assert all_of(fail, passing).classify(out) is Verdict.PASS  # not all FAIL
    assert any_of(fail, passing).classify(out) is Verdict.FAIL  # one FAIL
    err = PredicateOracle(lambda o: Verdict.ERROR)
    assert all_of(fail, err).classify(out) is Verdict.ERROR
    assert any_of(passing, err).classify(out) is Verdict.ERROR
