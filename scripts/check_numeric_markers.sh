#!/usr/bin/env bash
# Numeric-marker gate: every metric-like number in docs must carry a source tag
# ([measured] / [synthetic-benchmark] / [illustrative]). Prevents shipping an
# untagged figure that reads as a real-world measurement.
#
# "Metric-like" is restricted to percentages and ratios (e.g. 95%, 1.8x, 2.3×)
# so that version numbers (0.1.0a1, py3.10, Apache 2.0) are not flagged. The
# gate self-tests its return code before trusting the result.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 2

DOCS=("README.md")
shopt -s nullglob
DOCS+=(docs/*.md)
shopt -u nullglob

MARKER='\[(measured|synthetic-benchmark|illustrative)\]'
METRIC='[0-9]+(\.[0-9]+)?%|[0-9]+(\.[0-9]+)?[x×]'

violations() { grep -rEnH "$METRIC" "$@" 2>/dev/null | grep -Ev "$MARKER"; }

# --- return-code self-test --------------------------------------------------
poison="$(mktemp)"; clean="$(mktemp)"
trap 'rm -f "$poison" "$clean"' EXIT
printf 'recall reached 95%% on the benchmark\n' >"$poison"
printf 'recall reached 95%% [synthetic-benchmark]\n' >"$clean"
if ! violations "$poison" >/dev/null; then
  echo "SELF-TEST FAIL: did not catch an untagged metric" >&2; exit 2
fi
if violations "$clean" >/dev/null; then
  echo "SELF-TEST FAIL: false-fired on a tagged metric:" >&2; violations "$clean" >&2; exit 2
fi

# --- real docs --------------------------------------------------------------
v="$(violations "${DOCS[@]}")"
if [ -n "$v" ]; then
  echo "numeric-marker gate FAILED — untagged metric-like numbers:" >&2
  echo "$v" >&2
  exit 1
fi
echo "numeric-marker gate OK (self-test passed)"
