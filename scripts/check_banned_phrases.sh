#!/usr/bin/env bash
# Banned-phrase gate: fail if an overclaim phrase appears in user-facing docs.
#
# Uses a single ERE alternation with `grep -E` (never BRE `\|`), and self-tests
# its own return code against a poison fixture (must fire) and a clean fixture
# (must stay silent) before trusting the result. Do not write any banned phrase
# into the docs even to negate it — the gate cannot read intent.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 2

DOCS=("README.md")
shopt -s nullglob
DOCS+=(docs/*.md)
shopt -u nullglob

# Each pattern is an overclaim with no honest use in our docs.
PATTERNS='root[ -]?cause|diagnoses?|diagnostic|guarantees? (determinism|to find)|deterministic(ally)?|any framework|every framework|universal|the first|first (open|oss|tool|framework|library|package)|SOTA|state[ -]of[ -]the[ -]art|silver bullet|fixes the|repairs the'

scan() { grep -rEniH "$PATTERNS" "$@" 2>/dev/null; }

# --- return-code self-test --------------------------------------------------
poison="$(mktemp)"; clean="$(mktemp)"
trap 'rm -f "$poison" "$clean"' EXIT
printf 'this finds the root cause and is the first universal SOTA debugger\n' >"$poison"
printf 'returns a 1-minimal reproducer re-verified by re-execution\n' >"$clean"
if ! scan "$poison" >/dev/null; then
  echo "SELF-TEST FAIL: gate did not fire on poison fixture" >&2; exit 2
fi
if scan "$clean" >/dev/null; then
  echo "SELF-TEST FAIL: gate false-fired on clean fixture:" >&2; scan "$clean" >&2; exit 2
fi

# --- real docs --------------------------------------------------------------
hits="$(scan "${DOCS[@]}")"
if [ -n "$hits" ]; then
  echo "banned-phrase gate FAILED — overclaim phrases found:" >&2
  echo "$hits" >&2
  exit 1
fi
echo "banned-phrase gate OK (${#DOCS[@]} doc(s) clean; rc self-test passed)"
