# tracemin

**Re-execution-verified, dependency-aware delta-debugging for failed LLM-agent runs.**

Given a failed agent run and a `replay_fn`, `tracemin` shrinks the trajectory's
context — messages, tool definitions, retrieved files, instructions — to a
**1-minimal** subset by *re-executing each candidate* and keeping only those that
still reproduce the **same** failure (matched by a normalized failure signature).

`tracemin` returns a **minimal reproducer**, not a root-cause explanation.

> **Status:** pre-alpha (`0.1.0a1`). API and outputs may change.

<!-- numbers are generated from results/ at S6/S7; no hand-written metrics here -->

## What it does (and does not) claim

**Does:**
- Returns a context subset whose failure is **re-verified by re-execution** (not by static heuristic).
- Keeps every candidate **well-formed by construction** via dependency-aware closure removal.
- With the `[stochastic]` extra, treats a stochastic policy honestly: an atom is
  reported as necessary only when a statistical interval-separation test passes.
- Targets the **same** failure as the original run via a normalized failure signature.

**Does not:**
- Explain *why* the run failed or attribute blame to a step (that is failure attribution, a different problem).
- Promise a unique answer under a non-reproducible (flaky) `replay_fn`.
- Run on every framework out of the box — see the adapter table.

The single-shot core reports `certified: false`. Certification is available only
through the `[stochastic]` extra and only when interval separation passes.

## Install

```bash
pip install tracemin                 # core (re-execution engine + adapters)
pip install "tracemin[stochastic]"   # + statistical certification (numpy/scipy)
pip install "tracemin[all]"          # + seedloop / context-sieve integrations
```

## Quickstart

<!-- QUICKSTART@S7 -->

## Adapters

<!-- ADAPTER-TABLE@S7 -->

## How it works

<!-- ENGINE-NOTES@S7 -->

## Benchmarks

<!-- BENCH@S7: all figures sourced from results/ and tagged [synthetic-benchmark] -->

## Related work

<!-- RELATED-WORK@S7 -->

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
