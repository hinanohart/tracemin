# Benchmark

All figures below are `[synthetic-benchmark]` results: they come from a
failure-injection suite where the ground-truth minimal set is known **by
construction**, and are produced by running the real engine over those cases.
The canonical numbers live in [`results/v0.1.0a1_bench.json`](results/v0.1.0a1_bench.json)
and are reproduced by:

```bash
python -m tracemin.bench.metrics --seed 0 --n 20 --date <DATE> --out results/v0.1.0a1_bench.json
```

> **Disclaimer (verbatim):** This is a synthetic benchmark demonstrating that the
> algorithm recovers a known injected ground truth; it is not a prediction of
> real-world performance.

## What is measured

Four trigger families with known ground truth — single-atom, conjunctive (AND),
dependency-entangled (closure must keep the producer), and distractor-padding —
plus a decoy family (removing the real trigger yields a *different* failure) and a
fixed-seed stochastic family.

## Results (seed 0, n=20 per family) `[synthetic-benchmark]`

| Metric | Value | Meaning |
|---|---|---|
| recovery precision | `1.0` `[synthetic-benchmark]` | recovered atoms are exactly the injected ones |
| recovery recall | `1.0` `[synthetic-benchmark]` | the injected ground truth is always recovered |
| reduction ratio | `0.77` `[synthetic-benchmark]` | fraction of atoms removed |
| 1-minimality (wf-constrained) verify rate | `1.0` `[synthetic-benchmark]` | every result is wf-constrained 1-minimal |
| false-reproducer rate, **without** signature | `0.45` `[synthetic-benchmark]` | decoy cases where the wrong failure is "reproduced" |
| false-reproducer rate, **with** signature | `0.0` `[synthetic-benchmark]` | the signature eliminates those false reproducers |
| stochastic minimal reproduction | `39/40` `[synthetic-benchmark]` | fixed-seed controlled-noise family |

## The headline honesty result

The signature-gated engine drops the false-reproducer rate from `0.45`
`[synthetic-benchmark]` to `0.0` `[synthetic-benchmark]` on the decoy family. The
difference is statistically significant — paired exact McNemar p = `0.0039`
`[synthetic-benchmark]`, and the Wilson 95% intervals (`false_reproducer_wilson_*`
in the results JSON) are disjoint. This is the evidence behind the claim that
matching a normalized failure signature targets the *same* failure rather than
*a* failure.

Numbers are reproducible for a fixed seed; metadata (date, host) is environmental
and not part of the metric comparison.
