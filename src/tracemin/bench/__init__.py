"""Synthetic failure-injection benchmark.

The benchmark constructs trajectories whose ground-truth minimal set is known *by
construction* (not derived from any model), so recovery metrics are non-circular.
It is the single source of every number reported by tracemin.
"""

from __future__ import annotations
