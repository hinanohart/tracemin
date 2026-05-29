"""``tracemin doctor`` — an honest environment report.

Every component is reported as LIVE / MOCK / MISSING (never a bare check mark):
  * LIVE    — present and fully functional.
  * MOCK    — present but degraded; results are illustrative, not live
              (e.g. the hf adapter with no ``HF_TOKEN``).
  * MISSING — not installed; the feature is unavailable.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass

from tracemin import __version__
from tracemin.scrub import active_mode


def _installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


@dataclass(frozen=True)
class DoctorRow:
    component: str
    status: str  # "LIVE" | "MOCK" | "MISSING"
    detail: str


@dataclass
class DoctorReport:
    rows: list[DoctorRow]

    def status_of(self, component: str) -> str | None:
        for row in self.rows:
            if row.component == component:
                return row.status
        return None

    def render(self) -> str:
        width = max((len(r.component) for r in self.rows), default=0)
        lines = ["tracemin doctor", ""]
        for r in self.rows:
            lines.append(f"  {r.component.ljust(width)}  [{r.status:7}]  {r.detail}")
        return "\n".join(lines)


def run_doctor() -> DoctorReport:
    rows = [DoctorRow("tracemin", "LIVE", f"v{__version__}")]

    hf_hub = _installed("huggingface_hub")
    token = bool(os.environ.get("HF_TOKEN"))
    if not hf_hub:
        rows.append(DoctorRow("adapter:hf", "MISSING", "huggingface_hub not importable"))
    elif token:
        rows.append(
            DoctorRow("adapter:hf", "LIVE", "HF_TOKEN present (format not connection-verified)")
        )
    else:
        rows.append(DoctorRow("adapter:hf", "MOCK", "HF_TOKEN absent — live replay unavailable"))
    rows.append(DoctorRow("adapter:openhands", "LIVE", "stdlib ingest (replay via hf)"))
    rows.append(DoctorRow("adapter:claude", "LIVE", "stdlib ingest (reduction-only)"))

    stochastic = _installed("numpy") and _installed("scipy")
    rows.append(
        DoctorRow(
            "extra:stochastic",
            "LIVE" if stochastic else "MISSING",
            "pass^k / Beta / interval separation"
            if stochastic
            else "pip install tracemin[stochastic]",
        )
    )
    sieve_live = active_mode() == "context-sieve"
    rows.append(
        DoctorRow(
            "extra:sieve",
            "LIVE" if sieve_live else "MISSING",
            "context-sieve scrubber"
            if sieve_live
            else "builtin scrubber active (still default-ON)",
        )
    )
    seedloop_live = _installed("seedloop")
    rows.append(
        DoctorRow(
            "extra:seedloop",
            "LIVE" if seedloop_live else "MISSING",
            "async-race byte replay" if seedloop_live else "pip install tracemin[seedloop]",
        )
    )
    rows.append(DoctorRow("scrub", "LIVE", f"mode={active_mode()} (default-ON)"))
    return DoctorReport(rows)
