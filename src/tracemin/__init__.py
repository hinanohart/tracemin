"""tracemin — re-execution-verified, dependency-aware delta-debugging for failed LLM-agent runs.

Given a failed agent run and a user-supplied ``replay_fn``, tracemin shrinks the
trajectory's context (messages / tool definitions / retrieved files / instructions)
to a *1-minimal* subset by re-executing each candidate and keeping only those that
still reproduce the *same* failure. The result is a minimal reproducer — not a
root-cause explanation.

The package is import-light: ``import tracemin`` pulls in only the core dependency
(``huggingface_hub``). Statistical certification lives in the optional ``[stochastic]``
extra; the single-shot core never claims certification.
"""

from __future__ import annotations

__version__ = "0.1.0a1"

__all__ = ["__version__"]
