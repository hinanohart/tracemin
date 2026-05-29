"""Adapters: ingest agent trajectories and (where possible) re-execute them.

* ``hf``        — stateless re-prompt at temperature 0 (the default replay engine).
* ``openhands`` — ingest OpenHands V0/V1 trajectories (replay via the hf engine).
* ``claude``    — ingest Claude Code JSONL transcripts (reduction-only by default).
"""

from __future__ import annotations
