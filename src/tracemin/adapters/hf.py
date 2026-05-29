"""HuggingFace Inference adapter: stateless re-prompt as the default replay engine.

Each trial renders the candidate atoms into a chat request and calls
``InferenceClient.chat_completion`` at ``temperature=0`` — so "writing a replay
function" is not required to try tracemin. A live HF endpoint is non-deterministic
even at temperature 0, which is exactly why the engine's flakiness double-check
exists.

The client is injectable so the request-builder and response-parser are unit-tested
against recorded responses with no network (live calls are opt-in via
``@pytest.mark.live``). ``HF_TOKEN`` is read from the environment only — never baked
into code or artifacts.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from tracemin.adapters._render import render_chat
from tracemin.atoms import Atom
from tracemin.replay import RawOutput, ReplayError
from tracemin.scrub import scrub

replay_capable = True


def parse_response(resp: Any) -> RawOutput:
    """Extract assistant text from a chat-completion response (object- or dict-shaped)."""
    try:
        choices = resp.choices if hasattr(resp, "choices") else resp["choices"]
        choice = choices[0]
        message = choice.message if hasattr(choice, "message") else choice["message"]
        if hasattr(message, "content"):
            content = message.content
        elif isinstance(message, dict):
            content = message.get("content", "")
        else:
            content = str(message)
    except (AttributeError, KeyError, IndexError, TypeError):
        content = str(resp)
    return RawOutput(text=content or "")


class HFReplay:
    """A callable ``replay_fn``: ``HFReplay(model)(subset) -> RawOutput``."""

    replay_capable = True

    def __init__(
        self,
        model: str,
        *,
        token: str | None = None,
        client: Any | None = None,
        temperature: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self._token = token
        self._client = client

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        from huggingface_hub import InferenceClient  # core dependency

        self._client = InferenceClient(
            model=self.model, token=self._token or os.environ.get("HF_TOKEN")
        )
        return self._client

    def __call__(self, subset: Sequence[Atom]) -> RawOutput:
        messages, tools = render_chat(subset)
        try:
            resp = self._get_client().chat_completion(
                messages=messages,
                model=self.model,
                tools=tools or None,
                temperature=self.temperature,
                top_p=1.0,
                seed=self.seed,
            )
        except Exception as exc:  # network / quota / 5xx -> transport error, not a verdict
            # Defense-in-depth: scrub the message in case the client embedded a token.
            raise ReplayError(scrub(f"HF inference failed: {type(exc).__name__}: {exc}")) from exc
        return parse_response(resp)


def has_live_token() -> bool:
    return bool(os.environ.get("HF_TOKEN"))
