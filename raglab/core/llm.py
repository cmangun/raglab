"""Model access. One small protocol, an OpenAI-compatible client, and a scripted stand-in for tests."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Callable, Protocol

from .types import Usage


@dataclass
class LLMResponse:
    text: str
    usage: Usage = field(default_factory=Usage)


class LLM(Protocol):
    name: str

    def complete(self, system: str, user: str, *, images: list[bytes] | None = None, max_tokens: int = 800) -> LLMResponse: ...


class OpenAICompatLLM:
    """Chat completions against any OpenAI-compatible endpoint (OpenRouter by default)."""

    def __init__(self, api_key: str, model: str, base_url: str = "https://openrouter.ai/api/v1", temperature: float = 0.0):
        import httpx

        self.name = model
        self._model = model
        self._temperature = temperature
        self._client = httpx.Client(base_url=base_url, headers={"Authorization": f"Bearer {api_key}"}, timeout=120)

    def complete(self, system, user, *, images=None, max_tokens=800) -> LLMResponse:
        content: object = user
        if images:
            content = [{"type": "text", "text": user}] + [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(img).decode()}} for img in images
            ]
        r = self._client.post(
            "/chat/completions",
            json={
                "model": self._model,
                "temperature": self._temperature,
                "max_tokens": max_tokens,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
                "usage": {"include": True},
            },
        )
        r.raise_for_status()
        body = r.json()
        u = body.get("usage") or {}
        return LLMResponse(
            text=body["choices"][0]["message"]["content"] or "",
            usage=Usage(input_tokens=u.get("prompt_tokens", 0), output_tokens=u.get("completion_tokens", 0), cost_usd=float(u.get("cost") or 0.0)),
        )


class ScriptedLLM:
    """Returns whatever the given function returns. For tests and for demonstrating failure paths."""

    def __init__(self, fn: Callable[[str, str], str], name: str = "scripted"):
        self.name = name
        self._fn = fn
        self.calls: list[tuple[str, str]] = []

    def complete(self, system, user, *, images=None, max_tokens=800) -> LLMResponse:
        self.calls.append((system, user))
        return LLMResponse(text=self._fn(system, user), usage=Usage(input_tokens=len(user) // 4, output_tokens=20))
