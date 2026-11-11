"""Vision agent: build a provider by name (so the API / settings can say "gemini" without importing every SDK)."""
from __future__ import annotations

from carparking.vision_agent.base import BaseVisionAgent

PROVIDERS = ("gemini",)


def make_agent(provider: str = "gemini", **options) -> BaseVisionAgent:
    """`options` go to the provider's constructor (api_key, model, timeout_s, retries ...). Unknown names raise ValueError."""
    name = (provider or "").strip().lower()
    if name == "gemini":
        from carparking.vision_agent.gemini import GeminiVisionAgent  # imported here: the SDK is only needed when it is used
        return GeminiVisionAgent(**options)
    raise ValueError(f"unknown vision provider {provider!r}; available: {', '.join(PROVIDERS)}")
