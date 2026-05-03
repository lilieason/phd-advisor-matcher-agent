"""
Unified LLM client — Anthropic-compatible interface backed by litellm.

Supported providers and their recommended models:
  anthropic  → claude-haiku-4-5-20251001   (recommended, best value)
  openai     → gpt-4o-mini
  gemini     → gemini/gemini-1.5-flash

Usage (drop-in for anthropic.Anthropic):
  from mcp_servers.llm_client import make_client
  client = make_client(provider="anthropic", api_key="sk-ant-...")
  resp = client.messages.create(model=..., max_tokens=..., messages=[...], system=...)
  text = resp.content[0].text
"""

import os

PROVIDER_MODELS = {
    "anthropic": "claude-haiku-4-5-20251001",
    "openai":    "gpt-4o-mini",
    "gemini":    "gemini/gemini-1.5-flash",
}

PROVIDER_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai":    "OPENAI_API_KEY",
    "gemini":    "GEMINI_API_KEY",
}


class _Response:
    """Minimal Anthropic-compatible response wrapper."""
    def __init__(self, text: str):
        self.content = [type("Block", (), {"text": text})()]


class _Messages:
    def __init__(self, provider: str, api_key: str):
        self._provider = provider
        self._api_key  = api_key

    def create(self, model: str = "", max_tokens: int = 1024,
               messages: list = None, system: str = "", **kwargs) -> _Response:
        import litellm
        litellm.suppress_debug_info = True

        os.environ[PROVIDER_ENV[self._provider]] = self._api_key

        _model = model or PROVIDER_MODELS[self._provider]
        _msgs  = []
        if system:
            _msgs.append({"role": "system", "content": system})
        _msgs.extend(messages or [])

        resp = litellm.completion(
            model=_model,
            messages=_msgs,
            max_tokens=max_tokens,
            **{k: v for k, v in kwargs.items()
               if k not in ("stop_sequences",)},  # Anthropic-only params
        )
        return _Response(resp.choices[0].message.content or "")


class UnifiedClient:
    """Drop-in replacement for anthropic.Anthropic."""
    def __init__(self, provider: str, api_key: str):
        if provider not in PROVIDER_MODELS:
            raise ValueError(f"Unsupported provider '{provider}'. "
                             f"Choose from: {list(PROVIDER_MODELS)}")
        self.messages = _Messages(provider, api_key)


def make_client(provider: str = "anthropic", api_key: str = "") -> UnifiedClient:
    if not api_key:
        api_key = os.environ.get(PROVIDER_ENV.get(provider, "ANTHROPIC_API_KEY"), "")
    return UnifiedClient(provider, api_key)
