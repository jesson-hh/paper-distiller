"""OpenAI-compatible Chat Completions client.

Works with any provider exposing /v1/chat/completions: Aliyun Bailian,
DeepSeek, OpenRouter, local Ollama, etc.
"""

from __future__ import annotations

import httpx


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 120.0,
    ):
        if not api_key:
            raise ValueError("api_key is required")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.total_tokens_in = 0
        self.total_tokens_out = 0
        self._client = httpx.Client(timeout=timeout)

    def complete(
        self,
        messages: list,
        temperature: float = 0.7,
        response_format: str | None = None,
    ) -> str:
        """Send messages to the LLM, return the assistant content string.

        response_format="json" requests strict JSON object output.
        """
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format == "json":
            body["response_format"] = {"type": "json_object"}

        try:
            r = self._client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=body,
            )
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise LLMError(f"LLM request failed: {e}") from e

        data = r.json()
        if "usage" in data:
            self.total_tokens_in += data["usage"].get("prompt_tokens", 0)
            self.total_tokens_out += data["usage"].get("completion_tokens", 0)
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise LLMError(f"unexpected LLM response shape: {data}") from e

    def __del__(self):
        try:
            self._client.close()
        except Exception:
            pass
