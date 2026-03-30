from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class OllamaLLMConfig:
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"  # required by OpenAI SDK but ignored by Ollama
    model: str = "llama3.1"
    timeout_s: float = 30.0


class OllamaLLMError(RuntimeError):
    pass


def _extract_json_object(text: str) -> Dict[str, Any]:
    """
    Best-effort extraction of a single JSON object from model output.
    """
    if not text:
        raise OllamaLLMError("Empty model response; expected JSON object.")

    # Fast path: already valid JSON
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # Common pattern: fenced code block
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise OllamaLLMError("Could not locate JSON object in model response.")

    candidate = text[start : end + 1]
    try:
        data = json.loads(candidate)
    except Exception as e:
        raise OllamaLLMError(f"Failed to parse JSON from model response: {e}") from e

    if not isinstance(data, dict):
        raise OllamaLLMError("Parsed JSON was not an object.")

    return data


class OllamaLLMClient:
    """
    Minimal client for Ollama's OpenAI-compatible Chat Completions API.

    Uses the OpenAI Python SDK with:
    - base_url: http://localhost:11434/v1
    - api_key: any non-empty string (ignored by Ollama)
    """

    def __init__(self, config: OllamaLLMConfig):
        self.config = config

        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise OllamaLLMError(
                "Missing dependency 'openai'. Install it with: "
                "python -m pip install -r requirements/requirements.txt"
            ) from e

        self._client = OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            timeout=self.config.timeout_s,
        )

    def chat_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        extra_messages: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        if extra_messages:
            messages.extend(extra_messages)
        messages.append({"role": "user", "content": user_prompt})

        try:
            resp = self._client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=temperature,
            )
        except Exception as e:
            raise OllamaLLMError(f"LLM call failed: {e}") from e

        content = None
        try:
            content = resp.choices[0].message.content
        except Exception as e:
            raise OllamaLLMError(f"Unexpected response shape from LLM: {e}") from e

        return _extract_json_object(content or "")

