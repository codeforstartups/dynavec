"""Pluggable LLM judge implementations (OpenAI, Bedrock, Gemini, Custom, Mock)."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from ..credentials import resolve_session
from ..exceptions import MissingDependencyError
from .base import BaseJudge


class OpenAIJudge(BaseJudge):
    """LLM judge using OpenAI's chat completions API (bring your own API key).

    Parameters
    ----------
    model:
        Model name, e.g. ``"gpt-4o-mini"`` or ``"gpt-4o"``.
    api_key:
        Optional; falls back to the ``OPENAI_API_KEY`` environment variable.
    base_url:
        Optional custom endpoint URL (e.g. for Ollama / vLLM / Azure).
    temperature:
        Sampling temperature (default 0.0 for deterministic evaluation).
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - import guard
            raise MissingDependencyError("OpenAIJudge", "openai", "openai") from exc

        self.model = model
        self.temperature = temperature
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def judge(self, prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""


class BedrockJudge(BaseJudge):
    """LLM judge using Amazon Bedrock foundation models (Claude 3, Titan, etc.).

    Parameters
    ----------
    model_id:
        Bedrock model ID, e.g. ``"anthropic.claude-3-haiku-20240307-v1:0"`` or
        ``"amazon.titan-text-express-v1"``.
    region:
        AWS region for the Bedrock endpoint.
    temperature:
        Sampling temperature (default 0.0).
    boto_session:
        Optional custom boto3 Session.
    """

    def __init__(
        self,
        model_id: str = "anthropic.claude-3-haiku-20240307-v1:0",
        region: str | None = None,
        temperature: float = 0.0,
        boto_session=None,
    ) -> None:
        session = resolve_session(None, boto_session)
        kwargs: dict[str, Any] = {}
        if region:
            kwargs["region_name"] = region
        self._client = session.client("bedrock-runtime", **kwargs)
        self.model_id = model_id
        self.temperature = temperature

    def judge(self, prompt: str) -> str:
        if "anthropic.claude" in self.model_id:
            payload = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 2048,
                "temperature": self.temperature,
                "messages": [{"role": "user", "content": prompt}],
            }
            resp = self._client.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(payload),
            )
            data = json.loads(resp["body"].read())
            contents = data.get("content", [])
            return contents[0].get("text", "") if contents else ""

        # Default / Titan payload format
        payload = {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": 2048,
                "temperature": self.temperature,
            },
        }
        resp = self._client.invoke_model(
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(payload),
        )
        data = json.loads(resp["body"].read())
        results = data.get("results", [])
        return results[0].get("outputText", "") if results else ""


class GeminiJudge(BaseJudge):
    """LLM judge using Google Gemini models (bring your own API key).

    Parameters
    ----------
    model:
        Model name, e.g. ``"gemini-1.5-flash"`` or ``"gemini-1.5-pro"``.
    api_key:
        Optional; falls back to ``GEMINI_API_KEY`` or ``GOOGLE_API_KEY``.
    temperature:
        Sampling temperature (default 0.0).
    """

    def __init__(
        self,
        model: str = "gemini-1.5-flash",
        api_key: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        try:
            import google.generativeai as genai
        except ImportError as exc:  # pragma: no cover - import guard
            raise MissingDependencyError("GeminiJudge", "google-generativeai", "gemini") from exc

        if api_key:
            genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(
            model_name=model,
            generation_config={"temperature": temperature},
        )

    def judge(self, prompt: str) -> str:
        resp = self._model.generate_content(prompt)
        return resp.text or ""


class CustomJudge(BaseJudge):
    """LLM judge wrapping any custom Python callable or function.

    Parameters
    ----------
    fn:
        Callable taking a string prompt and returning a string response.
    """

    def __init__(self, fn: Callable[[str], str]) -> None:
        if not callable(fn):
            raise TypeError("fn must be a callable taking a prompt string.")
        self._fn = fn

    def judge(self, prompt: str) -> str:
        return self._fn(prompt)


class MockJudge(BaseJudge):
    """Deterministic mock judge for unit testing and offline benchmarking.

    Parameters
    ----------
    responses:
        Optional list of sequential responses to return on consecutive calls.
    default_response:
        Default fallback response when responses list is empty or exhausted.
    """

    def __init__(
        self,
        responses: list[str | dict[str, Any]] | None = None,
        default_response: str | dict[str, Any] | None = None,
    ) -> None:
        self._responses: list[str] = [
            json.dumps(r) if isinstance(r, dict) else str(r) for r in (responses or [])
        ]
        self._default: str = (
            json.dumps(default_response)
            if isinstance(default_response, dict)
            else str(default_response or '{"score": 1.0, "reasoning": "Mock evaluation"}')
        )
        self.call_history: list[str] = []

    def judge(self, prompt: str) -> str:
        self.call_history.append(prompt)
        if self._responses:
            return self._responses.pop(0)
        return self._default
