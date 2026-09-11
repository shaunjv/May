"""NVIDIA-hosted implementation of the shared LLM provider contract."""

import json
from typing import Any, Type

from pydantic import BaseModel

from agent.config import Settings
from .llm_interface import LLMProvider
from .provider_errors import ProviderUnavailableError


class NVIDIAProvider(LLMProvider):
    """Generate Pydantic-validated JSON through NVIDIA's compatible API."""

    base_url = "https://integrate.api.nvidia.com/v1"

    def __init__(self, api_key: str, model: str, client: Any | None = None):
        if not api_key:
            raise ValueError("NVIDIA_API_KEY must be configured to use the NVIDIA provider.")
        if not model:
            raise ValueError("NVIDIA_MODEL must be configured to use the NVIDIA provider.")

        self.model = model
        self._client = client or self._create_client(api_key)

    @classmethod
    def from_settings(cls, settings: Settings, client: Any | None = None) -> "NVIDIAProvider":
        """Construct a provider from application settings with clear errors."""
        if settings.nvidia_api_key is None:
            raise ValueError("NVIDIA_API_KEY is required to run the agent CLI.")
        if not settings.nvidia_model:
            raise ValueError("NVIDIA_MODEL is required to run the agent CLI.")

        return cls(
            api_key=settings.nvidia_api_key.get_secret_value(),
            model=settings.nvidia_model,
            client=client,
        )

    @classmethod
    def _create_client(cls, api_key: str) -> Any:
        try:
            from openai import AsyncOpenAI
        except ImportError as error:
            raise RuntimeError(
                "The OpenAI-compatible SDK is not installed. Install project dependencies before running the CLI."
            ) from error

        return AsyncOpenAI(
            base_url=cls.base_url,
            api_key=api_key,
            timeout=30.0,
            max_retries=0,
        )

    async def generate_structured_output(
        self,
        prompt: str,
        response_model: Type[BaseModel],
        max_retries: int = 1,
    ) -> BaseModel:
        """Request JSON only, then validate it against the requested Pydantic model."""
        del max_retries  # Retry policy belongs to the calling component.

        schema = json.dumps(response_model.model_json_schema(), separators=(",", ":"))
        system_prompt = (
            "Return only one valid JSON object, with no markdown or explanation. "
            "The object must validate against this JSON Schema: "
            f"{schema}"
        )
        response = await self._completion(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=1200,
            extra_body={"reasoning_budget": 768},
            stream=False,
        )
        choices = getattr(response, "choices", None)
        output_text = (
            choices[0].message.content
            if isinstance(choices, list) and choices and hasattr(choices[0], "message")
            else None
        )
        if not isinstance(output_text, str) or not output_text.strip():
            raise RuntimeError("NVIDIA returned no structured output.")

        try:
            return response_model.model_validate_json(output_text)
        except Exception as error:
            raise RuntimeError(
                f"NVIDIA returned output that does not match {response_model.__name__}."
            ) from error

    async def generate_text(self, prompt: str) -> str:
        """Generate a plain response without invoking the planning/tool pipeline."""
        response = await self._completion(
            model=self.model,
            messages=[
                {"role": "system", "content": (
                    "You are Personal AI Agent, a local-first assistant. Be concise and helpful. "
                    "You can answer questions normally. For local workspace tasks, the application first creates "
                    "a bounded plan and requires user approval before any action; do not claim a task already ran. "
                    "Available guarded capabilities include reading, searching, listing, editing and writing files, "
                    "plus approved test commands and read-only Git inspection, all restricted to the chosen workspace."
                )},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            # NVIDIA otherwise defaults to 16K output and 16K internal-thinking
            # tokens, which is unsuitable for a responsive chat interface.
            max_tokens=700,
            extra_body={"reasoning_budget": 384},
            stream=False,
        )
        choices = getattr(response, "choices", None)
        text = choices[0].message.content if isinstance(choices, list) and choices else None
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("NVIDIA returned no conversational output.")
        return text

    async def _completion(self, **kwargs):
        from openai import APIError
        try:
            return await self._client.chat.completions.create(**kwargs)
        except (APIError, TimeoutError, ConnectionError) as error:
            # Never propagate response bodies, headers or credentials to logs/UI.
            raise ProviderUnavailableError(getattr(error, "status_code", None)) from None
