"""Tests for the NVIDIA LLM provider without making network requests."""

import json
from types import SimpleNamespace

import pytest

from agent.config import Settings
from agent.intent_manager.models import Intent, IntentType
from agent.intent_manager.nvidia_provider import NVIDIAProvider


class FakeCompletions:
    def __init__(self, output_text: str):
        self.output_text = output_text
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.output_text))]
        )


class FakeClient:
    def __init__(self, output_text: str):
        self.chat = SimpleNamespace(completions=FakeCompletions(output_text))


@pytest.mark.asyncio
async def test_provider_requests_json_and_validates_output():
    client = FakeClient(json.dumps({"primary_intent": "CONVERSATION"}))
    provider = NVIDIAProvider(api_key="test-key", model="test-model", client=client)

    result = await provider.generate_structured_output("hello", Intent)

    assert result == Intent(primary_intent=IntentType.CONVERSATION)
    request = client.chat.completions.calls[0]
    assert request["model"] == "test-model"
    assert request["temperature"] == 0
    assert request["max_tokens"] == 1200
    assert request["extra_body"] == {"reasoning_budget": 768}
    assert request["stream"] is False
    assert "JSON Schema" in request["messages"][0]["content"]


@pytest.mark.asyncio
async def test_provider_bounds_chat_latency_and_output():
    client = FakeClient("hello")
    provider = NVIDIAProvider(api_key="test-key", model="test-model", client=client)

    assert await provider.generate_text("hello") == "hello"

    request = client.chat.completions.calls[0]
    assert request["max_tokens"] == 700
    assert request["extra_body"] == {"reasoning_budget": 384}
    assert request["stream"] is False


def test_provider_requires_api_key_and_model_from_settings():
    missing_key = Settings(_env_file=None, nvidia_model="test-model")
    with pytest.raises(ValueError, match="NVIDIA_API_KEY"):
        NVIDIAProvider.from_settings(missing_key)

    missing_model = Settings(_env_file=None, nvidia_api_key="test-key")
    with pytest.raises(ValueError, match="NVIDIA_MODEL"):
        NVIDIAProvider.from_settings(missing_model)


@pytest.mark.asyncio
async def test_provider_rejects_invalid_structured_output():
    provider = NVIDIAProvider(
        api_key="test-key",
        model="test-model",
        client=FakeClient("not json"),
    )

    with pytest.raises(RuntimeError, match="does not match Intent"):
        await provider.generate_structured_output("hello", Intent)
