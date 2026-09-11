from unittest.mock import AsyncMock
import httpx
import pytest
from openai import PermissionDeniedError

from agent.intent_manager.nvidia_provider import NVIDIAProvider
from agent.intent_manager.provider_errors import ProviderUnavailableError
from agent.intent_manager.models import Intent
from agent.web.runtime import Runtime
from agent.web.store import Store


@pytest.mark.asyncio
@pytest.mark.parametrize("structured", [False, True])
async def test_nvidia_access_error_is_safe(structured):
    client = AsyncMock()
    client.chat.completions.create.side_effect = PermissionDeniedError(
        "PRIVATE_RESPONSE", response=httpx.Response(403, request=httpx.Request("POST", "https://example.test")), body=None)
    provider = NVIDIAProvider("test-key", "test-model", client=client)
    with pytest.raises(ProviderUnavailableError) as caught:
        if structured:
            await provider.generate_structured_output("hello", Intent)
        else:
            await provider.generate_text("hello")
    assert "403" in str(caught.value)
    assert "PRIVATE_RESPONSE" not in str(caught.value)
    assert client.chat.completions.create.await_count == 1


@pytest.mark.asyncio
async def test_access_failure_is_failed_not_chat_or_retried(tmp_path):
    provider = AsyncMock()
    provider.generate_structured_output.side_effect = ProviderUnavailableError(403)
    planner = AsyncMock()
    runtime = Runtime(Store(tmp_path / "db.sqlite"), provider, planner=planner)
    conversation = runtime.store.create(str(tmp_path))
    task = runtime.submit(conversation["id"], "Read a file")
    await runtime.jobs[task["id"]]
    task = runtime.store.task(task["id"])
    assert task["state"] == "FAILED"
    assert "403" in task["error"]
    assert provider.generate_structured_output.await_count == 1
    planner.create_plan.assert_not_called()
    assert len(runtime.store.conversation(conversation["id"])["messages"]) == 1
    runtime.store.close()
