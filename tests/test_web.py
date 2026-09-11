import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from agent.intent_manager.models import Intent, IntentType, TaskObjective
from agent.planner.models import Plan, Phase, PlanStep
from agent.web.app import create_app
from agent.web.runtime import Runtime, Resolver, Manifest, ProposedAction
from agent.web.store import Store
from agent.web.tools import Boundary, ExactPermission
from agent.tool_system.models import ToolRequest, PermissionDecision
from agent.executor.models import ActionRequest


def plan():
    return Plan(plan_id="p", task_id="t", goal="Create hello.txt", phases=[Phase(phase_id="phase", name="Create", objective="Create hello.txt", steps=[PlanStep(step_id="s", description="Create hello.txt containing hello", objective="Create file", rationale="Requested", required_capabilities=["filesystem.write"], expected_outcome="File exists")])])


def provider():
    return AsyncMock(generate_text=AsyncMock(return_value="Hello, from your agent."), generate_structured_output=AsyncMock(return_value=Manifest(actions=[ProposedAction(step_id="s", tool_name="file_write", tool_input={"path":"hello.txt", "content":"hello"})])))


def runtime(tmp_path, task=True):
    p = provider()
    return Runtime(Store(tmp_path / "state.db"), p,
                   intent=AsyncMock(process_intent=AsyncMock(return_value=Intent(primary_intent=IntentType.TASK if task else IntentType.CONVERSATION, objective=TaskObjective.CREATE if task else None, goal="Create hello.txt"))),
                   planner=AsyncMock(create_plan=AsyncMock(return_value=plan())))


async def prepared(rt, path):
    conversation = rt.store.create(str(path))
    task = rt.submit(conversation["id"], "Create hello.txt")
    await rt.jobs[task["id"]]
    return rt.store.task(task["id"])


@pytest.mark.asyncio
async def test_approved_real_file_flow(tmp_path):
    rt = runtime(tmp_path)
    task = await prepared(rt, tmp_path)
    assert task["state"] == "AWAITING_APPROVAL"
    assert not (tmp_path / "hello.txt").exists()
    rt.approve(task["id"], task["review_hash"])
    with pytest.raises(ValueError):
        rt.approve(task["id"], task["review_hash"])
    await rt.jobs[task["id"]]
    assert (tmp_path / "hello.txt").read_text() == "hello"
    result = rt.store.task(task["id"])
    assert result["state"] == "COMPLETED"
    assert len(result["action_results"]) == 1
    assert result["approved_hash"] == task["review_hash"]
    with pytest.raises(ValueError):
        rt.approve(task["id"], task["review_hash"])
    rt.store.close()


@pytest.mark.asyncio
async def test_chat_history_no_planning(tmp_path):
    rt = runtime(tmp_path, False)
    task = await prepared(rt, tmp_path)
    assert task["state"] == "COMPLETED"
    rt.planner.create_plan.assert_not_called()
    data = rt.store.conversation(task["conversation_id"])
    assert [m["role"] for m in data["messages"]] == ["user", "assistant"]
    assert "Create hello.txt" in rt.provider.generate_text.call_args.args[0]
    rt.store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["plan", "package", "hash"])
async def test_approval_tampering_rejected(tmp_path, mutation):
    rt = runtime(tmp_path)
    task = await prepared(rt, tmp_path)
    original = task["review_hash"]
    if mutation == "plan":
        task["plan"]["goal"] = "Changed"
    elif mutation == "package":
        task["package"]["actions"][0]["tool_input"]["content"] = "injected"
    else:
        original = "0" * 64
    rt.store.save(task)
    with pytest.raises(ValueError):
        rt.approve(task["id"], original)
    assert not (tmp_path / "hello.txt").exists()
    rt.store.close()


@pytest.mark.asyncio
async def test_changed_file_stops_execution(tmp_path):
    (tmp_path / "hello.txt").write_text("original")
    rt = runtime(tmp_path)
    task = await prepared(rt, tmp_path)
    (tmp_path / "hello.txt").write_text("external edit")
    rt.approve(task["id"], task["review_hash"])
    await rt.jobs[task["id"]]
    assert rt.store.task(task["id"])["state"] == "FAILED"
    assert (tmp_path / "hello.txt").read_text() == "external edit"
    rt.store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("reject", [False, True])
async def test_stop_waiting_idempotent(tmp_path, reject):
    rt = runtime(tmp_path)
    task = await prepared(rt, tmp_path)
    for _ in range(2):
        rt.stop(task["id"], reject=reject)
    assert rt.store.task(task["id"])["state"] == ("REJECTED" if reject else "CANCELLED")
    with pytest.raises(ValueError):
        rt.approve(task["id"], task["review_hash"])
    rt.store.close()


@pytest.mark.asyncio
async def test_cancel_planning(tmp_path):
    rt = runtime(tmp_path)
    async def wait(*args):
        await asyncio.Event().wait()
    rt.intent.process_intent.side_effect = wait
    cid = rt.store.create(str(tmp_path))["id"]
    task = rt.submit(cid, "hello")
    await asyncio.sleep(0)
    rt.stop(task["id"])
    await rt.jobs[task["id"]]
    assert rt.store.task(task["id"])["state"] == "CANCELLED"
    rt.planner.create_plan.assert_not_called()
    rt.store.close()


@pytest.mark.asyncio
async def test_cancel_before_execution_starts(tmp_path):
    rt = runtime(tmp_path)
    task = await prepared(rt, tmp_path)
    rt.approve(task["id"], task["review_hash"])
    rt.stop(task["id"])
    rt.stop(task["id"])
    await rt.jobs[task["id"]]
    assert rt.store.task(task["id"])["state"] == "CANCELLED"
    assert not (tmp_path / "hello.txt").exists()
    rt.store.close()


@pytest.mark.asyncio
async def test_restart_restores_approval_package(tmp_path):
    rt = runtime(tmp_path)
    task = await prepared(rt, tmp_path)
    rt.store.close()
    restored = runtime(tmp_path)
    restored.approve(task["id"], task["review_hash"])
    await restored.jobs[task["id"]]
    assert restored.store.task(task["id"])["state"] == "COMPLETED"
    assert len(restored.store.conversation(task["conversation_id"])["messages"]) == 2
    restored.store.close()


@pytest.mark.parametrize("state", ["PLANNING", "EXECUTING"])
def test_recovery_never_resumes(tmp_path, state):
    store = Store(tmp_path / "state.db")
    task = store.begin(store.create(str(tmp_path))["id"], "hello")
    task["state"] = state
    store.save(task)
    store.close()
    store = Store(tmp_path / "state.db")
    store.recover()
    assert store.task(task["id"])["state"] == "FAILED"
    assert "nothing was resumed" in store.task(task["id"])["error"]
    store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("component", ["intent", "planner", "resolver"])
async def test_component_errors_are_safe(tmp_path, component):
    rt = runtime(tmp_path)
    secret = "nvapi-" + "X" * 30
    if component == "intent":
        rt.intent.process_intent.side_effect = RuntimeError(secret)
    elif component == "planner":
        rt.planner.create_plan.side_effect = RuntimeError(secret)
    else:
        rt.resolver = AsyncMock(resolve=AsyncMock(side_effect=RuntimeError(secret)))
    task = await prepared(rt, tmp_path)
    assert task["state"] == "FAILED"
    assert secret not in json.dumps(task)
    assert not (tmp_path / "hello.txt").exists()
    rt.store.close()


@pytest.mark.parametrize("value", ["../escape", "nested/../../escape", "C:relative.txt", "Z:\\outside.txt", "\\\\server\\share\\file", ".env", ".git/config", "test.pem"])
def test_boundary_rejects_unsafe_paths(tmp_path, value):
    with pytest.raises(ValueError):
        Boundary(tmp_path).arguments("file_write", {"path":value, "content":"x"})


def test_boundary_valid_paths_and_root(tmp_path):
    (tmp_path / "nested").mkdir()
    guard = Boundary(tmp_path)
    assert guard.path("nested/new.txt") == tmp_path / "nested" / "new.txt"
    assert guard.path(str(tmp_path / "a.txt")) == tmp_path / "a.txt"
    assert guard.arguments("directory_list", {"path":"."})["path"] == str(tmp_path)
    with pytest.raises(ValueError):
        guard.arguments("file_write", {"path":".", "content":"x"})
    with pytest.raises(ValueError):
        guard.path(str(tmp_path.parent / (tmp_path.name + "sibling") / "file.txt"))


def test_symlink_escape(tmp_path):
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    outside.mkdir()
    link = tmp_path / "linked"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        # Windows junctions require no developer-mode symlink privilege.
        import os
        if os.name != "nt":
            raise
        import subprocess
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(outside)], check=True, capture_output=True)
    with pytest.raises(ValueError):
        Boundary(tmp_path).path("linked/escape.txt")


@pytest.mark.asyncio
@pytest.mark.parametrize("name,args", [("terminal", {"command":"whoami"}), ("file_write", {"path":"x", "content":3}), ("file_write", {"path":"x", "content":"y", "extra":1}), ("file_read", {"path":"missing"})])
async def test_resolver_rejects_invalid_actions(tmp_path, name, args):
    p = provider()
    p.generate_structured_output.return_value = Manifest(actions=[ProposedAction(step_id="s", tool_name=name, tool_input=args)])
    with pytest.raises(ValueError):
        await Resolver(p).resolve(plan(), Boundary(tmp_path))


@pytest.mark.asyncio
async def test_permission_binds_content_not_id(tmp_path):
    action = ActionRequest(action_id="a", step_id="s", tool_name="file_write", tool_input={"path":str(tmp_path / "x"), "content":"yes"})
    permission = ExactPermission([action], Boundary(tmp_path), asyncio.Event())
    request = ToolRequest(request_id="a", tool_name="file_write", tool_input=action.tool_input)
    assert await permission.check_permission(request) == PermissionDecision.ALLOW
    request.tool_input["content"] = "different"
    assert await permission.check_permission(request) == PermissionDecision.DENY


def test_api_security_and_persistence(tmp_path):
    rt = runtime(tmp_path, False)
    with TestClient(create_app(rt)) as client:
        h = {"X-Agent-Client":"local-web-v1"}
        assert client.post("/api/conversations", json={"workspace":str(tmp_path)}).status_code == 403
        assert client.post("/api/conversations", json={"workspace":str(tmp_path)}, headers={**h,"origin":"https://evil.test"}).status_code == 403
        assert client.get("/api/status", headers={"host":"evil.test"}).status_code == 400
        c = client.post("/api/conversations", json={"workspace":str(tmp_path)}, headers=h).json()
        assert client.get("/api/conversations/" + c["id"]).json()["workspace"] == str(tmp_path)
        assert client.post("/api/conversations/" + c["id"] + "/messages", json={"content":" "}, headers=h).status_code == 422
        assert client.get("/api/status").json()["provider_ready"]


def test_native_workspace_picker_returns_only_selected_folder(tmp_path, monkeypatch):
    import agent.web.app as web_app
    monkeypatch.setattr(web_app, "choose_workspace", lambda initial: str(tmp_path))
    rt = runtime(tmp_path, False)
    with TestClient(create_app(rt)) as client:
        headers = {"X-Agent-Client": "local-web-v1"}
        response = client.post("/api/workspace-picker", json={"initial_directory": str(tmp_path)}, headers=headers)
        assert response.status_code == 200
        assert response.json() == {"selected": str(tmp_path.resolve())}
    rt.store.close()


def test_secret_redaction(tmp_path):
    store = Store(tmp_path / "state.db")
    cid = store.create(str(tmp_path))["id"]
    secret = "nvapi-" + "q" * 25
    store.begin(cid, secret)
    store.message(cid, "assistant", "password=secretvalue")
    dumped = json.dumps(store.conversation(cid))
    assert secret not in dumped and "secretvalue" not in dumped
    store.close()
