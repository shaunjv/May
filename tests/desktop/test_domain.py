import os
from pathlib import Path

import pytest

from agent.desktop.models import ApprovedExecutionPlan, TaskRecord, TaskState
from agent.desktop.persistence import SQLiteStore
from agent.desktop.state_machine import InvalidTaskTransition, TaskStateMachine
from agent.desktop.workspace import WorkspaceGuard, WorkspaceViolation
from agent.executor.models import ActionRequest
from agent.planner.models import Plan, Phase, PlanStep
from agent.tool_system.models import ToolMetadata, ToolRequest, OperationType, RiskLevel
from agent.tool_system.models import PermissionDecision
from agent.desktop.approval import ApprovedActionAuthorizer, ApprovedActionPermissionChecker


def plan(version=1):
    return Plan(plan_id="plan", task_id="task", goal="goal", version=version, phases=[Phase(phase_id="p", name="p", objective="o", steps=[PlanStep(step_id="s", description="d", objective="o", rationale="r", expected_outcome="e")])])


def test_state_machine_valid_and_terminal_idempotence():
    task = TaskRecord(task_id="t", conversation_id="c", workspace_path=".")
    TaskStateMachine.transition(task, TaskState.AWAITING_APPROVAL)
    TaskStateMachine.transition(task, TaskState.EXECUTING)
    TaskStateMachine.transition(task, TaskState.COMPLETED)
    assert TaskStateMachine.transition(task, TaskState.COMPLETED) is False
    with pytest.raises(InvalidTaskTransition):
        TaskStateMachine.transition(task, TaskState.EXECUTING)


def test_cancellation_and_approval_are_guarded():
    task = TaskRecord(task_id="t", conversation_id="c", workspace_path=".")
    TaskStateMachine.transition(task, TaskState.CANCELLED)
    assert TaskStateMachine.transition(task, TaskState.CANCELLED) is False
    with pytest.raises(InvalidTaskTransition):
        TaskStateMachine.transition(task, TaskState.EXECUTING)


def test_approved_manifest_rejects_plan_change():
    action = ActionRequest(action_id="a", step_id="s", tool_name="file_read", tool_input={"file_path": "a.txt"})
    approved = ApprovedExecutionPlan.create("task", plan(), [action])
    assert approved.verify(plan())
    assert not approved.verify(plan(version=2))


def test_sqlite_recovers_interrupted_but_preserves_approval(tmp_path):
    db = tmp_path / "agent.db"
    store = SQLiteStore(db)
    running = TaskRecord(task_id="run", conversation_id="c", workspace_path=str(tmp_path), state=TaskState.EXECUTING)
    waiting = TaskRecord(task_id="wait", conversation_id="c", workspace_path=str(tmp_path), state=TaskState.AWAITING_APPROVAL)
    store.save_task(running); store.save_task(waiting)
    recovered = SQLiteStore(db)
    assert recovered.get_task("run").state == TaskState.FAILED
    assert recovered.get_task("wait").state == TaskState.AWAITING_APPROVAL


def test_workspace_guard_blocks_traversal_and_accepts_nested(tmp_path):
    root = tmp_path / "workspace"; root.mkdir(); (root / "nested").mkdir()
    guard = WorkspaceGuard(root)
    assert guard.resolve_path("nested/file.txt") == (root / "nested" / "file.txt")
    with pytest.raises(WorkspaceViolation): guard.resolve_path("../outside.txt")
    metadata = ToolMetadata(name="file_read", description="r", operation_type=OperationType.READ, risk_level=RiskLevel.LOW, is_read_only=True, required_permission="READ", target_resource="file_path")
    with pytest.raises(WorkspaceViolation):
        guard.validate(ToolRequest(request_id="r", tool_name="file_read", tool_input={"file_path": str(tmp_path / "outside")}), metadata)


def test_workspace_guard_allows_only_bounded_terminal_commands(tmp_path):
    guard = WorkspaceGuard(tmp_path)
    terminal = ToolMetadata(name="terminal", description="t", operation_type=OperationType.EXECUTE, risk_level=RiskLevel.HIGH, is_read_only=False, required_permission="EXECUTE", target_resource="command")
    guard.validate(ToolRequest(request_id="1", tool_name="terminal", tool_input={"command": ["pytest", "-q"], "cwd": str(tmp_path)}), terminal)
    with pytest.raises(WorkspaceViolation):
        guard.validate(ToolRequest(request_id="2", tool_name="terminal", tool_input={"command": ["pytest", ";", "whoami"], "cwd": str(tmp_path)}), terminal)


@pytest.mark.asyncio
async def test_approval_grant_applies_only_to_exact_action():
    class RequiresApproval:
        async def check_permission(self, request): return PermissionDecision.REQUIRE_APPROVAL
    authorizer = ApprovedActionAuthorizer()
    checker = ApprovedActionPermissionChecker(RequiresApproval(), authorizer)
    authorizer.grant({"approved"})
    assert await checker.check_permission(ToolRequest(request_id="approved", tool_name="file_write")) == PermissionDecision.ALLOW
    assert await checker.check_permission(ToolRequest(request_id="other", tool_name="file_write")) == PermissionDecision.REQUIRE_APPROVAL
    authorizer.revoke({"approved"})
    assert await checker.check_permission(ToolRequest(request_id="approved", tool_name="file_write")) == PermissionDecision.REQUIRE_APPROVAL
