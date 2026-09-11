"""UI-independent conversation and approved-task services."""

import asyncio
import uuid
from typing import Dict, Optional

from agent.executor.executor import Executor
from agent.intent_manager.intent_manager import IntentManager
from agent.intent_manager.llm_interface import LLMProvider
from agent.planner.context import PlannerContext
from agent.planner.models import Plan, PlanStatus
from agent.planner.planner import Planner

from .models import ApprovedExecutionPlan, TaskRecord, TaskState, canonical_hash, utcnow
from .persistence import SQLiteStore
from .resolver import ActionResolver
from .state_machine import InvalidTaskTransition, TaskStateMachine


class ConversationService:
    def __init__(self, provider: LLMProvider):
        self.provider = provider

    async def reply(self, user_input: str) -> str:
        return await self.provider.generate_text(user_input)


class TaskService:
    """Single owner of task state; UI callers cannot invoke Executor directly."""
    def __init__(self, intent_manager: IntentManager, planner: Planner, resolver: ActionResolver,
                 executor: Executor, store: SQLiteStore, context_manager=None, approval_authorizer=None):
        self.intent_manager, self.planner, self.resolver = intent_manager, planner, resolver
        self.executor, self.store = executor, store
        self.context_manager = context_manager
        self.approval_authorizer = approval_authorizer
        self._plans: Dict[str, Plan] = {}
        self._approved: Dict[str, ApprovedExecutionPlan] = {}
        self._events: Dict[str, asyncio.Event] = {}
        self._locks: Dict[str, asyncio.Lock] = {}

    def _lock(self, task_id: str) -> asyncio.Lock:
        return self._locks.setdefault(task_id, asyncio.Lock())

    async def prepare(self, conversation_id: str, workspace_path: str, user_input: str) -> TaskRecord:
        task = TaskRecord(task_id=str(uuid.uuid4()), conversation_id=conversation_id, workspace_path=workspace_path,
                          planning_started_at=utcnow())
        self.store.save_task(task)
        event = self._events.setdefault(task.task_id, asyncio.Event())
        try:
            intent = await self.intent_manager.process_intent(user_input)
            if event.is_set():
                TaskStateMachine.transition(task, TaskState.CANCELLED)
                self.store.save_task(task); return task
            if self.context_manager is not None:
                await self.context_manager.create_context_bundle([], [], [], user_input, 4096)
            context = PlannerContext(intent=intent, task_id=task.task_id, task_state={}, active_plan=None,
                recent_observations=[], available_capabilities=["filesystem.read", "filesystem.write", "filesystem.list", "terminal.execute", "git.inspect"], constraints=[], relevant_context={}, planning_mode="INITIAL")
            plan = await self.planner.create_plan(context)
            if plan.status != PlanStatus.PENDING:
                TaskStateMachine.transition(task, TaskState.FAILED, reason=plan.revision_reason or "Planning failed")
                self.store.save_task(task); return task
            actions = await self.resolver.resolve(plan)
            approved = ApprovedExecutionPlan.create(task.task_id, plan, actions)
            self._plans[task.task_id], self._approved[task.task_id] = plan, approved
            task.plan_id, task.plan_version = approved.plan_id, approved.plan_version
            task.plan_hash, task.action_manifest_hash = approved.plan_hash, approved.action_manifest_hash
            self.store.save_plan(task.task_id, plan, approved.plan_hash)
            if event.is_set():
                TaskStateMachine.transition(task, TaskState.CANCELLED)
            else:
                TaskStateMachine.transition(task, TaskState.AWAITING_APPROVAL)
        except Exception as error:
            if task.state == TaskState.PLANNING:
                TaskStateMachine.transition(task, TaskState.FAILED, reason=str(error))
        self.store.save_task(task)
        return task

    async def approve_and_execute(self, task_id: str) -> TaskRecord:
        async with self._lock(task_id):
            task = self._require(task_id)
            if task.state != TaskState.AWAITING_APPROVAL:
                if task.state == TaskState.EXECUTING:
                    return task
                raise InvalidTaskTransition(f"Task cannot be approved from {task.state}.")
            plan, approved = self._plans.get(task_id), self._approved.get(task_id)
            if plan is None or approved is None or not approved.verify(plan):
                TaskStateMachine.transition(task, TaskState.FAILED, reason="Approved plan was changed or unavailable.")
                self.store.save_task(task); return task
            self.store.save_approval(task_id, approved.plan_hash, approved.action_manifest_hash)
            TaskStateMachine.transition(task, TaskState.EXECUTING)
            self.store.save_task(task)
        action_ids = {action.action_id for action in approved.actions}
        if self.approval_authorizer is not None:
            self.approval_authorizer.grant(action_ids)
        try:
            result = await self.executor.execute_approved_plan(approved, plan, self._events.setdefault(task_id, asyncio.Event()))
        finally:
            if self.approval_authorizer is not None:
                self.approval_authorizer.revoke(action_ids)
        async with self._lock(task_id):
            task = self._require(task_id)
            if task.state == TaskState.CANCELLED:
                return task
            TaskStateMachine.transition(task, TaskState.COMPLETED if result.status.value == "SUCCEEDED" else (TaskState.CANCELLED if result.status.value == "CANCELLED" else TaskState.FAILED), reason=result.error_message)
            self.store.save_task(task)
            return task

    async def reject(self, task_id: str) -> TaskRecord:
        async with self._lock(task_id):
            task = self._require(task_id)
            if task.state == TaskState.REJECTED: return task
            TaskStateMachine.transition(task, TaskState.REJECTED)
            self.store.save_task(task); return task

    async def cancel(self, task_id: str, reason: str = "Cancelled by user") -> TaskRecord:
        async with self._lock(task_id):
            task = self._require(task_id)
            if task.state == TaskState.CANCELLED: return task
            self._events.setdefault(task_id, asyncio.Event()).set()
            if task.state in (TaskState.PLANNING, TaskState.AWAITING_APPROVAL, TaskState.EXECUTING):
                TaskStateMachine.transition(task, TaskState.CANCELLED, reason=reason)
                self.store.save_task(task)
            else:
                raise InvalidTaskTransition(f"Task cannot be cancelled from {task.state}.")
            return task

    def _require(self, task_id: str) -> TaskRecord:
        task = self.store.get_task(task_id)
        if task is None: raise KeyError(task_id)
        return task
