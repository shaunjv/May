"""Browser-facing orchestration. All mutation occurs on one asyncio event loop."""
import asyncio
import json
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field
from agent.context_manager.context_manager import ContextManager
from agent.context_manager.models import ContextItem, ContextPriority
from agent.desktop.models import ApprovedExecutionPlan, canonical_hash, utcnow
from agent.executor.executor import Executor
from agent.executor.models import ActionRequest
from agent.executor.tool_system_adapter import ToolSystemToolExecutor, ToolSystemPermissionChecker
from agent.intent_manager.intent_manager import IntentManager
from agent.intent_manager.provider_errors import ProviderUnavailableError
from agent.planner.context import PlannerContext
from agent.planner.models import Plan
from agent.planner.planner import Planner
from agent.tool_system.manager import ToolManagerImpl
from agent.tool_system.registry import ToolRegistryImpl
from .store import redact
from .tools import Boundary, CONTRACTS, ExactPermission, WorkspaceTool, metadata


class ProposedAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    step_id: str
    tool_name: str
    tool_input: dict


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    actions: list[ProposedAction] = Field(min_length=1, max_length=20)


class Resolver:
    def __init__(self, provider):
        self.provider = provider

    async def resolve(self, plan, boundary):
        manifest = await self.provider.generate_structured_output(
            "Bind each plan step to exactly one concrete registered tool. No extra actions. "
            "Only use the path argument for file/directory names. File writes must contain complete final content. "
            "Do not assume contents of files you have not read. No secrets. "
            f"Workspace: {boundary.root}\nContracts: {json.dumps([metadata(n).model_dump(mode='json') for n in CONTRACTS])}\n"
            f"Plan: {plan.model_dump_json()}", Manifest,
        )
        steps = {s.step_id: s for p in plan.phases for s in p.steps}
        if len(steps) != len(manifest.actions) or {a.step_id for a in manifest.actions} != set(steps):
            raise ValueError("Resolver did not cover the exact plan.")
        actions = {}
        paths = set()
        for item in manifest.actions:
            args = boundary.arguments(item.tool_name, item.tool_input)
            if redact(json.dumps(args)) != json.dumps(args):
                raise ValueError("Credentials cannot be included in stored actions.")
            if not set(steps[item.step_id].required_capabilities).issubset({CONTRACTS[item.tool_name][1]}):
                raise ValueError("Action does not implement the plan capability.")
            # Multiple writes to one file have ambiguous preconditions; require a new plan.
            if item.tool_name == "file_write":
                if args["path"] in paths:
                    raise ValueError("Only one write per file is allowed in a plan.")
                paths.add(args["path"])
            actions[item.step_id] = ActionRequest(action_id=str(uuid4()), step_id=item.step_id, tool_name=item.tool_name, tool_input=args, timeout_seconds=30)
        ordered, done = [], set()
        while len(done) < len(steps):
            ready = [sid for sid, step in steps.items() if sid not in done and set(step.dependencies).issubset(done)]
            if not ready:
                raise ValueError("Plan dependencies cannot be resolved.")
            for sid in ready:
                ordered.append(actions[sid])
                done.add(sid)
        return ordered


class Counter:
    def count_tokens(self, text):
        return len(text) // 3 + 1


class ContextualIntentManager(IntentManager):
    """Classify a conversation envelope without applying raw-message regexes to it."""
    async def process_intent(self, user_input):
        return await self._classify_with_llm(user_input)


class Runtime:
    def __init__(self, store, provider, intent=None, planner=None, resolver=None, context=None):
        self.store, self.provider = store, provider
        self.intent = intent or (ContextualIntentManager(provider) if provider else None)
        self.planner = planner or (Planner(provider) if provider else None)
        self.resolver = resolver or Resolver(provider)
        self.context = context or ContextManager(token_counter=Counter())
        self.jobs, self.events = {}, {}
        self.store.recover()

    def submit(self, cid, text):
        task = self.store.begin(cid, text)
        self.events[task["id"]] = asyncio.Event()
        self.jobs[task["id"]] = asyncio.create_task(self._prepare(task))
        return task

    def update(self, task, **values):
        task.update(values)
        self.store.save(task)

    async def _prepare(self, task):
        try:
            async with asyncio.timeout(120):
                if self.provider is None:
                    raise RuntimeError("Provider unavailable")
                conversation = self.store.conversation(task["conversation_id"])
                history = [{"role": m["role"], "content": m["content"]} for m in conversation["messages"][-12:]]
                # IntentManager owns this decision. Its deterministic preprocessor
                # can now send ordinary questions straight to conversational chat.
                intent = await self.intent.process_intent(task["input"])
                if intent.requires_clarification:
                    reply = intent.clarification_reason
                elif intent.primary_intent.value not in {"TASK", "COMPUTER_ACTION", "AUTOMATION"}:
                    reply = await self.provider.generate_text(
                        "Respond to the latest message using this conversation. Actual web-runtime capabilities: "
                        "read text files, list folders, create/replace text files within the chosen workspace, ONLY after explicit plan approval. "
                        "No terminal execution, internet browsing, delete/move operations or Git tools in this web release. "
                        "For code examples, reply with properly indented Markdown code; do not claim to have saved or executed anything. "
                        "Never claim an action happened without a stored successful execution result.\n" + json.dumps(history)[-18000:]
                    )
                else:
                    reply = None
                if reply is not None:
                    self.store.message(task["conversation_id"], "assistant", reply)
                    self.update(task, state="COMPLETED", kind="chat", completed_at=utcnow(), stage="Ready")
                    return
                self.update(task, kind="task", stage="Preparing a plan", planning_started_at=utcnow())
                boundary = Boundary(conversation["workspace"])
                bundle = await self.context.create_context_bundle(
                    [ContextItem(content=str(boundary.root), priority=ContextPriority.MANDATORY, source="workspace")],
                    [ContextItem(content=json.dumps(history)[-12000:], priority=ContextPriority.HIGH, source="conversation")],
                    [], task["input"], 6000,
                )
                plan = await self.planner.create_plan(PlannerContext(
                    intent=intent, task_id=task["id"], planning_mode="INITIAL",
                    available_capabilities=[value[1] for value in CONTRACTS.values()],
                    constraints=["One bounded file or directory action per step. No unbounded fixes, shell, Git, or invented file contents.",
                                 "All concrete actions will be resolved and reviewed before any execution; results cannot dynamically expand a plan.",
                                 "Runtime context: " + json.dumps(bundle.get_included_content())],
                ))
                if plan.status.value == "FAILED" or not plan.phases:
                    raise ValueError("Planning failed")
                plan.task_id = task["id"]
                plan.plan_id = str(uuid4())
                if redact(plan.model_dump_json()) != plan.model_dump_json():
                    raise ValueError("Plan contains credentials")
                self.update(task, stage="Validating concrete actions")
                actions = await self.resolver.resolve(plan, boundary)
                package = ApprovedExecutionPlan.create(task["id"], plan, actions)
                preconditions = {a.tool_input["path"]: boundary.fingerprint(a.tool_input["path"]) for a in actions if a.tool_name == "file_write"}
                self.update(task, state="AWAITING_APPROVAL", stage="Review required", plan=plan.model_dump(mode="json"),
                            package=package.model_dump(mode="json"), preconditions=preconditions,
                            review_hash=canonical_hash({"package": package.model_dump(mode="json"), "preconditions": preconditions}),
                            planning_completed_at=utcnow(), approval_requested_at=utcnow())
        except asyncio.CancelledError:
            self.update(task, state="CANCELLED", completed_at=utcnow(), stage="Cancelled")
        except ProviderUnavailableError as error:
            self.update(task, state="FAILED", completed_at=utcnow(), stage="Provider unavailable", error=str(error))
        except Exception:
            self.update(task, state="FAILED", completed_at=utcnow(), stage="Stopped", error="The request could not be prepared safely. Check your provider connection/model configuration and ensure the request uses supported files and actions. You can send it again to retry.")

    def approve(self, tid, review_hash):
        task = self.store.task(tid)
        if task["state"] != "AWAITING_APPROVAL":
            raise ValueError("Only a plan awaiting approval can execute.")
        actual = canonical_hash({"package": task["package"], "preconditions": task["preconditions"]})
        if review_hash != task["review_hash"] or actual != review_hash:
            raise ValueError("The reviewed plan changed. A new plan and approval are required.")
        plan = Plan.model_validate(task["plan"])
        package = ApprovedExecutionPlan.model_validate(task["package"])
        if not package.verify(plan) or package.task_id != tid or plan.task_id != tid:
            raise ValueError("Plan identity does not match approval.")
        # No await between compare, state commit and job creation: duplicate requests cannot race.
        self.update(task, state="EXECUTING", stage="Executing approved actions", approved_at=utcnow(), execution_started_at=utcnow(), approved_hash=review_hash)
        self.events[tid] = asyncio.Event()
        self.jobs[tid] = asyncio.create_task(self._execute(task, plan, package))
        return task

    async def _execute(self, task, plan, package):
        event = self.events[task["id"]]
        try:
            boundary = Boundary(self.store.conversation(task["conversation_id"])["workspace"])
            for path, expected in task["preconditions"].items():
                if boundary.fingerprint(path) != expected:
                    raise ValueError("File changed since review")
            registry = ToolRegistryImpl()
            permissions = ExactPermission(package.actions, boundary, event)
            manager = ToolManagerImpl(registry, permissions)
            for name in CONTRACTS:
                await manager.register_tool(WorkspaceTool(metadata=metadata(name), boundary=boundary))
            adapter = ToolSystemToolExecutor(manager)
            store, tid = self.store, task["id"]

            class RecordingAdapter:
                async def execute(self, action):
                    if action.tool_name == "file_write" and boundary.fingerprint(action.tool_input["path"]) != task["preconditions"][action.tool_input["path"]]:
                        raise ValueError("File changed since review")
                    latest = store.task(tid)
                    latest["stage"] = f"Running {action.tool_name}"
                    store.save(latest)
                    result = await adapter.execute(action)
                    latest = store.task(tid)
                    latest.setdefault("action_results", []).append(json.loads(redact(result.model_dump_json())))
                    store.save(latest)
                    return result

            executor = Executor(RecordingAdapter(), ToolSystemPermissionChecker(manager))
            result = await executor.execute_approved_plan(package, plan, event)
            task = self.store.task(task["id"])
            state = "CANCELLED" if event.is_set() else ("COMPLETED" if result.status.value == "SUCCEEDED" else "FAILED")
            self.update(task, state=state, stage=state.title(), completed_at=utcnow(), execution=json.loads(redact(result.model_dump_json())))
            summary = f"{state.title()}: {len(result.step_results)} action(s) returned results."
            for item in result.step_results.values():
                if item.output is not None:
                    summary += "\n\n```text\n" + str(item.output)[:12000] + "\n```"
                if item.error:
                    summary += "\n\n" + item.error
            self.store.message(task["conversation_id"], "assistant", summary)
        except Exception:
            task = self.store.task(task["id"])
            self.update(task, state="CANCELLED" if event.is_set() else "FAILED", completed_at=utcnow(), error="Execution stopped. The workspace or file contents may have changed since review. Check action results before preparing a new plan.")

    def stop(self, tid, reject=False):
        task = self.store.task(tid)
        if task["state"] in {"CANCELLED", "REJECTED"}:
            return task
        if reject:
            if task["state"] != "AWAITING_APPROVAL":
                raise ValueError("Only a waiting plan can be rejected.")
            self.update(task, state="REJECTED", completed_at=utcnow())
        elif task["state"] == "EXECUTING":
            self.events[tid].set()
            self.update(task, cancel_requested=True, stage="Stopping after the current file operation")
        elif task["state"] == "PLANNING":
            self.jobs[tid].cancel()
            self.update(task, state="CANCELLED", completed_at=utcnow())
        elif task["state"] == "AWAITING_APPROVAL":
            self.update(task, state="CANCELLED", completed_at=utcnow())
        else:
            raise ValueError("This operation has already finished.")
        return task

    async def close(self):
        for tid, job in self.jobs.items():
            if not job.done():
                if self.store.task(tid)["state"] == "EXECUTING":
                    self.events[tid].set()
                else:
                    job.cancel()
        await asyncio.gather(*self.jobs.values(), return_exceptions=True)
        if self.provider and hasattr(self.provider, "_client"):
            await self.provider._client.close()
        self.store.close()
