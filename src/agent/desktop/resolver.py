"""Validated conversion from approved intent steps to concrete tool actions."""

from abc import ABC, abstractmethod
from typing import List

from pydantic import BaseModel, Field

from agent.executor.models import ActionRequest
from agent.intent_manager.llm_interface import LLMProvider
from agent.planner.models import Plan
from agent.tool_system.registry import ToolRegistryImpl
from agent.tool_system.models import ToolMetadata

from .workspace import WorkspaceGuard, WorkspaceViolation


class ProposedAction(BaseModel):
    step_id: str
    tool_name: str
    tool_input: dict = Field(default_factory=dict)
    timeout_seconds: int | None = 30


class ProposedActionManifest(BaseModel):
    actions: List[ProposedAction]


class ActionResolver(ABC):
    @abstractmethod
    async def resolve(self, plan: Plan) -> List[ActionRequest]:
        pass


class LLMActionResolver(ActionResolver):
    """LLM-assisted resolver with deterministic registry/schema/boundary checks."""

    def __init__(self, provider: LLMProvider, registry: ToolRegistryImpl, workspace_guard: WorkspaceGuard):
        self.provider = provider
        self.registry = registry
        self.workspace_guard = workspace_guard

    async def resolve(self, plan: Plan) -> List[ActionRequest]:
        tools = await self.registry.list_tools()
        metadata = {tool.metadata.name: tool.metadata for tool in tools}
        manifest = await self.provider.generate_structured_output(
            self._prompt(plan, list(metadata.values())), ProposedActionManifest
        )
        actions: List[ActionRequest] = []
        known_steps = {step.step_id: step for phase in plan.phases for step in phase.steps}
        seen_steps: set[str] = set()
        for index, item in enumerate(manifest.actions):
            if item.step_id not in known_steps or item.step_id in seen_steps:
                raise ValueError("Action manifest must contain exactly one known action per plan step.")
            seen_steps.add(item.step_id)
            tool = metadata.get(item.tool_name)
            if tool is None:
                raise ValueError(f"Unregistered tool requested: {item.tool_name}")
            required = set(known_steps[item.step_id].required_capabilities)
            if not required.issubset(tool.capabilities):
                raise ValueError(f"Tool {item.tool_name} cannot satisfy required capabilities.")
            self._validate_arguments(item.tool_input, tool)
            action = ActionRequest(
                action_id=f"{plan.plan_id}:{item.step_id}:{index}", step_id=item.step_id,
                tool_name=item.tool_name, tool_input=item.tool_input, timeout_seconds=item.timeout_seconds,
            )
            from agent.tool_system.models import ToolRequest
            self.workspace_guard.validate(ToolRequest(request_id=action.action_id, tool_name=action.tool_name, tool_input=action.tool_input, timeout_seconds=action.timeout_seconds), tool)
            actions.append(action)
        if seen_steps != set(known_steps):
            raise ValueError("Action manifest omitted one or more approved plan steps.")
        return actions

    @staticmethod
    def _validate_arguments(data: dict, metadata: ToolMetadata) -> None:
        required = metadata.argument_schema.get("required", [])
        missing = [name for name in required if name not in data]
        if missing:
            raise ValueError(f"Invalid arguments for {metadata.name}: missing {', '.join(missing)}")

    @staticmethod
    def _prompt(plan: Plan, metadata: List[ToolMetadata]) -> str:
        return (
            "Convert every plan step to exactly one concrete action using only these tool contracts. "
            "Never create additional actions. Return JSON only.\n"
            f"PLAN={plan.model_dump_json()}\nTOOLS={[m.model_dump(mode='json') for m in metadata]}"
        )
