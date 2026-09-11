"""
Data models for the Planner component.
"""

from enum import Enum
from typing import List, Optional, Set
from pydantic import BaseModel, Field, model_validator, ValidationInfo


class PlanStatus(str, Enum):
    """Overall plan status."""
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class PhaseStatus(str, Enum):
    """Status of a planning phase."""
    PENDING = "PENDING"
    READY = "READY"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"


class StepStatus(str, Enum):
    """Status of a plan step."""
    PLANNED = "PLANNED"
    READY = "READY"
    BLOCKED = "BLOCKED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    CANCELLED = "CANCELLED"


class RiskLevel(str, Enum):
    """Risk level for planning steps."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class PlanStep(BaseModel):
    """An executable step within a phase."""
    step_id: str = Field(description="Unique identifier for the step")
    description: str = Field(description="Human-readable description of what to do")
    objective: str = Field(description="Specific objective of this step")
    rationale: str = Field(description="Why this step is necessary")
    required_capabilities: List[str] = Field(
        default_factory=list,
        description="Abstract capabilities needed to execute this step (e.g., 'filesystem.read')"
    )
    dependencies: List[str] = Field(
        default_factory=list,
        description="List of step IDs that must be completed before this step can start"
    )
    expected_outcome: str = Field(description="What should be accomplished by this step")
    success_criteria: List[str] = Field(
        default_factory=list,
        description="Measurable criteria for determining step completion"
    )
    status: StepStatus = Field(default=StepStatus.PLANNED, description="Current execution status")
    risk_level: RiskLevel = Field(default=RiskLevel.LOW, description="Risk level associated with this step")

    @model_validator(mode='after')
    def validate_step(self) -> 'PlanStep':
        """Validate step after all fields are parsed."""
        # Ensure step_id is not empty
        if not self.step_id.strip():
            raise ValueError('step_id cannot be empty')

        # Ensure description is not empty
        if not self.description.strip():
            raise ValueError('description cannot be empty')

        return self


class Phase(BaseModel):
    """A phase containing related plan steps."""
    phase_id: str = Field(description="Unique identifier for the phase")
    name: str = Field(description="Human-readable name of the phase")
    objective: str = Field(description="Overall objective of this phase")
    success_criteria: List[str] = Field(
        default_factory=list,
        description="Criteria for determining phase completion"
    )
    status: PhaseStatus = Field(default=PhaseStatus.PENDING, description="Current phase status")
    steps: List[PlanStep] = Field(default_factory=list, description="Steps within this phase")

    @model_validator(mode='after')
    def validate_phase(self) -> 'Phase':
        """Validate phase after all fields are parsed."""
        # Ensure phase_id is not empty
        if not self.phase_id.strip():
            raise ValueError('phase_id cannot be empty')

        # Ensure name is not empty
        if not self.name.strip():
            raise ValueError('name cannot be empty')

        # Ensure objective is not empty
        if not self.objective.strip():
            raise ValueError('objective cannot be empty')

        # Validate step IDs are unique within phase
        step_ids = [step.step_id for step in self.steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError('Step IDs must be unique within a phase')

        return self


class Plan(BaseModel):
    """Hierarchical, adaptive plan for accomplishing a task."""
    plan_id: str = Field(description="Unique identifier for this plan")
    task_id: str = Field(description="Identifier of the task this plan belongs to")
    goal: str = Field(description="High-level goal to accomplish (from user intent)")
    phases: List[Phase] = Field(default_factory=list, description="Phases of the plan")
    current_phase: Optional[str] = Field(
        default=None,
        description="ID of the currently active phase"
    )
    overall_success_criteria: List[str] = Field(
        default_factory=list,
        description="Criteria for determining overall plan completion"
    )
    status: PlanStatus = Field(default=PlanStatus.PENDING, description="Overall plan status")
    version: int = Field(default=1, description="Version number of this plan")
    revision_reason: Optional[str] = Field(
        default=None,
        description="Reason for this plan revision (if applicable)"
    )

    @model_validator(mode='after')
    def validate_plan(self) -> 'Plan':
        """Validate plan after all fields are parsed."""
        # Ensure plan_id is not empty
        if not self.plan_id.strip():
            raise ValueError('plan_id cannot be empty')

        # Ensure task_id is not empty
        if not self.task_id.strip():
            raise ValueError('task_id cannot be empty')

        # Ensure goal is not empty
        if not self.goal.strip():
            raise ValueError('goal cannot be empty')

        # Validate phase IDs are unique
        phase_ids = [phase.phase_id for phase in self.phases]
        if len(phase_ids) != len(set(phase_ids)):
            raise ValueError('Phase IDs must be unique within a plan')

        # Validate that current_phase (if set) refers to an existing phase
        if self.current_phase and self.current_phase not in phase_ids:
            raise ValueError(f'current_phase "{self.current_phase}" does not match any phase ID')

        # Validate step IDs are unique across entire plan
        all_step_ids = []
        for phase in self.phases:
            all_step_ids.extend([step.step_id for step in phase.steps])
        if len(all_step_ids) != len(set(all_step_ids)):
            raise ValueError('Step IDs must be unique across the entire plan')

        # Validate dependencies reference existing steps
        for phase in self.phases:
            for step in phase.steps:
                for dep_id in step.dependencies:
                    # Check if dependency exists in any phase
                    dep_exists = any(
                        dep_id == s.step_id
                        for p in self.phases
                        for s in p.steps
                    )
                    if not dep_exists:
                        raise ValueError(
                            f'Step "{step.step_id}" depends on non-existent step "{dep_id}"'
                        )

        # Validate no circular dependencies
        self._validate_no_circular_dependencies()

        return self

    def _validate_no_circular_dependencies(self) -> None:
        """Validate that there are no circular dependencies in the plan."""
        # Build adjacency list of dependencies
        graph = {}
        for phase in self.phases:
            for step in phase.steps:
                graph[step.step_id] = set(step.dependencies)

        # Check for cycles using DFS
        visited = set()
        rec_stack = set()

        def has_cycle(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)

            for neighbor in graph.get(node, set()):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for node in graph:
            if node not in visited:
                if has_cycle(node):
                    raise ValueError('Circular dependency detected in plan steps')

    def get_step_by_id(self, step_id: str) -> Optional[PlanStep]:
        """Get a step by its ID."""
        for phase in self.phases:
            for step in phase.steps:
                if step.step_id == step_id:
                    return step
        return None

    def get_phase_by_id(self, phase_id: str) -> Optional[Phase]:
        """Get a phase by its ID."""
        for phase in self.phases:
            if phase.phase_id == phase_id:
                return phase
        return None