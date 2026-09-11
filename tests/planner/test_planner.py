"""
Tests for the Planner component.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
from agent.planner.planner import Planner
from agent.planner.models import Plan, Phase, PlanStep, PlanStatus, PhaseStatus, StepStatus, RiskLevel
from agent.planner.context import PlannerContext
from agent.intent_manager.models import Intent, IntentType, TaskObjective, Priority
from agent.intent_manager.llm_interface import LLMProvider


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing."""

    def __init__(self, responses=None):
        self.responses = responses or {}
        self.call_count = 0

    async def generate_structured_output(self, prompt, response_model, max_retries=1):
        self.call_count += 1

        # Handle callable responses (function that takes prompt and returns response data)
        if callable(self.responses):
            response_data = self.responses(prompt)
            # Validate that the response data matches the expected model
            try:
                return response_model(**response_data)
            except Exception as e:
                raise Exception(f"Failed to validate mock response: {e}")

        # Handle dictionary responses (mapping triggers to response data)
        for trigger, response_data in self.responses.items():
            if trigger in prompt:
                # Validate that the response data matches the expected model
                try:
                    return response_model(**response_data)
                except Exception as e:
                    raise Exception(f"Failed to validate mock response: {e}")

        # Return a default valid plan if no specific response
        return response_model(
            plan_id="test-plan-id",
            task_id="test-task-id",
            goal="Test goal",
            phases=[
                Phase(
                    phase_id="test-phase-1",
                    name="Test Phase",
                    objective="Test phase objective",
                    success_criteria=["Test criteria"],
                    status=PhaseStatus.PENDING,
                    steps=[
                        PlanStep(
                            step_id="test-step-1",
                            description="Test step description",
                            objective="Test step objective",
                            rationale="Test step rationale",
                            required_capabilities=["test.capability"],
                            dependencies=[],
                            expected_outcome="Test expected outcome",
                            success_criteria=["Test success criteria"],
                            status=StepStatus.PLANNED,
                            risk_level=RiskLevel.LOW
                        )
                    ]
                )
            ],
            current_phase="test-phase-1",
            overall_success_criteria=["Test overall criteria"],
            status=PlanStatus.PENDING,
            version=1,
            revision_reason=None
        )


@pytest.fixture
def mock_llm_provider():
    """Fixture providing a mock LLM provider."""
    return MockLLMProvider()


@pytest.fixture
def planner(mock_llm_provider):
    """Fixture providing a Planner instance."""
    return Planner(llm_provider=mock_llm_provider)


@pytest.fixture
def sample_intent():
    """Fixture providing a sample intent."""
    return Intent(
        primary_intent=IntentType.TASK,
        objective=TaskObjective.CREATE,
        goal="Create a simple web application",
        domain="web development",
        constraints=["Must be responsive"],
        success_criteria=["Application loads in browser"],
        priority=Priority.NORMAL
    )


@pytest.fixture
def sample_context(sample_intent):
    """Fixture providing a sample planner context."""
    return PlannerContext(
        intent=sample_intent,
        task_id="test-task-123",
        task_state={},
        active_plan=None,
        recent_observations=[],
        available_capabilities=["filesystem.read", "filesystem.write", "terminal.execute"],
        constraints=["Must be responsive"],
        relevant_context={},
        planning_mode="INITIAL"
    )


class TestPlannerInitialization:
    """Tests for Planner initialization."""

    def test_planner_init_with_provider(self, mock_llm_provider):
        """Test initializing planner with custom LLM provider."""
        planner = Planner(llm_provider=mock_llm_provider)
        assert planner.llm_provider == mock_llm_provider

    def test_planner_init_without_provider(self):
        """Test initializing planner without LLM provider (uses mock)."""
        # This test needs to be updated since Planner now requires an explicit provider
        # We'll skip this test as the requirement is to depend only on LLMProvider abstraction
        # and tests should provide their own mock
        pass  # Test skipped - Planner now requires explicit LLMProvider dependency


class TestPlannerCreatePlan:
    """Tests for the create_plan method."""

    @pytest.mark.asyncio
    async def test_create_plan_initial_mode(self, planner, sample_context):
        """Test creating a plan in INITIAL mode."""
        # Set up mock to return a plan compatible with the intent goal
        # Match on the goal substring that will be in the prompt
        planner.llm_provider.responses = {
            "Goal: Create a simple web application": {
                "plan_id": "test-plan-id",
                "task_id": "test-task-123",
                "goal": "Create a simple web application",
                "phases": [
                    {
                        "phase_id": "test-phase-1",
                        "name": "Test Phase",
                        "objective": "Test phase objective",
                        "success_criteria": ["Test criteria"],
                        "status": "PENDING",
                        "steps": [
                            {
                                "step_id": "test-step-1",
                                "description": "Test step description",
                                "objective": "Test step objective",
                                "rationale": "Test step rationale",
                                "required_capabilities": ["filesystem.read"],  # Use available capability
                                "dependencies": [],
                                "expected_outcome": "Test expected outcome",
                                "success_criteria": ["Test success criteria"],
                                "status": "PLANNED",
                                "risk_level": "LOW"
                            }
                        ]
                    }
                ],
                "current_phase": "test-phase-1",
                "overall_success_criteria": ["Test overall criteria"],
                "status": "PENDING",
                "version": 1,
                "revision_reason": None
            }
        }

        plan = await planner.create_plan(sample_context)

        assert isinstance(plan, Plan)
        assert plan.task_id == sample_context.task_id
        assert plan.goal == sample_context.intent.goal
        assert plan.status == PlanStatus.PENDING
        assert len(plan.phases) > 0
        assert planner.llm_provider.call_count > 0

    @pytest.mark.asyncio
    async def test_create_plan_continue_mode_no_active_plan(self, planner, sample_context):
        """Test continuing plan when no active plan exists."""
        sample_context.planning_mode = "CONTINUE"
        sample_context.active_plan = None

        plan = await planner.create_plan(sample_context)

        assert isinstance(plan, Plan)
        assert plan.task_id == sample_context.task_id
        # Should fall back to initial planning
        assert planner.llm_provider.call_count > 0

    @pytest.mark.asyncio
    async def test_create_plan_replan_mode_no_active_plan(self, planner, sample_context):
        """Test replanning when no active plan exists."""
        sample_context.planning_mode = "REPLAN"
        sample_context.active_plan = None

        plan = await planner.create_plan(sample_context)

        assert isinstance(plan, Plan)
        assert plan.task_id == sample_context.task_id
        # Should fall back to initial planning
        assert planner.llm_provider.call_count > 0

    @pytest.mark.asyncio
    async def test_create_plan_invalid_mode(self, planner, sample_context):
        """Test creating plan with invalid planning mode."""
        sample_context.planning_mode = "INVALID_MODE"

        with pytest.raises(ValueError, match="Invalid planning_mode"):
            await planner.create_plan(sample_context)


class TestPlannerValidation:
    """Tests for planner validation logic."""

    @pytest.mark.asyncio
    async def test_goal_compatibility_validation(self, mock_llm_provider):
        """Test that plan goal must be compatible with intent goal."""
        # Set up mock to return a plan with incompatible goal
        mock_llm_provider.responses = {
            "test": {
                "plan_id": "test-plan-id",
                "task_id": "test-task-id",
                "goal": "Completely different goal",  # Incompatible with intent goal
                "phases": [
                    {
                        "phase_id": "test-phase-1",
                        "name": "Test Phase",
                        "objective": "Test phase objective",
                        "success_criteria": ["Test criteria"],
                        "status": "PENDING",
                        "steps": [
                            {
                                "step_id": "test-step-1",
                                "description": "Test step description",
                                "objective": "Test step objective",
                                "rationale": "Test step rationale",
                                "required_capabilities": ["test.capability"],
                                "dependencies": [],
                                "expected_outcome": "Test expected outcome",
                                "success_criteria": ["Test success criteria"],
                                "status": "PLANNED",
                                "risk_level": "LOW"
                            }
                        ]
                    }
                ],
                "current_phase": "test-phase-1",
                "overall_success_criteria": ["Test overall criteria"],
                "status": "PENDING",
                "version": 1,
                "revision_reason": None
            }
        }

        planner = Planner(llm_provider=mock_llm_provider)
        intent = Intent(
            primary_intent=IntentType.TASK,
            objective=TaskObjective.CREATE,
            goal="Create a web application",  # Different from plan goal
            domain="web"
        )
        context = PlannerContext(
            intent=intent,
            task_id="test-task-id",
            task_state={},
            active_plan=None,
            recent_observations=[],
            available_capabilities=["test.capability"],
            constraints=[],
            relevant_context={},
            planning_mode="INITIAL"
        )

        # Should fail validation and create failure plan after retries
        plan = await planner.create_plan(context)
        assert plan.status == PlanStatus.FAILED  # Failure plan
        assert plan.goal == intent.goal  # Original goal should be preserved

    @pytest.mark.asyncio
    async def test_capability_validation(self, mock_llm_provider):
        """Test that required capabilities must be available."""
        # Set up mock to return a plan requiring unavailable capability
        mock_llm_provider.responses = {
            "test": {
                "plan_id": "test-plan-id",
                "task_id": "test-task-id",
                "goal": "Test goal",
                "phases": [
                    {
                        "phase_id": "test-phase-1",
                        "name": "Test Phase",
                        "objective": "Test phase objective",
                        "success_criteria": ["Test criteria"],
                        "status": "PENDING",
                        "steps": [
                            {
                                "step_id": "test-step-1",
                                "description": "Test step description",
                                "objective": "Test step objective",
                                "rationale": "Test step rationale",
                                "required_capabilities": ["unavailable.capability"],  # Not in available capabilities
                                "dependencies": [],
                                "expected_outcome": "Test expected outcome",
                                "success_criteria": ["Test success criteria"],
                                "status": "PLANNED",
                                "risk_level": "LOW"
                            }
                        ]
                    }
                ],
                "current_phase": "test-phase-1",
                "overall_success_criteria": ["Test overall criteria"],
                "status": "PENDING",
                "version": 1,
                "revision_reason": None
            }
        }

        planner = Planner(llm_provider=mock_llm_provider)
        intent = Intent(
            primary_intent=IntentType.TASK,
            objective=TaskObjective.CREATE,
            goal="Test goal"
        )
        context = PlannerContext(
            intent=intent,
            task_id="test-task-id",
            task_state={},
            active_plan=None,
            recent_observations=[],
            available_capabilities=["filesystem.read"],  # Does NOT include unavailable.capability
            constraints=[],
            relevant_context={},
            planning_mode="INITIAL"
        )

        # Should fail validation and create failure plan after retries
        plan = await planner.create_plan(context)
        assert plan.status == PlanStatus.FAILED  # Failure plan
        assert plan.goal == intent.goal  # Original goal should be preserved


class TestPlannerRetryMechanism:
    """Tests for the planner retry mechanism."""

    @pytest.mark.asyncio
    async def test_retry_on_validation_failure_then_success(self, mock_llm_provider):
        """Test that planner retries on validation failure and succeeds."""
        attempt_count = 0

        def failing_then_succeeding_response(prompt):
            nonlocal attempt_count
            attempt_count += 1

            if attempt_count == 1:
                # First attempt returns plan with validation error (incompatible goal)
                return {
                    "plan_id": "test-plan-id",
                    "task_id": "test-task-id",
                    "goal": "Different goal",  # Incompatible with "Create a web application"
                    "phases": [
                        {
                            "phase_id": "test-phase-1",
                            "name": "Test Phase",
                            "objective": "Test phase objective",
                            "success_criteria": ["Test criteria"],
                            "status": PhaseStatus.PENDING,
                            "steps": [
                                {
                                    "step_id": "test-step-1",
                                    "description": "Test step description",
                                    "objective": "Test step objective",
                                    "rationale": "Test step rationale",
                                    "required_capabilities": ["test.capability"],
                                    "dependencies": [],
                                    "expected_outcome": "Test expected outcome",
                                    "success_criteria": ["Test success criteria"],
                                    "status": StepStatus.PLANNED,
                                    "risk_level": RiskLevel.LOW
                                }
                            ]
                        }
                    ],
                    "current_phase": "test-phase-1",
                    "overall_success_criteria": ["Test overall criteria"],
                    "status": PlanStatus.PENDING,
                    "version": 1,
                    "revision_reason": None
                }
            else:
                # Second attempt returns valid plan
                return {
                    "plan_id": "test-plan-id",
                    "task_id": "test-task-id",
                    "goal": "Create a web application",  # Compatible
                    "phases": [
                        {
                            "phase_id": "test-phase-1",
                            "name": "Test Phase",
                            "objective": "Test phase objective",
                            "success_criteria": ["Test criteria"],
                            "status": PhaseStatus.PENDING,
                            "steps": [
                                {
                                    "step_id": "test-step-1",
                                    "description": "Test step description",
                                    "objective": "Test step objective",
                                    "rationale": "Test step rationale",
                                    "required_capabilities": ["test.capability"],
                                    "dependencies": [],
                                    "expected_outcome": "Test expected outcome",
                                    "success_criteria": ["Test success criteria"],
                                    "status": StepStatus.PLANNED,
                                    "risk_level": RiskLevel.LOW
                                }
                            ]
                        }
                    ],
                    "current_phase": "test-phase-1",
                    "overall_success_criteria": ["Test overall criteria"],
                    "status": PlanStatus.PENDING,
                    "version": 1,
                    "revision_reason": None
                }

        # Set the mock provider to use the callable response
        mock_llm_provider.responses = failing_then_succeeding_response
        planner = Planner(llm_provider=mock_llm_provider)
        intent = Intent(
            primary_intent=IntentType.TASK,
            objective=TaskObjective.CREATE,
            goal="Create a web application"
        )
        context = PlannerContext(
            intent=intent,
            task_id="test-task-id",
            task_state={},
            active_plan=None,
            recent_observations=[],
            available_capabilities=["test.capability"],
            constraints=[],
            relevant_context={},
            planning_mode="INITIAL"
        )

        plan = await planner.create_plan(context)

        # Should have succeeded on second attempt
        assert attempt_count == 2
        assert plan.goal == "Create a web application"
        assert plan.status == PlanStatus.PENDING

    @pytest.mark.asyncio
    async def test_failure_after_max_attempts(self, mock_llm_provider):
        """Test that planner returns failure plan after max attempts."""
        def always_failing_response(prompt):
            # Always return plan with validation error (incompatible goal with intent)
            return {
                "plan_id": "test-plan-id",
                "task_id": "test-task-id",
                "goal": "Different goal",  # Will fail goal compatibility validation
                "phases": [
                    {
                        "phase_id": "test-phase-1",
                        "name": "Test Phase",
                        "objective": "Test phase objective",
                        "success_criteria": ["Test criteria"],
                        "status": "PENDING",
                        "steps": [
                            {
                                "step_id": "test-step-1",
                                "description": "Test step description",
                                "objective": "Test step objective",
                                "rationale": "Test step rationale",
                                "required_capabilities": ["test.capability"],
                                "dependencies": [],
                                "expected_outcome": "Test expected outcome",
                                "success_criteria": ["Test success criteria"],
                                "status": "PLANNED",
                                "risk_level": "LOW"
                            }
                        ]
                    }
                ],
                "current_phase": "test-phase-1",
                "overall_success_criteria": ["Test overall criteria"],
                "status": "PENDING",
                "version": 1,
                "revision_reason": None
            }

        # Set the mock provider to use the callable response
        mock_llm_provider.responses = always_failing_response
        planner = Planner(llm_provider=mock_llm_provider)
        intent = Intent(
            primary_intent=IntentType.TASK,
            objective=TaskObjective.CREATE,
            goal="Test goal"
        )
        context = PlannerContext(
            intent=intent,
            task_id="test-task-id",
            task_state={},
            active_plan=None,
            recent_observations=[],
            available_capabilities=["test.capability"],
            constraints=[],
            relevant_context={},
            planning_mode="INITIAL"
        )

        plan = await planner.create_plan(context)

        # Should have created failure plan after 3 attempts
        assert plan.status == PlanStatus.FAILED
        assert plan.goal == intent.goal  # Original goal should be preserved
        assert planner.llm_provider.call_count == 3  # Should have tried exactly 3 times


if __name__ == "__main__":
    pytest.main([__file__, "-v"])