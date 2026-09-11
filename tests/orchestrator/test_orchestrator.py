"""Tests for the Orchestrator component."""

import asyncio
from unittest.mock import AsyncMock, MagicMock
import pytest
from pydantic import ValidationError

from agent.orchestrator.orchestrator import Orchestrator
from agent.orchestrator.models import OrchestrationResult
from agent.intent_manager.models import Intent, IntentType, TaskObjective, Priority
from agent.planner.models import Plan, PlanStatus, Phase, PlanStep, StepStatus, PhaseStatus, RiskLevel
from agent.executor.models import ExecutionResult, ExecutionState, ToolResult


class MockIntentManager:
    """Mock IntentManager for testing."""

    def __init__(self, return_intent: Intent | None = None, should_require_clarification: bool = False):
        self.return_intent = return_intent
        self.should_require_clarification = should_require_clarification
        self.process_intent_called = False

    async def process_intent(self, user_input: str) -> Intent:
        self.process_intent_called = True
        if self.return_intent is None:
            # Default to a valid TASK intent
            return Intent(
                primary_intent=IntentType.TASK,
                objective=TaskObjective.CREATE,
                goal="Test goal",
                priority=Priority.NORMAL,
            )
        intent = self.return_intent
        if self.should_require_clarification:
            intent.requires_clarification = True
            intent.clarification_reason = "Test clarification needed"
        return intent


class MockPlanner:
    """Mock Planner for testing."""

    def __init__(self, return_plan: Plan | None = None, should_fail: bool = False):
        self.return_plan = return_plan
        self.should_fail = should_fail
        self.create_plan_called = False

    async def create_plan(self, context) -> Plan:
        self.create_plan_called = True
        if self.return_plan is None:
            # Default to a plan based on should_fail flag
            if self.should_fail:
                # Return a failed plan
                return Plan(
                    plan_id="failed-plan",
                    task_id="test-task",
                    goal="Test goal",
                    phases=[
                        Phase(
                            phase_id="failed-phase",
                            name="Failed Phase",
                            objective="Indicates failure",
                            success_criteria=["Failure acknowledged"],
                            status=PhaseStatus.FAILED,
                            steps=[
                                PlanStep(
                                    step_id="failed-step",
                                    description="Failed step",
                                    objective="Acknowledge failure",
                                    rationale="All planning attempts exhausted",
                                    required_capabilities=[],
                                    dependencies=[],
                                    expected_outcome="Failure acknowledged",
                                    success_criteria=["Failure acknowledged"],
                                    status=StepStatus.FAILED,
                                    risk_level=RiskLevel.LOW,
                                )
                            ],
                        )
                    ],
                    current_phase="failed-phase",
                    overall_success_criteria=["Failure acknowledged"],
                    status=PlanStatus.FAILED,
                    version=1,
                    revision_reason="Planning failed after 3 attempts: test error",
                )
            else:
                # Default to a valid plan
                return Plan(
                    plan_id="test-plan",
                    task_id=context.task_id,
                    goal=context.intent.goal or "Test goal",
                    phases=[
                        Phase(
                            phase_id="test-phase",
                            name="Test Phase",
                            objective="Test phase objective",
                            success_criteria=["Test criteria"],
                            status=PhaseStatus.PENDING,
                            steps=[
                                PlanStep(
                                    step_id="test-step",
                                    description="Test step description",
                                    objective="Test step objective",
                                    rationale="Test step rationale",
                                    required_capabilities=["filesystem.read"],
                                    dependencies=[],
                                    expected_outcome="Test expected outcome",
                                    success_criteria=["Test success criteria"],
                                    status=StepStatus.PLANNED,
                                    risk_level=RiskLevel.LOW,
                                )
                            ],
                        )
                    ],
                    current_phase="test-phase",
                    overall_success_criteria=["Test overall criteria"],
                    status=PlanStatus.PENDING,
                    version=1,
                    revision_reason=None,
                )
        return self.return_plan


class MockContextManager:
    """Mock ContextManager for testing."""

    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail
        self.create_context_bundle_called = False

    async def create_context_bundle(
        self,
        core_context,
        working_context,
        retrieved_context,
        query: str,
        token_budget: int,
    ):
        self.create_context_bundle_called = True
        if self.should_fail:
            raise Exception("ContextManager failed")
        # Return a dummy context bundle (we don't use it in the orchestrator V1)
        from agent.context_manager.models import ContextBundle, ContextItem, ContextPriority
        return ContextBundle(
            core=[],
            working=[],
            retrieved=[],
            summaries=[],
            dropped_items=[],
            token_estimate=0,
            budget_used=0.0,
        )


class MockExecutor:
    """Mock Executor for testing."""

    def __init__(self, return_result: ExecutionResult | None = None, should_fail: bool = False):
        self.return_result = return_result
        self.should_fail = should_fail
        self.execute_plan_called = False

    async def execute_plan(self, plan: Plan) -> ExecutionResult:
        self.execute_plan_called = True
        if self.should_fail:
            # Return a failed execution result
            return ExecutionResult(
                execution_id="test-exec-failed",
                plan_id=plan.plan_id,
                task_id=plan.task_id,
                status=ExecutionState.FAILED,
                started_at="2026-08-25T00:00:00Z",
                completed_at="2026-08-25T00:00:01Z",
                step_results={},
                failed_steps=["test-step"],
                cancelled_steps=[],
                error_message="Test execution error",
            )
        if self.return_result is None:
            # Default to a successful execution result
            return ExecutionResult(
                execution_id="test-exec-success",
                plan_id=plan.plan_id,
                task_id=plan.task_id,
                status=ExecutionState.SUCCEEDED,
                started_at="2026-08-25T00:00:00Z",
                completed_at="2026-08-25T00:00:01Z",
                step_results={
                    "test-step": ToolResult(
                        action_id="action_test-step",
                        step_id="test-step",
                        success=True,
                        output={"result": "success"}
                    )
                },
                failed_steps=[],
                cancelled_steps=[],
                error_message=None,
            )
        return self.return_result


@pytest.fixture
def default_intent():
    """Fixture providing a default intent."""
    return Intent(
        primary_intent=IntentType.TASK,
        objective=TaskObjective.CREATE,
        goal="Test goal",
        priority=Priority.NORMAL,
    )


@pytest.fixture
def default_plan():
    """Fixture providing a default plan."""
    return Plan(
        plan_id="test-plan",
        task_id="test-task",
        goal="Test goal",
        phases=[
            Phase(
                phase_id="test-phase",
                name="Test Phase",
                objective="Test phase objective",
                success_criteria=["Test criteria"],
                status=PhaseStatus.PENDING,
                steps=[
                    PlanStep(
                        step_id="test-step",
                        description="Test step description",
                        objective="Test step objective",
                        rationale="Test step rationale",
                        required_capabilities=["filesystem.read"],
                        dependencies=[],
                        expected_outcome="Test expected outcome",
                        success_criteria=["Test success criteria"],
                        status=StepStatus.PLANNED,
                        risk_level=RiskLevel.LOW,
                    )
                ],
            )
        ],
        current_phase="test-phase",
        overall_success_criteria=["Test overall criteria"],
        status=PlanStatus.PENDING,
        version=1,
        revision_reason=None,
    )


@pytest.fixture
def default_execution_result():
    """Fixture providing a default execution result."""
    return ExecutionResult(
        execution_id="test-exec-success",
        plan_id="test-plan",
        task_id="test-task",
        status=ExecutionState.SUCCEEDED,
        started_at="2026-08-25T00:00:00Z",
        completed_at="2026-08-25T00:00:01Z",
        step_results={
            "test-step": ToolResult(
                action_id="action_test-step",
                step_id="test-step",
                success=True,
                output={"result": "success"}
            )
        },
        failed_steps=[],
        cancelled_steps=[],
        error_message=None,
    )


class TestOrchestratorInitialization:
    """Tests for Orchestrator initialization."""

    def test_orchestrator_init_with_dependencies(self, default_intent, default_plan, default_execution_result):
        """Test initializing orchestrator with all dependencies."""
        intent_manager = MockIntentManager(return_intent=default_intent)
        planner = MockPlanner(return_plan=default_plan)
        context_manager = MockContextManager()
        executor = MockExecutor(return_result=default_execution_result)

        orchestrator = Orchestrator(
            intent_manager=intent_manager,
            planner=planner,
            context_manager=context_manager,
            executor=executor,
        )

        assert orchestrator.intent_manager == intent_manager
        assert orchestrator.planner == planner
        assert orchestrator.context_manager == context_manager
        assert orchestrator.executor == executor

    def test_orchestrator_has_available_capabilities(self):
        """Test that orchestrator has the expected available capabilities."""
        # Dependencies can be mocks
        intent_manager = MockIntentManager()
        planner = MockPlanner()
        context_manager = MockContextManager()
        executor = MockExecutor()

        orchestrator = Orchestrator(
            intent_manager=intent_manager,
            planner=planner,
            context_manager=context_manager,
            executor=executor,
        )

        assert hasattr(orchestrator, "AVAILABLE_CAPABILITIES")
        assert isinstance(orchestrator.AVAILABLE_CAPABILITIES, set)
        assert "filesystem.read" in orchestrator.AVAILABLE_CAPABILITIES
        assert "filesystem.write" in orchestrator.AVAILABLE_CAPABILITIES
        assert "filesystem.list" in orchestrator.AVAILABLE_CAPABILITIES
        assert "terminal.execute" in orchestrator.AVAILABLE_CAPABILITIES
        assert "git.inspect" in orchestrator.AVAILABLE_CAPABILITIES
        assert "git.modify" in orchestrator.AVAILABLE_CAPABILITIES


class TestOrchestratorRun:
    """Tests for the run method."""

    @pytest.mark.asyncio
    async def test_successful_end_to_end(
        self, default_intent, default_plan, default_execution_result
    ):
        """Test successful end-to-end orchestration."""
        intent_manager = MockIntentManager(return_intent=default_intent)
        planner = MockPlanner(return_plan=default_plan)
        context_manager = MockContextManager()
        executor = MockExecutor(return_result=default_execution_result)

        orchestrator = Orchestrator(
            intent_manager=intent_manager,
            planner=planner,
            context_manager=context_manager,
            executor=executor,
        )

        user_input = "Create a test file"
        result = await orchestrator.run(user_input)

        # Verify dependencies were called
        assert intent_manager.process_intent_called
        assert planner.create_plan_called
        assert executor.execute_plan_called

        # Verify result
        assert isinstance(result, OrchestrationResult)
        assert result.task_id is not None
        assert result.user_input == user_input
        assert result.intent == default_intent
        assert result.plan == default_plan
        assert result.execution_result == default_execution_result
        assert result.status == "completed"
        assert result.error is None

    @pytest.mark.asyncio
    async def test_intent_manager_returns_clarification(self, default_intent, default_plan, default_execution_result):
        """Test handling when Intent Manager returns clarification."""
        # Create an intent that requires clarification
        clarification_intent = Intent(
            primary_intent=IntentType.TASK,
            objective=TaskObjective.CREATE,
            goal="Test goal",
            priority=Priority.NORMAL,
            requires_clarification=True,
            clarification_reason="Ambiguous input",
        )
        intent_manager = MockIntentManager(
            return_intent=clarification_intent, should_require_clarification=True
        )
        planner = MockPlanner(return_plan=default_plan)
        context_manager = MockContextManager()
        executor = MockExecutor(return_result=default_execution_result)

        orchestrator = Orchestrator(
            intent_manager=intent_manager,
            planner=planner,
            context_manager=context_manager,
            executor=executor,
        )

        user_input = "Do something unclear"
        result = await orchestrator.run(user_input)

        # Verify dependencies were called
        assert intent_manager.process_intent_called
        # Planner and executor should NOT be called because we fail early
        assert not planner.create_plan_called
        assert not executor.execute_plan_called

        # Verify result
        assert result.intent == clarification_intent
        assert result.status == "failed_intent"
        assert result.error is not None
        assert "clarification" in result.error.lower()
        assert result.plan is None
        assert result.execution_result is None

    @pytest.mark.asyncio
    async def test_planner_fails(self, default_intent, default_execution_result):
        """Test handling when Planner fails (returns a failed plan)."""
        intent_manager = MockIntentManager(return_intent=default_intent)
        planner = MockPlanner(should_fail=True)  # This will return a failed plan
        context_manager = MockContextManager()
        executor = MockExecutor(return_result=default_execution_result)

        orchestrator = Orchestrator(
            intent_manager=intent_manager,
            planner=planner,
            context_manager=context_manager,
            executor=executor,
        )

        user_input = "Create a test file"
        result = await orchestrator.run(user_input)

        # Verify dependencies were called
        assert intent_manager.process_intent_called
        assert planner.create_plan_called
        # Executor should NOT be called because planning failed
        assert not executor.execute_plan_called

        # Verify result
        assert result.intent == default_intent
        assert result.plan is not None
        assert result.plan.status == PlanStatus.FAILED
        assert result.status == "failed_planning"
        assert result.error is not None
        assert "planning" in result.error.lower()
        assert result.execution_result is None

    @pytest.mark.asyncio
    async def test_executor_fails(self, default_intent, default_plan):
        """Test handling when Executor fails."""
        intent_manager = MockIntentManager(return_intent=default_intent)
        planner = MockPlanner(return_plan=default_plan)
        context_manager = MockContextManager()
        executor = MockExecutor(should_fail=True)  # This will return a failed execution

        orchestrator = Orchestrator(
            intent_manager=intent_manager,
            planner=planner,
            context_manager=context_manager,
            executor=executor,
        )

        user_input = "Create a test file"
        result = await orchestrator.run(user_input)

        # Verify dependencies were called
        assert intent_manager.process_intent_called
        assert planner.create_plan_called
        assert executor.execute_plan_called

        # Verify result
        assert result.intent == default_intent
        assert result.plan == default_plan
        result.execution_result is not None
        assert result.execution_result.status == ExecutionState.FAILED
        assert result.status == "failed_execution"
        assert result.error is not None
        assert "execution" in result.error.lower()

    @pytest.mark.asyncio
    async def test_unexpected_error_in_intent_manager(self, default_intent, default_plan, default_execution_result):
        """Test handling of unexpected error in Intent Manager."""
        intent_manager = MockIntentManager()
        # Make the process_intent method raise an exception
        async def failing_process_intent(user_input: str) -> Intent:
            intent_manager.process_intent_called = True
            raise ValueError("Test unexpected error")
        intent_manager.process_intent = failing_process_intent
        planner = MockPlanner(return_plan=default_plan)
        context_manager = MockContextManager()
        executor = MockExecutor(return_result=default_execution_result)

        orchestrator = Orchestrator(
            intent_manager=intent_manager,
            planner=planner,
            context_manager=context_manager,
            executor=executor,
        )

        user_input = "Test input"
        result = await orchestrator.run(user_input)

        # Verify dependencies were called
        assert intent_manager.process_intent_called
        # Planner and executor should NOT be called
        assert not planner.create_plan_called
        assert not executor.execute_plan_called

        # Verify result
        assert result.intent is None  # Because the exception was caught and intent remains None
        assert result.status == "failed"
        assert result.error is not None
        assert "Unexpected error" in result.error
        assert result.plan is None
        assert result.execution_result is None

    @pytest.mark.asyncio
    async def test_context_manager_fails_but_orchestrator_continues(self, default_intent, default_plan, default_execution_result):
        """Test that Orchestrator continues even if ContextManager fails."""
        intent_manager = MockIntentManager(return_intent=default_intent)
        planner = MockPlanner(return_plan=default_plan)
        context_manager = MockContextManager(should_fail=True)  # This will raise an exception
        executor = MockExecutor(return_result=default_execution_result)

        orchestrator = Orchestrator(
            intent_manager=intent_manager,
            planner=planner,
            context_manager=context_manager,
            executor=executor,
        )

        user_input = "Test input"
        result = await orchestrator.run(user_input)

        # Verify dependencies were called
        assert intent_manager.process_intent_called
        assert planner.create_plan_called
        assert executor.execute_plan_called
        assert context_manager.create_context_bundle_called

        # Verify result is successful (because we ignore ContextManager failure in V1)
        assert result.status == "completed"
        assert result.error is None

    @pytest.mark.asyncio
    async def test_orchestrator_respects_dependency_injection(self):
        """Test that orchestrator uses the provided dependencies and does not create its own."""
        # Create unique mock instances
        intent_manager = MockIntentManager()
        planner = MockPlanner()
        context_manager = MockContextManager()
        executor = MockExecutor()

        orchestrator = Orchestrator(
            intent_manager=intent_manager,
            planner=planner,
            context_manager=context_manager,
            executor=executor,
        )

        # Verify that the orchestrator's attributes are the exact instances we passed in
        assert orchestrator.intent_manager is intent_manager
        assert orchestrator.planner is planner
        assert orchestrator.context_manager is context_manager
        assert orchestrator.executor is executor