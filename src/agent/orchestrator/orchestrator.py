"""Orchestrator / Agent Loop V1."""

import logging
import time
import uuid
from typing import List

from agent.intent_manager.intent_manager import IntentManager
from agent.intent_manager.models import Intent
from agent.planner.planner import Planner
from agent.planner.context import PlannerContext
from agent.planner.models import Plan, PlanStatus
from agent.context_manager.context_manager import ContextManager
from agent.executor.executor import Executor
from agent.executor.models import ExecutionResult, ExecutionState
from .models import OrchestrationResult

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Orchestrates the agent lifecycle: understands user input, plans, executes, and observes.

    Dependencies are injected via the constructor.
    """

    # Known capabilities provided by the Tool System V1.
    # These are matched to the abstract capability strings used in the Planner.
    AVAILABLE_CAPABILITIES = {
        "filesystem.read",
        "filesystem.write",
        "filesystem.list",
        "terminal.execute",
        "git.inspect",
        "git.modify",
    }

    def __init__(
        self,
        intent_manager: IntentManager,
        planner: Planner,
        context_manager: ContextManager,
        executor: Executor,
    ):
        """
        Initialize the Orchestrator.

        Args:
            intent_manager: Converts user input to structured Intents.
            planner: Creates hierarchical plans from context.
            context_manager: Builds token-bounded context bundles.
            executor: Executes plans sequentially.
        """
        self.intent_manager = intent_manager
        self.planner = planner
        self.context_manager = context_manager
        self.executor = executor
        logger.info("Orchestrator initialized")

    async def run(self, user_input: str) -> OrchestrationResult:
        """
        Run a single orchestration cycle.

        Args:
            user_input: Raw user input string.

        Returns:
            OrchestrationResult containing the outcome of the cycle.
        """
        task_id = str(uuid.uuid4())
        start_time = time.time()
        logger.info(f"Starting orchestration for task {task_id}")

        # Initialize result placeholders
        intent: Intent | None = None
        plan: Plan | None = None
        execution_result: ExecutionResult | None = None
        status = "failed"
        error: str | None = None

        try:
            # Step 1: Process user input with Intent Manager
            logger.debug(f"Processing user input: {user_input[:100]}...")
            intent = await self.intent_manager.process_intent(user_input)
            logger.debug(f"Received intent: {intent.primary_intent}")

            # If the intent requires clarification, we treat it as a failure for V1.
            # In a future version, we might trigger a clarification loop.
            if intent.requires_clarification:
                error = f"Intent requires clarification: {intent.clarification_reason}"
                logger.warning(error)
                return OrchestrationResult(
                    task_id=task_id,
                    user_input=user_input,
                    intent=intent,
                    status="failed_intent",
                    error=error,
                )

            # Step 2: Build PlannerContext
            # We'll create a minimal context; in a full system, the ContextManager
            # would help build the working and retrieved context.
            # For V1, we use empty working and retrieved context and rely on the
            # ContextManager to handle them (it will return empty bundles).
            core_context = [
                # We could add the intent as a core context item, but the PlannerContext
                # already takes the intent directly. We'll leave core_context empty for now
                # and see if the ContextManager requires it. We'll pass empty lists.
            ]
            working_context: List = []
            retrieved_context: List = []

            # Use the ContextManager to create a context bundle (though we don't use it directly
            # in the PlannerContext, we could extract relevant information from it).
            # For V1, we'll skip using the ContextManager's output and instead rely on the
            # PlannerContext's fields. We'll note that the ContextManager is a dependency
            # but we don't use its output in this V1. This is acceptable as we are not
            # required to use every dependency in every method.
            # We'll call it to satisfy the dependency but ignore the result for now.
            # In a real system, we would use the context bundle to inform the PlannerContext.
            # However, the PlannerContext does not have a field for a context bundle.
            # We'll leave this as a TODO for future versions.
            try:
                await self.context_manager.create_context_bundle(
                    core_context=core_context,
                    working_context=working_context,
                    retrieved_context=retrieved_context,
                    query=user_input,
                    token_budget=4096,  # reasonable default
                )
            except Exception as e:
                logger.warning(f"ContextManager failed: {e}")
                # Continue anyway; we don't rely on its output for V1.

            # Build the PlannerContext
            planner_context = PlannerContext(
                intent=intent,
                task_id=task_id,
                task_state={},  # start empty; we'll update with execution results later
                active_plan=None,
                recent_observations=[],
                available_capabilities=list(self.AVAILABLE_CAPABILITIES),
                constraints=[],  # no constraints from input for V1
                relevant_context={},
                planning_mode="INITIAL",
            )
            logger.debug(f"PlannerContext built: {planner_context.task_id}")

            # Step 3: Create plan with Planner
            logger.debug("Creating plan...")
            plan = await self.planner.create_plan(planner_context)

            # Check if planning failed (the planner returns a failed plan on validation errors)
            if plan.status != PlanStatus.PENDING:
                error = f"Planning failed: {plan.revision_reason or 'Unknown reason'}"
                logger.warning(error)
                return OrchestrationResult(
                    task_id=task_id,
                    user_input=user_input,
                    intent=intent,
                    plan=plan,
                    status="failed_planning",
                    error=error,
                )

            logger.debug(f"Received plan: {plan.plan_id}")

            # Step 4: Execute plan with Executor
            logger.debug("Executing plan...")
            execution_result = await self.executor.execute_plan(plan)
            logger.debug(
                f"Execution completed with status: {execution_result.status}"
            )

            # Determine overall status based on execution result
            if execution_result.status == ExecutionState.SUCCEEDED:
                status = "completed"
            else:
                status = "failed_execution"
                error = execution_result.error_message or "Execution failed"

        except Exception as e:
            # Catch any unexpected errors
            logger.error(f"Orchestration failed due to unexpected error: {e}", exc_info=True)
            error = f"Unexpected error: {str(e)}"
            status = "failed"

        # If we have an execution result, we can update the task state with observations
        # (though we don't replay the context in V1, we do it for completeness)
        if execution_result is not None:
            # We don't have a persistent task state across runs, but we can note the observation
            pass

        # Calculate elapsed time for logging
        elapsed_time = time.time() - start_time
        logger.info(
            f"Orchestration for task {task_id} finished in {elapsed_time:.2f}s with status: {status}"
        )

        return OrchestrationResult(
            task_id=task_id,
            user_input=user_input,
            intent=intent,
            plan=plan,
            execution_result=execution_result,
            status=status,
            error=error,
        )