"""
Main Planner component that creates hierarchical, adaptive plans from user intent.
"""

import logging
from typing import List, Optional
from pydantic import ValidationError
from .models import Plan, Phase, PlanStep, PlanStatus, PhaseStatus, StepStatus, RiskLevel
from .context import PlannerContext
from ..intent_manager.llm_interface import LLMProvider

logger = logging.getLogger(__name__)


class Planner:
    """
    Main Planner component that creates hierarchical, adaptive plans from user intent.

    The Planner decides HOW to accomplish the user's goal by creating validated plans
    with phases and steps, using LLM generation followed by deterministic validation.
    """

    def __init__(self, llm_provider: LLMProvider):
        """
        Initialize the Planner.

        Args:
            llm_provider: LLM provider to use for plan generation.
        """
        self.llm_provider = llm_provider
        logger.info("Planner initialized")

    async def create_plan(self, context: PlannerContext) -> Plan:
        """
        Create a plan based on the provided context.

        Args:
            context: PlannerContext containing intent and planning context

        Returns:
            Validated Plan instance

        Raises:
            Exception: If planning fails after all attempts
        """
        logger.debug(f"Creating plan in mode: {context.planning_mode}")

        if context.planning_mode == "INITIAL":
            return await self._create_initial_plan(context)
        elif context.planning_mode == "CONTINUE":
            return await self._continue_plan(context)
        elif context.planning_mode == "REPLAN":
            return await self._replan(context)
        else:
            raise ValueError(f"Invalid planning_mode: {context.planning_mode}")

    async def _create_initial_plan(self, context: PlannerContext) -> Plan:
        """Create an initial plan from scratch."""
        logger.debug("Creating initial plan")

        # Generate plan using LLM
        plan = await self._generate_plan_with_llm(context, is_revision=False)

        # Validate the generated plan
        validated_plan = await self._validate_and_retry_plan(plan, context, max_attempts=3)

        return validated_plan

    async def _continue_plan(self, context: PlannerContext) -> Plan:
        """Continue an existing active plan when it remains valid."""
        logger.debug("Continuing existing plan")

        # If there's no active plan, fall back to initial planning
        if not context.active_plan:
            logger.warning("No active plan to continue, falling back to initial planning")
            return await self._create_initial_plan(context)

        # TODO: Implement actual continuation logic based on task state
        # For now, we'll treat it as initial planning to keep implementation focused
        # In a full implementation, this would check if the active plan is still valid
        # and only replan if necessary
        logger.info("Continue planning not fully implemented, treating as initial planning")
        return await self._create_initial_plan(context)

    async def _replan(self, context: PlannerContext) -> Plan:
        """Generate a revised plan when observations or state invalidate the current approach."""
        logger.debug("Replanning based on new observations or state")

        # If there's no active plan, fall back to initial planning
        if not context.active_plan:
            logger.warning("No active plan to replan, falling back to initial planning")
            return await self._create_initial_plan(context)

        # Generate revised plan using LLM with context about what needs to change
        plan = await self._generate_plan_with_llm(context, is_revision=True)

        # Validate the generated plan
        validated_plan = await self._validate_and_retry_plan(plan, context, max_attempts=3)

        return validated_plan

    async def _generate_plan_with_llm(self, context: PlannerContext, is_revision: bool = False) -> Plan:
        """
        Generate a plan using the LLM provider.

        Args:
            context: PlannerContext for planning
            is_revision: Whether this is a revision of an existing plan

        Returns:
            Generated Plan (may need validation)
        """
        # Build prompt for LLM
        prompt = self._build_planning_prompt(context, is_revision)

        # Generate structured output from LLM
        plan = await self.llm_provider.generate_structured_output(
            prompt=prompt,
            response_model=Plan,
            max_retries=1  # We handle retries at the planner level
        )

        logger.debug(f"LLM generated plan: {plan.plan_id}")
        return plan

    async def _validate_and_retry_plan(self, plan: Plan, context: PlannerContext, max_attempts: int = 3) -> Plan:
        """
        Validate a plan and retry generation if validation fails.

        Args:
            plan: Initial plan to validate (generated by LLM)
            context: PlannerContext for retrying generation
            max_attempts: Maximum number of validation attempts

        Returns:
            Validated Plan

        Raises:
            Exception: If validation fails after all attempts
        """
        last_error = None
        current_plan = plan

        for attempt in range(max_attempts):
            try:
                logger.debug(f"Plan validation attempt {attempt + 1}/{max_attempts}")

                # Validate the plan (Pydantic validation happens in constructor)
                # Additional validation is done in the model validators
                # We'll also do deterministic validation here
                self._perform_deterministic_validation(current_plan, context)

                logger.debug(f"Plan validation successful on attempt {attempt + 1}")
                return current_plan

            except Exception as e:
                last_error = e
                logger.warning(f"Plan validation attempt {attempt + 1} failed: {str(e)}")

                # If this was our last attempt, break
                if attempt == max_attempts - 1:
                    break

                # Otherwise, generate a new plan for the next attempt
                logger.debug(f"Generating new plan for attempt {attempt + 2}")
                is_revision = context.planning_mode == "REPLAN"
                current_plan = await self._generate_plan_with_llm(context, is_revision=is_revision)

        # If we get here, all attempts failed
        logger.error(f"All plan validation attempts failed. Last error: {last_error}")

        # Return a structured planning failure
        return self._create_failure_plan(context, str(last_error))

    def _perform_deterministic_validation(self, plan: Plan, context: PlannerContext) -> None:
        """
        Perform deterministic validation on a plan.

        Args:
            plan: Plan to validate
            context: PlannerContext for validation

        Raises:
            ValidationError: If validation fails
        """
        # Validate that the plan goal aligns with the original intent goal
        if not self._goals_are_compatible(plan.goal, context.intent.goal):
            raise ValueError(f"Plan goal '{plan.goal}' is not compatible with intent goal '{context.intent.goal}'")

        # Validate that required capabilities are available
        self._validate_required_capabilities(plan, context.available_capabilities)

        # Additional validation is already handled by Pydantic model validators
        # (unique IDs, dependency validation, circular dependency detection, etc.)

    def _goals_are_compatible(self, plan_goal: str, intent_goal: Optional[str]) -> bool:
        """
        Check if plan goal is compatible with intent goal.

        Args:
            plan_goal: Goal from the generated plan
            intent_goal: Goal from the original intent

        Returns:
            True if goals are compatible, False otherwise
        """
        if not intent_goal:
            # If intent has no goal, accept any plan goal
            return True

        # Simple compatibility check - in a real implementation, this might be more sophisticated
        # For now, we'll check if the intent goal is contained in the plan goal (case-insensitive)
        return intent_goal.lower() in plan_goal.lower()

    def _validate_required_capabilities(self, plan: Plan, available_capabilities: List[str]) -> None:
        """
        Validate that all required capabilities in the plan are available.

        Args:
            plan: Plan to validate
            available_capabilities: List of capabilities that are available

        Raises:
            ValueError: If any required capability is not available
        """
        # Collect all required capabilities from all steps
        required_capabilities = set()
        for phase in plan.phases:
            for step in phase.steps:
                required_capabilities.update(step.required_capabilities)

        # Check if all required capabilities are available
        unavailable_capabilities = required_capabilities - set(available_capabilities)
        if unavailable_capabilities:
            raise ValueError(f"Required capabilities not available: {unavailable_capabilities}")

    def _create_failure_plan(self, context: PlannerContext, error_message: str) -> Plan:
        """
        Create a structured planning failure plan.

        Args:
            context: PlannerContext
            error_message: Error message from validation failures

        Returns:
            Plan indicating planning failure
        """
        logger.error(f"Creating failure plan due to: {error_message}")

        # Create a minimal plan that indicates planning failed
        failure_plan = Plan(
            plan_id=f"failed-plan-{context.task_id}",
            task_id=context.task_id,
            goal=context.intent.goal or "Planning failed",
            phases=[
                Phase(
                    phase_id="failure-phase",
                    name="Planning Failed",
                    objective="Indicates that planning failed after all attempts",
                    success_criteria=["Planning process acknowledged failure"],
                    status=PhaseStatus.FAILED,
                    steps=[
                        PlanStep(
                            step_id="failure-step",
                            description="Planning process failed",
                            objective="Acknowledge planning failure",
                            rationale="All planning attempts were exhausted",
                            required_capabilities=[],
                            dependencies=[],
                            expected_outcome="Failure acknowledged",
                            success_criteria=["Failure acknowledged"],
                            status=StepStatus.FAILED,
                            risk_level=RiskLevel.LOW
                        )
                    ]
                )
            ],
            current_phase="failure-phase",
            overall_success_criteria=["Failure acknowledged"],
            status=PlanStatus.FAILED,
            version=1,
            revision_reason=f"Planning failed after 3 attempts: {error_message}"
        )

        return failure_plan

    def _build_planning_prompt(self, context: PlannerContext, is_revision: bool = False) -> str:
        """
        Build a prompt for the LLM to generate a plan.

        Args:
            context: PlannerContext for planning
            is_revision: Whether this is a revision of an existing plan

        Returns:
            Formatted prompt string for LLM
        """
        # Define the output format description
        plan_format = """{
    "plan_id": "unique string identifier",
    "task_id": "string matching the context task_id",
    "goal": "string describing the high-level goal",
    "phases": [
        {
            "phase_id": "unique string identifier",
            "name": "human-readable phase name",
            "objective": "string describing phase objective",
            "success_criteria": ["list of strings"],
            "status": "one of: PENDING, READY, IN_PROGRESS, COMPLETED, FAILED, BLOCKED",
            "steps": [
                {
                    "step_id": "unique string identifier",
                    "description": "human-readable description",
                    "objective": "string describing step objective",
                    "rationale": "string explaining why this step is needed",
                    "required_capabilities": ["list of strings like 'filesystem.read'"],
                    "dependencies": ["list of step IDs"],
                    "expected_outcome": "string describing what should be accomplished",
                    "success_criteria": ["list of strings"],
                    "status": "one of: PLANNED, READY, BLOCKED, IN_PROGRESS, COMPLETED, FAILED, SKIPPED, CANCELLED",
                    "risk_level": "one of: LOW, MEDIUM, HIGH, CRITICAL"
                }
            ]
        }
    ],
    "current_phase": "string matching a phase_id or null",
    "overall_success_criteria": ["list of strings"],
    "status": "one of: PENDING, ACTIVE, PAUSED, COMPLETED, FAILED, CANCELLED",
    "version": 1,
    "revision_reason": "string explaining why this is a revision (if applicable) or null"
}"""

        # Build context information for the LLM
        intent_info = f"""
    Intent:
    - Primary intent: {context.intent.primary_intent}
    - Objective: {context.intent.objective or 'None'}
    - Goal: {context.intent.goal or 'None'}
    - Domain: {context.intent.domain or 'None'}
"""

        context_info = f"""
    Context:
    - Task ID: {context.task_id}
    - Available capabilities: {', '.join(context.available_capabilities) if context.available_capabilities else 'None'}
    - Constraints: {', '.join(context.constraints) if context.constraints else 'None'}
    - Recent observations count: {len(context.recent_observations)}
"""

        if is_revision and context.active_plan:
            revision_info = f"""
    Active Plan Info (for revision):
    - Plan ID: {context.active_plan.get('plan_id', 'Unknown')}
    - Current phase: {context.active_plan.get('current_phase', 'None')}
    - Plan status: {context.active_plan.get('status', 'Unknown')}
"""
        else:
            revision_info = ""

        prompt = f"""You are a Planning System. Your job is to analyze user intent and context to create a hierarchical, adaptive plan.

{intent_info}
{context_info}{revision_info}

TASK:
Create a structured plan that breaks down the goal into executable phases and steps.

PLANNING PRINCIPLES:
1. Break down the goal into logical phases
2. Each phase should contain executable steps
3. Steps should have clear dependencies when needed
4. Steps should specify required abstract capabilities (not concrete tools)
5. Include appropriate success criteria for measurement
6. Assess risk levels for each step
7. Ensure the plan preserves the original user goal
8. For revisions: preserve the original goal but adjust approach based on new information

AVAILABLE CAPABILITIES EXAMPLES (use these abstract names, not concrete implementations):
- filesystem.read
- filesystem.write
- filesystem.list
- terminal.execute
- git.inspect
- git.modify
- web.search
- browser.navigate
- data.process
- notify.user

OUTPUT FORMAT:
Return a JSON object that conforms exactly to this structure:
{plan_format}

RULES:
1. Preserve the user's original goal from the intent
2. Steps must have unique IDs across the entire plan
3. Phase IDs must be unique within the plan
4. Dependencies must reference existing step IDs only
5. No circular dependencies allowed
6. All required capabilities must be from the available capabilities list
7. If uncertain, create simpler, safer plans rather than complex risky ones
8. For revisions: explain why the plan is being revised in revision_reason
9. Do not guess or infer information that isn't reasonably inferable
10. Respond ONLY with the JSON object, no additional text."""

        return prompt