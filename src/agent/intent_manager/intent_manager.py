"""
Intent Manager - Converts raw user input into validated, structured Intents.
"""

import logging
from typing import Optional
from .models import Intent, IntentType
from .llm_interface import LLMProvider, MockLLMProvider
from .preprocessor import DeterministicPreprocessor
from .provider_errors import ProviderUnavailableError

logger = logging.getLogger(__name__)


class IntentManager:
    """
    Main Intent Manager class that processes user input into structured Intents.

    Uses a two-step approach:
    1. Deterministic preprocessor for high-confidence, straightforward cases
    2. LLM classifier for complex or uncertain cases
    """

    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        """
        Initialize the Intent Manager.

        Args:
            llm_provider: LLM provider to use for classification.
                         If None, a mock provider will be used (useful for testing).
        """
        self.preprocessor = DeterministicPreprocessor()
        self.llm_provider = llm_provider or MockLLMProvider()
        logger.info("Intent Manager initialized")

    async def process_intent(self, user_input: str) -> Intent:
        """
        Process raw user input into a structured Intent.

        Args:
            user_input: Raw user input string

        Returns:
            Validated Intent instance
        """
        # Handle empty or invalid input
        if not user_input or not isinstance(user_input, str):
            logger.warning("Received empty or invalid input")
            return Intent(
                primary_intent=IntentType.CLARIFICATION,
                requires_clarification=True,
                clarification_reason="Empty or invalid input provided"
            )

        user_input = user_input.strip()
        if not user_input:
            logger.warning("Received empty input after stripping")
            return Intent(
                primary_intent=IntentType.CLARIFICATION,
                requires_clarification=True,
                clarification_reason="Empty input provided"
            )

        # Step 1: Try deterministic preprocessor first
        logger.debug(f"Processing input with deterministic preprocessor: {user_input[:50]}...")
        deterministic_result = self.preprocessor.process(user_input)

        if deterministic_result is not None:
            logger.debug("Deterministic preprocessor produced confident result")
            return deterministic_result

        # Step 2: Fall back to LLM classifier
        logger.debug("Routing to LLM classifier for uncertain input")
        return await self._classify_with_llm(user_input)

    async def _classify_with_llm(self, user_input: str) -> Intent:
        """
        Classify user input using the LLM provider.

        Implements retry logic for validation failures.

        Args:
            user_input: Raw user input string

        Returns:
            Validated Intent instance
        """
        # Construct prompt for LLM
        prompt = self._build_classification_prompt(user_input)

        max_attempts = 3
        last_error = None

        for attempt in range(max_attempts):
            try:
                logger.debug(f"LLM classification attempt {attempt + 1}/{max_attempts}")

                # Generate structured output from LLM
                intent = await self.llm_provider.generate_structured_output(
                    prompt=prompt,
                    response_model=Intent,
                    max_retries=1  # We handle retries at this level
                )

                # Validate the intent (Pydantic validation happens in generate_structured_output)
                # Additional validation can be added here if needed
                logger.debug(f"LLM classification successful on attempt {attempt + 1}")
                return intent

            except ProviderUnavailableError:
                # Authorization/transport errors cannot be repaired by classifying again.
                raise
            except Exception as e:
                last_error = e
                logger.warning(f"LLM classification attempt {attempt + 1} failed: {str(e)}")

                # If this was our last attempt, break
                if attempt == max_attempts - 1:
                    break

                # Otherwise, continue to next attempt (the LLM provider might adjust based on error)
                continue

        # If we get here, all attempts failed
        logger.error(f"All LLM classification attempts failed. Last error: {last_error}")

        # Fail safely - return a clarification intent
        return Intent(
            primary_intent=IntentType.CLARIFICATION,
            requires_clarification=True,
            clarification_reason=f"Unable to classify intent after {max_attempts} attempts: {str(last_error)}"
        )

    def _build_classification_prompt(self, user_input: str) -> str:
        """
        Build a prompt for the LLM to classify user intent.

        Args:
            user_input: Raw user input string

        Returns:
            Formatted prompt string for LLM
        """
        # Define the intent types and their descriptions
        intent_descriptions = {
            "CONVERSATION": "Casual conversation, greetings, acknowledgements, or social interaction",
            "INFORMATION": "Requests for facts, explanations, descriptions, or knowledge",
            "TASK": "Requests to perform some work or activity that requires effort",
            "AUTOMATION": "Requests to set up automated actions, reminders, schedules, or recurring tasks",
            "COMPUTER_ACTION": "Direct commands to control the computer or software applications",
            "CLARIFICATION": "Input that is unclear, ambiguous, or needs more information to understand"
        }

        # Define task objectives
        task_objectives = {
            "CREATE": "Make something new",
            "MODIFY": "Change or update something existing",
            "FIX": "Repair, correct, or resolve something broken",
            "ANALYZE": "Examine, investigate, or study something in detail",
            "RESEARCH": "Investigate or gather information about a topic",
            "EXECUTE": "Carry out or perform an action"
        }

        # Define priority levels
        priority_levels = {
            "LOW": "Not urgent, can be done at leisure",
            "NORMAL": "Standard priority",
            "HIGH": "Urgent, should be done soon",
            "CRITICAL": "Extremely urgent, needs immediate attention"
        }

        prompt = f"""You are an Intent Classification System. Your job is to analyze user input and convert it into a structured intent representation.

USER INPUT:
"{user_input}"

TASK:
Classify the user's intent into one of the primary intent types and extract relevant information.

PRIMARY INTENT TYPES:
{chr(10).join([f"- {intent}: {desc}" for intent, desc in intent_descriptions.items()])}

TASK OBJECTIVES (only applies when primary_intent is TASK):
{chr(10).join([f"- {obj}: {desc}" for obj, desc in task_objectives.items()])}

PRIORITY LEVELS:
{chr(10).join([f"- {level}: {desc}" for level, desc in priority_levels.items()])}

OUTPUT FORMAT:
Return a JSON object that conforms to this structure:
{{
    "primary_intent": "One of: CONVERSATION, INFORMATION, TASK, AUTOMATION, COMPUTER_ACTION, CLARIFICATION",
    "sub_intents": ["Array of additional intent types from the primary intent list"],
    "domain": "Optional string indicating domain/topic (e.g., 'code', 'email', 'calendar')",
    "objective": "One of: CREATE, MODIFY, FIX, ANALYZE, RESEARCH, EXECUTE (required if primary_intent is TASK)",
    "goal": "Optional string describing the high-level goal",
    "constraints": ["Array of constraint strings mentioned in the input"],
    "success_criteria": ["Array of strings describing how to know when the intent is fulfilled"],
    "deadline": "Optional string with explicit time/deadline (e.g., 'today', 'by 5 PM', 'tomorrow')",
    "priority": "One of: LOW, NORMAL, HIGH, CRITICAL (defaults to NORMAL)",
    "requires_clarification": "Boolean indicating if clarification is needed",
    "clarification_reason": "String explaining why clarification is needed (required if requires_clarification is true)"
}}

RULES:
1. If the input is clearly a greeting, acknowledgement, or casual conversation, use CONVERSATION
2. If the input is asking for information, facts, or explanations, use INFORMATION
3. If the input is requesting work to be done, use TASK
4. If the input is about setting up automation, reminders, or schedules, consider adding AUTOMATION as sub-intent
5. If the input is direct computer/software control commands, consider COMPUTER_ACTION
6. If unsure or ambiguous, use CLARIFICATION and set requires_clarification to true
7. For TASK intent, you MUST specify an objective
8. Extract explicit priority signals (ASAP, urgent, low priority, etc.)
9. Extract explicit deadline expressions (today, tomorrow, by 5 PM, etc.)
10. Do not guess or infer information that isn't clearly present
11. If validation fails, the system will retry with feedback

Respond ONLY with the JSON object, no additional text."""

        return prompt
