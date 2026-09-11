"""
Deterministic preprocessor for the Intent Manager.

Handles straightforward, high-confidence extraction/classification for:
- greetings/simple conversational acknowledgements
- obvious information requests
- explicit automation signals
- explicit computer commands
- explicit priority signals
- explicit deadline/date/time expressions
"""

import re
from typing import List, Optional
from .models import Intent, IntentType, TaskObjective, Priority


class DeterministicPreprocessor:
    """
    Rule-based preprocessor that handles high-confident intent classification
    without requiring LLM inference.
    """

    def __init__(self):
        # Conversational patterns (greetings, acknowledgements)
        self.conversation_patterns = [
            r'^\s*(hi|hello|hey|good\s+(morning|afternoon|evening)|greetings)\b',
            r'^\s*(thanks?|thank\s+you|thx|ty)\b',
            r'^\s*(bye|goodbye|see\s+you|later)\b',
            r'^\s*(ok|okay|k|yes|yeah|yep|no|nope|nah)\b',
        ]

        # Information request patterns
        self.information_patterns = [
            r'^\s*(what|who|when|where|why|how)\s+',
            r'^\s*(can\s+you\s+tell\s+me|do\s+you\s+know|is\s+it\s+true\s+that)\s+',
            r'^\s*(explain|describe|define|list|show\s+me)\s+',
            r'^\s*(what\'s|what\s+is)\s+',
        ]

        # Automation patterns
        self.automation_patterns = [
            r'\b(remind|schedule|every\s+(day|week|month|year)|daily|weekly|monthly|yearly)\b',
            r'\b(timer|alarm|wake\s+up)\b',
            r'\b(at\s+\d{1,2}:\d{2}\s*(am|pm)?|at\s+\d{1,2}\s*(am|pm))\b',
            r'\b(in\s+\d+\s*(minutes?|hours?|days?|weeks?))\b',
            r'\b(tomorrow|today|tonight|next\s+(week|month|year))\b',
        ]

        # Computer action patterns
        self.computer_action_patterns = [
            r'\b(open|launch|start|run|execute)\b',
            r'\b(close|quit|exit|terminate|kill)\b',
            r'\b(create|make|new|delete|remove)\s+(file|folder|directory|window|tab)\b',
            r'\b(copy|paste|cut|save|print)\b',
        ]

        # Uncertainty patterns - if we detect these, we're not confident
        self.uncertainty_patterns = [
            r'\b(somewhat|maybe|perhaps|possibly|i\s+think|i\s+am\s+not\s+sure|not\s+sure|uncertain|unsure)\b',
            r'\b(what\s+to\s+do|how\s+to\s+|whether\s+)\b',
        ]

        # Priority patterns
        self.priority_patterns = [
            (r'\b(asap|urgent|urgently|critical|emergency|important|high\s+priority)\b', Priority.HIGH),
            (r'\b(low\s+priority|not\s+urgent|not\s+urgently|when\s+you\s+can)\b', Priority.LOW),
        ]

        # Deadline patterns
        self.deadline_patterns = [
            r'\b(today|tomorrow|tonight)\b',
            r'\b(this\s+(morning|afternoon|evening|night|week|month|year))\b',
            r'\b(next\s+(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|week|month|year))\b',
            r'\b(last\s+(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|week|month|year))\b',
            r'\b(\d{1,2}:\d{2}\s*(am|pm)?)\b',
            r'\b(\d{1,2}\s*(am|pm))\b',
            r'\b(by\s+(today|tomorrow|\d{1,2}:\d{2}|\d{1,2}\s*(am|pm)))\b',
            r'\b(at\s+(today|tomorrow|\d{1,2}:\d{2}|\d{1,2}\s*(am|pm)))\b',
            r'\b(in\s+\d+\s*(minutes?|hours?|days?|weeks?))\b',
        ]

        # Task objective patterns
        self.task_objective_patterns = [
            (r'\b(fix|repair|correct|resolve)\b', TaskObjective.FIX),
            (r'\b(create|make|build|new|add)\b', TaskObjective.CREATE),
            (r'\b(modify|change|update|alter|edit)\b', TaskObjective.MODIFY),
            (r'\b(analyze|examine|investigate|review|audit)\b', TaskObjective.ANALYZE),
            (r'\b(research|investigate|look\s+up|find\s+information)\b', TaskObjective.RESEARCH),
            (r'\b(execute|run|perform|carry\s+out|do)\b', TaskObjective.EXECUTE),
        ]

        # Compile regex patterns for efficiency
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile all regex patterns for better performance."""
        self._compiled_conversation = [re.compile(p, re.IGNORECASE) for p in self.conversation_patterns]
        self._compiled_information = [re.compile(p, re.IGNORECASE) for p in self.information_patterns]
        self._compiled_automation = [re.compile(p, re.IGNORECASE) for p in self.automation_patterns]
        self._compiled_computer_action = [re.compile(p, re.IGNORECASE) for p in self.computer_action_patterns]
        self._compiled_priority = [(re.compile(p, re.IGNORECASE), pri) for p, pri in self.priority_patterns]
        self._compiled_deadline = [re.compile(p, re.IGNORECASE) for p in self.deadline_patterns]
        self._compiled_task_objective = [(re.compile(p, re.IGNORECASE), obj) for p, obj in self.task_objective_patterns]
        self._compiled_uncertainty = [re.compile(p, re.IGNORECASE) for p in self.uncertainty_patterns]

    def process(self, user_input: str) -> Optional[Intent]:
        """
        Process user input through deterministic rules.

        Returns:
            Intent if confidently classifiable, None otherwise
        """
        if not user_input or not user_input.strip():
            return None

        input_lower = user_input.lower().strip()

        # Check for uncertainty patterns - if we detect significant uncertainty,
        # don't make a confident classification
        uncertainty_count = sum(1 for pattern in self._compiled_uncertainty
                              if pattern.search(input_lower))
        # If we have uncertainty indicators, be more conservative
        has_uncertainty = uncertainty_count > 0

        # Check for conversational intent
        if self._matches_any_pattern(input_lower, self._compiled_conversation):
            return Intent(
                primary_intent=IntentType.CONVERSATION,
                priority=Priority.NORMAL
            )

        # Check for information request intent
        if self._matches_any_pattern(input_lower, self._compiled_information):
            return Intent(
                primary_intent=IntentType.INFORMATION,
                priority=Priority.NORMAL
            )

        # Initialize intent data
        intent_data = {
            'primary_intent': IntentType.TASK,  # Default to TASK for action-oriented requests
            'priority': Priority.NORMAL,
            'sub_intents': [],
            'objective': None,
            'goal': user_input[:100],  # Truncate for simplicity
            'domain': None,
            'constraints': [],
            'success_criteria': [],
            'deadline': None,
            'requires_clarification': False,
            'clarification_reason': None
        }

        # Check for automation signals
        if self._matches_any_pattern(input_lower, self._compiled_automation):
            intent_data['sub_intents'].append(IntentType.AUTOMATION)

        # Check for computer action signals
        computer_action_matches = []
        for pattern in self._compiled_computer_action:
            if pattern.search(input_lower):
                computer_action_matches.append(pattern)

        if computer_action_matches:
            intent_data['sub_intents'].append(IntentType.COMPUTER_ACTION)

            # If we have computer action patterns and no other strong signals,
            # this might be the primary intent
            automation_matches = []
            for pattern in self._compiled_automation:
                if pattern.search(input_lower):
                    automation_matches.append(pattern)

            # If we have more computer action matches than automation matches,
            # consider computer action as primary
            if len(computer_action_matches) > len(automation_matches) and len(intent_data['sub_intents']) <= 1:
                intent_data['primary_intent'] = IntentType.COMPUTER_ACTION

        # Extract priority
        for pattern, priority in self._compiled_priority:
            if pattern.search(input_lower):
                intent_data['priority'] = priority
                break

        # Extract deadline
        deadline_match = self._extract_deadline(input_lower)
        if deadline_match:
            intent_data['deadline'] = deadline_match.group(0)

        # Extract task objective if primary_intent is TASK
        if intent_data['primary_intent'] == IntentType.TASK:
            for pattern, objective in self._compiled_task_objective:
                if pattern.search(input_lower):
                    intent_data['objective'] = objective
                    break

            # If no objective found but we have action-oriented sub-intents, default to EXECUTE
            if intent_data['objective'] is None and intent_data['sub_intents']:
                intent_data['objective'] = TaskObjective.EXECUTE
            # If still no objective, check for any action indicators (priority, deadline, etc.)
            elif intent_data['objective'] is None:
                # Check if we have any indicators that suggest this is an action/task
                has_action_indicators = (
                    self._matches_any_pattern(input_lower, self._compiled_priority) or
                    self._matches_any_pattern(input_lower, self._compiled_deadline) or
                    self._matches_any_pattern(input_lower, self._compiled_automation) or
                    self._matches_any_pattern(input_lower, self._compiled_computer_action)
                )

                if has_action_indicators:
                    intent_data['objective'] = TaskObjective.EXECUTE

        # Determine if we have enough confidence to return an intent
        confidence_indicators = 0

        # Count how many pattern categories we matched
        if self._matches_any_pattern(input_lower, self._compiled_conversation):
            confidence_indicators += 1
        if self._matches_any_pattern(input_lower, self._compiled_information):
            confidence_indicators += 1
        if self._matches_any_pattern(input_lower, self._compiled_automation):
            confidence_indicators += 1
        if self._matches_any_pattern(input_lower, self._compiled_computer_action):
            confidence_indicators += 1
        if self._matches_any_pattern(input_lower, self._compiled_priority):
            confidence_indicators += 1
        if self._matches_any_pattern(input_lower, self._compiled_deadline):
            confidence_indicators += 1
        if self._matches_any_pattern(input_lower, self._compiled_task_objective):
            confidence_indicators += 1

        # If we detected action-based objective, that's also a confidence indicator
        if intent_data['objective'] is not None:
            confidence_indicators += 1

        # Apply uncertainty penalty - if we detect uncertainty, don't make confident classification
        if has_uncertainty:
            confidence_indicators = 0

        # If we have at least one strong signal, return the intent
        if confidence_indicators > 0:
            return Intent(**intent_data)

        # Not confident enough - return None to let LLM handle it
        return None

    def _matches_any_pattern(self, text: str, patterns: List) -> bool:
        """Check if text matches any of the compiled patterns."""
        if not patterns:
            return False

        # Handle different pattern types
        if patterns and isinstance(patterns[0], tuple):
            # For priority patterns which are (pattern, priority) tuples
            return any(pattern.search(text) for pattern, _ in patterns)
        else:
            # For regular pattern lists
            return any(pattern.search(text) for pattern in patterns)

    def _extract_deadline(self, text: str) -> Optional[re.Match]:
        """Extract deadline expression from text."""
        for pattern in self._compiled_deadline:
            match = pattern.search(text)
            if match:
                return match
        return None