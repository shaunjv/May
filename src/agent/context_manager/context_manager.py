"""
Context Manager implementation that creates structured ContextBundles.
"""

import logging
from typing import List, Optional, Tuple
from .interfaces import ContextManager as ContextManagerInterface, TokenCounter, RelevanceScorer, Summarizer
from .models import ContextItem, ContextBundle, ContextPriority

logger = logging.getLogger(__name__)


class ContextManager(ContextManagerInterface):
    """
    Context Manager that decides what information should be provided to the next agent component.

    Implements a layered approach:
    Core Context → Working Context → Retrieved Context → Hybrid Relevance Scoring →
    Context Budget → Context Compression → ContextBundle
    """

    def __init__(
        self,
        token_counter: TokenCounter,
        relevance_scorer: Optional[RelevanceScorer] = None,
        summarizer: Optional[Summarizer] = None
    ):
        """
        Initialize the Context Manager.

        Args:
            token_counter: Token counter for estimating token usage
            relevance_scorer: Optional relevance scorer for hybrid scoring
            summarizer: Optional summarizer for compressing context
        """
        self.token_counter = token_counter
        self.relevance_scorer = relevance_scorer
        self.summarizer = summarizer
        logger.info("Context Manager initialized")

    async def create_context_bundle(
        self,
        core_context: List[ContextItem],
        working_context: List[ContextItem],
        retrieved_context: List[ContextItem],
        query: str,
        token_budget: int
    ) -> ContextBundle:
        """
        Create a context bundle from the provided contexts.

        Args:
            core_context: Mandatory context that must be preserved
            working_context: Current execution state context
            retrieved_context: Externally supplied context
            query: Current query or task description
            token_budget: Maximum tokens allowed in the bundle

        Returns:
            ContextBundle containing the processed context
        """
        logger.debug(f"Creating context bundle with budget {token_budget}")

        # Start with all context items
        all_items = core_context + working_context + retrieved_context

        # Apply deterministic priority (core context is always MANDATORY)
        prioritized_items = self._apply_deterministic_priority(
            core_context, working_context, retrieved_context
        )

        # Apply hybrid relevance scoring if scorer is available
        if self.relevance_scorer and len(prioritized_items) > 0:
            prioritized_items = await self._apply_relevance_scoring(
                prioritized_items, query
            )

        # Apply context budgeting and compression
        bundle_items, dropped_items, summaries = await self._apply_budget_and_compression(
            prioritized_items, token_budget
        )

        # Calculate token estimate and budget usage
        total_content = " ".join(str(item.content) for item in bundle_items)
        token_estimate = self.token_counter.count_tokens(total_content)
        budget_used = min(token_estimate / token_budget, 1.0) if token_budget > 0 else 0.0

        # Separate by context type for the bundle
        core_items = [item for item in bundle_items if item.priority == ContextPriority.MANDATORY]
        working_items = [item for item in bundle_items if item.priority == ContextPriority.HIGH]
        retrieved_items = [item for item in bundle_items if item.priority in [ContextPriority.MEDIUM, ContextPriority.LOW]]

        bundle = ContextBundle(
            core=core_items,
            working=working_items,
            retrieved=retrieved_items,
            summaries=summaries,
            dropped_items=dropped_items,
            token_estimate=token_estimate,
            budget_used=budget_used
        )

        logger.debug(
            f"Context bundle created: {len(core_items)} core, {len(working_items)} working, "
            f"{len(retrieved_items)} retrieved, {len(summaries)} summaries, {len(dropped_items)} dropped"
        )

        return bundle

    def _apply_deterministic_priority(
        self,
        core_context: List[ContextItem],
        working_context: List[ContextItem],
        retrieved_context: List[ContextItem]
    ) -> List[ContextItem]:
        """
        Apply deterministic priority levels to context items.

        Args:
            core_context: Mandatory context items
            working_context: Working context items
            retrieved_context: Retrieved context items

        Returns:
            List of all context items with priority levels set
        """
        # Core context is always MANDATORY
        for item in core_context:
            item.priority = ContextPriority.MANDATORY

        # Working context is HIGH priority by default
        for item in working_context:
            if item.priority not in [ContextPriority.MANDATORY]:
                item.priority = ContextPriority.HIGH

        # Retrieved context defaults to MEDIUM priority
        for item in retrieved_context:
            if item.priority not in [ContextPriority.MANDATORY, ContextPriority.HIGH]:
                item.priority = ContextPriority.MEDIUM

        return core_context + working_context + retrieved_context

    async def _apply_relevance_scoring(
        self,
        items: List[ContextItem],
        query: str
    ) -> List[ContextItem]:
        """
        Apply LLM-based relevance scoring to context items.

        Args:
            items: Context items to score
            query: Query to score relevance against

        Returns:
            List of context items with relevance scores set
        """
        if not items or not self.relevance_scorer:
            return items

        try:
            # Extract text content for scoring
            texts = [str(item.content) for item in items]
            # Batch score all items at once
            scores = self.relevance_scorer.score_relevance(items, query)

            # Apply scores to items
            for item, score in zip(items, scores):
                item.relevance_score = score

            # Sort by relevance score (descending) for items that aren't MANDATORY
            mandatory_items = [item for item in items if item.priority == ContextPriority.MANDATORY]
            non_mandatory_items = [item for item in items if item.priority != ContextPriority.MANDATORY]
            non_mandatory_items.sort(key=lambda x: x.relevance_score or 0.0, reverse=True)

            return mandatory_items + non_mandatory_items

        except Exception as e:
            logger.warning(f"Relevance scoring failed: {e}")
            # Return items in original order if scoring fails
            return items

    async def _apply_budget_and_compression(
        self,
        items: List[ContextItem],
        token_budget: int
    ) -> Tuple[List[ContextItem], List[ContextItem], List[str]]:
        """
        Apply context budget constraints and compression.

        Args:
            items: Prioritized context items
            token_budget: Maximum tokens allowed

        Returns:
            Tuple of (included_items, dropped_items, summaries)
        """
        if token_budget <= 0:
            return [], items, []

        # Separate mandatory items (never dropped)
        mandatory_items = [item for item in items if item.priority == ContextPriority.MANDATORY]
        optional_items = [item for item in items if item.priority != ContextPriority.MANDATORY]

        # Start with all mandatory items
        included_items = mandatory_items.copy()
        current_tokens = sum(
            self.token_counter.count_tokens(str(item.content))
            for item in included_items
        )

        # Process optional items by priority: HIGH → MEDIUM → LOW
        for priority_level in [ContextPriority.HIGH, ContextPriority.MEDIUM, ContextPriority.LOW]:
            priority_items = [item for item in optional_items if item.priority == priority_level]

            for item in priority_items:
                item_tokens = self.token_counter.count_tokens(str(item.content))

                # Check if we can add this item without exceeding budget
                if current_tokens + item_tokens <= token_budget:
                    included_items.append(item)
                    current_tokens += item_tokens
                else:
                    # Try to summarize if we have a summarizer and it's not already summarized
                    if (self.summarizer and
                        priority_level in [ContextPriority.MEDIUM, ContextPriority.LOW] and
                        item.metadata.get("summarized", False) is not True):

                        try:
                            summary = self.summarizer.summarize([item])
                            summary_tokens = self.token_counter.count_tokens(summary)

                            if current_tokens + summary_tokens <= token_budget:
                                # Create summarized item
                                summarized_item = ContextItem(
                                    content=summary,
                                    priority=item.priority,
                                    source=f"{item.source}_summary",
                                    relevance_score=item.relevance_score,
                                    token_estimate=summary_tokens,
                                    metadata={**item.metadata, "summarized": True, "original_item": item.content}
                                )
                                included_items.append(summarized_item)
                                current_tokens += summary_tokens
                                continue
                        except Exception as e:
                            logger.warning(f"Summarization failed: {e}")

                    # If we can't fit it (even summarized), drop it
                    item.metadata["dropped_reason"] = "exceeded_token_budget"

        # Remaining optional items are dropped
        dropped_items = [
            item for item in optional_items
            if item not in included_items
        ]

        # Extract summaries from included items
        summaries = [
            item.content for item in included_items
            if item.metadata.get("summarized", False) is True
        ]

        return included_items, dropped_items, summaries