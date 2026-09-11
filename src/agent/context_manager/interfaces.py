"""
Interfaces for the Context Manager components.
"""

from abc import ABC, abstractmethod
from typing import List, Protocol, runtime_checkable
from .models import ContextItem, ContextBundle


@runtime_checkable
class TokenCounter(Protocol):
    """Interface for counting tokens in text."""

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in the given text.

        Args:
            text: Text to count tokens for

        Returns:
            Estimated number of tokens
        """
        ...


@runtime_checkable
class RelevanceScorer(Protocol):
    """Interface for scoring context relevance."""

    def score_relevance(self, items: List[ContextItem], query: str) -> List[float]:
        """
        Score relevance of context items to a query.

        Args:
            items: List of context items to score
            query: Query to score relevance against

        Returns:
            List of relevance scores (0.0 to 1.0) corresponding to each item
        """
        ...


@runtime_checkable
class Summarizer(Protocol):
    """Interface for summarizing context items."""

    def summarize(self, items: List[ContextItem]) -> str:
        """
        Summarize a list of context items.

        Args:
            items: List of context items to summarize

        Returns:
            Summarized text
        """
        ...


class ContextManager(ABC):
    """Abstract base class for Context Manager."""

    @abstractmethod
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
        ...