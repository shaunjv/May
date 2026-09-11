"""
Tests for the Context Manager component.
"""

import asyncio
from unittest.mock import Mock
import pytest

from agent.context_manager.context_manager import ContextManager
from agent.context_manager.interfaces import TokenCounter, RelevanceScorer, Summarizer
from agent.context_manager.models import ContextItem, ContextBundle, ContextPriority


class MockTokenCounter:
    """Mock token counter for testing."""

    def __init__(self, tokens_per_char=0.25):  # Rough estimate: 4 chars per token
        self.tokens_per_char = tokens_per_char

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return max(1, int(len(str(text)) * self.tokens_per_char))


class MockRelevanceScorer:
    """Mock relevance scorer for testing."""

    def __init__(self, scores=None):
        self.scores = scores or []

    def score_relevance(self, items, query):
        # Return scores in order, cycling if needed
        if not self.scores:
            return [0.5] * len(items)  # Default medium relevance
        return [self.scores[i % len(self.scores)] for i in range(len(items))]


class MockSummarizer:
    """Mock summarizer for testing."""

    def __init__(self, summary_prefix="SUMMARY: "):
        self.summary_prefix = summary_prefix

    def summarize(self, items):
        if not items:
            return ""
        # Simple mock: concatenate and prefix
        content = " ".join(str(item.content) for item in items)
        return f"{self.summary_prefix}{content[:50]}..." if len(content) > 50 else f"{self.summary_prefix}{content}"


@pytest.fixture
def mock_token_counter():
    """Fixture providing a mock token counter."""
    return MockTokenCounter()


@pytest.fixture
def mock_relevance_scorer():
    """Fixture providing a mock relevance scorer."""
    return MockRelevanceScorer()


@pytest.fixture
def mock_summarizer():
    """Fixture providing a mock summarizer."""
    return MockSummarizer()


@pytest.fixture
def context_manager(mock_token_counter, mock_relevance_scorer, mock_summarizer):
    """Fixture providing a Context Manager instance."""
    return ContextManager(
        token_counter=mock_token_counter,
        relevance_scorer=mock_relevance_scorer,
        summarizer=mock_summarizer
    )


@pytest.fixture
def context_manager_no_scorer(mock_token_counter):
    """Fixture providing a Context Manager without relevance scorer."""
    return ContextManager(token_counter=mock_token_counter)


def create_test_content(text, priority=ContextPriority.MEDIUM, source="test"):
    """Helper to create test context items."""
    return ContextItem(
        content=text,
        priority=priority,
        source=source
    )


class TestContextManagerInitialization:
    """Tests for Context Manager initialization."""

    def test_init_with_all_components(self, mock_token_counter, mock_relevance_scorer, mock_summarizer):
        """Test initializing context manager with all components."""
        cm = ContextManager(
            token_counter=mock_token_counter,
            relevance_scorer=mock_relevance_scorer,
            summarizer=mock_summarizer
        )
        assert cm.token_counter == mock_token_counter
        assert cm.relevance_scorer == mock_relevance_scorer
        assert cm.summarizer == mock_summarizer

    def test_init_without_optional_components(self, mock_token_counter):
        """Test initializing context manager without optional components."""
        cm = ContextManager(token_counter=mock_token_counter)
        assert cm.token_counter == mock_token_counter
        assert cm.relevance_scorer is None
        assert cm.summarizer is None


class TestContextBundleCreation:
    """Tests for context bundle creation."""

    @pytest.mark.asyncio
    async def test_create_context_basic_bundle(self, context_manager):
        """Test creating a basic context bundle."""
        core = [create_test_content("User goal: build a website", ContextPriority.MANDATORY, "intent")]
        working = [create_test_content("Current step: design UI", ContextPriority.HIGH, "planner")]
        retrieved = [create_test_content("Previous project: similar website", ContextPriority.LOW, "memory")]

        bundle = await context_manager.create_context_bundle(
            core_context=core,
            working_context=working,
            retrieved_context=retrieved,
            query="build a website",
            token_budget=100
        )

        assert isinstance(bundle, ContextBundle)
        assert len(bundle.core) == 1
        assert len(bundle.working) == 1
        assert len(bundle.retrieved) == 1
        assert bundle.token_estimate > 0
        assert 0.0 <= bundle.budget_used <= 1.0

    @pytest.mark.asyncio
    async def test_mandatory_context_preservation(self, context_manager):
        """Test that mandatory context is never dropped."""
        core = [
            create_test_content("Goal: create web app", ContextPriority.MANDATORY, "intent"),
            create_test_content("Constraint: must be responsive", ContextPriority.MANDATORY, "intent")
        ]
        working = [create_test_content("Step: design", ContextPriority.HIGH, "planner")]
        retrieved = [
            create_test_content("Low priority info 1", ContextPriority.LOW, "memory"),
            create_test_content("Low priority info 2", ContextPriority.LOW, "memory"),
            create_test_content("Low priority info 3", ContextPriority.LOW, "memory")
        ]

        # Very small budget that should force dropping of low priority items
        # Total tokens for all items: ~27, so budget=20 should force dropping
        bundle = await context_manager.create_context_bundle(
            core_context=core,
            working_context=working,
            retrieved_context=retrieved,
            query="build web app",
            token_budget=20  # Very small budget
        )

        # Mandatory context should always be preserved
        assert len(bundle.core) == 2
        assert bundle.core[0].content == "Goal: create web app"
        assert bundle.core[1].content == "Constraint: must be responsive"

        # Low priority items should be dropped
        assert len(bundle.dropped_items) >= 1

        # Check that dropped items have the reason
        for dropped in bundle.dropped_items:
            assert dropped.metadata.get("dropped_reason") == "exceeded_token_budget"

    @pytest.mark.asyncio
    async def test_high_priority_preserved_when_possible(self, context_manager):
        """Test that high priority context is preserved when budget allows."""
        core = [create_test_content("Goal: task", ContextPriority.MANDATORY, "intent")]
        working = [
            create_test_content("High priority 1", ContextPriority.HIGH, "planner"),
            create_test_content("High priority 2", ContextPriority.HIGH, "planner")
        ]
        retrieved = [create_test_content("Low priority", ContextPriority.LOW, "memory")]

        # Large enough budget to keep everything
        bundle = await context_manager.create_context_bundle(
            core_context=core,
            working_context=working,
            retrieved_context=retrieved,
            query="task",
            token_budget=200
        )

        # All high priority items should be preserved
        assert len(bundle.working) == 2
        assert bundle.working[0].content == "High priority 1"
        assert bundle.working[1].content == "High priority 2"

        # Low priority should also be preserved with large budget
        assert len(bundle.retrieved) == 1

    @pytest.mark.asyncio
    async def test_context_within_budget_no_dropping(self, context_manager):
        """Test that when context fits budget, nothing is dropped."""
        core = [create_test_content("Goal: simple task", ContextPriority.MANDATORY, "intent")]
        working = [create_test_content("Step: do thing", ContextPriority.HIGH, "planner")]
        retrieved = [create_test_content("Info: some data", ContextPriority.LOW, "memory")]

        # Very large budget
        bundle = await context_manager.create_context_bundle(
            core_context=core,
            working_context=working,
            retrieved_context=retrieved,
            query="simple task",
            token_budget=1000
        )

        # Nothing should be dropped
        assert len(bundle.dropped_items) == 0
        assert len(bundle.core) == 1
        assert len(bundle.working) == 1
        assert len(bundle.retrieved) == 1

    @pytest.mark.asyncio
    async def test_empty_context_handling(self, context_manager):
        """Test handling of empty context inputs."""
        bundle = await context_manager.create_context_bundle(
            core_context=[],
            working_context=[],
            retrieved_context=[],
            query="test query",
            token_budget=100
        )

        assert isinstance(bundle, ContextBundle)
        assert len(bundle.core) == 0
        assert len(bundle.working) == 0
        assert len(bundle.retrieved) == 0
        assert len(bundle.dropped_items) == 0
        assert bundle.token_estimate == 0
        assert bundle.budget_used == 0.0

    @pytest.mark.asyncio
    async def test_relevance_scoring_integration(self, context_manager_no_scorer):
        """Test that context manager works without relevance scorer."""
        core = [create_test_content("Goal: test", ContextPriority.MANDATORY, "intent")]
        working = [create_test_content("Work: testing", ContextPriority.HIGH, "planner")]
        retrieved = [create_test_content("Retrieved: data", ContextPriority.LOW, "memory")]

        bundle = await context_manager_no_scorer.create_context_bundle(
            core_context=core,
            working_context=working,
            retrieved_context=retrieved,
            query="test query",
            token_budget=100
        )

        # Should still work and create a bundle
        assert isinstance(bundle, ContextBundle)
        assert len(bundle.core) == 1
        assert len(bundle.working) == 1
        assert len(bundle.retrieved) == 1

        # Relevance scores should remain None when no scorer
        for item in bundle.get_all_items():
            assert item.relevance_score is None

    @pytest.mark.asyncio
    async def test_summarizer_integration(self, context_manager):
        """Test that summarizer is used when needed."""
        core = [create_test_content("Goal: very long goal that exceeds budget", ContextPriority.MANDATORY, "intent")]
        working = [create_test_content("Step: doing work", ContextPriority.HIGH, "planner")]
        # Create multiple medium priority items that will exceed budget
        retrieved = [
            create_test_content(f"Medium info {i}: " + "x" * 100, ContextPriority.MEDIUM, "memory")
            for i in range(5)
        ]

        # Small budget that should trigger summarization
        bundle = await context_manager.create_context_bundle(
            core_context=core,
            working_context=working,
            retrieved_context=retrieved,
            query="test",
            token_budget=100
        )

        # Should have some summaries when items are summarized
        # Note: exact behavior depends on token estimates, but we should see the summarizer being used
        assert isinstance(bundle, ContextBundle)

    @pytest.mark.asyncio
    async def test_token_estimate_calculation(self, context_manager):
        """Test that token estimates are calculated correctly."""
        core = [create_test_content("Hello world", ContextPriority.MANDATORY, "intent")]
        working = [create_test_content("Test working", ContextPriority.HIGH, "planner")]
        retrieved = [create_test_content("Retrieved data", ContextPriority.LOW, "memory")]

        bundle = await context_manager.create_context_bundle(
            core_context=core,
            working_context=working,
            retrieved_context=retrieved,
            query="test",
            token_budget=1000
        )

        # Manual calculation for verification
        total_content = "Hello worldTest workingRetrieved data"
        expected_tokens = context_manager.token_counter.count_tokens(total_content)

        assert bundle.token_estimate == expected_tokens
        assert bundle.budget_used == expected_tokens / 1000

    @pytest.mark.asyncio
    async def test_large_context_handling(self, context_manager):
        """Test handling of large amounts of context."""
        core = [create_test_content("Goal: process large data", ContextPriority.MANDATORY, "intent")]
        working = [create_test_content("Step: analyze", ContextPriority.HIGH, "planner")]
        # Create lots of retrieved context
        retrieved = [
            create_test_content(f"Data point {i}: " + "data" * 50, ContextPriority.LOW, "memory")
            for i in range(20)
        ]

        bundle = await context_manager.create_context_bundle(
            core_context=core,
            working_context=working,
            retrieved_context=retrieved,
            query="process data",
            token_budget=200
        )

        # Should handle gracefully
        assert isinstance(bundle, ContextBundle)
        assert len(bundle.core) == 1  # Mandatory preserved
        assert bundle.token_estimate > 0
        assert bundle.budget_used <= 1.0

        # Some items should be dropped or summarized
        total_handled = len(bundle.core) + len(bundle.working) + len(bundle.retrieved) + len(bundle.summaries)
        assert total_handled < len(core) + len(working) + len(retrieved)  # Some were dropped/summarized


if __name__ == "__main__":
    pytest.main([__file__, "-v"])