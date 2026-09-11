"""
Data models for the Context Manager.
"""

from enum import Enum
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field


class ContextPriority(str, Enum):
    """Priority levels for context items."""
    MANDATORY = "MANDATORY"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class ContextItem(BaseModel):
    """
    A single item of context with metadata for processing.

    This represents a piece of information that can be included in a ContextBundle.
    """
    content: Any = Field(description="The actual context content")
    priority: ContextPriority = Field(description="Priority level of this context item")
    source: str = Field(description="Source of this context item (e.g., 'intent', 'plan', 'observation')")
    relevance_score: Optional[float] = Field(
        default=None,
        description="Relevance score from 0.0 to 1.0 (None if not scored)"
    )
    token_estimate: Optional[int] = Field(
        default=None,
        description="Estimated token count for this item"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about this context item"
    )


class ContextBundle(BaseModel):
    """
    Structured context bundle for supplying information to agent components.

    This bundle contains processed context organized by type, with tracking
    of what was included, summarized, or dropped.
    """
    core: List[ContextItem] = Field(
        default_factory=list,
        description="Mandatory context that must be preserved"
    )
    working: List[ContextItem] = Field(
        default_factory=list,
        description="Current execution state context"
    )
    retrieved: List[ContextItem] = Field(
        default_factory=list,
        description="Externally supplied context"
    )
    summaries: List[str] = Field(
        default_factory=list,
        description="Summarized context items"
    )
    dropped_items: List[ContextItem] = Field(
        default_factory=list,
        description="Context items that were dropped due to budget constraints"
    )
    token_estimate: int = Field(
        default=0,
        description="Estimated total token count of the bundle"
    )
    budget_used: float = Field(
        default=0.0,
        description="Fraction of token budget used (0.0 to 1.0)"
    )

    def get_all_items(self) -> List[ContextItem]:
        """
        Get all context items in the bundle (core, working, retrieved).

        Returns:
            List of all non-summarized, non-dropped context items
        """
        return self.core + self.working + self.retrieved

    def get_included_content(self) -> List[Any]:
        """
        Get the actual content of all included context items.

        Returns:
            List of context content from core, working, and retrieved items
        """
        return [item.content for item in self.get_all_items()]