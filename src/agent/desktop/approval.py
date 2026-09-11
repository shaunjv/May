"""Exact-action approval bridge for the desktop runtime only."""

from typing import Set

from agent.tool_system.interfaces import PermissionChecker
from agent.tool_system.models import PermissionDecision, ToolRequest


class ApprovedActionAuthorizer:
    """Short-lived in-memory grants for one hash-verified execution manifest."""
    def __init__(self):
        self._request_ids: Set[str] = set()

    def grant(self, request_ids: set[str]) -> None:
        self._request_ids.update(request_ids)

    def revoke(self, request_ids: set[str]) -> None:
        self._request_ids.difference_update(request_ids)

    def contains(self, request_id: str) -> bool:
        return request_id in self._request_ids


class ApprovedActionPermissionChecker(PermissionChecker):
    """Converts REQUIRE_APPROVAL to ALLOW only for exact approved action IDs."""
    def __init__(self, delegate: PermissionChecker, authorizer: ApprovedActionAuthorizer):
        self.delegate, self.authorizer = delegate, authorizer

    async def check_permission(self, request: ToolRequest) -> PermissionDecision:
        decision = await self.delegate.check_permission(request)
        if decision == PermissionDecision.REQUIRE_APPROVAL and self.authorizer.contains(request.request_id):
            return PermissionDecision.ALLOW
        return decision
