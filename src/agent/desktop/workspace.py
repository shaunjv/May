"""Deterministic workspace and bounded-command validation."""

from pathlib import Path
import re
from typing import Iterable

from agent.tool_system.models import ToolMetadata, ToolRequest


class WorkspaceViolation(ValueError):
    pass


class WorkspaceGuard:
    """Tool System boundary guard; never trusts an LLM supplied path or command."""
    allowed_commands = {"pytest", "python", "git"}
    shell_metacharacters = re.compile(r"[|&;<>`$(){}\n\r]")

    def __init__(self, workspace: str | Path):
        self.root = Path(workspace).expanduser().resolve(strict=True)
        if not self.root.is_dir():
            raise WorkspaceViolation("Workspace must be an existing directory.")

    def resolve_path(self, value: str | Path) -> Path:
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        # strict=False resolves existing parent symlinks and safely handles new files.
        resolved = candidate.resolve(strict=False)
        try:
            resolved.relative_to(self.root)
        except ValueError as error:
            raise WorkspaceViolation("Path is outside the selected workspace.") from error
        return resolved

    def validate(self, request: ToolRequest, metadata: ToolMetadata) -> None:
        data = request.tool_input
        if request.tool_name == "terminal":
            self._validate_terminal(data)
            return
        if request.tool_name == "git":
            if data.get("operation") not in {"status", "diff", "log"}:
                raise WorkspaceViolation("Only read-only Git inspection is allowed in V1.")
            self.resolve_path(data.get("cwd", self.root))
            return
        target = metadata.target_resource
        if metadata.allowed_workspace_only and target and isinstance(data.get(target), str):
            self.resolve_path(data[target])

    def _validate_terminal(self, data: dict) -> None:
        command = data.get("command")
        if not isinstance(command, list) or not command or not all(isinstance(p, str) for p in command):
            raise WorkspaceViolation("Terminal command must be a non-empty argument list.")
        if command[0] not in self.allowed_commands or any(self.shell_metacharacters.search(p) for p in command):
            raise WorkspaceViolation("Terminal command is not in the approved no-shell allowlist.")
        self.resolve_path(data.get("cwd", self.root))
