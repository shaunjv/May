"""Narrow browser-runtime tool contracts; no shell or implicit tool access."""
import asyncio
import hashlib
import os
from pathlib import Path, PureWindowsPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from agent.desktop.models import canonical_hash
from agent.executor.models import ActionRequest
from agent.tool_system.models import Tool, ToolMetadata, ToolRequest, ToolResult, PermissionDecision


class ReadArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    path: str = Field(min_length=1, max_length=1024)


class WriteArgs(ReadArgs):
    content: str = Field(max_length=100_000)


CONTRACTS = {
    "file_read": (ReadArgs, "filesystem.read", "Read a UTF-8 text file (up to 100 KB)."),
    "file_write": (WriteArgs, "filesystem.write", "Create or replace one UTF-8 file with the exact displayed content; parent folder must exist."),
    "directory_list": (ReadArgs, "filesystem.list", "List a directory, without recursion, up to 300 entries."),
}


class Boundary:
    def __init__(self, workspace):
        self.root = Path(workspace).resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("Workspace must be an existing folder.")

    def path(self, value):
        win = PureWindowsPath(value)
        if "\x00" in value or value.startswith(("\\\\", "//")) or (win.drive and (os.name != "nt" or not win.is_absolute())):
            raise ValueError("Unsupported path.")
        raw = Path(value)
        if ".." in raw.parts or ".." in win.parts:
            raise ValueError("Parent traversal is not allowed.")
        path = raw if raw.is_absolute() else self.root / raw
        resolved = path.resolve(strict=False)
        try:
            parts = resolved.relative_to(self.root).parts
        except ValueError:
            raise ValueError("Path is outside the selected workspace.") from None
        # Block reparse points (Windows junctions too) and aliases, not just escapes.
        current = self.root
        for part in path.relative_to(self.root).parts:
            current = current / part
            if current.is_symlink() or (hasattr(current, "is_junction") and current.is_junction()):
                raise ValueError("Linked paths are not supported.")
        for part in parts:
            lower = part.lower()
            if lower in {".git", ".ssh", ".aws", ".azure"} or lower.startswith(".env") or lower.endswith((".key", ".pem", ".pfx", ".p12")) or ":" in part or part.endswith((".", " ")):
                raise ValueError("Protected path.")
        if resolved.is_file() and resolved.stat().st_nlink > 1:
            raise ValueError("Hard-linked files are not supported.")
        return resolved

    def arguments(self, name, data):
        if name not in CONTRACTS:
            raise ValueError("Tool is not available in this runtime.")
        args = CONTRACTS[name][0].model_validate(data).model_dump()
        path = self.path(args["path"])
        if name == "directory_list":
            if not path.is_dir():
                raise ValueError("Directory does not exist.")
        elif path == self.root or (path.exists() and not path.is_file()):
            raise ValueError("Expected a file path.")
        elif name == "file_read" and not path.is_file():
            raise ValueError("File does not exist.")
        elif not path.parent.is_dir():
            raise ValueError("Parent folder does not exist.")
        if path.is_file() and path.stat().st_size > 100_000:
            raise ValueError("File exceeds the 100 KB V1 limit.")
        args["path"] = str(path)
        return args

    def fingerprint(self, path):
        target = self.path(path)
        if target.is_file():
            if target.stat().st_size > 100_000:
                raise ValueError("File exceeds the V1 limit.")
            return hashlib.sha256(target.read_bytes()).hexdigest()
        return None


def metadata(name):
    schema, capability, description = CONTRACTS[name]
    return ToolMetadata(name=name, description=description, operation_type="WRITE" if name == "file_write" else "READ", risk_level="MEDIUM" if name == "file_write" else "LOW", is_read_only=name != "file_write", required_permission="APPROVAL", capabilities={capability}, argument_schema=schema.model_json_schema())


class WorkspaceTool(Tool):
    boundary: Boundary

    def execute(self, request):
        try:
            args = self.boundary.arguments(request.tool_name, request.tool_input)
            path = Path(args["path"])
            if request.tool_name == "file_read":
                output = path.read_text(encoding="utf-8")
            elif request.tool_name == "file_write":
                path.write_text(args["content"], encoding="utf-8")
                output = f"Saved {path.name}."
            else:
                output = []
                for child in path.iterdir():
                    try:
                        self.boundary.path(str(child))
                    except ValueError:
                        continue
                    output.append({"name": child.name, "directory": child.is_dir()})
                    if len(output) == 300:
                        break
                output.sort(key=lambda entry: (not entry["directory"], entry["name"].lower()))
            return ToolResult(request_id=request.request_id, tool_name=request.tool_name, success=True, output=output)
        except Exception:
            return ToolResult(request_id=request.request_id, tool_name=request.tool_name, success=False, error="File operation failed. The path may have changed, be protected, or be unreadable.")


class ExactPermission:
    """Permission is scoped to exact serialized requests, never just an action ID."""
    def __init__(self, actions, boundary, cancelled):
        self.allowed = {a.action_id: canonical_hash(ToolRequest(request_id=a.action_id, tool_name=a.tool_name, tool_input=a.tool_input, timeout_seconds=a.timeout_seconds)) for a in actions}
        self.boundary, self.cancelled = boundary, cancelled

    async def check_permission(self, request):
        try:
            self.boundary.arguments(request.tool_name, request.tool_input)
            if not self.cancelled.is_set() and self.allowed.get(request.request_id) == canonical_hash(request):
                return PermissionDecision.ALLOW
        except Exception:
            pass
        return PermissionDecision.DENY
