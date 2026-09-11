"""
agent_cli.py – Simple command-line wrapper to launch the iterative agent.

This file wires together the *already‑implemented* components:

* IntentManager
* Planner
* ContextManager
* Executor

It then launches the V1 ``Orchestrator`` and prints the final result.

The wrapper is deliberately minimal – it does **not** modify any of the
locked components, it only calls their public APIs, and it respects all
safety boundaries (iteration limits, timeout handling, etc.).
"""
import asyncio
import json
import logging
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
#   Component instantiation helpers – each returns a ready‑to‑use instance
#   that respects the public contracts of the locked components.
# --------------------------------------------------------------------------- #
async def _load_llm_provider(settings_instance=None):
    """Load the configured real model provider without a mock fallback."""
    from agent.config import settings
    from agent.intent_manager.nvidia_provider import NVIDIAProvider

    return NVIDIAProvider.from_settings(settings_instance or settings)


async def _load_intent_manager(llm_provider):
    """
    IntentManager receives the configured provider through dependency injection.
    """
    from agent.intent_manager.intent_manager import IntentManager
    return IntentManager(llm_provider=llm_provider)


async def _load_planner(llm_provider):
    """
    Planner receives the same configured provider as the Intent Manager.
    """
    from agent.planner.planner import Planner
    return Planner(llm_provider=llm_provider)


async def _load_context_manager() -> "ContextManager":
    """
    ContextManager needs a ``TokenCounter`` (required) and optionally
    ``RelevanceScorer`` and ``Summarizer``.  The repository ships a
    ``DefaultTokenCounter`` that implements the ``TokenCounter`` protocol.
    If such a class does not exist we fall back to a tiny local implementation
    (it only counts whitespace‑separated tokens).
    """
    from agent.context_manager.context_manager import ContextManager  # type: ignore

    # Try the real token‑counter first; if it does not exist we fall back
    # to a tiny local implementation that satisfies the Protocol.
    try:
        from agent.context_manager.token_counter import TokenCounter as RealTokenCounter  # type: ignore
        token_counter = RealTokenCounter()
    except Exception:  # pragma: no cover – fallback when the symbol is missing
        class DummyTokenCounter:
            def count_tokens(self, text: str) -> int:  # pragma: no cover
                # Very naive tokenisation – useful only for placeholder work
                return max(1, len(text.split()))
        token_counter = DummyTokenCounter()  # type: ignore

    # The other dependencies are optional; we pass ``None``.
    return ContextManager(token_counter=token_counter)  # type: ignore


async def _load_executor() -> "Executor":
    """
    Build the real Executor → Tool System integration used by the CLI.

    Tools are registered centrally, permission enforcement uses the current
    working directory as its workspace boundary, and the Executor receives
    Tool System adapters through its existing public interfaces.
    """
    from agent.executor.executor import Executor
    from agent.executor.tool_system_adapter import (
        ToolSystemPermissionChecker,
        ToolSystemToolExecutor,
    )
    from agent.tool_system.manager import ToolManagerImpl
    from agent.tool_system.permission import PermissionCheckerImpl, PermissionPolicyImpl
    from agent.tool_system.registry import ToolRegistryImpl
    from agent.tool_system.tools import (
        DirectoryListTool,
        FileEditTool,
        FileReadTool,
        FileSearchTool,
        FileWriteTool,
        GitTool,
        TerminalTool,
    )

    registry = ToolRegistryImpl()
    permission_policy = PermissionPolicyImpl(workspace_root=str(Path.cwd()))
    permission_checker = PermissionCheckerImpl(permission_policy, registry=registry)
    tool_manager = ToolManagerImpl(registry, permission_checker)

    for tool in (
        FileReadTool(),
        FileWriteTool(),
        FileEditTool(),
        FileSearchTool(),
        DirectoryListTool(),
        TerminalTool(),
        GitTool(),
    ):
        await tool_manager.register_tool(tool)

    return Executor(
        tool_executor=ToolSystemToolExecutor(tool_manager),
        permission_checker=ToolSystemPermissionChecker(tool_manager),
    )


# --------------------------------------------------------------------------- #
#   CLI entry point
# --------------------------------------------------------------------------- #
async def main() -> None:
    """
    Entry point used by ``python -m agent.cli`` or by the console script
    defined in ``pyproject.toml``.  It:

    1. Instantiates the four core components.
    2. Instantiates the V1 ``Orchestrator``.
    3. Calls its ``run`` method with the user-supplied text.
    4. Prints the structured orchestration outcome.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the V1 agent on a user-provided query."
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="What can you do for me?",
        help="Raw user input to process.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------- #
    #   1️⃣  Build the component graph (all constructions are side‑effect‑free)
    # --------------------------------------------------------------------------- #
    try:
        llm_provider = await _load_llm_provider()
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))
    intent_manager = await _load_intent_manager(llm_provider)
    planner = await _load_planner(llm_provider)
    context_manager = await _load_context_manager()
    executor = await _load_executor()

    # --------------------------------------------------------------------------- #
    #   2️⃣  Create the V1 orchestrator and run it
    # --------------------------------------------------------------------------- #
    from agent.orchestrator.orchestrator import Orchestrator

    agent = Orchestrator(
        intent_manager=intent_manager,
        planner=planner,
        context_manager=context_manager,
        executor=executor,
    )

    # --------------------------------------------------------------------------- #
    #   3️⃣  Execute and return the result
    # --------------------------------------------------------------------------- #
    orchestration_result = await agent.run(args.query)

    if orchestration_result.status == "completed":
        print("\n=== ORCHESTRATION COMPLETED ===\n")
        print(json.dumps(orchestration_result.model_dump(mode="json"), indent=2))
    else:
        print("\n=== ERROR ===\n")
        print(orchestration_result.error or "An unknown error occurred.")


# --------------------------------------------------------------------------- #
#   Synchronous wrapper required by the console‑script entry point
# --------------------------------------------------------------------------- #
def main_sync() -> None:
    """
    A tiny synchronous wrapper that runs the async ``main`` coroutine.
    This function is what the ``console_scripts`` entry point invokes.
    """
    asyncio.run(main())


if __name__ == "__main__":
    # ``__main__`` is used when the module is executed directly
    # (e.g. ``python -m agent.cli``).  It simply forwards to ``main_sync``.
    main_sync()
