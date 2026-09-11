# Personal AI Agent

A local-first personal AI agent with a browser interface, NVIDIA-hosted model support, and an approval gate before supported workspace actions run.

## Supported runtime

The browser app is the supported runtime:

```powershell
uv run --extra web agent-web
```

Or double-click `Launch Personal AI Agent.vbs`. The app opens locally at `http://127.0.0.1:8765`.

The older PySide6 desktop implementation remains in the repository as legacy code only. It is not the supported path while the browser runtime is being hardened.

## What it can do today

- Answer ordinary questions and generate writing or code.
- Keep conversations locally in SQLite.
- Give every conversation its own selected workspace folder.
- Create a plan and concrete action preview for a workspace task.
- Run supported file actions only after approval: list folders, read text files, create text files, and replace text files.
- Block workspace traversal, linked-path escapes, protected paths, and unapproved actions.

The current browser runtime intentionally does **not** delete, move, rename, copy files, run terminal commands, use Git, browse the web, or edit binary documents/images.

## Setup

1. Install [uv](https://docs.astral.sh/uv/).
2. Install Python dependencies:

   ```powershell
   uv sync --extra web
   ```

3. In the project folder, copy `.env.example` to `.env` and set:

   ```text
   NVIDIA_API_KEY=your_key_here
   NVIDIA_MODEL=nvidia/nemotron-3.5-lightning-30b-a3b
   ```

   Do not commit `.env` or paste keys into chat.

4. Install/build the browser UI:

   ```powershell
   cd web
   npm ci
   npm run build
   cd ..
   ```

5. Start the app using the command above or the launcher.

## Development checks

```powershell
uv run --no-sync pytest -q

cd web
npm test
npm run build
```

## Architecture

The active browser path is:

```text
React browser UI
  -> loopback FastAPI API
  -> browser Runtime
  -> Intent Manager / Planner / Action Resolver
  -> Executor / Tool System
  -> workspace-bound file tools
```

Plans and exact action manifests are stored locally and hashed before approval. A task interrupted during planning or execution is marked failed after restart and is never resumed automatically.

## Security notes

- The app binds to `127.0.0.1` only.
- It never stores the NVIDIA API key in its SQLite database.
- Workspace actions are restricted to the folder selected for that conversation.
- File actions require explicit approval of the displayed plan.

## Current status

This is a V1 prototype, not yet a full unrestricted computer-control agent. Before enabling additional tools, the browser runtime and its security checks will be extended and tested incrementally.
