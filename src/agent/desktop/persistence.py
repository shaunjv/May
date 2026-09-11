"""Small, versioned SQLite persistence with no provider-secret fields."""

import json
import sqlite3
from pathlib import Path
from typing import Iterable, Optional

from .models import TaskRecord, TaskState, utcnow


class SQLiteStore:
    schema_version = 1

    def __init__(self, database_path: str | Path):
        self.path = Path(database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self._migrate()
        self.recover_interrupted_tasks()

    def _migrate(self) -> None:
        with self.connection:
            self.connection.execute("CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY)")
            self.connection.execute("""CREATE TABLE IF NOT EXISTS conversations (
                conversation_id TEXT PRIMARY KEY, workspace_path TEXT NOT NULL, created_at TEXT NOT NULL)""")
            self.connection.execute("""CREATE TABLE IF NOT EXISTS messages (
                message_id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL, role TEXT NOT NULL,
                content TEXT NOT NULL, created_at TEXT NOT NULL)""")
            self.connection.execute("""CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL, workspace_path TEXT NOT NULL,
                state TEXT NOT NULL, payload_json TEXT NOT NULL, updated_at TEXT NOT NULL)""")
            self.connection.execute("""CREATE TABLE IF NOT EXISTS plans (
                task_id TEXT PRIMARY KEY, plan_id TEXT NOT NULL, plan_version INTEGER NOT NULL,
                plan_hash TEXT NOT NULL, payload_json TEXT NOT NULL)""")
            self.connection.execute("""CREATE TABLE IF NOT EXISTS approvals (
                task_id TEXT PRIMARY KEY, plan_hash TEXT NOT NULL, action_manifest_hash TEXT NOT NULL,
                approved_at TEXT NOT NULL)""")
            self.connection.execute("""CREATE TABLE IF NOT EXISTS task_actions (
                task_id TEXT NOT NULL, action_id TEXT NOT NULL, payload_json TEXT NOT NULL,
                PRIMARY KEY(task_id, action_id))""")
            self.connection.execute("""CREATE TABLE IF NOT EXISTS execution_summaries (
                task_id TEXT PRIMARY KEY, payload_json TEXT NOT NULL, created_at TEXT NOT NULL)""")
            self.connection.execute("INSERT OR IGNORE INTO schema_migrations(version) VALUES (?)", (self.schema_version,))

    def save_task(self, task: TaskRecord) -> None:
        payload = task.model_dump(mode="json")
        with self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO tasks(task_id, conversation_id, workspace_path, state, payload_json, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (task.task_id, task.conversation_id, task.workspace_path, task.state.value, json.dumps(payload), utcnow()),
            )

    def get_task(self, task_id: str) -> Optional[TaskRecord]:
        row = self.connection.execute("SELECT payload_json FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        return TaskRecord.model_validate_json(row["payload_json"]) if row else None

    def save_plan(self, task_id: str, plan: object, plan_hash: str) -> None:
        data = plan.model_dump(mode="json")
        with self.connection:
            self.connection.execute("INSERT OR REPLACE INTO plans VALUES (?, ?, ?, ?, ?)", (task_id, data["plan_id"], data["version"], plan_hash, json.dumps(data)))

    def save_approval(self, task_id: str, plan_hash: str, action_hash: str) -> None:
        with self.connection:
            self.connection.execute("INSERT OR REPLACE INTO approvals VALUES (?, ?, ?, ?)", (task_id, plan_hash, action_hash, utcnow()))

    def recover_interrupted_tasks(self) -> int:
        rows = self.connection.execute("SELECT task_id, payload_json FROM tasks WHERE state IN ('PLANNING','EXECUTING')").fetchall()
        for row in rows:
            task = TaskRecord.model_validate_json(row["payload_json"])
            task.state = TaskState.FAILED
            task.error = "Application restarted while task was in progress; it was not resumed."
            task.completed_at = utcnow()
            self.save_task(task)
        return len(rows)
