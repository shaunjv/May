"""Single-process SQLite repository. Transactions contain no asynchronous work."""
import json
import re
import sqlite3
from pathlib import Path
from uuid import uuid4

from agent.desktop.models import utcnow


def redact(text: str) -> str:
    text = re.sub(r"(?i)(?:nvapi-|sk-)[A-Za-z0-9_\-]{12,}", "[redacted credential]", text)
    return re.sub(r"(?i)((?:api[_ -]?key|authorization|password|access_token|secret)\s*[:=]\s*)([^\s,;]+)", r"\1[redacted]", text)


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA journal_mode=WAL")
        version = self.db.execute("PRAGMA user_version").fetchone()[0]
        if version > 1:
            raise RuntimeError("This database needs a newer application version.")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY, title TEXT NOT NULL, workspace TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL REFERENCES conversations(id),
                role TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS messages_conversation ON messages(conversation_id);
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL REFERENCES conversations(id),
                state TEXT NOT NULL, data TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS tasks_conversation ON tasks(conversation_id);
            CREATE UNIQUE INDEX IF NOT EXISTS one_active_task ON tasks(conversation_id)
                WHERE state IN ('PLANNING','AWAITING_APPROVAL','EXECUTING');
            PRAGMA user_version=1;
        """)

    def recover(self):
        with self.db:
            for row in self.db.execute("SELECT data FROM tasks WHERE state IN ('PLANNING','EXECUTING')").fetchall():
                task = json.loads(row[0])
                task.update(state="FAILED", error="Application stopped during this operation. Review any file changes before retrying; nothing was resumed.", completed_at=utcnow())
                self._save(task)

    def conversations(self):
        return [dict(r) for r in self.db.execute("SELECT * FROM conversations ORDER BY created_at DESC")]

    def conversation(self, cid):
        row = self.db.execute("SELECT * FROM conversations WHERE id=?", (cid,)).fetchone()
        if row is None:
            raise KeyError("Conversation not found")
        result = dict(row)
        result["messages"] = [dict(r) for r in self.db.execute("SELECT * FROM messages WHERE conversation_id=? ORDER BY rowid", (cid,))]
        result["tasks"] = [json.loads(r[0]) for r in self.db.execute("SELECT data FROM tasks WHERE conversation_id=? ORDER BY rowid", (cid,))]
        return result

    def create(self, workspace):
        cid = str(uuid4())
        with self.db:
            self.db.execute("INSERT INTO conversations VALUES (?,?,?,?)", (cid, "New conversation", workspace, utcnow()))
        return self.conversation(cid)

    def message(self, cid, role, text):
        with self.db:
            self.db.execute("INSERT INTO messages VALUES (?,?,?,?,?)", (str(uuid4()), cid, role, redact(text), utcnow()))

    def begin(self, cid, text):
        self.conversation(cid)
        task = dict(id=str(uuid4()), conversation_id=cid, state="PLANNING", stage="Understanding your message", created_at=utcnow(), input=redact(text), cancel_requested=False)
        with self.db:
            self._save(task)
            self.db.execute("INSERT INTO messages VALUES (?,?,?,?,?)", (str(uuid4()), cid, "user", redact(text), utcnow()))
            self.db.execute("UPDATE conversations SET title=? WHERE id=? AND title='New conversation'", (redact(text)[:60], cid))
        return task

    def task(self, tid):
        row = self.db.execute("SELECT data FROM tasks WHERE id=?", (tid,)).fetchone()
        if row is None:
            raise KeyError("Task not found")
        return json.loads(row[0])

    def _save(self, task):
        self.db.execute("INSERT INTO tasks VALUES (?,?,?,?) ON CONFLICT(id) DO UPDATE SET state=excluded.state,data=excluded.data", (task["id"], task["conversation_id"], task["state"], json.dumps(task)))

    def save(self, task):
        with self.db:
            self._save(task)

    def close(self):
        self.db.close()
