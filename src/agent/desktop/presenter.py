"""Thread-safe UI facade; widgets never call agent components directly."""
import asyncio
import threading
import uuid
from concurrent.futures import Future
from typing import Callable
from .runtime import DesktopRuntime
from agent.intent_manager.models import IntentType

class DesktopPresenter:
    def __init__(self, runtime: DesktopRuntime):
        self.runtime, self.conversation_id = runtime, str(uuid.uuid4())
        self.on_message: Callable[[str], None] = lambda _x: None
        self.on_status: Callable[[str], None] = lambda _x: None
        self.on_task: Callable[[object], None] = lambda _x: None
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
    def _serve(self):
        asyncio.set_event_loop(self._loop); self._loop.run_forever()
    def _run(self, coro) -> Future: return asyncio.run_coroutine_threadsafe(coro, self._loop)
    def submit(self, text: str) -> Future: return self._run(self._submit_message(text))
    def submit_chat(self, text: str) -> Future: return self.submit(text)
    async def _submit_message(self, text: str):
        """Route local-work requests automatically; users never choose a mode."""
        self.on_message(f"You: {text}")
        try:
            self.on_status("Understanding your request...")
            intent = await self.runtime.tasks.intent_manager.process_intent(text)
            if intent.primary_intent == IntentType.TASK:
                await self._prepare(text, echo=False)
            else:
                await self._chat(text, echo=False)
        except Exception as error:
            self.on_status(f"Request failed: {error}")
    async def _chat(self, text, echo=True):
        try:
            if echo: self.on_message(f"You: {text}")
            self.on_status("Thinking...")
            self.on_message(f"Agent: {await self.runtime.conversations.reply(text)}"); self.on_status("Ready")
        except Exception as error: self.on_status(f"Model unavailable: {error}")
    def prepare_task(self, text: str) -> Future: return self._run(self._prepare(text))
    async def _prepare(self, text, echo=True):
        if echo: self.on_message(f"You: {text}")
        self.on_status("Preparing plan...")
        task = await self.runtime.tasks.prepare(self.conversation_id, str(self.runtime.workspace), text)
        self.on_task(task); self.on_status(task.state.value)
    def approve(self, task_id: str) -> Future: return self._run(self._approve(task_id))
    async def _approve(self, task_id):
        self.on_status("Executing approved plan..."); task = await self.runtime.tasks.approve_and_execute(task_id)
        self.on_task(task); self.on_status(task.state.value)
    def reject(self, task_id: str) -> Future: return self._run(self._reject(task_id))
    async def _reject(self, task_id):
        task = await self.runtime.tasks.reject(task_id); self.on_task(task); self.on_status(task.state.value)
    def cancel(self, task_id: str) -> Future: return self._run(self._cancel(task_id))
    async def _cancel(self, task_id):
        task = await self.runtime.tasks.cancel(task_id); self.on_task(task); self.on_status(task.state.value)
    def close(self): self._loop.call_soon_threadsafe(self._loop.stop)
