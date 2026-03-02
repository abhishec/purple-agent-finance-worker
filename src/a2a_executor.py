"""
A2A Executor for Purple Finance Agent.
Calls the real worker brain (run_worker) so the full FSM + ACE pipeline
runs during OfficeQA and other A2A assessments.
"""
import logging
import os
import sys

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import InvalidRequestError, Part, TaskState, TextPart, UnsupportedOperationError
from a2a.utils import get_message_text, new_agent_text_message, new_task
from a2a.utils.errors import ServerError

# Add /app (repo root) to sys.path so "src.xxx" imports inside worker_brain work.
_app_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _app_root not in sys.path:
    sys.path.insert(0, _app_root)

from src.worker_brain import run_worker  # noqa: E402

logger = logging.getLogger(__name__)

TERMINAL_STATES = {TaskState.completed, TaskState.canceled, TaskState.failed, TaskState.rejected}

# Default tools endpoint: container name used in docker-compose assessment network.
# Can be overridden via env var for local testing.
_DEFAULT_TOOLS_ENDPOINT = os.environ.get("TOOLS_ENDPOINT", "http://green-agent:9009")


class A2AExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        msg = context.message
        if not msg:
            raise ServerError(error=InvalidRequestError(message="Missing message"))

        task = context.current_task
        if task and task.status.state in TERMINAL_STATES:
            raise ServerError(error=InvalidRequestError(message="Task already in terminal state"))

        if not task:
            task = new_task(msg)
            await event_queue.enqueue_event(task)

        context_id = task.context_id
        task_id = task.id
        updater = TaskUpdater(event_queue, task_id, context_id)
        await updater.start_work()

        try:
            task_text = get_message_text(msg) or ""
            if not task_text:
                await updater.complete(
                    new_agent_text_message("No task text provided.",
                                          context_id=context_id, task_id=task_id)
                )
                return

            # Pull metadata the green agent may have sent (tools_endpoint, policy_doc, session_id)
            metadata: dict = {}
            if hasattr(task, "metadata") and isinstance(task.metadata, dict):
                metadata = task.metadata

            tools_endpoint = metadata.get("tools_endpoint", _DEFAULT_TOOLS_ENDPOINT)
            policy_doc = metadata.get("policy_doc", "")
            session_id = metadata.get("session_id", context_id or task_id)

            logger.info(
                f"[A2AExecutor] task={task_id} tools={tools_endpoint} text_len={len(task_text)}"
            )

            # Run the full worker brain — FSM synthesis, ACE, RL loop, tool calls
            answer = await run_worker(
                task_text=task_text,
                policy_doc=policy_doc,
                tools_endpoint=tools_endpoint,
                task_id=task_id,
                session_id=session_id,
            )

            await updater.add_artifact(
                parts=[Part(root=TextPart(text=answer))],
                name="result",
                artifact_id="result-001",
            )
            await updater.complete()

        except Exception as e:
            logger.error(f"[A2AExecutor] error: {e}", exc_info=True)
            await updater.failed(
                new_agent_text_message(
                    f"Agent error: {str(e)}",
                    context_id=context_id,
                    task_id=task_id,
                )
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise ServerError(error=UnsupportedOperationError())
