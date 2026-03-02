"""
A2A Executor for Purple Finance Agent.
Manages agent lifecycle and task execution.
"""
import logging
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import InvalidRequestError, TaskState, UnsupportedOperationError
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError

from a2a_agent import A2AAgent

logger = logging.getLogger(__name__)

TERMINAL_STATES = {TaskState.completed, TaskState.canceled, TaskState.failed, TaskState.rejected}


class A2AExecutor(AgentExecutor):
    def __init__(self):
        self.agents: dict[str, A2AAgent] = {}

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
        updater = TaskUpdater(event_queue, task.id, context_id)
        await updater.start_work()

        try:
            agent = self.agents.get(context_id)
            if not agent:
                agent = A2AAgent()
                self.agents[context_id] = agent
            await agent.run(msg, updater)
        except Exception as e:
            logger.error(f"Finance agent error: {e}", exc_info=True)
            await updater.failed(
                new_agent_text_message(
                    f"Agent error: {str(e)}",
                    context_id=context_id,
                    task_id=task.id
                )
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise ServerError(error=UnsupportedOperationError())
