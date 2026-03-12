from typing import Any
from vinagent.executor.guardrail import GuardrailExecutor
from vinagent.memory.history import InConversationHistory
from vinagent.memory.memory import Memory
from vinagent.graph.operator import FlowStateGraph
from vinagent.logger.logger import logger
from vinagent.executor.base import GraphExecutorBase


class GraphExecutor(GraphExecutorBase):
    def __init__(
        self,
        compiled_graph: FlowStateGraph,
        guardrail_executor: GuardrailExecutor = None,
        user_id: str = None,
        thread_id: str = None,
    ):
        self.compiled_graph = compiled_graph
        self.guardrail_executor = guardrail_executor
        self.user_id = user_id
        self.thread_id = thread_id

    def initialize_state(
        self, query: str, user_id: str, thread_id: str = "123", **kwargs
    ):
        input_state = (
            kwargs["input_state"]
            if "input_state" in kwargs
            else {"messages": [{"role": "user", "content": query}]}
        )

        config = (
            kwargs["config"]
            if "config" in kwargs
            else {
                "configurable": {"user_id": user_id},
                "thread_id": thread_id,
            }
        )
        return {"input": input_state, "config": config}

    def _invoke_compiled_graph(
        self,
        query: str,
        user_id: str,
        history: InConversationHistory,
        memory: Memory,
        is_save_memory: bool,
        **kwargs,
    ) -> Any:
        """
        Invoke the compiled LangGraph graph when a flow is defined.
        Handles state initialization, output guardrail, history, and memory saving.

        Returns:
            Any: The result from the compiled graph invocation.
        """
        thread_id = kwargs.get("thread_id", "123")
        input_state = self.initialize_state(
            query=query, user_id=user_id, thread_id=thread_id
        )

        try:
            result = self.compiled_graph.invoke(**input_state)
            self.guardrail_executor.check_output_guardrail(result)
            history.add_message(result)
            if memory and is_save_memory:
                memory.save_memory(message=result, user_id=self.user_id)
            return result
        except ValueError as e:
            logger.error(f"Error in compiled_graph.invoke: {e}")
            return None

    async def _invoke_compiled_graph_async(
        self,
        query: str,
        user_id: str,
        history: InConversationHistory,
        memory: Memory,
        is_save_memory: bool,
        **kwargs,
    ) -> Any:
        """
        Async variant of _invoke_compiled_graph.
        """
        thread_id = kwargs.get("thread_id", "123")
        input_state = self.initialize_state(
            query=query, user_id=user_id, thread_id=thread_id
        )

        try:
            result = await self.compiled_graph.ainvoke(**input_state)
            self.guardrail_executor.check_output_guardrail(result)
            history.add_message(result)
            if memory and is_save_memory:
                memory.save_memory(message=result, user_id=user_id)
            return result
        except ValueError as e:
            logger.error(f"Error in compiled_graph.ainvoke: {e}")
            return None
