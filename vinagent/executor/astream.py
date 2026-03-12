from typing import Union, AsyncGenerator
import asyncio
from langchain_core.messages import HumanMessage, AIMessage
from langchain_together import ChatTogether
from langchain_core.language_models.base import BaseLanguageModel
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_core.messages import ToolMessage
from langchain_core.messages.ai import AIMessageChunk
from vinagent.executor.base import AgentResponse
from vinagent.executor.base import MessageHandler
from vinagent.executor.base import AsyncStreamInvokeExecutorBase
from vinagent.logger.logger import logger
from vinagent.register.tool import ToolCall
from vinagent.executor.guardrail import GuardrailExecutor
from vinagent.register.tool import ToolManager
from vinagent.message.adapter import adapter_ai_response_with_tool_calls
from vinagent.prompt.agent_prompt import PromptHandler
from vinagent.memory.history import InConversationHistory
from vinagent.memory.memory import Memory
from vinagent.mcp.client import DistributedMCPClient


class AsyncStreamInvokeExecutor(
    AsyncStreamInvokeExecutorBase, MessageHandler, PromptHandler
):
    """
    Async streaming variant of the agent executor.

    Provides async generator methods for Steps 1 and 3 so that tokens are
    yielded to the caller (e.g. a WebSocket handler) as soon as they arrive
    from the LLM.  Step 2 (tool execution) is purely async I/O and does not
    stream tokens, so it is identical to ``AsyncInvokeExecutor``.
    """

    def __init__(
        self,
        llm: Union[ChatTogether, BaseLanguageModel, BaseChatOpenAI],
        guardrail_executor: GuardrailExecutor = None,
        *args,
        **kwargs,
    ):
        self.llm = llm
        self.guardrail_executor = guardrail_executor

    # ------------------------------------------------------------------
    # Step 1 — determine tool vs. direct answer (structured output, sync-safe)
    # ------------------------------------------------------------------

    async def define_tools_async(
        self,
        messages: list[Union[AIMessage, ToolMessage, HumanMessage]] = [],
        tools_manager: ToolManager = None,
    ) -> AgentResponse:
        """Invoke the LLM with structured output to get an AgentResponse."""
        messages = self._sanitize_history(messages)
        structured_llm = self.llm.with_structured_output(
            AgentResponse, method="function_calling"
        )
        try:
            response = await structured_llm.ainvoke(messages)
            if isinstance(response, AgentResponse) and isinstance(
                response.requires_tool, bool
            ):
                response = self._set_tool_metadata(response, tools_manager)
                return response
            raise ValueError("Invalid response structure")
        except Exception:
            # Fallback: plain LLM call + manual parsing
            raw = await self.llm.ainvoke(messages)
            content = raw.content if hasattr(raw, "content") else str(raw)
            return self._parse_agent_response(content, tools_manager)

    # ------------------------------------------------------------------
    # Step 1 — stream variant: yield raw tokens for direct answers
    # ------------------------------------------------------------------
    async def _step1_llm_define_tool_async(
        self,
        iteration: int = 1,
        max_history: int = 5,
        user_id: str = "unknown_user",
        message: str = "",
        tools_manager: ToolManager = None,
        memory: str = "",
        skills: list = [],
        description: str = "",
        instruction: str = "",
        history: InConversationHistory = None,
    ) -> AgentResponse:
        """
        Step 1: Build prompt and invoke LLM to get AgentResponse.
        On the first iteration, initializes conversation history with system + user messages.
        On subsequent iterations, appends the updated query as a new HumanMessage.
        """
        _history = self._preprocessing_messages(
            iteration=iteration,
            max_history=max_history,
            user_id=user_id,
            message=message,
            tools_manager=tools_manager,
            memory=memory,
            skills=skills,
            description=description,
            instruction=instruction,
            history=history,
        )
        return await self.define_tools_async(
            messages=_history, tools_manager=tools_manager
        )

    # ------------------------------------------------------------------
    # Step 2 — tool execution (no streaming, pure async I/O)
    # ------------------------------------------------------------------

    async def _step2_tool_invoke_async(
        self,
        current_query: str,
        response: AgentResponse,
        tools_manager: ToolManager,
        history: InConversationHistory,
        mcp_client: DistributedMCPClient,
        mcp_server_name: str,
    ) -> tuple[str, ToolMessage | None, bool]:
        """
        Async variant of Step 2: execute the tool call (or fix_bug_command)
        from the ``AgentResponse``.

        Returns:
            current_query (str): Updated query for the next iteration.
            tool_message (ToolMessage | None): Result of tool execution.
            should_continue (bool): True if the loop should proceed.
        """
        # --- 2a. Handle fix_bug_command ---
        fix_cmd = getattr(response, "fix_bug_command", None)
        if fix_cmd:
            next_query, fix_msg, should_continue = self._handle_fix_bug_command(
                fix_cmd=fix_cmd, query=current_query, response=response
            )
            return next_query, fix_msg, should_continue

        # --- 2b. No tool call — direct answer ---
        if not getattr(response, "requires_tool", False) or not getattr(
            response, "tool_call", None
        ):
            return current_query, None, False

        # --- 2c. Validate tool data ---
        tool_data = response.tool_call.model_dump()
        if not tool_data:
            logger.warning(
                "LLM generated empty or invalid tool_call. Prompting for correction."
            )
            history.add_message(AIMessage(content=str(response)))
            return (
                "Your response was invalid. You must provide either a valid `tool_call`, "
                "a valid `answer`, or a `fix_bug_command`.",
                None,
                True,
            )

        # --- 2d. Guardrail check ---
        logger.info(f"Executing async-stream tool call: {tool_data}")
        try:
            is_valid_tool_permission = self.guardrail_executor.check_tool_guardrail(
                llm=self.llm,
                tool_name=tool_data.get("tool_name"),
                user_input=current_query,
            )
        except Exception:
            is_valid_tool_permission = False

        # --- 2e. Wrap AIMessage with tool_calls metadata ---
        content_val = response.answer if getattr(response, "answer", None) else ""
        ai_message = adapter_ai_response_with_tool_calls(
            tools_manager.load_tools(), AIMessage(content=content_val), tool_data
        )
        history.add_message(ai_message)

        # --- 2f. Execute tool asynchronously ---
        if is_valid_tool_permission:
            tool_message = await tools_manager._execute_tool(
                tool_name=tool_data["tool_name"],
                tool_type=tool_data["tool_type"],
                arguments=tool_data["arguments"],
                module_path=tool_data["module_path"],
                mcp_client=mcp_client,
                mcp_server_name=mcp_server_name,
            )
            if tool_message is None:
                tool_message = ToolMessage(
                    content="Tool execution success without artifact",
                    additional_kwargs={"is_error": False},
                    tool_call_id=ai_message.tool_calls[0].get("id"),
                )
        else:
            tool_message = ToolMessage(
                content="Tool is not permitted by security rules.",
                additional_kwargs={"is_error": True},
                tool_call_id=ai_message.tool_calls[0].get("id"),
            )

        history.add_message(tool_message)
        _history = history.get_history()

        # --- 2g. Build next-iteration context ---
        next_query = self.prompt_tool(current_query, tool_data, tool_message, _history)
        return next_query, tool_message, True

    # ------------------------------------------------------------------
    # Step 3 — stream final LLM summarisation
    # ------------------------------------------------------------------

    async def _step3_final_response_astream(
        self,
        query: str,
        tool_message: ToolMessage | None,
        is_tool_formatted: bool,
        is_save_memory: bool,
        max_history: int = None,
        history: InConversationHistory = None,
        memory: Memory = None,
        user_id: str = None,
    ) -> AsyncGenerator[AIMessageChunk, None]:
        """
        Step 3 (async stream variant): Stream or yield the final response
        after the tool-calling loop ends.

        If ``is_tool_formatted=True``, asks the LLM to summarise tool
        results and yields chunks in real time via ``llm.astream()``.
        Otherwise yields the raw ``tool_message`` directly.

        Yields:
            AIMessageChunk | ToolMessage: Streamed chunks or raw tool message.
        """
        if is_tool_formatted:
            history.add_message(
                HumanMessage(
                    content=f"Based on the previous tool executions, please provide a final response to: {query}"
                )
            )
            _history = history.get_history(max_history=max_history)

            full_content = AIMessageChunk(content="")
            async for chunk in self.llm.astream(_history):
                full_content += chunk
                yield chunk

            history.add_message(full_content)
            if memory and is_save_memory:
                memory.save_memory(message=full_content.content, user_id=user_id)
        else:
            self.guardrail_executor.check_output_guardrail(tool_message)
            if memory and is_save_memory:
                content = (
                    tool_message.content
                    if hasattr(tool_message, "content")
                    else str(tool_message)
                )
                memory.save_memory(message=content, user_id=user_id)
            yield tool_message
