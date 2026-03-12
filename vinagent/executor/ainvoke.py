from typing import Union, List
import asyncio
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_together import ChatTogether
from langchain_core.language_models.base import BaseLanguageModel
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_core.messages import ToolMessage, BaseMessage
from vinagent.executor.base import AgentResponse
from vinagent.executor.base import MessageHandler
from vinagent.logger.logger import logger
from vinagent.register.tool import ToolCall
from vinagent.executor.guardrail import GuardrailExecutor
from vinagent.register.tool import ToolManager
from vinagent.message.adapter import adapter_ai_response_with_tool_calls
from vinagent.prompt.agent_prompt import PromptHandler
from vinagent.memory.history import InConversationHistory
from vinagent.memory.memory import Memory
from vinagent.mcp.client import DistributedMCPClient
from vinagent.executor.base import AsyncInvokeExecutorBase


class AsyncInvokeExecutor(AsyncInvokeExecutorBase, MessageHandler, PromptHandler):
    def __init__(
        self,
        llm: Union[ChatTogether, BaseLanguageModel, BaseChatOpenAI],
        guardrail_executor: GuardrailExecutor = None,
        *args,
        **kwargs,
    ):
        self.llm = llm
        self.guardrail_executor = guardrail_executor

    async def define_tools_async(
        self,
        messages: list[Union[AIMessage, ToolMessage, HumanMessage]] = [],
        tools_manager: ToolManager = None,
    ) -> AgentResponse:
        messages = self._sanitize_history(messages)
        self.structured_llm = self.llm.with_structured_output(
            AgentResponse, method="function_calling"
        )
        try:
            response = await self.structured_llm.ainvoke(messages)
            # Validate it's a proper AgentResponse, not a mis-packed string
            if isinstance(response, AgentResponse) and isinstance(
                response.requires_tool, bool
            ):
                response = self._set_tool_metadata(response, tools_manager)
                return response
            raise ValueError("Invalid response structure")
        except Exception:
            # Fallback: call plain LLM and parse manually
            raw = self.llm.invoke(messages)
            content = raw.content if hasattr(raw, "content") else str(raw)
            return self._parse_agent_response(content, tools_manager)

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
        Async variant of _step2_tool_invoke.
        Same logic but uses await for tool execution instead of asyncio.run().
        """
        # --- 2a. Handle fix_bug_command if present ---
        fix_cmd = getattr(response, "fix_bug_command", None)
        if fix_cmd:
            next_query, fix_msg, should_continue = self._handle_fix_bug_command(
                fix_cmd=fix_cmd, query=current_query, response=response
            )
            return next_query, fix_msg, should_continue

        # --- 2b. No tool call — agent has a direct answer ---
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

        # --- 2d. Check tool guardrail permission ---
        logger.info(f"Executing async tool call: {tool_data}")
        try:
            is_valid_tool_permission = self.guardrail_executor.check_tool_guardrail(
                llm=self.llm,
                tool_name=tool_data.get("tool_name"),
                user_input=current_query,
            )
        except Exception:
            is_valid_tool_permission = False

        # --- 2e. Adapt AIMessage to carry tool_calls metadata ---
        content_val = response.answer if getattr(response, "answer", None) else ""
        ai_message = adapter_ai_response_with_tool_calls(
            tools_manager.load_tools(), AIMessage(content=content_val), tool_data
        )
        history.add_message(ai_message)

        # --- 2f. Execute tool asynchronously ---
        if is_valid_tool_permission:
            tool_message = (
                await tools_manager._execute_tool(  # ← only difference from sync
                    tool_name=tool_data["tool_name"],
                    tool_type=tool_data["tool_type"],
                    arguments=tool_data["arguments"],
                    module_path=tool_data["module_path"],
                    mcp_client=mcp_client,
                    mcp_server_name=mcp_server_name,
                )
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
        # --- 2g. Build next iteration context ---
        next_query = self.prompt_tool(current_query, tool_data, tool_message, _history)
        return next_query, tool_message, True

    async def _step3_final_response_async(
        self,
        query: str,
        tool_message: ToolMessage | None,
        is_tool_formatted: bool,
        is_save_memory: bool,
        max_history: int = None,
        history: InConversationHistory = None,
        memory: Memory = None,
        user_id: str = None,
    ) -> AIMessage:
        """
        Async variant of _step3_final_response.
        Uses llm.ainvoke instead of llm.invoke for the summary call.
        """
        if is_tool_formatted:
            history.add_message(
                HumanMessage(
                    content=f"Based on the previous tool executions, please provide a final response to: {query}"
                )
            )
            history = history.get_history(max_history=max_history)
            final_message = await self.llm.ainvoke(
                history
            )  # ← only difference from sync
            history.add_message(final_message)
        else:
            final_message = tool_message

        self.guardrail_executor.check_output_guardrail(final_message)

        if memory and is_save_memory:
            final_content = (
                final_message.content
                if hasattr(final_message, "content")
                else str(final_message)
            )
            memory.save_memory(message=final_content, user_id=user_id)

        return (
            final_message
            if hasattr(final_message, "content")
            else AIMessage(content=final_message)
        )
