from abc import ABC, abstractmethod
from typing import Optional, Union, List, Any
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from vinagent.register.tool import ToolCall
from vinagent.memory.history import InConversationHistory
from vinagent.register.tool import ToolManager
from vinagent.logger.logger import logger
from vinagent.mcp.client import DistributedMCPClient
from vinagent.memory.memory import Memory
from vinagent.prompt.agent_prompt import PromptHandler


class AgentResponse(BaseModel):
    requires_tool: bool = Field(
        default=None,
        description=(
            "Set True if a tool is needed. "
            "IMPORTANT: If requires_tool=True, you MUST populate tool_call with a valid ToolCall object. "
            "Returning requires_tool=True with tool_call=None is INVALID."
        ),
    )
    answer: Optional[str] = Field(
        default="Sorry I can not answer this question",
        description=(
            "Directly provide the final answer when requires_tool=False."
            "This field is used for conversational responses that do not require a tool."
            "Do not return any tool JSON definition like."
            "{'tool_name': <tool_name>,''tool_type': <tool_type>,...}"
        ),
    )
    tool_call: Optional[ToolCall] = Field(
        default=None,
        description=(
            "The tool call object to be executed. "
            "MUST be populated when requires_tool=True. "
            "MUST be None when requires_tool=False."
        ),
    )
    fix_bug_command: Optional[str] = Field(
        default=None,
        description=(
            "A multi-line executable bash script to fix environment errors from the previous tool execution. "
            "May contain multiple commands separated by newlines or '&&', including package installation "
            "(e.g. pip install <lib>), system dependencies (e.g. apt-get install <pkg>), "
            "path configuration (e.g. export PATH=...), environment variables (e.g. export KEY=value), "
            "file downloads (e.g. wget/curl <url>), or any setup steps required for the tool to run successfully. "
            "Example:\n"
            "pip install pandas numpy\n"
            "apt-get install -y libgomp1\n"
            "export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH"
        ),
    )
    model_config = {"extra": "forbid"}


class MessageHandler(PromptHandler):
    def _sanitize_history(
        self,
        messages: list,
    ) -> list:
        """
        Ensure the message list is valid for OpenAI's chat API.

        Rules enforced:
        1. Every ToolMessage must be immediately preceded by an AIMessage
           that has a ``tool_calls`` list containing the matching id.
        2. The list must NOT start with a ToolMessage.

        Any ToolMessage that violates these rules — along with its preceding
        AIMessage (if it exists but has no tool_calls) — is quietly dropped so
        the request doesn't get a 400.
        """
        # Collect ids of tool_calls present in the history
        valid_tool_call_ids: set = set()
        for msg in messages:
            if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                for tc in msg.tool_calls:
                    tc_id = (
                        tc.get("id")
                        if isinstance(tc, dict)
                        else getattr(tc, "id", None)
                    )
                    if tc_id:
                        valid_tool_call_ids.add(tc_id)

        sanitized = []
        for msg in messages:
            if isinstance(msg, ToolMessage):
                # Only keep ToolMessages whose tool_call_id has a matching AIMessage
                tc_id = getattr(msg, "tool_call_id", None)
                if tc_id and tc_id in valid_tool_call_ids:
                    sanitized.append(msg)
                # else: silently drop this orphaned ToolMessage
            else:
                sanitized.append(msg)

        # Strip any leading ToolMessages (shouldn't remain after above, but be safe)
        while sanitized and isinstance(sanitized[0], ToolMessage):
            sanitized.pop(0)

        return sanitized

    def _run_fix_bug_command(self, fix_cmd: str, history: InConversationHistory):
        """Run a fix_bug_command script and persist env variable changes to os.environ."""
        import os as _os, subprocess as _subprocess, uuid as _uuid, re as _re

        logger.info(f"Resolve error by setup:\n{fix_cmd}")

        tool_call_id = "fix_" + str(_uuid.uuid4())[:8]

        # --- 1. Persist `export KEY=value` lines directly in os.environ ---
        # This is critical: subprocess.run() runs in a child shell that exits
        # immediately, so `export` commands inside it have no effect on the
        # parent process or subsequent subprocess calls. We extract them and
        # apply them to os.environ so subsequent tool invocations inherit them.
        export_pattern = _re.compile(
            r"^\s*export\s+([A-Za-z_][A-Za-z0-9_]*)=(.*)$", _re.MULTILINE
        )
        for key, value in export_pattern.findall(fix_cmd):
            # Strip surrounding quotes if any
            value = value.strip().strip('"').strip("'")
            # If value references existing vars like $PATH, expand them first
            expanded = _os.path.expandvars(value)
            _os.environ[key] = expanded

        # --- 2. Run the full command so install/download steps also execute ---
        try:
            _fix_res = _subprocess.run(
                fix_cmd,
                shell=True,
                capture_output=True,
                text=True,
                env=_os.environ.copy(),  # inherit any exports we just set
            )
            _fix_stdout = _fix_res.stdout.strip()
            _fix_stderr = _fix_res.stderr.strip()
            if _fix_res.returncode == 0:
                fix_artifact = (
                    _fix_stdout or "Fix executed successfully with no output."
                )
                fix_content = (
                    f"Applied fix_bug_command successfully.\nOutput: {fix_artifact}"
                )
                fix_is_err = False
            else:
                fix_artifact = f"STDOUT:\n{_fix_stdout}\nSTDERR:\n{_fix_stderr}"
                fix_content = (
                    f"fix_bug_command failed (returncode={_fix_res.returncode}).\n"
                    f"STDERR: {_fix_stderr}"
                )
                fix_is_err = True
        except Exception as _e:
            fix_content = f"Exception running fix_bug_command: {_e}"
            fix_artifact = None
            fix_is_err = True

        # Synthetic AIMessage with tool_calls must precede the ToolMessage
        synthetic_ai_msg = AIMessage(
            content="",
            tool_calls=[
                {
                    "id": tool_call_id,
                    "name": "fix_bug_command",
                    "args": {"command": fix_cmd},
                }
            ],
        )

        fix_msg = ToolMessage(
            content=fix_content,
            artifact=fix_artifact,
            tool_call_id=tool_call_id,
            additional_kwargs={"is_error": fix_is_err},
        )

        # Inject both into conversation history so the tool role has a valid preceding tool_calls
        history.add_message(synthetic_ai_msg)
        history.add_message(fix_msg)

        return fix_msg, history

    def _set_tool_metadata(self, response: AgentResponse, tools_manager: ToolManager):
        if response.requires_tool and response.tool_call:
            tool_name = response.tool_call.tool_name
            if tool_name:
                meta = tools_manager.get(tool_name, {})
                response.tool_call.return_type = meta.get("return", "str")
                response.tool_call.module_path = meta.get("module_path", "")
                response.tool_call.tool_type = meta.get("tool_type", "function")
                response.tool_call.tool_call_id = meta.get(
                    "tool_call_id", f"tool_{tool_name}"
                )
                response.tool_call.is_runtime = meta.get("is_runtime", False)
        return response

    def _parse_agent_response(self, content: str, tools_manager: ToolManager):
        # Attempt to parse the whole string as AgentResponse JSON
        import json, re

        match = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
        json_str = match.group(1) if match else content
        try:
            # logger.info(f"json_str: {json_str}")
            parsed = json.loads(json_str)
            if isinstance(parsed, dict) and (
                "requires_tool" in parsed or "tool_call" in parsed or "answer" in parsed
            ):
                tc = parsed.get("tool_call")
                if tc and isinstance(tc, str):
                    import ast

                    try:
                        tc = ast.literal_eval(tc)
                    except Exception:
                        pass

                if tc and isinstance(tc, dict):
                    tool_name = tc.get("tool_name")
                    if tool_name:
                        meta = tools_manager.get(tool_name, {})
                        tc.setdefault("return", meta.get("return", "str"))
                        tc.setdefault("module_path", meta.get("module_path", ""))
                        tc.setdefault("tool_type", meta.get("tool_type", "function"))
                        tc.setdefault(
                            "tool_call_id",
                            meta.get("tool_call_id", f"tool_{tool_name}"),
                        )
                        tc.setdefault("is_runtime", meta.get("is_runtime", False))
                    parsed["tool_call"] = ToolCall(**tc)
                elif "tool_call" in parsed:
                    del parsed["tool_call"]

                return AgentResponse(**parsed)
        except Exception as e:
            logger.debug(f"Could not parse as AgentResponse json: {e}")
            pass

        # If that fails, fallback to extracting just the tool call
        tool_data = tools_manager.extract_tool(content)
        logger.info(f"tool_data: {tool_data}")
        if tool_data:
            if isinstance(tool_data, str):
                try:
                    tool_data = json.loads(tool_data)
                except json.JSONDecodeError:
                    return AgentResponse(requires_tool=False, answer=content)
            if isinstance(tool_data, dict):
                tool_name = tool_data.get("tool_name")
                if tool_name:
                    registered = tools_manager.load_tools()
                    meta = registered.get(tool_name, {})
                    tool_data.setdefault("return", meta.get("return", "str"))
                    tool_data.setdefault("module_path", meta.get("module_path", ""))
                    tool_data.setdefault("tool_type", meta.get("tool_type", "function"))
                    tool_data.setdefault(
                        "tool_call_id", meta.get("tool_call_id", f"tool_{tool_name}")
                    )
                    tool_data.setdefault("is_runtime", meta.get("is_runtime", False))
                try:
                    return AgentResponse(
                        requires_tool=True, tool_call=ToolCall(**tool_data)
                    )
                except Exception as e:
                    import traceback

                    logger.error(
                        f"Error parsing AgentResponse from tool_data: {e}\n{traceback.format_exc()}"
                    )
                    return AgentResponse(requires_tool=False, answer=content)
        return AgentResponse(requires_tool=False, answer=content)

    def _preprocessing_messages(
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
        tools = tools_manager.load_tools()
        prompt = self.build_prompt(user_id, message, tools, memory)
        _system_prompt = self.system_prompt(skills, description, instruction)

        if iteration == 1:
            messages = [_system_prompt, HumanMessage(content=prompt)]
            history.add_messages(messages)
        else:
            history.add_message(HumanMessage(content=prompt))

        _history = history.get_history(max_history=max_history)
        return _history

    def _handle_fix_bug_command(
        self, fix_cmd: str, query: str, response: AgentResponse
    ):
        fix_msg = self._run_fix_bug_command(fix_cmd=fix_cmd)
        if not getattr(response, "requires_tool", False) or not getattr(
            response, "tool_call", None
        ):
            next_query = (
                f"Original task: {query}\n\n"
                f"I executed your fix_bug_command.\n"
                f"Result: {fix_msg.content}\n"
                f"Output: {fix_msg.artifact}\n\n"
                f"Please now RETRY the PREVIOUS tool call that failed. You MUST set "
                f"`requires_tool`: true and provide the exact `tool_call` so it can execute "
                f"in the newly repaired environment. DO NOT provide an `answer` explanation."
            )
            return next_query, fix_msg, True


class InvokeExecutorBase(ABC):
    @abstractmethod
    def define_tools(
        self,
        messages: list[Union[AIMessage, ToolMessage, HumanMessage]] = [],
        tools_manager: ToolManager = None,
    ) -> AgentResponse:
        pass

    @abstractmethod
    def _step1_llm_define_tool(
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
        pass

    @abstractmethod
    def _step2_tool_invoke(
        self,
        current_query: str,
        response: AgentResponse,
        tools_manager: ToolManager,
        history: InConversationHistory,
        mcp_client: DistributedMCPClient,
        mcp_server_name: str,
    ) -> tuple[str, ToolMessage | None, bool]:
        pass

    def _step3_final_response(
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
        pass


class AsyncInvokeExecutorBase(ABC):
    @abstractmethod
    async def _step1_llm_define_tool_async(
        self,
        current_query: str,
        iteration: int,
        max_history: int = None,
    ) -> AgentResponse:
        pass

    @abstractmethod
    async def _step2_tool_invoke_async(
        self,
        query: str,
        current_query: str,
        response: AgentResponse,
    ) -> tuple[str, ToolMessage | None, bool]:
        pass

    @abstractmethod
    async def _step3_final_response_async(
        self,
        query: str,
        tool_message: ToolMessage | None,
        is_tool_formatted: bool,
        is_save_memory: bool,
        max_history: int = None,
    ) -> AIMessage:
        pass


class StreamInvokeExecutorBase(ABC):
    @abstractmethod
    def define_tools(
        self,
        messages: list[Union[AIMessage, ToolMessage, HumanMessage]] = [],
        tools_manager: ToolManager = None,
    ) -> AgentResponse:
        pass

    @abstractmethod
    def _step1_llm_define_tool(
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
        pass

    @abstractmethod
    def _step2_tool_invoke(
        self,
        current_query: str,
        response: AgentResponse,
        tools_manager: ToolManager,
        history: InConversationHistory,
        mcp_client: DistributedMCPClient,
        mcp_server_name: str,
    ) -> tuple[str, ToolMessage | None, bool]:
        pass

    def _step3_final_response_stream(
        self,
        query: str,
        tool_message: ToolMessage | None,
        is_tool_formatted: bool,
        is_save_memory: bool,
        max_history: int = None,
        history: InConversationHistory = None,
        memory: Memory = None,
        user_id: str = None,
    ):
        pass


class AsyncStreamInvokeExecutorBase(ABC):
    @abstractmethod
    async def define_tools_async(
        self,
        messages: list[Union[AIMessage, ToolMessage, HumanMessage]] = [],
        tools_manager: ToolManager = None,
    ) -> AgentResponse:
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    async def _step2_tool_invoke_async(
        self,
        current_query: str,
        response: "AgentResponse",
        tools_manager: ToolManager,
        history: "InConversationHistory",
        mcp_client: Any,
        mcp_server_name: str,
    ) -> "tuple[str, Any, bool]":
        pass

    @abstractmethod
    async def _step3_final_response_astream(
        self,
        query: str,
        tool_message: Any,
        is_tool_formatted: bool,
        is_save_memory: bool,
        max_history: int = None,
        history: "InConversationHistory" = None,
        memory: "Memory" = None,
        user_id: str = None,
    ):
        """Async generator: stream final LLM summary chunks."""
        pass


class GraphExecutorBase(ABC):
    @abstractmethod
    def initialize_state(
        self, query: str, user_id: str, thread_id: str = "123", **kwargs
    ):
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass
