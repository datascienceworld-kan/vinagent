from abc import ABC, abstractmethod
from typing import Any, Awaitable, List, AsyncGenerator, Generator, TypedDict, Optional, Union
from typing_extensions import is_typeddict
import asyncio
import json
import re
import yaml
from pydantic import BaseModel, Field
from pathlib import Path
from langchain_together import ChatTogether
from langchain_core.language_models.base import BaseLanguageModel
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.messages.tool import ToolMessage
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver
import logging
import mlflow
from mlflow.entities import SpanType
from vinagent.register.tool import ToolManager
from vinagent.memory.memory import Memory
from vinagent.memory.history import InConversationHistory
from vinagent.mcp.client import DistributedMCPClient
from vinagent.graph.function_graph import FunctionStateGraph
from vinagent.graph.operator import FlowStateGraph
from vinagent.oauth2.client import AuthenCard
from vinagent.guardrail import (
    GuardRailBase,
    GuardrailDecision,
    OutputGuardrailDecision,
    GuardrailManager,
)
from vinagent.register.tool import ToolCall
from vinagent.prompt.agent_prompt import PromptHandler
from vinagent.message.adapter import adapter_ai_response_with_tool_calls
from vinagent.executor.guardrail import GuardrailExecutor
from vinagent.executor.graph_executor import GraphExecutor
from vinagent.executor.invoke import InvokeExecutor
from vinagent.executor.ainvoke import AsyncInvokeExecutor
from vinagent.executor.stream import StreamInvokeExecutor
from vinagent.executor.astream import AsyncStreamInvokeExecutor
from vinagent.executor.base import AgentResponse

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def is_jupyter_notebook():
    try:
        from IPython import get_ipython

        ipython = get_ipython()
        if ipython is None:
            return False
        # Check if it's a Jupyter Notebook (ZMQInteractiveShell is used in Jupyter)
        return "ZMQInteractiveShell" in str(type(ipython))
    except ImportError:
        return False


if is_jupyter_notebook():
    import nest_asyncio

    nest_asyncio.apply()


class AgentMeta(ABC):
    """Abstract base class for agents"""

    @abstractmethod
    def __init__(
        self,
        llm: Union[ChatTogether, BaseLanguageModel, BaseChatOpenAI],
        tools: List[Union[str, BaseTool]] = [],
        *args,
        **kwargs,
    ):
        """Initialize a new Agent with LLM and tools"""
        pass

    @abstractmethod
    def invoke(self, query: str, *args, **kwargs) -> Any:
        """Synchronously invoke the agent's main function"""
        pass

    @abstractmethod
    async def ainvoke(self, query: str, *args, **kwargs) -> Awaitable[Any]:
        """Asynchronously invoke the agent's main function"""
        pass

    @abstractmethod
    def stream(self, query: str, *args, **kwargs) -> Generator[Any, None, None]:
        """Streaming the agent's main function"""
        pass

    @abstractmethod
    async def astream(self, query: str, *args, **kwargs) -> AsyncGenerator[Any, None]:
        """Asynchronously stream the agent's main function"""
        pass


class Agent(AgentMeta):
    """The Agent class is a concrete implementation of an AI agent with tool-calling capabilities, inheriting from AgentMeta. It integrates a language model, tools, memory, and flow management to process queries, execute tools, and maintain conversational context."""

    @classmethod
    def load_agent(
        cls,
        agent_paths: Union[List[Union[str, Path]], Union[str, Path], None] = None,
        llm: Union["ChatTogether", "BaseLanguageModel", "BaseChatOpenAI", None] = None,
        **kwargs,
    ) -> "Agent":
        """
        Load an Agent from one or more AgentSkill directories (Anthropic structure).

        The directory must follow the layout::

            <agent_path>/
            ├── SKILL.md      # YAML front-matter + markdown usage guidance
            └── scripts/      # Shell scripts / Python helpers invoked by the agent

        **SKILL.md structure**::

            ---
            name: <skill_name>
            description: "<short description shown to the user>"
            license: ...
            ---

            # <Skill title>

            <Full markdown body — examples, shell commands, workflow steps.>
            This becomes the agent's `instruction` AND the tool's `docstring`
            so the LLM knows exactly which commands to construct.

        If `agent_paths` is provided, the metadata, descriptions, instructions,
        and tools are merged into a single capable agent.

        Args:
            agent_paths (List[Union[str, Path]], optional): List of paths to multiple agentskills.
            llm: Language model instance.
            **kwargs: Additional keyword arguments forwarded to :class:`Agent`.

        Returns:
            Agent: A fully initialised Agent with the skills registered as tools.

        Raises:
            FileNotFoundError: If ``SKILL.md`` is not found in a provided path.
            ValueError: If the YAML front-matter cannot be parsed or no paths are given.
        """
        paths_to_load = []
        if agent_paths:
            if isinstance(agent_paths, list):
                paths_to_load.extend(agent_paths)
            else:
                paths_to_load.append(agent_paths)

        if not paths_to_load:
            raise ValueError("Must provide at least one `agent_paths`.")

        names = []
        descriptions = []
        instructions = []
        tools_list = []

        for p in paths_to_load:
            agent_dir = Path(p).resolve()
            skill_file = agent_dir / "SKILL.md"
            if not skill_file.exists():
                raise FileNotFoundError(f"SKILL.md not found in {agent_dir}")

            content = skill_file.read_text(encoding="utf-8")

            # --- Parse YAML front-matter ---
            match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", content, re.DOTALL)
            if not match:
                # No front-matter: treat the whole file as instruction
                instructions.append(
                    f"=== Instructions for {agent_dir.name} ===\n{content.strip()}"
                )
                tools_list.append(str(agent_dir))
                names.append(agent_dir.name)
                continue

            frontmatter_text, markdown_body = match.groups()
            try:
                metadata = yaml.safe_load(frontmatter_text) or {}
            except yaml.YAMLError as e:
                raise ValueError(
                    f"Failed to parse YAML front-matter in {skill_file}: {e}"
                )

            name = metadata.get("name", agent_dir.name)
            desc = metadata.get(
                "description",
                "You are a helpful assistant who can use the following tools to complete a task.",
            )

            names.append(name)
            descriptions.append(f"{name}: {desc}")

            inst = markdown_body.strip()
            if inst:
                instructions.append(f"=== Instructions for {name} ===\n{inst}")

            tools_list.append(str(agent_dir))

        # Join the collected metadata
        final_name = "_".join(names) if names else "MultiSkillAgent"

        if len(descriptions) == 1:
            final_description = descriptions[0].split(": ", 1)[-1]
        elif descriptions:
            final_description = (
                "You are a helpful assistant with these skills:\n"
                + "\n".join(f"- {d}" for d in descriptions)
            )
        else:
            final_description = "You are a helpful assistant who can use the following tools to complete a task."

        if len(instructions) == 1:
            # Strip the added header for backwards compatibility on single tools
            idx = instructions[0].find("===\n")
            if idx != -1:
                final_instruction = instructions[0][idx + 4 :].strip()
            else:
                final_instruction = instructions[0]
        else:
            final_instruction = (
                "\n\n".join(instructions)
                if instructions
                else (
                    "You should answer the question relying on your determined skills."
                )
            )

        return cls(
            name=final_name,
            llm=llm,
            tools=tools_list,
            tools_path=Path("templates/tools.json"),
            is_reset_tools=True,
            description=final_description,
            instruction=final_instruction,
            **kwargs,
        )

    def __init__(
        self,
        llm: Union[ChatTogether, BaseLanguageModel, BaseChatOpenAI],
        name: str = "default_name",
        tools: List[Union[str, BaseTool]] = [],
        tools_path: Path = Path("templates/tools.json"),
        is_reset_tools=False,
        description: str = "You are a helpful assistant who can use the following tools to complete a task.",
        instruction: str = "You should answer the question relying on your determined skills",
        skills: list[str] = ["You can answer the user question with tools"],
        flow: list[str] = [],
        state_schema: type[Any] = None,
        config_schema: type[Any] = None,
        thread_id: str = "123",
        memory_path: Path = None,
        is_reset_memory: bool = False,
        num_buffered_messages: int = 10,
        mcp_client: DistributedMCPClient = None,
        mcp_server_name: str = None,
        is_pii: bool = False,
        authen_card: AuthenCard = None,
        input_guardrail: GuardrailDecision = None,
        output_guardrail: OutputGuardrailDecision = None,
        guardrail_manager: GuardrailManager = None,
        *args,
        **kwargs,
    ):
        """
        Initialize the agent with a language model, a list of tools, a description, and a set of skills.
        Parameters:
        ----------
        name : str = "default_name"
            Name of agent
        llm : Union[ChatTogether, BaseLanguageModel, BaseChatOpenAI]
            An instance of a language model used by the agent to process and generate responses.

        tools : List, optional
            A list of tools that the agent can utilize when performing tasks. Defaults to an empty list.

        tools_path: Path, optional
            The path to the file containing the tools. Defaults to a template file.

        description : str, optional
            A brief description of the assistant's capabilities. Defaults to a general helpful assistant message.

        instruction: str, optional
            An instruction for the assistant to follow. Defaults to an empty string.

        skills : list[str], optional
            A list of skills or abilities describing what the assistant can do. Defaults to a basic tool-usage skill.

        flow: list[str], optional
            A list of routes in the graph that defines start_node >> end_node. Defaults empty.

        state_schema: type[Any] = None,
            Only define if flow exists. It setups a state storage for agent workflow.
            This will save all intermediate messages which is returned at each node to access them after the graph execution is finished.
            By default, the state_schema is langgraph.graph.MessageState.

        config_schema: type[Any] = None,
            A schema for the config of the graph. Defaults to None.
             Define a config schema which pass before the workflow triggering to manage the thread_id and user_id the agent is chatting with.
            class ConfigSchema(TypedDict):
                user_id: str

        thread_id: str = None,
            A thread id for the graph. Defaults to 123.

        is_reset_tools : bool, optional
            A flag indicating whether the agent should override its existing tools with the provided list of tools. Defaults to False.

        memory_path : Path, optional
            The path to the file containing the graph memory. Defaults to a template file. Only valid if memory is not None.

        is_reset_memory : bool, optional
            A flag indicating whether the agent should reset its graph memory when re-initializes it's memory. Defaults to False. Only valid if memory is not None.

        num_buffered_messages: int
            An buffered memory, which is not stored to memory, just existed in a runtime conversation. Default is a list of last 10 messages.

        mcp_client : DistributedMCPClient, optional
            An instance of a DistributedMCPClient used to register tools with the memory. Defaults to None.

        mcp_name: str, optional
            The name of the memory server. Defaults to None.

        is_pii: bool, optional
            A flag indicating whether the assistant should be able to recognize a person who is chatting with. Defaults to False.

        authen_card: AuthenCard, optional
            An instance of AuthenCard used to authenticate the assistant. Defaults to None.

        input_guardrail: GuardrailDecision, optional
            An instance of GuardrailDecision used to decide whether the input is safe. Defaults to None.

        output_guardrail: OutputGuardrailDecision, optional
            An instance of OutputGuardrailDecision used to decide whether the output is safe. Defaults to None.

        guardrail_manager: GuardrailManager, optional
            An instance of GuardrailManager used to manage the guardrails. Defaults to None.

        *args, **kwargs : Any
            Additional arguments passed to the superclass or future extensions.
        """
        # Initialize Agent llm and tools
        self.name = name
        self.llm = llm
        self.tools = tools
        self.description = description
        self.instruction = instruction
        self.skills = skills

        # Initialize Agent flow by Langgraph
        self.compiled_graph = None
        self.flow = flow
        if self.flow:
            self.initialize_flow(state_schema=state_schema, config_schema=config_schema)

        # Initialize Tools
        self.tools_path = None
        if tools_path:
            self.tools_path = (
                Path(tools_path) if isinstance(tools_path, str) else tools_path
            )
        else:
            self.tools_path = Path("templates/tools.json")
        if self.tools_path and (self.tools_path.suffix != ".json"):
            raise ValueError(
                "tools_path must be json format ending with .json. For example, 'templates/tools.json'"
            )
        self.tools_path.parent.mkdir(parents=True, exist_ok=True)
        self.is_reset_tools = is_reset_tools
        self.tools_manager = ToolManager(
            llm=self.llm, tools_path=self.tools_path, is_reset_tools=self.is_reset_tools
        )
        self.register_tools(self.tools)
        self.mcp_client = mcp_client
        self.mcp_server_name = mcp_server_name

        # Initialize memory
        self.memory_path = (
            Path(memory_path) if isinstance(memory_path, str) else memory_path
        )
        if self.memory_path and (self.memory_path.suffix not in [".json", ".jsonl"]):
            raise ValueError(
                "memory_path must be json format ending with .json or .jsonl. For example, 'templates/memory.json'"
            )
        self.is_reset_memory = is_reset_memory
        self.memory = None
        if self.memory_path:
            self.memory = Memory(
                memory_path=self.memory_path, is_reset_memory=self.is_reset_memory
            )
        self.in_conversation_history = InConversationHistory(
            messages=[], max_length=num_buffered_messages
        )

        # Identify user
        self.is_pii = is_pii
        self._user_id = None
        if not self.is_pii:
            self._user_id = "unknown_user"
        self._thread_id = thread_id
        # OAuth2 authentication if enabled
        self.authen_card = authen_card

        self.input_guardrail = input_guardrail
        self.output_guardrail = output_guardrail
        self.guardrail_manager = guardrail_manager
        self.guardrail_executor = GuardrailExecutor(
            input_guardrail=self.input_guardrail,
            output_guardrail=self.output_guardrail,
            guardrail_manager=self.guardrail_manager,
        )
        self.invoke_executor = InvokeExecutor(
            llm=self.llm, guardrail_executor=self.guardrail_executor
        )
        self.async_invoke_executor = AsyncInvokeExecutor(
            llm=self.llm, guardrail_executor=self.guardrail_executor
        )
        self.graph_executor = GraphExecutor(
            compiled_graph=self.compiled_graph,
            guardrail_executor=self.guardrail_executor,
            user_id=self._user_id,
            thread_id=self._thread_id,
        )
        self.stream_invoke_executor = StreamInvokeExecutor(
            llm=self.llm, guardrail_executor=self.guardrail_executor
        )
        self.async_stream_invoke_executor = AsyncStreamInvokeExecutor(
            llm=self.llm, guardrail_executor=self.guardrail_executor
        )

    def authenticate(self):
        if self.authen_card is None:
            logger.info("No authentication card provided, skipping authentication")
            return True

        is_enable_access = self.authen_card.verify_access_token()
        if is_enable_access:
            logger.info(f"Successfully authenticated!")
        else:
            logger.info(f"Authentication failed!")
            raise Exception("Authentication failed!")
        return is_enable_access

    async def connect_mcp_tool(self):
        logger.info(f"{self.mcp_client}: {self.mcp_server_name}")
        if self.mcp_client and self.mcp_server_name:
            mcp_tools = await self.tools_manager.register_mcp_tool(
                self.mcp_client, self.mcp_server_name
            )
            logger.info(f"Successfully connected to mcp server {self.mcp_server_name}!")
        elif self.mcp_client:
            mcp_tools = await self.tools_manager.register_mcp_tool(self.mcp_client)
            logger.info(f"Successfully connected to mcp server!")
        return f"Successfully connected to mcp server! mcp_tools: {mcp_tools}"

    def initialize_flow(self, state_schema: TypedDict, config_schema: TypedDict):
        # Validate state_schema if provided
        if state_schema is not None and not is_typeddict(state_schema):
            raise TypeError("state_schema must be a TypedDict subclass")

        # Validate config_schema if provided
        if config_schema is not None and not is_typeddict(config_schema):
            raise TypeError("config_schema must be a TypedDict subclass")

        self.graph = (
            FunctionStateGraph(state_schema=state_schema, config_schema=config_schema)
            if isinstance(self.flow, FunctionStateGraph)
            else FlowStateGraph(state_schema=state_schema, config_schema=config_schema)
        )

        self.checkpoint = MemorySaver()
        self.compiled_graph = self.graph.compile(
            checkpointer=self.checkpoint, flow=self.flow
        )

    def register_tools(self, tools: List[str]) -> Any:
        """
        Register a list of tools.

        Automatically detects whether each entry is an AgentSkill directory
        (contains a ``SKILL.md`` file) and routes it to
        :meth:`~vinagent.register.tool.ToolManager.register_agentskill_tool`.
        All other paths are treated as regular Python modules and forwarded to
        :meth:`~vinagent.register.tool.ToolManager.register_module_tool`.
        """
        for tool in tools:
            tool_path = Path(tool)
            if tool_path.is_dir() and (tool_path / "SKILL.md").exists():
                self.tools_manager.register_agentskill_tool(tool)
            else:
                self.tools_manager.register_module_tool(tool)

    @property
    def user_id(self):
        return self._user_id

    @user_id.setter
    def user_id(self, new_user_id):
        self._user_id = new_user_id

    def invoke(
        self,
        query: str,
        is_save_memory: bool = False,
        user_id: str = "unknown_user",
        max_iterations: int = 10,
        is_tool_formatted: bool = True,
        max_history: int = None,
        **kwargs,
    ) -> Any:
        # --- Auth & user setup ---
        self.authenticate()
        self._user_id = user_id or self._user_id
        logger.info(f"I am chatting with {self._user_id}")

        if self.memory and is_save_memory:
            self.save_memory(query, user_id=self._user_id)

        self.guardrail_executor.check_input_guardrail(query)

        # --- Compiled graph path ---
        if getattr(self, "compiled_graph", None):
            return self.graph_executor._invoke_compiled_graph(
                query=query,
                user_id=user_id,
                history=self.in_conversation_history,
                memory=self.memory,
                is_save_memory=is_save_memory,
                **kwargs,
            )

        # --- Tool calling loop ---
        current_query = query
        tool_message = None

        for iteration in range(1, max_iterations + 1):
            logger.info(f"Tool calling iteration {iteration}/{max_iterations}")

            # Step 1: LLM invoke
            response = self.invoke_executor._step1_llm_define_tool(
                iteration=iteration,
                max_history=max_history,
                user_id=user_id,
                message=current_query,
                tools_manager=self.tools_manager,
                memory=self.memory,
                skills=self.skills,
                description=self.description,
                instruction=self.instruction,
                history=self.in_conversation_history,
            )

            # Early exit: direct answer with no tool needed
            if not getattr(response, "requires_tool", False) and getattr(
                response, "answer", None
            ):
                answer = response.answer
                logger.info(
                    f"No more tool calls needed. Completed in {iteration} iterations."
                )
                self.guardrail_executor.check_output_guardrail(answer)
                if self.memory and is_save_memory:
                    self.save_memory(message=answer, user_id=self._user_id)
                return AIMessage(content=answer)

            # Step 2: Tool invoke
            current_query, tool_message, should_continue = (
                self.invoke_executor._step2_tool_invoke(
                    current_query=current_query,
                    response=response,
                    tools_manager=self.tools_manager,
                    history=self.in_conversation_history,
                    mcp_client=self.mcp_client,
                    mcp_server_name=self.mcp_server_name,
                )
            )

            if not should_continue:
                # Step 2 found a direct answer (no tool needed)
                answer = getattr(response, "answer", None) or ""
                self.guardrail_executor.check_output_guardrail(answer)
                if self.memory and is_save_memory:
                    self.save_memory(message=answer, user_id=self._user_id)
                return AIMessage(content=answer)

        # Step 3: Max iterations reached — format final response
        logger.warning(
            f"Reached maximum iterations ({max_iterations}). Stopping tool calling loop."
        )
        return self.invoke_executor._step3_final_response(
            query=current_query,
            tool_message=tool_message,
            is_tool_formatted=is_tool_formatted,
            is_save_memory=is_save_memory,
            max_history=max_history,
            history=self.in_conversation_history,
            memory=self.memory,
            user_id=user_id,
        )

    async def ainvoke(
        self,
        query: str,
        is_save_memory: bool = False,
        user_id: str = "unknown_user",
        max_iterations: int = 10,
        is_tool_formatted: bool = True,
        max_history: int = None,
        **kwargs,
    ) -> Awaitable[Any]:
        # --- Auth & user setup ---
        self.authenticate()
        self._user_id = user_id or self._user_id
        logger.info(f"I am chatting with {self._user_id}")

        if self.memory and is_save_memory:
            self.save_memory(query, user_id=self._user_id)

        self.guardrail_executor.check_input_guardrail(query)

        # --- Compiled graph path ---
        if getattr(self, "compiled_graph", None):
            return await self.graph_executor._invoke_compiled_graph_async(
                query=query,
                user_id=user_id,
                history=self.in_conversation_history,
                memory=self.memory,
                is_save_memory=is_save_memory,
                **kwargs,
            )

        # --- Tool calling loop ---
        current_query = query
        tool_message = None

        for iteration in range(1, max_iterations + 1):
            logger.info(f"Async tool calling iteration {iteration}/{max_iterations}")

            # Step 1: LLM invoke (reused directly — sync, no await needed)
            response = await self.async_invoke_executor._step1_llm_define_tool_async(
                iteration=iteration,
                max_history=max_history,
                user_id=user_id,
                message=current_query,
                tools_manager=self.tools_manager,
                memory=self.memory,
                skills=self.skills,
                description=self.description,
                instruction=self.instruction,
                history=self.in_conversation_history,
            )

            # Early exit: direct answer with no tool needed
            if not getattr(response, "requires_tool", False) and getattr(
                response, "answer", None
            ):
                answer = response.answer
                logger.info(
                    f"No more tool calls needed. Completed in {iteration} iterations."
                )
                self.guardrail_executor.check_output_guardrail(answer)
                if self.memory and is_save_memory:
                    self.save_memory(message=answer, user_id=self._user_id)
                return AIMessage(content=answer)

            # Step 2: Tool invoke (async variant — await replaces asyncio.run)
            current_query, tool_message, should_continue = (
                await self.async_invoke_executor._step2_tool_invoke_async(
                    current_query=current_query,
                    response=response,
                    tools_manager=self.tools_manager,
                    history=self.in_conversation_history,
                    mcp_client=self.mcp_client,
                    mcp_server_name=self.mcp_server_name,
                )
            )

            if not should_continue:
                answer = getattr(response, "answer", None) or ""
                self.guardrail_executor.check_output_guardrail(answer)
                if self.memory and is_save_memory:
                    self.save_memory(message=answer, user_id=self._user_id)
                return AIMessage(content=answer)

        # Step 3: Max iterations reached — format final response (async variant)
        logger.warning(
            f"Reached maximum iterations ({max_iterations}). Stopping async tool calling loop."
        )
        return await self.async_invoke_executor._step3_final_response_async(
            query=current_query,
            tool_message=tool_message,
            is_tool_formatted=is_tool_formatted,
            is_save_memory=is_save_memory,
            max_history=max_history,
            history=self.in_conversation_history,
            memory=self.memory,
            user_id=user_id,
        )

    def stream(
        self,
        query: str,
        is_save_memory: bool = False,
        user_id: str = "unknown_user",
        max_iterations: int = 10,
        is_tool_formatted: bool = True,
        max_history: int = None,
        **kwargs,
    ) -> Generator[Any, None, None]:
        """
        Stream the agent response token-by-token with continuous tool-calling
        capability.  Follows the same 3-step loop as :meth:`invoke` but yields
        ``AIMessageChunk`` objects so the caller can push tokens to a live
        connection (WebSocket, SSE, etc.) as soon as they arrive.

        Workflow
        --------
        1. **Step 1** – Call the LLM with structured output to determine
           whether a tool is needed (``AgentResponse``).
           - If *no tool needed*: stream the direct answer token-by-token
             and return.
        2. **Step 2** – Execute the tool synchronously (same as ``invoke``).
        3. **Step 3** – After all tool iterations, stream the final LLM
           summary token-by-token.

        Args:
            query (str): User query.
            is_save_memory (bool): Save conversation to long-term memory.
            user_id (str): Identifier for the current user.
            max_iterations (int): Maximum tool-calling iterations.
            is_tool_formatted (bool): If True, streams a final LLM summary
                after tool execution; if False, yields the raw ToolMessage.
            max_history (int): Number of history messages to include.
            **kwargs: Forwarded to the compiled-graph path when applicable.

        Yields:
            AIMessageChunk | AIMessage | ToolMessage: Streamed LLM tokens
            or the final tool/LLM message.
        """
        # --- Auth & user setup ---
        self.authenticate()
        self._user_id = user_id or self._user_id
        logger.info(f"I am chatting with {self._user_id}")

        if self.memory and is_save_memory:
            self.save_memory(query, user_id=self._user_id)

        self.guardrail_executor.check_input_guardrail(query)

        try:
            # --- Compiled graph path ---
            if (
                getattr(self, "compiled_graph", None)
                and self.compiled_graph is not None
            ):
                result = []
                thread_id = kwargs.get("thread_id", self._thread_id)
                input_state = self.graph_executor.initialize_state(
                    query=query, user_id=user_id, thread_id=thread_id
                )
                for chunk in self.compiled_graph.stream(**input_state):
                    for v in chunk.values():
                        if v:
                            result += v["messages"]
                            yield v
                self.guardrail_executor.check_output_guardrail(result)
                if self.memory and is_save_memory:
                    self.save_memory(message=result, user_id=self._user_id)
                yield result
                return

            # --- Tool calling loop ---
            current_query = query
            tool_message = None

            for iteration in range(1, max_iterations + 1):
                logger.info(
                    f"Streaming tool calling iteration {iteration}/{max_iterations}"
                )

                # Step 1: Ask LLM (structured output) whether a tool is needed
                response = self.stream_invoke_executor._step1_llm_define_tool(
                    iteration=iteration,
                    max_history=max_history,
                    user_id=user_id,
                    message=current_query,
                    tools_manager=self.tools_manager,
                    memory=self.memory,
                    skills=self.skills,
                    description=self.description,
                    instruction=self.instruction,
                    history=self.in_conversation_history,
                )
                logger.info(f"response: {response}")
                # Early exit: direct answer — stream it token-by-token
                if not getattr(response, "requires_tool", False) and getattr(
                    response, "answer", None
                ):
                    logger.info(
                        f"No tool needed. Streaming direct answer (iteration {iteration})."
                    )
                    answer_text = response.answer
                    for chunk in answer_text:
                        chunk = AIMessageChunk(content=chunk)
                        yield chunk
                    self.guardrail_executor.check_output_guardrail(answer_text)
                    full_chunk = AIMessage(content=answer_text)
                    self.in_conversation_history.add_message(full_chunk)
                    # Stream the direct answer chunk-by-chunk via a short llm.stream call
                    # Re-add message to history so the next prompt is correct
                    if self.memory and is_save_memory:
                        self.save_memory(
                            message=full_chunk.content, user_id=self._user_id
                        )
                    if answer_text:
                        try:
                            tool_call = json.loads(answer_text)
                            logger.info(
                                "Detected JSON tool call in answer, preparing next iteration."
                            )
                            current_query = AIMessage(
                                content=f"Let's call tool: {json.dumps(tool_call)}"
                            )
                            continue

                        except json.JSONDecodeError:
                            return

                # Step 2: Execute the tool (synchronous — no streaming here)
                logging.info(f"current_query: {current_query}")
                current_query, tool_message, should_continue = (
                    self.stream_invoke_executor._step2_tool_invoke(
                        current_query=current_query,
                        response=response,
                        tools_manager=self.tools_manager,
                        history=self.in_conversation_history,
                        mcp_client=self.mcp_client,
                        mcp_server_name=self.mcp_server_name,
                    )
                )

                if not should_continue:
                    # No tool was executed — yield the direct answer
                    answer_text = getattr(response, "answer", None) or ""
                    self.guardrail_executor.check_output_guardrail(answer_text)
                    if self.memory and is_save_memory:
                        self.save_memory(message=answer_text, user_id=self._user_id)
                    yield AIMessage(content=answer_text)
                    return

            # Step 3: Max iterations — stream the final LLM summary
            logger.warning(
                f"Reached maximum iterations ({max_iterations}). Stopping streaming loop."
            )
            yield from self.stream_invoke_executor._step3_final_response_stream(
                query=current_query,
                tool_message=tool_message,
                is_tool_formatted=is_tool_formatted,
                is_save_memory=is_save_memory,
                max_history=max_history,
                history=self.in_conversation_history,
                memory=self.memory,
                user_id=user_id,
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"An error occurred during streaming: {str(e)}")
            yield AIMessage(content=f"An error occurred: {str(e)}")

    async def astream(
        self,
        query: str,
        is_save_memory: bool = False,
        user_id: str = "unknown_user",
        max_iterations: int = 10,
        is_tool_formatted: bool = True,
        max_history: int = None,
        **kwargs,
    ) -> AsyncGenerator[Any, None]:
        """
        Asynchronously stream the agent response token-by-token with
        continuous tool-calling capability.  Mirrors :meth:`stream` but uses
        ``async for`` / ``await`` throughout so it can run inside an existing
        async event loop (FastAPI, Starlette, etc.).

        Workflow
        --------
        1. **Step 1** – Async structured-output call to determine tool need.
           - If *no tool needed*: async-stream direct answer, then return.
        2. **Step 2** – Async tool execution (``await``).
        3. **Step 3** – Async-stream the final LLM summary.

        Args:
            query (str): User query.
            is_save_memory (bool): Save conversation to long-term memory.
            user_id (str): Identifier for the current user.
            max_iterations (int): Maximum tool-calling iterations.
            is_tool_formatted (bool): Stream a final LLM summary (True) or
                yield the raw ToolMessage (False) after tool execution.
            max_history (int): Number of history messages to include.
            **kwargs: Forwarded to the compiled-graph path when applicable.

        Yields:
            AIMessageChunk | AIMessage | ToolMessage: Async-streamed tokens
            or the final tool/LLM message.
        """
        # --- Auth & user setup ---
        self.authenticate()
        self._user_id = user_id or self._user_id
        logger.info(f"I am chatting with {self._user_id}")

        if self.memory and is_save_memory:
            self.save_memory(query, user_id=self._user_id)

        self.guardrail_executor.check_input_guardrail(query)

        try:
            # --- Compiled graph path ---
            if (
                getattr(self, "compiled_graph", None)
                and self.compiled_graph is not None
            ):
                result = await self.graph_executor._invoke_compiled_graph_async(
                    query=query,
                    user_id=user_id,
                    history=self.in_conversation_history,
                    memory=self.memory,
                    is_save_memory=is_save_memory,
                    **kwargs,
                )
                yield result
                return

            # --- Tool calling loop ---
            current_query = query
            tool_message = None

            for iteration in range(1, max_iterations + 1):
                logger.info(
                    f"Async streaming tool calling iteration {iteration}/{max_iterations}"
                )

                response = await self.async_stream_invoke_executor._step1_llm_define_tool_async(
                    iteration=iteration,
                    max_history=max_history,
                    user_id=user_id,
                    message=current_query,
                    tools_manager=self.tools_manager,
                    memory=self.memory,
                    skills=self.skills,
                    description=self.description,
                    instruction=self.instruction,
                    history=self.in_conversation_history,
                )

                logger.info(f"Response from LLM: {response}")
                # Early exit: direct answer — async-stream it token-by-token
                if not getattr(response, "requires_tool", False) and getattr(
                    response, "answer", None
                ):
                    logger.info(
                        f"No tool needed. Async-streaming direct answer (iteration {iteration})."
                    )
                    answer_text = response.answer
                    for chunk in answer_text:
                        chunk = AIMessageChunk(content=chunk)
                        yield chunk
                    self.guardrail_executor.check_output_guardrail(answer_text)
                    full_chunk = AIMessage(content=answer_text)
                    self.in_conversation_history.add_message(full_chunk)
                    # Stream the direct answer chunk-by-chunk via a short llm.stream call
                    # Re-add message to history so the next prompt is correct
                    if self.memory and is_save_memory:
                        self.save_memory(
                            message=full_chunk.content, user_id=self._user_id
                        )
                    if answer_text:
                        try:
                            tool_call = json.loads(answer_text)
                            logger.info(
                                "Detected JSON tool call in answer, preparing next iteration."
                            )
                            current_query = AIMessage(
                                content=f"Let's call tool: {json.dumps(tool_call)}"
                            )
                            continue

                        except json.JSONDecodeError:
                            return

                # Step 2: Execute tool asynchronously
                current_query, tool_message, should_continue = (
                    await self.async_stream_invoke_executor._step2_tool_invoke_async(
                        current_query=current_query,
                        response=response,
                        tools_manager=self.tools_manager,
                        history=self.in_conversation_history,
                        mcp_client=self.mcp_client,
                        mcp_server_name=self.mcp_server_name,
                    )
                )

                if not should_continue:
                    answer_text = getattr(response, "answer", None) or ""
                    self.guardrail_executor.check_output_guardrail(answer_text)
                    if self.memory and is_save_memory:
                        self.save_memory(message=answer_text, user_id=self._user_id)
                    yield AIMessage(content=answer_text)
                    return

            # Step 3: Max iterations — async-stream final LLM summary
            logger.warning(
                f"Reached maximum iterations ({max_iterations}). Stopping async streaming loop."
            )
            async for (
                chunk
            ) in self.async_stream_invoke_executor._step3_final_response_astream(
                query=current_query,
                tool_message=tool_message,
                is_tool_formatted=is_tool_formatted,
                is_save_memory=is_save_memory,
                max_history=max_history,
                history=self.in_conversation_history,
                memory=self.memory,
                user_id=user_id,
            ):
                yield chunk

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"An error occurred during async streaming: {str(e)}")
            yield AIMessage(content=f"An error occurred: {str(e)}")

    def save_memory(
        self, message: Union[ToolMessage, AIMessage], user_id: str = "unknown_user"
    ) -> None:
        """
        Save the tool message to the memory
        """
        if self.memory:
            if isinstance(message, str):
                self.memory.save_short_term_memory(self.llm, message, user_id=user_id)
                logging.info(f"Saved to memory the query: {message}")
            elif isinstance(message, AIMessage):
                self.memory.save_short_term_memory(
                    self.llm, message.content, user_id=user_id
                )
                logging.info(f"Saved to memory the ai message: {message.content}")
            elif isinstance(message.artifact, str):
                self.memory.save_short_term_memory(
                    self.llm, message.artifact, user_id=user_id
                )
                logging.info(f"Saved to memory the tool artifact: {message.artifact}")
            else:
                self.memory.save_short_term_memory(
                    self.llm, message.content, user_id=user_id
                )
                logging.info(f"Saved to memory the tool content: {message.content}")

    def function_tool(self, func: Any):
        return self.tools_manager.register_function_tool(func)
