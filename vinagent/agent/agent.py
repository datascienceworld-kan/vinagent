import asyncio
import json
from abc import ABC, abstractmethod
from typing import Any, Awaitable, List, AsyncGenerator, TypedDict
from typing_extensions import is_typeddict
from langchain_together import ChatTogether
from langchain_core.language_models.base import BaseLanguageModel
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.messages.tool import ToolMessage
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver
import logging
from pathlib import Path
from typing import Union
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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    def stream(self, query: str, *args, **kwargs) -> Any:
        """Streaming the agent's main function"""
        pass


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


class Agent(AgentMeta):
    """The Agent class is a concrete implementation of an AI agent with tool-calling capabilities, inheriting from AgentMeta. It integrates a language model, tools, memory, and flow management to process queries, execute tools, and maintain conversational context."""

    def __init__(
        self,
        llm: Union[ChatTogether, BaseLanguageModel, BaseChatOpenAI],
        tools: List[Union[str, BaseTool]] = [],
        tools_path: Path = Path("templates/tools.json"),
        is_reset_tools=False,
        description: str = "You are a helpful assistant who can use the following tools to complete a task.",
        instruction: str = "You should answer the question relying on your determined skills",
        skills: list[str] = ["You can answer the user question with tools"],
        flow: list[str] = [],
        state_schema: type[Any] = None,
        config_schema: type[Any] = None,
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
        self.llm = llm
        self.tools = tools
        self.description = description
        self.instruction = instruction
        self.skills = skills

        # Initialize Agent flow by Langgraph
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

        # OAuth2 authentication if enabled
        self.authen_card = authen_card

        self.input_guardrail = input_guardrail
        self.output_guardrail = output_guardrail
        self.guardrail_manager = guardrail_manager

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
        Register a list of tools
        """
        for tool in tools:
            self.tools_manager.register_module_tool(tool)

    @property
    def user_id(self):
        return self._user_id

    @user_id.setter
    def user_id(self, new_user_id):
        self._user_id = new_user_id

    def prompt_template(
        self, query: str, user_id: str = "unknown_user", *args, **kwargs
    ) -> str:
        try:
            tools = json.loads(self.tools_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            tools = {}
            self.tools_path.write_text(json.dumps({}, indent=4), encoding="utf-8")

        memory = ""
        if self.memory:
            memory_content = self.memory.load_memory_by_user(
                load_type="string", user_id=user_id
            )
            if memory_content:
                memory = f"- Memory: {memory_content}\n"

        prompt = (
            "You are given a task, a list of available tools, and the memory about user to have precise information.\n"
            f"- Task: {query}\n"
            f"- Tools list: {json.dumps(tools)}\n"
            f"{memory}\n"
            f"- User: {user_id}\n"
            "------------------------\n"
            "Instructions:\n"
            "- Let's answer in a natural, clear, and detailed way without providing reasoning or explanation.\n"
            f"- If user used I in Memory, let's replace by name {user_id} in User part.\n"
            "- You need to think about whether the question need to use Tools?\n"
            "- If it was daily normal conversation. Let's directly answer as a human with memory.\n"
            "- If the task requires a tool, select the appropriate tool with its relevant arguments from Tools list according to following format (no explanations, no markdown):\n"
            "{\n"
            '"tool_name": "Function name",\n'
            '"tool_type": "Type of tool. Only get one of three values ["function", "module", "mcp"]"\n'
            '"arguments": "A dictionary of keyword-arguments to execute tool_name",\n'
            '"module_path": "Path to import the tool"\n'
            "}\n"
            "- Let's say I don't know and suggest where to search if you are unsure the answer.\n"
            "- Not make up anything.\n"
        )
        return prompt

    def prompt_tool(
        self, query: str, tool_call: str, tool_message: ToolMessage, *args, **kwargs
    ) -> str:
        tool_template = (
            f"- Question: {query}\n"
            f"- Tool Used: {tool_call}\n"
            f"- Tool's Result: {tool_message.artifact}\n"
            "Let's answer:"
        )
        return tool_template

    def system_prompt(self):
        skills = "- " + "- ".join(self.skills)
        content = (
            f"{self.description}\nYour skills:\n{skills}\nInstruction:\n{self.instruction}"
            if self.instruction
            else f"{self.description}\nYour skills:\n{skills}"
        )
        system_prompt = SystemMessage(content=content)
        return system_prompt

    def initialize_state(
        self, query: str, user_id: str, thread_id: str = 123, **kwargs
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

    def check_input_guardrail(self, query: str):
        if self.guardrail_manager:
            decision = self.guardrail_manager.validate_input(self.llm, query)
            if not decision.allowed:
                raise ValueError(decision.reason)
        else:
            if self.input_guardrail:
                decision = self.input_guardrail.validate(self.llm, query)
                if not decision.allowed:
                    raise ValueError(decision.reason)
        return True

    def check_output_guardrail(self, output_text: str):
        if self.guardrail_manager:
            decision = self.guardrail_manager.validate_output(self.llm, output_text)
            if not decision.allowed:
                raise ValueError(decision.reason)
        else:
            if self.output_guardrail:
                decision = self.output_guardrail.validate(self.llm, output_text)
                logger.info(decision)
                if not decision.allowed:
                    raise ValueError(decision.reason)
        return True

    def check_tool_guardrail(self, tool_name: str):
        if self.guardrail_manager:
            decision = self.guardrail_manager.validate_tools(tool_name)
            if not decision.allowed:
                raise ValueError(decision.reason)
        return True

    def invoke(
        self,
        query: str,
        is_save_memory: bool = False,
        user_id: str = "unknown_user",
        max_iterations: int = 10,  # Add max iterations to prevent infinite loops
        is_tool_formatted: bool = True,
        max_history: int = None,
        **kwargs,
    ) -> Any:
        """
        Answer the user query synchronously with continuous tool calling capability.

        Args:
            query (str): The input query or task description provided by the user.
            is_save_memory (bool, optional): Flag to determine if the conversation should be saved to memory. Defaults to False.
            user_id (str, optional): Identifier for the user making the request. Defaults to "unknown_user".
            max_iterations (int, optional): Maximum number of tool call iterations to prevent infinite loops. Defaults to 10.
            is_tool_formatted (bool, optional): Modifying the output Tool to become more human-preferred. If True, it needs one next llm invoke to format answer, else directly return tool_message. Defaults to True.
            max_history (int, optional): The maximum number of messages. Defaults to None.
            **kwargs: Additional keyword arguments, including an optional `config` dictionary for graph execution.

        Returns:
            Any: The result of the tool execution, LLM response, or None if an error occurs during tool execution.

        Raises:
            json.JSONDecodeError: If the tool data cannot be parsed as valid JSON.
            KeyError: If required keys are missing in the tool data.
            ValueError: If the tool data is invalid or cannot be processed.
        """
        self.authenticate()
        if self._user_id:
            pass
        if user_id:  # user clarify their name
            self._user_id = user_id
        logger.info(f"I'am chatting with {self._user_id}")

        if self.memory and is_save_memory:
            self.save_memory(query, user_id=self._user_id)

        # Check guardrail
        self.check_input_guardrail(query)

        # Question and answering
        try:
            if hasattr(self, "compiled_graph"):
                thread_id = (
                    kwargs.get("thread_id") if hasattr(self, "thread_id") else "123"
                )
                input_state = self.initialize_state(
                    query=query, user_id=user_id, thread_id=thread_id
                )

                try:
                    result = self.compiled_graph.invoke(**input_state)
                    self.check_output_guardrail(result)
                    self.in_conversation_history.add_message(result)
                    if self.memory and is_save_memory:
                        self.save_memory(message=result, user_id=self._user_id)
                    return result
                except ValueError as e:
                    logger.error(f"Error in compiled_graph.invoke: {e}")
            else:
                # Initialize conversation with original query
                current_query = query
                iteration = 0

                while iteration < max_iterations:
                    iteration += 1
                    logger.info(f"Tool calling iteration {iteration}/{max_iterations}")

                    # Create messages for current iteration
                    prompt = self.prompt_template(
                        query=current_query, user_id=self._user_id
                    )
                    system_prompt = self.system_prompt()
                    messages = [
                        system_prompt,
                        HumanMessage(content=prompt),
                    ]

                    # Add conversation history to messages
                    self.in_conversation_history.add_messages(messages)

                    # Get LLM response
                    history = self.in_conversation_history.get_history(
                        max_history=max_history
                    )
                    response = self.llm.invoke(history)

                    # Extract tool call from response
                    tool_data = ""
                    if hasattr(response, "content"):
                        tool_data = self.tools_manager.extract_tool(response.content)
                    else:
                        tool_data = self.tools_manager.extract_tool(response)

                    # If no tool call is found, return the final response
                    if not tool_data or ("None" in tool_data) or (tool_data == "{}"):
                        logger.info(
                            f"No more tool calls needed. Completed in {iteration} iterations."
                        )
                        self.check_output_guardrail(response)
                        if self.memory and is_save_memory:
                            self.save_memory(message=response, user_id=self._user_id)
                        return response

                    # Parse and execute tool call
                    tool_call = json.loads(tool_data)
                    logger.info(f"Executing tool call: {tool_call}")
                    # Adapt tool-call for gpt4o model by adding tool_calls argument into response
                    # Note: It must be preceded to adding tool_message into history.
                    self.check_tool_guardrail(tool_name=tool_call.get("tool_name"))
                    response = self.adapter_ai_response_with_tool_calls(
                        response, tool_call
                    )
                    self.in_conversation_history.add_message(response)

                    tool_message = asyncio.run(
                        self.tools_manager._execute_tool(
                            tool_name=tool_call["tool_name"],
                            tool_type=tool_call["tool_type"],
                            arguments=tool_call["arguments"],
                            module_path=tool_call["module_path"],
                            mcp_client=self.mcp_client,
                            mcp_server_name=self.mcp_server_name,
                        )
                    )

                    # Add AI message and tool result to conversation history
                    self.in_conversation_history.add_message(tool_message)
                    # Prepare next iteration with tool result context
                    tool_template = self.prompt_tool(
                        current_query, tool_call, tool_message
                    )
                    current_query = tool_template

                # If we reach max iterations, return the last tool message with a warning
                logger.warning(
                    f"Reached maximum iterations ({max_iterations}). Stopping tool calling loop."
                )

                if is_tool_formatted:
                    user_query = HumanMessage(
                        content=f"Based on the previous tool executions, please provide a final response to: {query}"
                    )
                    self.in_conversation_history.add_message(user_query)
                    history = self.in_conversation_history.get_history(
                        max_history=max_history
                    )
                    final_message = self.llm.invoke(history)
                    self.in_conversation_history.add_message(final_message)
                else:
                    final_message = tool_message

                self.check_output_guardrail(final_message)
                # Save memory
                if self.memory and is_save_memory:
                    self.save_memory(message=final_message, user_id=self._user_id)

                return final_message

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Tool calling failed: {str(e)}")
            return None

    def stream(
        self,
        query: str,
        is_save_memory: bool = False,
        user_id: str = "unknown_user",
        max_iterations: int = 10,  # Add max iterations to prevent infinite loops
        is_tool_formatted: bool = True,
        max_history: int = None,
        **kwargs,
    ) -> AsyncGenerator[Any, None]:
        """
        Answer the user query by streaming with continuous tool calling capability. Yields streamed responses or the final tool execution result.

        Args:
            query (str): The input query or task description provided by the user.
            is_save_memory (bool, optional): Flag to determine if the conversation should be saved to memory. Defaults to False.
            user_id (str, optional): Identifier for the user making the request. Defaults to "unknown_user".
            max_iterations (int, optional): Maximum number of tool call iterations to prevent infinite loops. Defaults to 10.
            is_tool_formatted (bool, optional): Modifying the output Tool to become more human-preferred. If True, it needs one next llm invoke to format answer, else directly return tool_message. Defaults to True.
            max_history (int, optional): Number of last messages in the history. Defaults to None.
            **kwargs: Additional keyword arguments, including an optional `config` dictionary for graph execution.

        Returns:
            Any: The result of the tool execution, LLM response, or None if an error occurs during tool execution.

        Raises:
            json.JSONDecodeError: If the tool data cannot be parsed as valid JSON.
            KeyError: If required keys are missing in the tool data.
            ValueError: If the tool data is invalid or cannot be processed.
        """
        self.authenticate()
        if not self._user_id:
            self._user_id = user_id
        logger.info(f"I am chatting with {self._user_id}")

        if self.memory and is_save_memory:
            self.save_memory(query, user_id=self._user_id)

        self.check_input_guardrail(query)
        try:
            if hasattr(self, "compiled_graph"):
                result = []
                thread_id = (
                    kwargs.get("thread_id") if hasattr(self, "thread_id") else "123"
                )
                input_state = self.initialize_state(
                    query=query, user_id=user_id, thread_id=thread_id
                )

                for chunk in self.compiled_graph.stream(**input_state):
                    for v in chunk.values():
                        if v:
                            result += v["messages"]
                            yield v

                self.check_output_guardrail(result)
                # Save memory
                if self.memory and is_save_memory:
                    self.save_memory(message=result, user_id=self._user_id)
                yield result
            else:
                # Initialize conversation with original query
                current_query = query
                iteration = 0
                final_result = None

                while iteration < max_iterations:
                    iteration += 1
                    logger.info(
                        f"Streaming tool calling iteration {iteration}/{max_iterations}"
                    )

                    # Create messages for current iteration
                    prompt = self.prompt_template(
                        query=current_query, user_id=self._user_id
                    )
                    system_prompt = self.system_prompt()
                    messages = [
                        system_prompt,
                        HumanMessage(content=prompt),
                    ]

                    # Add conversation history to messages
                    self.in_conversation_history.add_messages(messages)

                    # Accumulate streamed content
                    full_content = AIMessageChunk(content="")
                    history = self.in_conversation_history.get_history(
                        max_history=max_history
                    )
                    for chunk in self.llm.stream(history):
                        full_content += chunk
                        yield chunk

                    response = AIMessage(content=full_content.content)

                    # After streaming is complete, process tool data
                    tool_data = ""
                    if hasattr(full_content, "content"):
                        tool_data = self.tools_manager.extract_tool(
                            full_content.content
                        )
                    else:
                        tool_data = self.tools_manager.extract_tool(full_content)

                    if (tool_data is None) or (tool_data == "{}"):
                        logger.info(
                            f"No more tool calls needed. Completed in {iteration} iterations."
                        )
                        self.check_output_guardrail(response)
                        final_result = full_content
                        if self.memory and is_save_memory:
                            self.save_memory(
                                message=full_content, user_id=self._user_id
                            )
                        return final_result

                    # Parse and execute tool
                    tool_call = json.loads(tool_data)
                    # Adapt tool-call for gpt4o model by adding tool_calls argument into response
                    # Note: It must be preceded to adding tool_message into history.
                    self.check_tool_guardrail(tool_name=tool_call.get("tool_name"))
                    response = self.adapter_ai_response_with_tool_calls(
                        response, tool_call
                    )
                    self.in_conversation_history.add_message(response)

                    logger.info(f"Executing streaming tool call: {tool_call}")
                    tool_message = asyncio.run(
                        self.tools_manager._execute_tool(
                            tool_name=tool_call["tool_name"],
                            tool_type=tool_call["tool_type"],
                            arguments=tool_call["arguments"],
                            module_path=tool_call["module_path"],
                            mcp_client=self.mcp_client,
                            mcp_server_name=self.mcp_server_name,
                        )
                    )

                    # Add AI message and tool result to conversation history
                    self.in_conversation_history.add_message(tool_message)

                    # Prepare next iteration with tool result context
                    tool_template = self.prompt_tool(
                        current_query, tool_call, tool_message
                    )
                    current_query = tool_template

                # If we reach max iterations without natural completion
                if iteration >= max_iterations and final_result is None:
                    logger.warning(
                        f"Reached maximum iterations ({max_iterations}). Stopping streaming tool calling loop."
                    )
                    if is_tool_formatted:
                        final_message_content = HumanMessage(
                            f"Based on the previous tool executions, please provide a final response to: {query}"
                        )
                        self.in_conversation_history.add_message(final_message_content)
                        history = self.in_conversation_history.get_history(
                            max_history=max_history
                        )
                        full_content = AIMessageChunk(content="")
                        for chunk in self.llm.stream(history):
                            full_content += chunk
                            yield chunk
                        self.in_conversation_history.add_message(full_content)
                    else:
                        full_content = tool_message
                        self.check_output_guardrail(full_content)
                    if self.memory and is_save_memory:
                        # Create a message from the streamed content for memory
                        self.save_memory(message=full_content, user_id=self._user_id)

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Tool calling failed: {str(e)}")
            yield None  # Yield None to indicate failure

    async def ainvoke(
        self,
        query: str,
        is_save_memory: bool = False,
        user_id: str = "unknown_user",
        max_iterations: int = 10,  # Add max iterations to prevent infinite loops
        is_tool_formatted: bool = True,
        max_history: int = None,
        **kwargs,
    ) -> Any:
        """
        Answer the user query asynchronously with continuous tool calling capability.

        Args:
            query (str): The input query or task description provided by the user.
            is_save_memory (bool, optional): Flag to determine if the conversation should be saved to memory. Defaults to False.
            user_id (str, optional): Identifier for the user making the request. Defaults to "unknown_user".
            max_iterations (int, optional): Maximum number of tool call iterations to prevent infinite loops. Defaults to 10.
            is_tool_formatted (bool, optional): Modifying the output Tool to become more human-preferred. If True, it needs one next llm invoke to format answer, else directly return tool_message. Defaults to True.
            max_history (int, None): Number of maximum history messages. Defaults to None.
            **kwargs: Additional keyword arguments, including an optional `config` dictionary for graph execution.

        Returns:
            Any: The result of the tool execution, LLM response, or None if an error occurs during tool execution.

        Raises:
            json.JSONDecodeError: If the tool data cannot be parsed as valid JSON.
            KeyError: If required keys are missing in the tool data.
            ValueError: If the tool data is invalid or cannot be processed.
        """
        self.authenticate()
        if self._user_id:
            pass
        else:  # user clarify their name
            self._user_id = user_id
        logger.info(f"I'am chatting with {self._user_id}")

        self.check_input_guardrail(query)
        if self.memory and is_save_memory:
            self.save_memory(query, user_id=self._user_id)

        try:
            if hasattr(self, "compiled_graph"):
                thread_id = (
                    kwargs.get("thread_id") if hasattr(self, "thread_id") else "123"
                )
                input_state = self.initialize_state(
                    query=query, user_id=user_id, thread_id=thread_id
                )

                try:
                    result = await self.compiled_graph.ainvoke(**input_state)
                    self.in_conversation_history.add_message(result)
                    self.check_output_guardrail(result)
                    if self.memory and is_save_memory:
                        self.save_memory(message=result, user_id=self._user_id)
                    return result
                except ValueError as e:
                    logger.error(f"Error in compiled_graph.invoke: {e}")
            else:
                # Initialize conversation with original query
                current_query = query
                iteration = 0

                while iteration < max_iterations:
                    iteration += 1
                    logger.info(
                        f"Async tool calling iteration {iteration}/{max_iterations}"
                    )

                    # Create messages for current iteration
                    prompt = self.prompt_template(
                        query=current_query, user_id=self._user_id
                    )
                    system_prompt = self.system_prompt()
                    messages = [
                        system_prompt,
                        HumanMessage(content=prompt),
                    ]

                    # Add conversation history to messages
                    self.in_conversation_history.add_messages(messages)
                    history = self.in_conversation_history.get_history(
                        max_history=max_history
                    )

                    # Get LLM response
                    response = asyncio.to_thread(self.llm.ainvoke(input=history))

                    # Extract tool call from response
                    tool_data = ""
                    if hasattr(response, "content"):
                        tool_data = self.tools_manager.extract_tool(response.content)
                    else:
                        tool_data = self.tools_manager.extract_tool(response)

                    # If no tool call is found, return the final response
                    if not tool_data or ("None" in tool_data) or (tool_data == "{}"):
                        logger.info(
                            f"No more tool calls needed. Completed in {iteration} iterations."
                        )
                        self.check_output_guardrail(response)
                        if self.memory and is_save_memory:
                            self.save_memory(message=response, user_id=self._user_id)
                        return response

                    # Parse and execute tool call
                    tool_call = json.loads(tool_data)
                    logger.info(f"Executing async tool call: {tool_call}")
                    # Adapt tool-call for gpt4o model by adding tool_calls argument into response
                    # Note: It must be preceded to adding tool_message into history.
                    self.check_tool_guardrail(tool_name=tool_call.get("tool_name"))
                    response = self.adapter_ai_response_with_tool_calls(
                        response, tool_call
                    )
                    self.in_conversation_history.add_message(response)

                    tool_message = await self.tools_manager._execute_tool(
                        tool_name=tool_call["tool_name"],
                        tool_type=tool_call["tool_type"],
                        arguments=tool_call["arguments"],
                        module_path=tool_call["module_path"],
                        mcp_client=self.mcp_client,
                        mcp_server_name=self.mcp_server_name,
                    )

                    # Add AI message and tool result to conversation history
                    self.in_conversation_history.add_message(tool_message)

                    # Prepare next iteration with tool result context
                    tool_template = self.prompt_tool(
                        current_query, tool_call, tool_message
                    )
                    current_query = tool_template

                # If we reach max iterations, return the last tool message with a warning
                logger.warning(
                    f"Reached maximum iterations ({max_iterations}). Stopping async tool calling loop."
                )
                user_query = HumanMessage(
                    content=f"Based on the previous tool executions, please provide a final response to: {query}"
                )
                self.in_conversation_history.add_message(user_query)
                history = self.in_conversation_history.get_history(
                    max_history=max_history
                )
                if is_tool_formatted:
                    final_message = self.llm.invoke(history)
                    self.in_conversation_history.add_message(final_message)
                else:
                    final_message = tool_message
                self.check_output_guardrail(final_message)
                if self.memory and is_save_memory:
                    self.save_memory(message=final_message, user_id=self._user_id)

                return final_message

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Tool calling failed: {str(e)}")
            return None

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

    def adapter_ai_response_with_tool_calls(self, response, tool_call):
        adapt_tool = {}
        all_tools = self.tools_manager.load_tools()
        selected_tool = all_tools[tool_call["tool_name"]]
        adapt_tool["name"] = selected_tool["tool_name"]
        adapt_tool["args"] = tool_call["arguments"]
        adapt_tool["type"] = "tool_call"
        adapt_tool["id"] = selected_tool["tool_call_id"]
        response.tool_calls = [adapt_tool]
        return response
