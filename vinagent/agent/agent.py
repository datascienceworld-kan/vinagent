import asyncio
import json
from abc import ABC, abstractmethod
from typing import Any, Awaitable, List, AsyncGenerator
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
from typing_extensions import is_typeddict
import mlflow
from mlflow.entities import SpanType

from vinagent.register.tool import ToolManager
from vinagent.memory.memory import Memory
from vinagent.mcp.client import DistributedMCPClient
from vinagent.graph.function_graph import FunctionStateGraph

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

def is_jupyter_notebook():
    try:
        from IPython import get_ipython
        ipython = get_ipython()
        if ipython is None:
            return False
        # Check if it's a Jupyter Notebook (ZMQInteractiveShell is used in Jupyter)
        return 'ZMQInteractiveShell' in str(type(ipython))
    except ImportError:
        return False

if is_jupyter_notebook():
    import nest_asyncio
    nest_asyncio.apply()

class Agent(AgentMeta):
    """Concrete implementation of an AI agent with tool-calling capabilities"""
    def __init__(
        self,
        llm: Union[ChatTogether, BaseLanguageModel, BaseChatOpenAI],
        tools: List[Union[str, BaseTool]] = [],
        tools_path: Path = Path("templates/tools.json"),
        is_reset_tools = False,
        description: str = "You are a helpful assistant who can use the following tools to complete a task.",
        skills: list[str] = ["You can answer the user question with tools"],
        flow: list[str] = [],
        state_schema: type[Any] = None,
        config_schema: type[Any] = None,
        memory_path: Path = None,
        is_reset_memory = False,
        mcp_client: DistributedMCPClient = None,
        mcp_server_name: str = None,
        is_pii: bool = False,
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

        skills : list[str], optional
            A list of skills or abilities describing what the assistant can do. Defaults to a basic tool-usage skill.

        flow: list[str], optional
            A list of routes in the graph that defines start_node >> end_node. Defaults empty.  
        
        is_reset_tools : bool, optional
            A flag indicating whether the agent should override its existing tools with the provided list of tools. Defaults to False.

        memory_path : Path, optional
            The path to the file containing the memory. Defaults to a template file. Only valid if memory is not None.

        is_reset_memory : bool, optional
            A flag indicating whether the agent should reset its memory when re-initializes it's memory. Defaults to False. Only valid if memory is not None.

        mcp_client : DistributedMCPClient, optional
            An instance of a DistributedMCPClient used to register tools with the memory. Defaults to None.
        
        mcp_name: str, optional
            The name of the memory server. Defaults to None.

        is_pii: bool, optional
            A flag indicating whether the assistant should be able to recognize a person who is chatting with. Defaults to False.
        
        *args, **kwargs : Any
            Additional arguments passed to the superclass or future extensions.
        """
        # Initialize Agent llm and tools
        self.llm = llm
        self.tools = tools
        self.description = description
        self.skills = skills
        
        # Initialize Agent flow by Langgraph
        self.flow = flow
        if self.flow:
            self.initialize_flow(state_schema=state_schema, config_schema=config_schema)
        
        # Initialize Tools
        self.tools_path = None
        if tools_path:
            self.tools_path = Path(tools_path) if isinstance(tools_path, str) else tools_path
        else:
            self.tools_path = Path("templates/tools.json")
        if self.tools_path and (self.tools_path.suffix != ".json"):
            raise ValueError("tools_path must be json format ending with .json. For example, 'templates/tools.json'")
        self.tools_path.parent.mkdir(parents=True, exist_ok=True)
        self.is_reset_tools = is_reset_tools
        self.tools_manager = ToolManager(tools_path=self.tools_path, is_reset_tools=self.is_reset_tools)
        self.register_tools(self.tools)
        self.mcp_client = mcp_client
        self.mcp_server_name = mcp_server_name
        
        # Initialize memory
        self.memory_path = Path(memory_path) if isinstance(memory_path, str) else memory_path
        if self.memory_path and (self.memory_path.suffix != ".json"):
            raise ValueError("memory_path must be json format ending with .json. For example, 'templates/memory.json'")
        self.is_reset_memory = is_reset_memory
        self.memory = None
        if self.memory_path:
            self.memory = Memory(
                memory_path=self.memory_path,
                is_reset_memory=self.is_reset_memory
            )
        
        # Identify user
        self.is_pii = is_pii
        self._user_id = None
        if not self.is_pii:
            self._user_id = 'unknown_user'
    
    async def connect_mcp_tool(self):
        logger.info(f"{self.mcp_client}: {self.mcp_server_name}")
        if self.mcp_client and self.mcp_server_name:
            mcp_tools = await self.tools_manager.register_mcp_tool(self.mcp_client, self.mcp_server_name)
            logger.info(f"Successfully connected to mcp server {self.mcp_server_name}!")
        elif self.mcp_client:
            mcp_tools = await self.tools_manager.register_mcp_tool(self.mcp_client)
            logger.info(f"Successfully connected to mcp server!")
        return "Successfully connected to mcp server!"

    def initialize_flow(self, 
                        state_schema: type[Any],
                        config_schema: type[Any]):
        # Validate state_schema if provided
        if state_schema is not None and not is_typeddict(state_schema):
            raise TypeError("state_schema must be a TypedDict subclass")
        
        # Validate config_schema if provided
        if config_schema is not None and not is_typeddict(config_schema):
            raise TypeError("config_schema must be a TypedDict subclass")
        
        if self.flow:
            self.graph = FunctionStateGraph(state_schema=state_schema, config_schema=config_schema) 
            self.checkpoint = MemorySaver()
            self.compiled_graph = self.graph.compile(checkpointer=self.checkpoint, flow=self.flow)
            
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


    def prompt_template(self, query: str, user_id: str = "unknown_user", *args, **kwargs) -> str:
        try:
            tools = json.loads(self.tools_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            tools = {}
            self.tools_path.write_text(json.dumps({}, indent=4), encoding="utf-8")
        
        memory = ""
        if self.memory:
            memory_content = self.memory.load_memory(load_type='string', user_id=user_id)
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
    
    def prompt_tool(self, query: str, tool_call: str, tool_message: ToolMessage, *args, **kwargs) -> str:
        tool_template=("You are a professional reporter. Your task is to deliver a clear and factual report that directly addresses the given question. Use the tool's name and result only to support your explanation. Do not fabricate any information or over-interpret the result.\n"
            f"- Question: {query}\n"
            f"- Tool Used: {tool_call}\n"
            f"- Result: {tool_message.artifact}\n"
            "Let's report:"
        )
        return tool_template

    def invoke(self, query: str, is_save_memory: bool = False, user_id: str = "unknown_user", **kwargs) -> Any:
        """
        Select and execute a tool based on the task description
        """
        if self._user_id:
            pass
        if user_id: # user clarify their name
            self._user_id=user_id
        logger.info(f"I'am chatting with {self._user_id}")

        prompt = self.prompt_template(query=query, user_id=self._user_id)
        skills = "- " + "- ".join(self.skills)
        messages = [
            SystemMessage(content=f"{self.description}\nHere is your skills:\n{skills}"),
            HumanMessage(content=prompt),
        ]

        if self.memory and is_save_memory:
            self.save_memory(query, user_id=self._user_id)

        try:
            if hasattr(self, 'compiled_graph'):
                if 'config' in kwargs:
                    config = kwargs['config']
                else:
                    config = {"configurable": {"user_id": user_id}, "thread_id": "123"} # Default config
                try:
                    result = self.compiled_graph.invoke(input=query, config=config)
                    if self.memory and is_save_memory:
                        self.save_memory(message=result, user_id=self._user_id)
                    return result
                except ValueError as e:
                    logger.error(f"Error in compiled_graph.invoke: {e}")
            else:
                response = self.llm.invoke(messages)
                tool_data = self.tools_manager.extract_tool(response.content)
                if not tool_data or ("None" in tool_data) or (tool_data == "{}"):
                    return response
                
                tool_call = json.loads(tool_data)
                logging.info(tool_call)
                tool_message = asyncio.run(
                    self.tools_manager._execute_tool(
                        tool_name=tool_call["tool_name"],
                        tool_type=tool_call["tool_type"], 
                        arguments=tool_call["arguments"],
                        module_path=tool_call["module_path"],
                        mcp_client=self.mcp_client,
                        mcp_server_name=self.mcp_server_name
                    )
                )

                tool_template = self.prompt_tool(query, tool_call, tool_message)
                message = self.llm.invoke(tool_template)
                tool_message.content = "\n" + message.content
                if self.memory and is_save_memory:
                    self.save_memory(message=tool_message, user_id=self._user_id)
                return tool_message
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Tool calling failed: {str(e)}")
            return None

    def stream(self, query: str, is_save_memory: bool = False, user_id: str = "unknown_user", **kwargs) -> AsyncGenerator[Any, None]:
        """
        Select and execute a tool based on the task description, using streaming.
        Yields streamed responses or the final tool execution result.
        """
        if not self._user_id:
            self._user_id = user_id
        logger.info(f"I am chatting with {self._user_id}")

        prompt = self.prompt_template(query=query, user_id=self._user_id)
        skills = "- " + "- ".join(self.skills)
        messages = [
            SystemMessage(content=f"{self.description}\nHere is your skills:\n{skills}"),
            HumanMessage(content=prompt),
        ]

        if self.memory and is_save_memory:
            self.save_memory(query, user_id=self._user_id)

        try:
            if hasattr(self, 'compiled_graph'):
                result = []
                if 'config' in kwargs:
                    config = kwargs['config']
                else:
                    config = {"configurable": {"user_id": user_id}, "thread_id": "123"} # Default config
                for chunk in self.compiled_graph.stream(input=query, config=config):
                    for v in chunk.values():
                        if v:
                            result += v 
                            yield v
                if self.memory and is_save_memory:
                    self.save_memory(message=result, user_id=self._user_id)
                yield result
            else:
                # Accumulate streamed content
                full_content = AIMessageChunk(content="")
                for chunk in self.llm.stream(messages):
                    full_content += chunk
                    yield chunk

                # After streaming is complete, process tool data
                tool_data = self.tools_manager.extract_tool(full_content.content)
                if (tool_data is None) or (tool_data == "{}"):
                    return full_content  # Return the full content if no tool is called
                
                # Parse and execute tool
                tool_call = json.loads(tool_data)
                logger.info(f"Tool call: {tool_call}")
                tool_message = asyncio.run(
                    self.tools_manager._execute_tool(
                        tool_name=tool_call["tool_name"],
                        tool_type=tool_call["tool_type"], 
                        arguments=tool_call["arguments"],
                        module_path=tool_call["module_path"],
                        mcp_client=self.mcp_client,
                        mcp_server_name=self.mcp_server_name
                    )
                )

                tool_template = self.prompt_tool(query, tool_call, tool_message)
                message = self.llm.invoke(tool_template)
                tool_message.content = "\n" + message.content            
                if self.memory and is_save_memory:
                    self.save_memory(message=tool_message, user_id=self._user_id)
                yield tool_message  # Yield the final tool execution result

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Tool calling failed: {str(e)}")
            yield None  # Yield None to indicate failure

    async def ainvoke(self, query: str, is_save_memory: bool = False, user_id: str = "unknown_user", **kwargs) -> Any:
        """
        Select and execute a tool based on the task description
        """
        if self._user_id:
            pass
        else: # user clarify their name
            self._user_id=user_id
        logger.info(f"I'am chatting with {self._user_id}")

        prompt = self.prompt_template(query=query, user_id=self._user_id)
        skills = "- " + "- ".join(self.skills)
        messages = [
            SystemMessage(content=f"{self.description}\nHere is your skills:\n{skills}"),
            HumanMessage(content=prompt),
        ]

        if self.memory and is_save_memory:
            self.save_memory(query, user_id=self._user_id)

        try:
            if hasattr(self, 'compiled_graph'):
                if 'config' in kwargs:
                    config = kwargs['config']
                else:
                    config = {"configurable": {"user_id": user_id}, "thread_id": "123"} # Default config
                result = await self.compiled_graph.ainvoke(input=query, config=config)
                if self.memory and is_save_memory:
                    self.save_memory(message=result, user_id=self._user_id)
                return result
            else:
                response = await self.llm.ainvoke(messages)
                tool_data = self.tools_manager.extract_tool(response.content)
                if not tool_data or ("None" in tool_data) or (tool_data == "{}"):
                    return response
                
                tool_call = json.loads(tool_data)
                tool_message = await self.tools_manager._execute_tool(
                    tool_name=tool_call["tool_name"],
                    tool_type=tool_call["tool_type"], 
                    arguments=tool_call["arguments"],
                    module_path=tool_call["module_path"],
                    mcp_client=self.mcp_client,
                    mcp_server_name=self.mcp_server_name
                )
                if self.memory and is_save_memory:
                    self.save_memory(message=tool_message, user_id=self._user_id)
                return tool_message
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Tool calling failed: {str(e)}")
            return None

    def save_memory(self, message: Union[ToolMessage, AIMessage], user_id: str = "unknown_user") -> None:
        """
        Save the tool message to the memory
        """
        if self.memory:
            if isinstance(message, str):
                self.memory.save_short_term_memory(self.llm, message, user_id=user_id)
                logging.info(f"Saved to memory the query: {message}")
            elif isinstance(message, AIMessage):
                self.memory.save_short_term_memory(self.llm, message.content, user_id=user_id)
                logging.info(f"Saved to memory the ai message: {message.content}")
            elif isinstance(message.artifact, str):
                self.memory.save_short_term_memory(self.llm, message.artifact, user_id=user_id)
                logging.info(f"Saved to memory the tool artifact: {message.artifact}")
            else:
                self.memory.save_short_term_memory(self.llm, message.content, user_id=user_id)
                logging.info(f"Saved to memory the tool content: {message.content}")
                                               
    def function_tool(self, func: Any):
        return self.tools_manager.register_function_tool(func)
