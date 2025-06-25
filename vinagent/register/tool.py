import sys
import os
import json
import inspect
import importlib
import logging
from functools import wraps
from typing import Dict, Any, Optional, Callable, Union, Literal
import ast
import uuid
from pathlib import Path
from .initialize import llm
from vinagent.mcp import load_mcp_tools
from vinagent.mcp.client import DistributedMCPClient
from langchain_core.messages.tool import ToolMessage
import asyncio

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ToolManager:
    """Centralized tool management class"""
    def __init__(self, tools_path: Path = Path("templates/tools.json"), is_reset_tools: bool=False):
        self.tools_path = tools_path
        self.is_reset_tools = is_reset_tools
        self.tools_path = Path(tools_path) if isinstance(tools_path, str) else tools_path
        if not self.tools_path.exists():
            self.tools_path.write_text(json.dumps({}, indent=4), encoding="utf-8")

        if self.is_reset_tools:
            self.tools_path.write_text(json.dumps({}, indent=4), encoding="utf-8")

        self._registered_functions: Dict[str, Callable] = {}
        
    def load_tools(self) -> Dict[str, Any]:
        """Load existing tools from JSON"""
        if self.tools_path:
            with open(self.tools_path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            return {}

    def save_tools(self, tools: Dict[str, Any]) -> None:
        """Save tools to JSON"""
        with open(self.tools_path, "w", encoding="utf-8") as f:
            json.dump(tools, f, indent=4, ensure_ascii=False)

    def register_function_tool(self, func):
        """Decorator to register a function as a tool
        # Example usage:
        @function_tool
        def sample_function(x: int, y: str) -> str:
            '''Sample function for testing'''
            return f"{y}: {x}"
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        # Get function metadata
        signature = inspect.signature(func)

        # Try to get module path, fall back to None if not available
        module_path = "__runtime__"

        # Create metadata
        if module_path == "__runtime__":
            metadata = {
                "tool_name": func.__name__,
                "arguments": {
                    name: (
                        str(param.annotation)
                        if param.annotation != inspect.Parameter.empty
                        else "Any"
                    )
                    for name, param in signature.parameters.items()
                },
                "return": (
                    str(signature.return_annotation)
                    if signature.return_annotation != inspect.Signature.empty
                    else "Any"
                ),
                "docstring": (func.__doc__ or "").strip(),
                "module_path": module_path,
                "tool_type": "function",
                "tool_call_id": "tool_" + str(uuid.uuid4()),
                "is_runtime": module_path == "__runtime__",
            }

            # Register both the function and its metadata
            self._registered_functions[func.__name__] = func
            tools = self.load_tools()
            tools[func.__name__] = metadata
            self.save_tools(tools)
            logger.info(
                f"Registered tool: {func.__name__} "
                f"({'runtime' if module_path == '__runtime__' else 'file-based'})"
            )
        return wrapper

    async def register_mcp_tool(self, client: DistributedMCPClient, server_name: str = None) -> list[Dict[str, Any]]:
        # Load all tools
        logger.info(f"Registering MCP tools")
        all_tools = []
        if server_name:
            all_tools = await client.get_tools(server_name=server_name)
            logger.info(f"Loaded MCP tools of {server_name}: {len(all_tools)}")
        else:
            try:
                all_tools = await client.get_tools()
                logger.info(f"Loaded MCP tools: {len(all_tools)}")
            except Exception as e:
                logger.error(f"Error loading MCP tools: {e}")
                return []
        # Convert MCP tools to our format
        def convert_mcp_tool(mcp_tool: Dict[str, Any]):
            tool_name = mcp_tool['name']
            arguments = dict([(k, v['type']) for (k,v) in mcp_tool['args_schema']['properties'].items()])
            docstring = mcp_tool['description']
            return_value = mcp_tool['response_format']
            tool = {}
            tool['tool_name'] = tool_name
            tool['arguments'] = arguments
            tool['return'] = return_value
            tool['docstring'] = docstring
            tool['module_path'] = '__mcp__'
            tool['tool_type'] = 'mcp'
            # tool['mcp_client_connections'] = client.connections
            # tool['mcp_server_name'] = server_name
            tool['tool_call_id'] = "tool_" + str(uuid.uuid4())
            return tool
        
        new_tools = [convert_mcp_tool(mcp_tool.__dict__) for mcp_tool in all_tools]
        tools = self.load_tools()
        for tool in new_tools:
            tools[tool["tool_name"]] = tool
            tools[tool["tool_name"]]["tool_call_id"] = "tool_" + str(uuid.uuid4())
            logger.info(f"Registered {tool['tool_name']}:\n{tool}")
        self.save_tools(tools)
        logger.info(f"Completed registration for mcp module {server_name}")
        return new_tools

    def register_module_tool(self, module_path: str) -> None:
        """Register tools from a module"""
        try:
            module = importlib.import_module(module_path, package=__package__)
            module_source = inspect.getsource(module)
        except (ImportError, ValueError) as e:
            raise ValueError(f"Failed to load module {module_path}: {str(e)}")

        prompt = (
            "Analyze this module and return a list of tools in JSON format:\n"
            "- Module code:\n"
            f"{module_source}\n"
            "- Format: Let's return a list of json format without further explaination and without ```json characters markdown and keep module_path unchange.\n"
            "[{{\n"
            '"tool_name": "The function",\n'
            '"arguments": "A dictionary of keyword-arguments to execute tool. Let\'s keep default value if it was set",\n'
            '"return": "Return value of this tool",\n'
            '"docstring": "Docstring of this tool",\n'
            '"dependencies": "List of libraries need to run this tool",\n'
            f'"module_path": "{module_path}"\n'
            "}}]\n"
        )

        response = llm.invoke(prompt)

        try:
            new_tools = ast.literal_eval(response.content.strip())
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"Invalid tool format from LLM: {str(e)}")

        tools = self.load_tools()
        for tool in new_tools:
            tool["module_path"] = module_path
            tool["tool_type"] = 'module'
            tools[tool["tool_name"]] = tool
            tools[tool["tool_name"]]["tool_call_id"] = "tool_" + str(uuid.uuid4())
            logger.info(f"Registered {tool['tool_name']}:\n{tool}")

        self.save_tools(tools)
        logger.info(f"Completed registration for module {module_path}")

    def extract_tool(self, text: str) -> Optional[str]:
        """Extract first valid JSON object from text"""
        stack = []
        start = text.find("{")
        if start == -1:
            return None

        for i in range(start, len(text)):
            if text[i] == "{":
                stack.append("{")
            elif text[i] == "}":
                stack.pop()
                if not stack:
                    return text[start : i + 1]
        return None


    async def _execute_tool(self, 
                            tool_name: str, 
                            arguments: dict,
                            mcp_client: DistributedMCPClient,
                            mcp_server_name: str,
                            module_path: str,
                            tool_type: str = Literal['function', 'mcp', 'module']
                            ) -> Any:
        """Execute the specified tool with given arguments"""
        if tool_type == 'function':
            message = await FunctionTool.execute(self, tool_name, arguments)
        elif tool_type == 'mcp':
            message = await MCPTool.execute(self, tool_name, arguments, mcp_client, mcp_server_name)
        elif tool_type == 'module':
            message = await ModuleTool.execute(self, tool_name, arguments, module_path)
        return message
            
    @staticmethod
    def _extract_json(text: str) -> Optional[str]:
        """Extract first valid JSON object from text using stack-based parsing"""
        start = text.find("{")
        if start == -1:
            return None

        stack = []
        for i in range(start, len(text)):
            if text[i] == "{":
                stack.append("{")
            elif text[i] == "}":
                stack.pop()
                if not stack:
                    return text[start : i + 1]
        return None


class FunctionTool:
    @classmethod
    async def execute(cls,
            tool_manager: ToolManager,
            tool_name: str,
            arguments: Dict[str, Any]
            ):
        registered_functions = tool_manager.load_tools()

        if tool_name in tool_manager._registered_functions:
            try:
                func = tool_manager._registered_functions[tool_name]
                # artifact = await func(**arguments)
                artifact = await asyncio.to_thread(func, **arguments)
                content = f"Completed executing function tool {tool_name}({arguments})"
                logger.info(content)
                tool_call_id = registered_functions[tool_name]["tool_call_id"]
                message = ToolMessage(
                    content=content, artifact=artifact, tool_call_id=tool_call_id
                )
                return message
            except Exception as e:
                content = f"Failed to execute function tool {tool_name}({arguments}): {str(e)}"
                logger.error(content)
                # raise {"error": content}
                return content


class MCPTool:
    @classmethod
    async def execute(cls,
            tool_manager: ToolManager,
            tool_name: str, 
            arguments: Dict[str, Any], 
            mcp_client: DistributedMCPClient,
            mcp_server_name: str):
        
        registered_functions = tool_manager.load_tools()
        """Call the MCP tool natively using the client session."""
        async with mcp_client.session(mcp_server_name) as session:
            payload = {
                "name": tool_name,
                "arguments": arguments
            }
            try:
                # Send the request to the MCP server
                # response = await session.call_tool(**payload)
                response = await session.call_tool(**payload)
                content = f"Completed executing mcp tool {tool_name}({arguments})"
                logger.info(content)
                tool_call_id = registered_functions[tool_name]["tool_call_id"]
                artifact = response
                message = ToolMessage(
                    content=content, artifact=artifact, tool_call_id=tool_call_id
                )
                return message
            except Exception as e:
                content = f"Failed to execute mcp tool {tool_name}({arguments}): {str(e)}"
                logger.error(content)
                # raise {"error": content}
                return content


class ModuleTool:
    @classmethod
    async def execute(cls, 
            tool_manager: ToolManager,
            tool_name: str, 
            arguments: Dict[str, Any], 
            module_path: Union[str, Path], *arg, **kwargs):
        
        registered_functions = tool_manager.load_tools()
        try:
            if tool_name in globals():
                return globals()[tool_name](**arguments)

            module = importlib.import_module(module_path, package=__package__)
            func = getattr(module, tool_name)
            # artifact = await func(**arguments)
            artifact = await asyncio.to_thread(func, **arguments)
            content = f"Completed executing module tool {tool_name}({arguments})"
            logger.info(content)
            tool_call_id = registered_functions[tool_name]["tool_call_id"]
            message = ToolMessage(
                content=content, artifact=artifact, tool_call_id=tool_call_id
            )
            return message
        except (ImportError, AttributeError) as e:
            content = f"Failed to execute module tool {tool_name}({arguments}): {str(e)}"
            logger.error(content)
            # raise {"error": content}
            return content
