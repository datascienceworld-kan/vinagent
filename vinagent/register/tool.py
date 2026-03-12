import sys
import os
import json
import inspect
import importlib
import importlib.util
import logging
from functools import wraps
from typing import Dict, Any, Optional, Callable, Union, Literal
from pydantic import BaseModel, Field, field_validator
import ast
import uuid
from pathlib import Path
import shutil
from vinagent.mcp import load_mcp_tools
from vinagent.mcp.client import DistributedMCPClient
from langchain_core.messages.tool import ToolMessage
from langchain_core.language_models.base import BaseLanguageModel
import asyncio
import re
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ToolManager:
    """
    Centralized tool management class for registering, loading, saving, and executing tools.
    Tools are stored in a JSON file and can be of type 'function', 'mcp', or 'module'.
    """

    def __init__(
        self,
        llm: BaseLanguageModel,
        tools_path: Path = Path("templates/tools.json"),
        is_reset_tools: bool = False,
    ):
        """
        Initialize the ToolManager with a path to the tools JSON file.

        Args:
            llm (BaseLanguageModel): Language model instance for tool analysis.
            tools_path (Path, optional): Path to the JSON file for storing tools. Defaults to Path("templates/tools.json").
            is_reset_tools (bool, optional): If True, resets the tools file to an empty JSON object. Defaults to False.

        Behavior:
            - Converts tools_path to a Path object if provided as a string.
            - Creates the tools file if it does not exist.
            - Resets the tools file if is_reset_tools is True.
        """
        self.llm = llm
        self.tools_path = tools_path
        self.is_reset_tools = is_reset_tools
        self.tools_path = (
            Path(tools_path) if isinstance(tools_path, str) else tools_path
        )
        if not self.tools_path.exists():
            self.tools_path.write_text(json.dumps({}, indent=4), encoding="utf-8")

        if self.is_reset_tools:
            self.tools_path.write_text(json.dumps({}, indent=4), encoding="utf-8")

        self._registered_functions: Dict[str, Callable] = {}

    def load_tools(self) -> Dict[str, Any]:
        """
        Load existing tools from the JSON file.

        Returns:
            Dict[str, Any]: A dictionary of tool metadata, where keys are tool names.
        """
        if self.tools_path:
            with open(self.tools_path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            return {}

    def save_tools(self, tools: Dict[str, Any]) -> None:
        """
        Save tools metadata to the JSON file.

        Args:
            tools (Dict[str, Any]): Dictionary of tool metadata to save.
        """
        with open(self.tools_path, "w", encoding="utf-8") as f:
            json.dump(tools, f, indent=4, ensure_ascii=False)

    def register_function_tool(self, func):
        """
        Decorator to register a function as a tool.

        Args:
            func: The function to register as a tool.

        Returns:
            Callable: A wrapped function that retains original behavior.

        Example:
            @tool_manager.register_function_tool
            def sample_function(x: int, y: str) -> str:
                '''Sample function for testing'''
                return f"{y}: {x}"

        Behavior:
            - Extracts function metadata (name, arguments, return type, docstring).
            - Assigns a unique tool_call_id.
            - Stores metadata in the tools JSON file.
            - Registers the function in _registered_functions for execution.
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        # Get function metadata
        signature = inspect.signature(func)

        # Try to get module path, fall back to None if not available
        module_path = "__runtime__"

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
            "tool_call_id": "tool_" + str(uuid.uuid4())[:35],
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

    async def register_mcp_tool(
        self, client: DistributedMCPClient, server_name: str = None
    ) -> list[Dict[str, Any]]:
        """
        Register tools from an MCP (Memory Compute Platform) server.

        Args:
            client (DistributedMCPClient): Client for interacting with the MCP server.
            server_name (str, optional): Name of the MCP server. Defaults to None.

        Returns:
            list[Dict[str, Any]]: List of registered MCP tool metadata.

        Behavior:
            - Fetches tools from the MCP server using the client.
            - Converts MCP tools to the internal tool format.
            - Assigns unique tool_call_id for each tool.
            - Saves tools to the JSON file.
        """
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
            tool_name = mcp_tool["name"]
            arguments = dict(
                [
                    (k, v["type"])
                    for (k, v) in mcp_tool["args_schema"]["properties"].items()
                ]
            )
            docstring = mcp_tool["description"]
            return_value = mcp_tool["response_format"]
            tool = {}
            tool["tool_name"] = tool_name
            tool["arguments"] = arguments
            tool["return"] = return_value
            tool["docstring"] = docstring
            tool["module_path"] = "__mcp__"
            tool["tool_type"] = "mcp"
            # tool['mcp_client_connections'] = client.connections
            # tool['mcp_server_name'] = server_name
            tool["tool_call_id"] = "tool_" + str(uuid.uuid4())[:35]
            tool["is_runtime"]: module_path == "__runtime__"
            return tool

        new_tools = [convert_mcp_tool(mcp_tool.__dict__) for mcp_tool in all_tools]
        tools = self.load_tools()
        for tool in new_tools:
            tools[tool["tool_name"]] = tool
            tools[tool["tool_name"]]["tool_call_id"] = "tool_" + str(uuid.uuid4())[:35]
            logger.info(f"Registered {tool['tool_name']}:\n{tool}")
        self.save_tools(tools)
        logger.info(f"Completed registration for mcp module {server_name}")
        return new_tools

    def register_module_tool(self, module_path: str) -> None:
        """
        Register tools from a Python module.

        Args:
            module_path (str): Path to the module or import path in module import format.

        Raises:
            ValueError: If the module cannot be loaded or tool format is invalid.

        Behavior:
            - Copies the module file to the tools directory if a file path is provided.
            - Imports the module and extracts tool metadata using the language model.
            - Assigns a unique tool_call_id for each tool.
            - Saves tools to the JSON file.
        """
        if os.path.isdir(module_path):
            module_dir = Path(module_path)
            absolute_lib_path = Path(os.path.dirname(os.path.abspath(__file__)))
            destination_dir = Path(os.path.join(absolute_lib_path.parent, "tools"))

            # Copy the entire directory tree into tools/, merging if it already exists
            for item in module_dir.rglob("*"):
                if item.is_file():
                    relative = item.relative_to(module_dir.parent)
                    dest_file = destination_dir / relative
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    if item.resolve() != dest_file.resolve():
                        shutil.copy2(item, dest_file)

            # Now register only top-level .py files (not __init__, not subdirs)
            for py_file in module_dir.glob("*.py"):
                if py_file.name.startswith("__"):
                    continue
                try:
                    # Build the import path relative to tools/
                    import_name = f"vinagent.tools.{module_dir.name}.{py_file.stem}"
                    module = importlib.import_module(import_name, package=__package__)
                    module_source = inspect.getsource(module)
                    # ... rest of your tool extraction logic here
                except (ImportError, ValueError) as e:
                    logger.warning(f"Skipping {py_file.name}: {str(e)}")

        try:
            if os.path.isfile(module_path):
                # This is a path of module import format
                module_path = Path(module_path)
                absolute_lib_path = Path(os.path.dirname(os.path.abspath(__file__)))
                destination_path = Path(
                    os.path.join(absolute_lib_path.parent, "tools", module_path.name)
                )
                if module_path.resolve(strict=False) == destination_path.resolve(
                    strict=False
                ):
                    pass
                else:
                    shutil.copy2(module_path, destination_path)
                module_path = f"vinagent.tools.{destination_path.name.split('.')[0]}"
            module = importlib.import_module(module_path, package=__package__)
            module_source = inspect.getsource(module)
        except (ImportError, ValueError) as e:
            raise ValueError(f"Failed to load module {module_path}: {str(e)}")

        prompt = (
            "Analyze this module and return a list of tools in JSON format:\n"
            "- Module code:\n"
            f"{module_source}\n"
            "- Extract only tools marked with the @primary_function decorator. For example @primary_function def function_name(): ...\n"
            "- Let's return a list of json format without further explaination and without ```json characters markdown and keep module_path unchange.\n"
            "- Return value must be able to convert into a list from string.\n"
            "[{{\n"
            '"tool_name": "The function",\n'
            '"arguments": "A dictionary of keyword-arguments to execute tool. Let\'s keep default value if it was set",\n'
            '"return": "Return value of this tool",\n'
            '"docstring": "Docstring of this tool",\n'
            '"dependencies": "List of libraries need to run this tool",\n'
            f'"module_path": "{module_path}"\n'
            "}}]\n"
        )

        response = self.llm.invoke(prompt)
        response_text = ""
        if hasattr(response, "content"):
            response_text = response.content.strip()
        else:
            response_text = response.strip()

        # Remove markdown code fences if present
        if response_text.startswith("```"):
            # Remove the first line (```json or ```)
            response_lines = response_text.splitlines()
            # Skip first line if it starts with ```
            if response_lines[0].startswith("```"):
                response_lines = response_lines[1:]
            # Remove last line if it's ```
            if response_lines and response_lines[-1].startswith("```"):
                response_lines = response_lines[:-1]
            response_text = "\n".join(response_lines)

        # Attempt to parse the entire text first
        try:
            new_tools = ast.literal_eval(response_text)
        except (ValueError, SyntaxError):
            # Fallback: extract the first JSON object/list from text
            extracted = self.extract_tool(response_text)
            if extracted:
                try:
                    new_tools = ast.literal_eval(extracted)
                except (ValueError, SyntaxError) as e:
                    raise ValueError(
                        f"Invalid tool format from LLM after extraction: {str(e)}"
                    )
            else:
                raise ValueError(
                    "Invalid tool format from LLM: could not find valid JSON list"
                )

        # Ensure new_tools is a list of dictionaries
        if isinstance(new_tools, dict):
            new_tools = [new_tools]
        if not isinstance(new_tools, list):
            raise ValueError(
                f"Invalid tool format from LLM: Expected list or dict, got {type(new_tools)}"
            )
        for idx, item in enumerate(new_tools):
            if not isinstance(item, dict):
                raise ValueError(
                    f"Invalid tool format from LLM: Element at index {idx} is {type(item)}, expected dict"
                )

        # Fallback to local introspection if required keys are missing or list is empty
        REQUIRED_KEYS = {"tool_name", "arguments", "return", "docstring"}

        def _introspect_module(module_obj, module_path_str):
            result = []
            for name, obj in inspect.getmembers(module_obj, inspect.isfunction):
                if inspect.getmodule(obj) != module_obj:
                    continue  # Skip imported functions
                sig = inspect.signature(obj)
                arguments = {
                    param_name: (
                        str(param.annotation)
                        if param.annotation != inspect.Parameter.empty
                        else "Any"
                    )
                    for param_name, param in sig.parameters.items()
                }
                return_type = (
                    str(sig.return_annotation)
                    if sig.return_annotation != inspect.Signature.empty
                    else "Any"
                )
                metadata = {
                    "tool_name": name,
                    "arguments": arguments,
                    "return": return_type,
                    "docstring": (obj.__doc__ or "").strip(),
                    "dependencies": [],
                    "module_path": module_path_str,
                }
                result.append(metadata)
            return result

        if len(new_tools) == 0 or any(
            not REQUIRED_KEYS.issubset(item.keys()) for item in new_tools
        ):
            logger.warning(
                "LLM did not return valid tool metadata, falling back to introspection."
            )
            new_tools = _introspect_module(module, module_path)

        tools = self.load_tools()
        for tool in new_tools:
            tool["module_path"] = module_path
            tool["tool_type"] = "module"
            tool["is_runtime"] = module_path == "__runtime__"
            tools[tool["tool_name"]] = tool
            tools[tool["tool_name"]]["tool_call_id"] = "tool_" + str(uuid.uuid4())[:35]
            logger.info(f"Registered {tool['tool_name']}:\n{tool}")

        self.save_tools(tools)
        logger.info(f"Completed registration for module {module_path}")

    def extract_tool(self, text: str) -> Optional[str]:
        """
        Extract the first valid JSON object from a text string.

        Args:
            text (str): The text to parse for a JSON object.

        Returns:
            Optional[str]: The extracted JSON string, or None if no valid JSON is found.
        """
        start = text.find("{")
        if start == -1:
            return None

        try:
            obj, _ = json.JSONDecoder().raw_decode(text, start)
            return json.dumps(obj, indent=2)
        except json.JSONDecodeError:
            return None

    def register_agentskill_tool(self, skill_path: Union[str, Path]) -> None:
        """
        Register an AgentSkill directory as a single tool.

        AgentSkill tools work differently from function/module tools: there are
        **no Python callables to introspect**.  Instead, the full content of
        ``SKILL.md`` — which contains usage examples, shell commands, and
        workflow guidance — is stored as the tool ``docstring``.  At runtime the
        LLM reads the docstring, reasons about the task, and constructs a shell
        command (e.g. ``python scripts/unpack.py document.docx unpacked/``).  The
        :class:`AgentSkillTool` executor then runs that command in a subprocess.

        One tool entry is registered per skill directory, keyed by the skill
        ``name`` extracted from the YAML front-matter (or the directory name as
        fallback).  The registered argument schema always has a single field:
        ``command`` (``str``), which the LLM populates with the exact shell
        command to run.

        Args:
            skill_path (Union[str, Path]): Path to the agentskill directory.

        Raises:
            FileNotFoundError: If ``SKILL.md`` is not found inside
                ``skill_path``.
            ValueError: If the YAML front-matter cannot be parsed.

        Example skill directory layout::

            my_skill/
            ├── SKILL.md          # YAML front-matter + usage examples
            └── scripts/
                ├── unpack.py
                └── pack.py

        Resulting JSON entry::

            {
                "tool_name": "<skill_name>",
                "arguments": {"command": "str"},
                "return": "str",
                "docstring": "<full SKILL.md content>",
                "module_path": "<absolute/path/to/skill/scripts/>",
                "tool_type": "agentskills",
                "tool_call_id": "tool_<uuid>",
                "is_runtime": false
            }
        """
        skill_dir = Path(skill_path)
        skill_file = skill_dir / "SKILL.md"

        if not skill_file.exists():
            raise FileNotFoundError(f"SKILL.md not found in {skill_dir}")

        # --- Read the full SKILL.md as docstring (includes examples & commands) ---
        full_content = skill_file.read_text(encoding="utf-8")

        # --- Parse YAML front-matter for skill name ---
        skill_name = skill_dir.name  # fallback
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n", full_content, re.DOTALL)
        if match:
            try:
                frontmatter = yaml.safe_load(match.group(1)) or {}
                skill_name = frontmatter.get("name", skill_name)
            except yaml.YAMLError as e:
                raise ValueError(
                    f"Failed to parse YAML front-matter in {skill_file}: {e}"
                )

        # --- scripts/ directory is used as the working directory at runtime ---
        scripts_dir = skill_dir
        module_path = (
            str(scripts_dir.resolve())
            if scripts_dir.exists()
            else str(skill_dir.resolve())
        )

        metadata = {
            "tool_name": skill_name,
            "arguments": {
                "command": "str: Complete, self-contained Python or bash script to accomplish the user's task using the patterns shown in this tool's docstring. Write new code tailored to the task — do not copy docstring examples verbatim."
            },
            "return": "str: The output of the tool",
            "docstring": full_content.strip(),
            "module_path": module_path,
            "tool_type": "agentskills",
            "tool_call_id": "tool_" + str(uuid.uuid4())[:35],
            "is_runtime": False,
        }

        tools = self.load_tools()
        tools[skill_name] = metadata
        self.save_tools(tools)
        logger.info(
            f"Registered agentskill tool: '{skill_name}' "
            f"(working dir: {module_path})"
        )

    async def _execute_tool(
        self,
        tool_name: str,
        arguments: dict,
        mcp_client: DistributedMCPClient = None,
        mcp_server_name: str = None,
        module_path: str = None,
        tool_type: str = "function",
    ) -> Any:
        """
        Execute the specified tool with the given arguments.

        Args:
            tool_name (str): Name of the tool to execute.
            arguments (dict): Dictionary of arguments to pass to the tool.
            mcp_client (DistributedMCPClient): Client for MCP tool execution.
            mcp_server_name (str): Name of the MCP server.
            module_path (str): Path to the module for module-type tools.
            tool_type (str): Type of tool ('function', 'mcp', 'module', or 'agentskills').

        Returns:
            Any: The result of the tool execution, typically a ToolMessage.
        """
        try:
            if tool_type == "function":
                message = await FunctionTool.execute(self, tool_name, arguments)
            elif tool_type == "mcp":
                message = await MCPTool.execute(
                    self, tool_name, arguments, mcp_client, mcp_server_name
                )
            elif tool_type == "module":
                message = await ModuleTool.execute(
                    self, tool_name, arguments, module_path
                )
            elif tool_type == "agentskills":
                message = await AgentSkillTool.execute(
                    self, tool_name, arguments, module_path
                )
            else:
                raise ValueError(f"Unknown tool_type: '{tool_type}'")

            # Validate that the returned message correctly propagates errors
            if isinstance(message, str):
                tools = self.load_tools()
                tool_call_id = tools.get(tool_name, {}).get(
                    "tool_call_id", f"tool_{uuid.uuid4()}"
                )
                message = ToolMessage(
                    content=message,
                    artifact=None,
                    tool_call_id=tool_call_id,
                    additional_kwargs={"is_error": True},
                )

            return message
        except Exception as e:
            content = (
                f"Error executing tool '{tool_name}': {type(e).__name__}: {str(e)}"
            )
            logger.error(content)
            tools = self.load_tools()
            tool_call_id = tools.get(tool_name, {}).get(
                "tool_call_id", f"tool_{uuid.uuid4()}"
            )
            return ToolMessage(
                content=content,
                artifact=None,
                tool_call_id=tool_call_id,
                additional_kwargs={"is_error": True},
            )

    @staticmethod
    def _extract_json(text: str) -> Optional[str]:
        """
        Extract the first valid JSON object from text using stack-based parsing.

        Args:
            text (str): The text to parse for a JSON object.

        Returns:
            Optional[str]: The extracted JSON string, or None if no valid JSON is found.
        """
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
    """
    Utility class for executing function-type tools.
    """

    @classmethod
    async def execute(
        cls, tool_manager: ToolManager, tool_name: str, arguments: Dict[str, Any]
    ):
        """
        Execute a registered function tool.

        Args:
            tool_manager (ToolManager): The ToolManager instance containing registered tools.
            tool_name (str): Name of the function tool to execute.
            arguments (Dict[str, Any]): Arguments to pass to the function.

        Returns:
            ToolMessage: A message containing the execution result or error details.

        Raises:
            Exception: If the function execution fails, logs the error and returns a message.
        """
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
                # Ensure it returns as ToolMessage
                tool_call_id = registered_functions.get(tool_name, {}).get(
                    "tool_call_id", f"tool_{uuid.uuid4()}"
                )
                return ToolMessage(
                    content=content,
                    artifact=None,
                    tool_call_id=tool_call_id,
                    additional_kwargs={"is_error": True},
                )


class MCPTool:
    """
    Utility class for executing MCP-type tools.
    """

    @classmethod
    async def execute(
        cls,
        tool_manager: ToolManager,
        tool_name: str,
        arguments: Dict[str, Any],
        mcp_client: DistributedMCPClient,
        mcp_server_name: str,
    ):
        """
        Execute an MCP tool using the provided client and server.

        Args:
            tool_manager (ToolManager): The ToolManager instance containing registered tools.
            tool_name (str): Name of the MCP tool to execute.
            arguments (Dict[str, Any]): Arguments to pass to the tool.
            mcp_client (DistributedMCPClient): Client for interacting with the MCP server.
            mcp_server_name (str): Name of the MCP server.

        Returns:
            ToolMessage: A message containing the execution result or error details.

        Raises:
            Exception: If the tool execution fails, logs the error and returns a message.
        """
        registered_functions = tool_manager.load_tools()
        """Call the MCP tool natively using the client session."""
        async with mcp_client.session(mcp_server_name) as session:
            payload = {"name": tool_name, "arguments": arguments}
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
                content = (
                    f"Failed to execute mcp tool {tool_name}({arguments}): {str(e)}"
                )
                logger.error(content)
                # raise {"error": content}
                return content


class ModuleTool:
    """
    Utility class for executing module-type tools.
    """

    @classmethod
    async def execute(
        cls,
        tool_manager: ToolManager,
        tool_name: str,
        arguments: Dict[str, Any],
        module_path: Union[str, Path],
        *arg,
        **kwargs,
    ):
        """
        Execute a module-based tool by importing and calling the specified function.

        Args:
            tool_manager (ToolManager): The ToolManager instance containing registered tools.
            tool_name (str): Name of the module tool to execute.
            arguments (Dict[str, Any]): Arguments to pass to the tool.
            module_path (Union[str, Path]): Path to the module containing the tool.

        Returns:
            ToolMessage: A message containing the execution result or error details.

        Raises:
            ImportError, AttributeError: If the module or function cannot be loaded, logs the error and returns a message.
        """
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
            content = (
                f"Failed to execute module tool {tool_name}({arguments}): {str(e)}"
            )
            logger.error(content)
            # raise {"error": content}
            return content


class AgentSkillTool:
    """
    Utility class for executing agentskills-type tools.

    AgentSkill tools do **not** call Python functions directly.
    Instead, the LLM reads the skill's ``SKILL.md`` docstring (which contains
    usage examples and shell command patterns) and constructs a shell command
    string.  This class receives that command and runs it in a subprocess,
    using the skill's ``scripts/`` directory as the working directory so that
    relative paths in the command (e.g. ``python scripts/unpack.py``) resolve
    correctly.

    The ``arguments`` schema for every agentskill tool is fixed::

        {"command": "str"}   # the shell command chosen by the LLM
    """

    # Patterns that strongly indicate Python code rather than a shell command
    _PYTHON_KEYWORDS: frozenset = frozenset(
        [
            "import ",
            "from ",
            "def ",
            "class ",
            "print(",
            "return ",
            "for ",
            "while ",
            "if ",
            "with ",
            "async ",
            "await ",
            "pd.",
            "df.",
            "np.",
            "plt.",
            "os.",
            "sys.",
            "open(",
        ]
    )

    @classmethod
    def _is_python_code(cls, command: str) -> bool:
        """
        Heuristically decide whether *command* is a Python code block (to be
        executed via ``python <tempfile>``) rather than a shell command.

        Rules applied in order (first match wins):

        1. Starts with a shell prefix (``python ``, ``bash ``, etc.) → **shell**.
        2. Contains a shell operator (``|``, ``&&``, ``>``, …) → **shell**.
        3. ``ast.parse`` succeeds **and** the code is multi-line or contains a
           recognisable Python keyword → **Python code**.
        4. Anything else → **shell**.
        """
        import ast as _ast

        stripped = command.strip()

        # Rule 1 — explicit shell invocations are always shell commands
        shell_prefixes = ("python ", "python3 ", "bash ", "sh ", "node ", "ruby ")
        if any(stripped.startswith(p) for p in shell_prefixes):
            return False

        # Rule 2 — shell operators
        if any(op in stripped for op in ("|", "&&", "||", " > ", " >> ", " < ")):
            return False

        # Rule 3 — valid Python AST with Python-like content
        try:
            _ast.parse(stripped)
            if "\n" in stripped or any(kw in stripped for kw in cls._PYTHON_KEYWORDS):
                return True
        except SyntaxError:
            pass

        return False

    @classmethod
    async def execute(
        cls,
        tool_manager: "ToolManager",
        tool_name: str,
        arguments: Dict[str, Any],
        module_path: Union[str, Path],
        *args,
        **kwargs,
    ):
        """
        Execute an agentskill tool as either a **Python code block** or a
        **shell command**, detected automatically from the ``command`` string.

        The LLM constructs the ``command`` argument by reasoning over the
        tool's ``docstring`` (the full ``SKILL.md`` content).

        **Shell command** (default)::

            python scripts/unpack.py document.docx unpacked/
            pandoc --track-changes=all document.docx -o output.md

        Runs via ``subprocess.run(command, shell=True, cwd=working_dir)``.

        **Python code block**::

            import pandas as pd
            df = pd.read_excel('file.xlsx')
            print(df.head())

        Written to a temporary ``.py`` file then executed as
        ``sys.executable <tempfile>`` so that ``print()`` output, imports, and
        tracebacks are captured correctly.

        Detection is handled by :meth:`_is_python_code` (see its docstring).

        Args:
            tool_manager (ToolManager): Used to look up ``tool_call_id``.
            tool_name (str): Registered agentskill tool name.
            arguments (Dict[str, Any]): Must contain ``"command": str``.
            module_path (Union[str, Path]): Working directory (skill root or
                its ``scripts/`` sub-directory).

        Returns:
            ToolMessage: ``content`` = status line; ``artifact`` = stdout
            (or combined stdout/stderr on failure).
        """
        import subprocess
        import tempfile

        registered_functions = tool_manager.load_tools()
        working_dir = Path(module_path).resolve()

        command = arguments.get("command", "")
        if not command:
            content = (
                f"AgentSkillTool '{tool_name}': no 'command' provided in arguments."
            )
            logger.error(content)
            return content

        is_python = cls._is_python_code(command)
        mode = "python-code" if is_python else "shell"
        logger.info(
            f"Executing agentskill tool '{tool_name}' [{mode}] "
            f"(cwd={working_dir}):\n{command}"
        )

        def _run() -> "subprocess.CompletedProcess[str]":
            if is_python:
                # Write the code to a temp file so that multi-line logic,
                # imports, and print() all behave like a normal Python script.
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".py",
                    dir=str(working_dir),
                    delete=False,
                    encoding="utf-8",
                ) as tmp:
                    tmp.write(command)
                    tmp_path = tmp.name
                try:
                    return subprocess.run(
                        [sys.executable, tmp_path],
                        capture_output=True,
                        text=True,
                        cwd=str(working_dir),
                        env=os.environ.copy(),  # inherit any env changes from fix_bug_command
                    )
                finally:
                    try:
                        Path(tmp_path).unlink()
                    except OSError:
                        pass
            else:
                return subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    cwd=str(working_dir),
                    env=os.environ.copy(),  # inherit any env changes from fix_bug_command
                )

        tool_call_id = registered_functions.get(tool_name, {}).get(
            "tool_call_id", "tool_" + str(uuid.uuid4())[:35]
        )

        try:
            result = await asyncio.to_thread(_run)
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            returncode = result.returncode

            out_parts = []
            if stdout:
                out_parts.append(f"STDOUT:\n{stdout}")
            if stderr:
                out_parts.append(f"STDERR:\n{stderr}")

            if returncode == 0:
                artifact = (
                    "\n\n".join(out_parts)
                    if out_parts
                    else "(Completed successfully with no output)"
                )
                content = (
                    f"Completed executing agentskill tool '{tool_name}' [{mode}]: "
                    f"returncode={returncode}"
                    f"\n{artifact}"
                )
                return ToolMessage(
                    content=content, artifact=artifact, tool_call_id=tool_call_id
                )
            else:
                artifact = (
                    "\n\n".join(out_parts) if out_parts else "(Failed with no output)"
                )
                if stderr:
                    content = (
                        f"AgentSkillTool '{tool_name}' [{mode}] exited with "
                        f"returncode={returncode} but output contained an error. STDERR: {stderr}"
                    )
                    logger.warning(content)
                else:
                    content = (
                        f"AgentSkillTool '{tool_name}' [{mode}] exited with "
                        f"returncode={returncode}. STDOUT: {stdout}"
                    )
                    logger.warning(content)

                return ToolMessage(
                    content=content,
                    artifact=artifact,
                    tool_call_id=tool_call_id,
                    additional_kwargs={"is_error": True},
                )

        except Exception as e:
            content = (
                f"Unexpected error in agentskill tool '{tool_name}' [{mode}]: "
                f"{type(e).__name__}: {e}"
            )
            logger.error(content)
            return ToolMessage(
                content=content,
                artifact=None,
                tool_call_id=tool_call_id,
                additional_kwargs={"is_error": True},
            )


class ToolCall(BaseModel):
    tool_name: str = Field(description="Function name of the tool to call")
    arguments: Dict[str, Any] = Field(
        default_factory=dict,
        description="""Dictionary of keyword arguments. Example: {"arg1": "value1", "arg2": "value2",...}""",
    )
    return_: str = Field(
        alias="return", default="str", description="The return of tool_name"
    )
    module_path: str = Field(default="", description="Path to import the tool")
    tool_type: Literal["function", "module", "mcp", "agentskills"] = Field(
        default="function", description="Type of tool"
    )
    tool_call_id: str = Field(default="", description="Tool calling ID")
    is_runtime: bool = Field(
        default=False, description="Runtime value, is True if the tool_type is function"
    )

    @field_validator("arguments", mode="before")
    @classmethod
    def fix_arguments(cls, v):
        if isinstance(v, dict):
            return v

        if isinstance(v, str):
            # Try JSON parse
            try:
                return json.loads(v)
            except:
                pass

            # Convert format: query: 'abc'
            m = re.findall(r"(\w+)\s*:\s*['\"](.+?)['\"]", v)
            if m:
                return {k: val for k, val in m}
        # raise ValueError("arguments must be a dictionary")

    class Config:
        populate_by_name = True
