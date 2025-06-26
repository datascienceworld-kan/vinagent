import json
from pathlib import Path
from typing import Any
from vinagent.agent import Agent
from vinagent.config.logger_config import setup_logger
from vinagent.util.tool_extractor import ToolMetadataExtractor
from langchain_core.messages.tool import ToolMessage
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import importlib

logger = setup_logger(__name__,"vinagent_analysis.log")

class VinAgent(Agent):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tool_dir = Path("templates")
        self.tool_dir.mkdir(parents=True, exist_ok=True)
        self.tools_metadata = self._load_or_generate_tool_metadata()

    def _load_or_generate_tool_metadata(self) -> dict:
        tools_metadata = {}

        for module_path in self.tools:
            try:
                module_name = module_path.split(".")[-1]
                tool_file = self.tool_dir / f"{module_name}.json"

                if tool_file.exists():
                    with open(tool_file, "r", encoding="utf-8") as f:
                        metadata = json.load(f)
                        logger.info(f"[VinAgent] Loaded metadata from {tool_file}")
                else:
                    extractor = ToolMetadataExtractor(module_path)
                    metadata = extractor.extract_metadata()
                    extractor.export_to_json()
                    logger.info(f"[VinAgent] Extracted and saved metadata to {tool_file}")

                tools_metadata.update(metadata)

            except Exception as e:
                logger.error(f"[VinAgent] Failed to load metadata from {module_path}: {e}", exc_info=True)

        return tools_metadata
    

    async def invoke_with_tool_async(self, query: str) -> Any:
        messages = self._build_prompt_messages(query, self.tools_metadata)
        try:
            response = await self.llm.ainvoke(messages)
            tool_data = self._extract_json(response.content)

            if not tool_data or ("None" in tool_data) or (tool_data == "{}"):
                return response
            
            tool_call = json.loads(tool_data)

            return self._execute_tool(
                tool_call["tool_name"], tool_call["arguments"], tool_call["module_path"]
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"[VinAgent] Async tool calling failed: {str(e)}")
            return AIMessage(content=f"[VinAggent] Failed to invoke agent: {str(e)}")
        

    def _build_prompt_messages(self, query: str, tools: dict) -> list:
        prompt = (
            "You are given a task and a list of available tools.\n"
            f"- Task: {query}\n"
            f"- Tools list: {json.dumps(tools)}\n\n"
            "Instructions:\n"
            "- If the task can be solved without tools, just return the answer without any explanation\n"
            "- If the task requires a tool, select the appropriate tool with its relevant arguments from Tools list according to following format (no explanations, no markdown):\n"
            "{\n"
            '"tool_name": "Function name",\n'
            '"arguments": "A dictionary of keyword-arguments to execute tool_name",\n'
            '"module_path": "Path to import the tool"\n'
            "}\n"
            "Let's say I don't know and suggest where to search if you are unsure the answer.\n"
            "Not make up anything.\n"
        )
        skills = "- ".join(self.skills)
        return [
            SystemMessage(content=f"{self.description}\nHere is your skills: {skills}"),
            HumanMessage(content=prompt),
        ]

    def _execute_tool(self, tool_name: str, arguments: dict, module_path: str) -> Any:
        """""
        Override: Execute the specified tool with given arguments, without relying on ToolManager
        """""
        try:
            if (
                    module_path == "__runtime__"
                    and tool_name in getattr(self, "_registered_functions", {})
            ):
                func = self._registered_functions[tool_name]
                content = f"Completed executing tool {tool_name}({arguments})"
                logger.info(f"[VinAgent] {content}")
                artifact = func(**arguments)
                tool_call_id = f"__runtime__.{tool_name}"
                return ToolMessage(content=content, artifact=artifact, tool_call_id=tool_call_id)

            module = importlib.import_module(module_path)
            func = getattr(module, tool_name)

            content = f"Completed executing tool {tool_name}({arguments})"
            logger.info(f"[VinAgent] {content}")

            artifact = func(**arguments)
            tool_call_id = f"{module_path}.{tool_name}"

            return ToolMessage(
                content=content,
                artifact=artifact,
                tool_call_id=tool_call_id
            )

        except Exception as e:
            logger.error(f"[VinAgent] Failed to execute tool '{tool_name}' from '{module_path}': {e}",
                         exc_info=True)
            raise