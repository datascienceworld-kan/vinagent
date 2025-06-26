import asyncio
import uuid
import pandas as pd
import plotly.graph_objects as go
from typing import List, Dict, Any, Union

from vinagent.config.logger_config import setup_logger
from vinagent.agent.agent_extend import VinAgent
from vinagent.util.llm_factory import LLMFactory
from vinagent.util.langchain_message import AIMessage, ToolMessage

from app.utils.util import Utils
from app.schemas.handle_query_response import QueryResponse, ChatMessage
from app.schemas.agent_config import AgentConfig

logger = setup_logger(__name__, "vinagent.log")

class AgentService:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.agent_description = config.description
        self.agent_skills = config.skils

    def create_agent(self, tools: List[str]) -> VinAgent:
        llm_instance = LLMFactory.create_llm(self.config.model)
        return VinAgent(
            description=self.agent_description,
            llm=llm_instance,
            skills=self.agent_skills,
            tools=tools,
        )
    
    async def invoke_agent_async(self, agent: VinAgent, query: str) -> Any:
        result = await agent.invoke_with_tool_async(query)
        return result
    
    def extract_artifact_data(self,artifact) -> Union[Dict[str, Any], None]:

        if isinstance(artifact, pd.DataFrame):
            return {
                "type": "table",
                "columns": artifact.columns.tolist(),
                "rows": artifact.values.tolist()
            }
        elif isinstance(artifact, go.Figure):
            return {
                "type": "plotly",
                "content": artifact.to_json()
            }
        elif isinstance(artifact, str):
            if Utils.look_like_markdown(artifact):
                return {
                    "type": "markdown",
                    "content": artifact
                }

        return None
    

    async def handle_query(self,  query: str, selected_tools: List[str]) -> Dict[str, Any]:
        try:
            logger.info(f"Handling query: '{query}' with tools: {selected_tools}")
            first_agent_tools = ['vinagent.tools.websearch_tools'] if selected_tools else []
            first_agent = self.create_agent(first_agent_tools)
            first_agent_task = asyncio.create_task(self.invoke_agent_async(first_agent, query))
            logger.debug("First agent task created.")

            second_agent_task = None
            use_second_agent = selected_tools and selected_tools[0] != 'vinagent.tools.websearch_tools'

            if use_second_agent:
                logger.debug(f"Second agent will be used with tools: {selected_tools}")
                second_agent = self.create_agent(selected_tools)
                second_agent_task = asyncio.create_task(self.invoke_agent_async(second_agent, query))

            first_agent_result = await first_agent_task
            logger.debug(f"First agent result: {first_agent_result}")
            if isinstance(first_agent_result, AIMessage):
                chat_message_text = first_agent_result.content
            elif isinstance(first_agent_result, ToolMessage):
                artifact = first_agent_result.artifact
                chat_message_text = artifact if isinstance(artifact, str) else str(artifact)
            else:
                chat_message_text = str(first_agent_result)

            second_agent_result = await second_agent_task if second_agent_task else None
            if second_agent_result is not None:
                logger.debug(f"Second agent result: {second_agent_result}")

            logger.info("Returning final response to user.")
            artifact_data = None
            if isinstance(second_agent_result, ToolMessage):
                artifact_data = self.extract_artifact_data(second_agent_result.artifact)

            return QueryResponse(
                query_id=str(uuid.uuid4()),
                chat_message=ChatMessage(from_="agent", text=chat_message_text),
                artifact_data=artifact_data
            )

        except Exception as e:
            logger.error(f"Error during agent invocation with tools {selected_tools}: {e}", exc_info=True)
            return QueryResponse(
                query_id=str(uuid.uuid4()),
                chat_message=ChatMessage(from_="agent", text=f"[Error]: Cannot process the request: {str(e)}"),
                artifact_data=None
            )
        
    async def process_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        query = payload.get("query")
        selected_tools = payload.get("selected_tools", [])

        if not query:
            return QueryResponse(
                query_id="error_" + str(uuid.uuid4()),
                chat_message=ChatMessage(from_="agent", text="[Error]: Empty query"),
                artifact_data=None
            )

        try:
            return await self.handle_query(query, selected_tools)
        except Exception as e:
            return QueryResponse(
                query_id="error_" + str(uuid.uuid4()),
                chat_message=ChatMessage(from_="agent", text=f"[Error]: {str(e)}"),
                artifact_data=None
            )
