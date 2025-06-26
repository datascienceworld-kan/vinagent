from app.services.tool_service import ToolService
from app.services.model_service import ModelService
from fastapi import WebSocket, WebSocketDisconnect
from app.services.agent_service import AgentService
from vinagent.config.logger_config import setup_logger
from app.core.config import get_config
from app.schemas.setting_request import SettingRequest
import json
import uuid
import together
from app.schemas.handle_query_response import QueryResponse, ChatMessage
from app.schemas.conversion_request import ConversionRequest
from fastapi.responses import FileResponse
from app.services.convert_service import convert_markdown_to_file, MarkdownConversionError

logger = setup_logger(__name__, "vinagent.log")
agent_service = AgentService(get_config())

class VinagentController:
    
    def __init__(self, env_service):
        self.env_service = env_service
        self.discover_tools = ToolService.discover_tools
        self.list_model = together.Models.list()

    def list_available_tools(self):
        return self.discover_tools
    
    def save_config(self, request_data: SettingRequest) -> dict:
        try:
            self.env_service.update_variable("LLM_MODEL", request_data.model_id)
            self.env_service.update_variable("AGENT_DESCRIPTION", request_data.description)
            skills_str = ",".join(request_data.skills)
            self.env_service.update_variable("AGENT_SKILLS", skills_str)
            logger.info("Environment variables updated successfully")
            return {"status": "success", "message": "Environment variables updated"}
        except Exception as e:
            logger.exception("Failed to update environment variables")
            raise e
        
    def list_models(self):
        return ModelService.filter_models(self.list_model)
        

    def convert_markdown(self, request_data: ConversionRequest) -> FileResponse:
        try:
            output_filepath = convert_markdown_to_file(
                request_data.markdown,
                request_data.format
            )

            media_type = (
                "application/pdf"
                if request_data.format == "pdf"
                else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
            filename = f"document.{request_data.format}"

            logger.info(f"Markdown converted to {filename}, saved at {output_filepath}")
            return FileResponse(
                path=output_filepath,
                filename=filename,
                media_type=media_type
            )
        except MarkdownConversionError as e:
            logger.exception("Markdown conversion failed due to a known error.")
            raise e
        except Exception as e:
            logger.exception("Unexpected error occurred during markdown conversion.")
            raise e

    
    async def websocket_handler(websocket: WebSocket):
        await websocket.accept()
        logger.info("WebSocket connected")
        try:
            while True:
                data = await websocket.receive_text()
                try:
                    payload = json.loads(data)
                    response_payload = await agent_service.process_payload(payload)
                    await websocket.send_text(json.dumps(response_payload))
                except json.JSONDecodeError:
                    logger.error(f"Received invalid JSON: {data}")
                    error_payload = QueryResponse(
                        query_id="error_" + str(uuid.uuid4()),
                        chat_message=ChatMessage(from_="agent", text="[Error]: Invalid JSON format"),
                        artifact_data=None
                    )
                    await websocket.send_text(json.dumps(error_payload))
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected")
        except Exception as e:
            logger.error(f"Unexpected WebSocket error: {e}")