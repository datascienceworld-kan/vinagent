from vinagent.config.logger_config import setup_logger
from app.controllers.vinagent_controller import VinagentController

from app.services.env_config_service import EnvConfigService
from app.schemas.conversion_request import ConversionRequest
from app.services.convert_service import MarkdownConversionError
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from app.schemas.setting_request import SettingRequest

logger = setup_logger(__name__, "vinagent.log")

api_router = APIRouter()
env_service = EnvConfigService(".env")
vinagent_controller = VinagentController(env_service=env_service)


@api_router.get("/tools")
async def list_tools():
    """Returns a list of available agent tool module names."""
    available_tools = vinagent_controller.discover_tools()
    logger.info(f"Discovered tools: {available_tools}")
    return JSONResponse(content=available_tools)

@api_router.post("/save-config")
async def save_config(request_data: SettingRequest):
    try:
        result = vinagent_controller.save_config(request_data=request_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@api_router.post("/convert-markdown")
async def convert_markdown(request_data: ConversionRequest):
    try:
        return vinagent_controller.convert_markdown(request_data)
    except MarkdownConversionError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")
    

@api_router.get("/models", response_class=JSONResponse)
async def list_models():
    try:
        return vinagent_controller.list_models()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching models: {str(e)}")