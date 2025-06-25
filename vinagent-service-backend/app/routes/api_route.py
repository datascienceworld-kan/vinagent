from vinagent.config.logger_config import setup_logger
from app.controllers.vinagent_controller import VinagentController
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

logger = setup_logger("api_router", "vinagent.log")

api_router = APIRouter()
vinagent_controller = VinagentController()

@api_router.get("/tools")
async def list_tools():
    """Returns a list of available agent tool module names."""
    available_tools = vinagent_controller.discover_tools()
    logger.info(f"Discovered tools: {available_tools}")
    return JSONResponse(content=available_tools)
