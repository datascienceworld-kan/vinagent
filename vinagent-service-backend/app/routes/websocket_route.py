from fastapi import APIRouter, WebSocket
from app.controllers.vinagent_controller import VinagentController
from app.services.env_config_service import EnvConfigService

router = APIRouter()
controller = VinagentController(EnvConfigService(".env"))

@router.websocket("/agent")
async def websocket_agent(websocket: WebSocket):
    await controller.websocket_handler(websocket)
