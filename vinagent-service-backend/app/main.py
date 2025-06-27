from fastapi import FastAPI
from app.routes.api_route import api_router as api_router
from app.routes.websocket_route import router as websocket_router
from app.core.setting import get_settings
from fastapi.middleware.cors import CORSMiddleware

config = get_settings()

frontend_origin = config.frontend_origin

app = FastAPI(title="Vinagent Service Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=frontend_origin,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(websocket_router, prefix="/ws")
app.include_router(api_router, prefix="/api")