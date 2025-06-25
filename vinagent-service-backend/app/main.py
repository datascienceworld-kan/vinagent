from fastapi import FastAPI
from app.routes.api_route import api_router as api_router

app = FastAPI(title="Vinagent Service Backend")

app.include_router(api_router, prefix="/api")