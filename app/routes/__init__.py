from fastapi import APIRouter

from .health import router as health_router
from .inference import router as inference_router
from .traffic import router as traffic_router

# Create a combined router for easy inclusion in main app
api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(inference_router, prefix="/v1")
api_router.include_router(traffic_router, prefix="/v1")