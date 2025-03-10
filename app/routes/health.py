from fastapi import APIRouter, Depends
from schemas import HealthResponse
from config import settings
from models import get_model

# Create router
router = APIRouter(tags=["Health"])

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint to verify that the API is running
    
    Returns:
        HealthResponse: Status information about the API
    """
    # Get model instance to verify it's loaded
    model = get_model()
    
    return HealthResponse(
        status="ok",
        version=settings.API.VERSION,
        is_model_loaded=model.model is not None
    )


@router.get("/readiness", response_model=HealthResponse)
async def readiness_check():
    """
    Readiness probe endpoint for Kubernetes
    
    Checks if the service is ready to accept traffic
    
    Returns:
        HealthResponse: Status information about the API
    """
    # Similar to health check but more focused on service readiness
    model = get_model()
    
    return HealthResponse(
        status="ready",
        version=settings.API.VERSION,
        is_model_loaded=model.model is not None
    )


@router.get("/liveness", response_model=HealthResponse)
async def liveness_check():
    """
    Liveness probe endpoint for Kubernetes
    
    Verifies that the application is running and not deadlocked
    
    Returns:
        HealthResponse: Minimal status information
    """
    # A very lightweight check to ensure the service is alive
    return HealthResponse(
        status="alive",
        version=settings.API.VERSION,
        is_model_loaded=True  # Simplified check for liveness
    )