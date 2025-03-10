import logging
import sys
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.exceptions import HTTPException as StarletteHTTPException

# Import application modules
from config import settings
from middleware import RequestIDMiddleware, LoggingMiddleware, PrometheusMiddleware
from schemas import ErrorResponse
from routes import api_router

# Configure logging
def configure_logging():
    """Configure application logging"""
    log_level = getattr(logging, settings.LOGGING.LEVEL.upper())
    
    # Configure basic logging
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

# Configure logging
configure_logging()

# Create application
app = FastAPI(
    title=settings.API.TITLE,
    description=settings.API.DESCRIPTION,
    version=settings.API.VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    debug=settings.API.DEBUG,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.API.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware
app.add_middleware(RequestIDMiddleware)
app.add_middleware(LoggingMiddleware)
app.add_middleware(PrometheusMiddleware)

# Add exception handlers
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions and return standardized response"""
    request_id = getattr(request.state, "request_id", "unknown")
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=str(exc.detail),
            request_id=request_id
        ).dict(),
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Handle unhandled exceptions"""
    request_id = getattr(request.state, "request_id", "unknown")
    logging.exception(f"Unhandled exception: {str(exc)}", extra={"request_id": request_id})
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="An unexpected error occurred",
            detail=str(exc) if settings.API.DEBUG else None,
            request_id=request_id
        ).dict(),
    )

# Add Prometheus metrics endpoint
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Include all API routes
app.include_router(api_router)

# Entry point for running the application directly
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.API.DEBUG,
        log_level=settings.LOGGING.LEVEL.lower(),
    )