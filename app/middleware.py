import time
import uuid
import logging
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

# Initialize logger
logger = logging.getLogger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware to add a unique request ID to each incoming request"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Add request ID to response headers
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        
        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for structured request/response logging"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Capture request information
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        path = request.url.path
        method = request.method
        
        # Process the request
        try:
            response = await call_next(request)
            status_code = response.status_code
            processing_time = time.time() - start_time
            
            # Log request information
            log_data = {
                "request_id": request_id,
                "method": method,
                "path": path,
                "status_code": status_code,
                "processing_time_ms": round(processing_time * 1000, 2)
            }
            
            if 200 <= status_code < 400:
                logger.info(f"Request completed successfully", extra=log_data)
            else:
                logger.warning(f"Request completed with error", extra=log_data)
                
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(
                f"Request failed with exception: {str(e)}",
                extra={
                    "request_id": request_id,
                    "method": method,
                    "path": path,
                    "processing_time_ms": round(processing_time * 1000, 2),
                    "exception": str(e)
                },
                exc_info=True
            )
            raise


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Middleware to collect metrics for Prometheus"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        # Placeholder for prometheus metrics implementation
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Process the request
        response = await call_next(request)
        
        # Record request duration - real implementation would use prometheus_client
        processing_time = time.time() - start_time
        
        # For now, just return the response
        return response