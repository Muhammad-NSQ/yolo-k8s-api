from prometheus_client import Counter, Histogram, Gauge
import time
from functools import wraps
import logging

# Initialize logger
logger = logging.getLogger(__name__)

# Initialize metrics
REQUEST_COUNT = Counter(
    "api_request_count_total",
    "Total count of API requests",
    ["method", "endpoint", "status"]
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "API request latency in seconds",
    ["method", "endpoint"]
)

INFERENCE_LATENCY = Histogram(
    "model_inference_latency_seconds",
    "Model inference latency in seconds",
    ["model_name"]
)

DETECTION_COUNT = Counter(
    "model_detection_count_total",
    "Total count of objects detected",
    ["model_name", "class_name"]
)

ACTIVE_REQUESTS = Gauge(
    "api_active_requests",
    "Number of active requests",
    ["endpoint"]
)

def measure_latency(func):
    """
    Decorator to measure function execution time
    
    Args:
        func: The function to measure
        
    Returns:
        The wrapped function with latency measurement
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            # Execute the function
            result = await func(*args, **kwargs)
            
            # Calculate latency
            latency = time.time() - start_time
            
            # Log metrics (in a real implementation, you'd use more context)
            logger.debug(f"Function {func.__name__} took {latency:.4f} seconds")
            
            return result
        except Exception as e:
            # Log exception
            latency = time.time() - start_time
            logger.error(f"Function {func.__name__} failed after {latency:.4f} seconds: {str(e)}")
            raise
    
    return wrapper


def track_inference_metrics(model_name, detections=None):
    """
    Track detection metrics
    
    Args:
        model_name: Name of the model used
        detections: List of detection results (optional)
    """
    if detections:
        # Count detections by class
        class_counts = {}
        for detection in detections:
            class_name = detection.class_name
            if class_name in class_counts:
                class_counts[class_name] += 1
            else:
                class_counts[class_name] = 1
        
        # Update metrics
        for class_name, count in class_counts.items():
            DETECTION_COUNT.labels(model_name=model_name, class_name=class_name).inc(count)


def track_request_metrics(method, endpoint, status_code):
    """
    Track request metrics
    
    Args:
        method: HTTP method
        endpoint: API endpoint
        status_code: HTTP status code
    """
    REQUEST_COUNT.labels(
        method=method,
        endpoint=endpoint,
        status=status_code
    ).inc()


def time_request(method, endpoint):
    """
    Context manager to time a request
    
    Args:
        method: HTTP method
        endpoint: API endpoint
        
    Returns:
        Context manager that times the request
    """
    class RequestTimer:
        def __enter__(self):
            self.start_time = time.time()
            ACTIVE_REQUESTS.labels(endpoint=endpoint).inc()
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            latency = time.time() - self.start_time
            REQUEST_LATENCY.labels(
                method=method,
                endpoint=endpoint
            ).observe(latency)
            ACTIVE_REQUESTS.labels(endpoint=endpoint).dec()
    
    return RequestTimer()