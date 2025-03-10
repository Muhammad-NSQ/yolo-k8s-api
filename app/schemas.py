from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uuid


class DetectionResult(BaseModel):
    """Schema for a single detection result"""
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float] = Field(..., description="Bounding box in format [x1, y1, x2, y2]")


class InferenceResponse(BaseModel):
    """Schema for inference response"""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    detections: List[DetectionResult]
    processing_time: float
    image_width: int
    image_height: int
    model_name: str
    total_detections: int


class ErrorResponse(BaseModel):
    """Schema for error responses"""
    error: str
    detail: Optional[str] = None
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class HealthResponse(BaseModel):
    """Schema for health check response"""
    status: str
    version: str
    is_model_loaded: bool = True
    
    model_config = {
        "protected_namespaces": ()
    }


class TrafficAnalysisResponse(BaseModel):
    """Schema for traffic analysis response extending inference response"""
    vehicle_counts: Dict[str, int]
    total_vehicles: int
    pedestrians: int
    traffic_signals: int