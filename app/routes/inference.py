import time
import logging
import os
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Response, Query, Path
from fastapi.responses import FileResponse, JSONResponse
from typing import List, Optional

from schemas import InferenceResponse, DetectionResult, ErrorResponse
from models import YOLOModel, get_model
from utils.image import validate_image, read_image, save_upload_file, draw_detections
from utils.monitoring import track_inference_metrics, measure_latency
from config import settings

# Initialize logger
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["Inference"])

@router.post("/detect", response_model=InferenceResponse)
async def detect_objects(
    file: UploadFile = File(...),
    model: YOLOModel = Depends(get_model),
    visualize: bool = Query(False, description="Whether to return the image with detections visualized")
):
    """
    Detect objects in an uploaded image
    
    Args:
        file: Image file to process
        model: YOLO model instance
        visualize: Whether to visualize detections
        
    Returns:
        InferenceResponse: Detection results
    """
    # Validate image
    await validate_image(file)
    
    # Measure start time
    start_time = time.time()
    
    try:
        # Read and preprocess image
        image, original_size = await read_image(file)
        
        # Run inference
        detections = model.predict(image)
        
        # Track metrics
        track_inference_metrics(model.model_name, detections)
        
        # Save upload if configured to do so
        file_path = None
        if visualize:
            file_path = await save_upload_file(file)
            
            # Draw detections on the image if requested
            result_image = draw_detections(image, detections)
            
            # Save result image
            result_path = os.path.join(
                os.path.dirname(file_path),
                f"result_{os.path.basename(file_path)}"
            )
            import cv2
            cv2.imwrite(result_path, result_image)
            
            # Serve the result image as a response
            return FileResponse(
                result_path,
                media_type="image/jpeg",
                filename=f"result_{file.filename}"
            )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Create response
        response = InferenceResponse(
            detections=detections,
            processing_time=processing_time,
            image_width=original_size[0],
            image_height=original_size[1],
            model_name=model.model_name,
            total_detections=len(detections)
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )