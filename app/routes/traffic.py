import time
import logging
import os
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from typing import List, Dict, Optional

from schemas import InferenceResponse, DetectionResult, TrafficAnalysisResponse
from models import YOLOModel, get_model
from utils.image import validate_image, read_image, save_upload_file, draw_detections
from utils.monitoring import track_inference_metrics
from config import settings

# Initialize logger
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["Traffic Analysis"])

@router.post("/detect/traffic", response_model=InferenceResponse)
async def detect_traffic(
    file: UploadFile = File(...),
    visualize: bool = Query(True, description="Whether to return the image with detections visualized"),
    count_vehicles: bool = Query(True, description="Whether to count vehicles by type"),
    model: YOLOModel = Depends(get_model)
):
    """
    Specialized endpoint for traffic analysis
    
    Args:
        file: Image file to process
        visualize: Whether to visualize detections
        count_vehicles: Whether to count vehicles by type
        model: YOLO model instance
        
    Returns:
        InferenceResponse or FileResponse: Detection results or visualized image
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
        
        # Filter for traffic-related classes
        traffic_classes = ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'person', 'traffic light', 'stop sign']
        filtered_detections = [d for d in detections if d.class_name.lower() in traffic_classes]
        
        # Track metrics
        track_inference_metrics(model.model_name, filtered_detections)
        
        # Count vehicles by type if requested
        vehicle_counts = {}
        if count_vehicles:
            for detection in filtered_detections:
                class_name = detection.class_name.lower()
                if class_name in vehicle_counts:
                    vehicle_counts[class_name] += 1
                else:
                    vehicle_counts[class_name] = 1
            
            logger.info(f"Vehicle counts: {vehicle_counts}")
        
        # Save and visualize if requested
        if visualize:
            file_path = await save_upload_file(file)
            
            # Draw detections on the image
            result_image = draw_detections(image, filtered_detections)
            
            # Add vehicle count text
            if count_vehicles:
                y_pos = 30
                import cv2
                for vehicle_type, count in vehicle_counts.items():
                    cv2.putText(
                        result_image,
                        f"{vehicle_type}: {count}",
                        (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA
                    )
                    y_pos += 25
            
            # Save result image
            result_path = os.path.join(
                os.path.dirname(file_path),
                f"traffic_{os.path.basename(file_path)}"
            )
            import cv2
            cv2.imwrite(result_path, result_image)
            
            # Serve the result image as a response
            return FileResponse(
                result_path,
                media_type="image/jpeg",
                filename=f"traffic_{file.filename}"
            )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Create response
        response = InferenceResponse(
            detections=filtered_detections,
            processing_time=processing_time,
            image_width=original_size[0],
            image_height=original_size[1],
            model_name=model.model_name,
            total_detections=len(filtered_detections)
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing traffic image: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing traffic image: {str(e)}"
        )