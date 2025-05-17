import time
import logging
import os
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Response, Query, Path, status
from fastapi.responses import FileResponse, JSONResponse
from typing import List, Optional

from schemas import InferenceResponse, DetectionResult, ErrorResponse, VideoInferenceResponse
from models import YOLOModel, get_model
from utils.image import validate_image, read_image, save_upload_file, draw_detections
from utils.video import validate_video, save_video_file, get_video_info, frame_generator, create_result_video
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

@router.post("/detect-video", response_model=VideoInferenceResponse)
async def detect_objects_in_video(
    file: UploadFile = File(...),
    model: YOLOModel = Depends(get_model),
    max_frames: Optional[int] = Query(None, description="Maximum number of frames to process"),
    skip_frames: int = Query(2, description="Number of frames to skip between processing"),
    start_time: float = Query(0, description="Start time in seconds"),
    end_time: Optional[float] = Query(None, description="End time in seconds"),
    visualize: bool = Query(True, description="Whether to create a result video with detections")
):
    """
    Detect objects in an uploaded video file
    
    Args:
        file: Video file to process
        model: YOLO model instance
        max_frames: Maximum number of frames to process
        skip_frames: Number of frames to skip between processed frames
        start_time: Start time in seconds
        end_time: End time in seconds
        visualize: Whether to create a result video
        
    Returns:
        VideoInferenceResponse: Detection results by frame
    """
    # Validate video
    await validate_video(file)
    
    # Measure start time
    start_time_processing = time.time()
    
    try:
        # Save the video file
        video_path = await save_video_file(file)
        logger.info(f"Video saved to {video_path}")
        
        # Get video info
        video_info = get_video_info(video_path)
        logger.info(f"Video info: {video_info}")
        
        # Limit max frames if not specified
        if max_frames is None:
            max_frames = settings.API.MAX_VIDEO_FRAMES
        
        # Initialize results
        frame_results = {}
        total_detections = 0
        processed_frames = 0
        
        # Process frames
        for frame_idx, frame in frame_generator(
            video_path, max_frames=max_frames, skip_frames=skip_frames,
            start_time=start_time, end_time=end_time
        ):
            # Run inference on frame
            detections = model.predict(frame)
            
            # Store results
            frame_results[frame_idx] = detections
            total_detections += len(detections)
            processed_frames += 1
            
            # Log progress
            if processed_frames % 10 == 0:
                logger.info(f"Processed {processed_frames} frames, found {total_detections} detections so far")
        
        # Calculate processing time
        processing_time = time.time() - start_time_processing
        
        # Create result video if requested
        result_video_path = None
        if visualize:
            try:
                # Ensure results directory exists
                os.makedirs(settings.API.RESULTS_DIR, exist_ok=True)
                
                result_video_path = create_result_video(video_path, frame_results)
                logger.info(f"Result video created at {result_video_path}")
            except Exception as e:
                logger.error(f"Error creating result video: {str(e)}")
        
        # Create response
        response = VideoInferenceResponse(
            video_info=video_info,
            processing_time=processing_time,
            total_frames_processed=processed_frames,
            total_detections=total_detections,
            model_name=model.model_name,
            frames_with_detections=len(frame_results),
            result_video=result_video_path
        )
        
        # Return result video file if requested
        if visualize and result_video_path and os.path.exists(result_video_path):
            return FileResponse(
                result_video_path,
                media_type="video/mp4",
                filename=f"result_{file.filename}"
            )
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing video: {str(e)}"
        )