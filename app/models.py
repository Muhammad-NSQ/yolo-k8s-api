import os
import time
import logging
import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional
from ultralytics import YOLO
from pathlib import Path
import cv2

# Import Triton client libraries
try:
    import tritonclient.grpc as triton_grpc
    import tritonclient.utils as triton_utils
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    logging.warning("Triton client libraries not available. Install with: pip install tritonclient[grpc]")

from config import settings
from schemas import DetectionResult

# Initialize logger
logger = logging.getLogger(__name__)

# COCO class names
COCO_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
    10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird',
    15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
    20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
    25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
    30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
    35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
    45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
    50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
    55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
    60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
    65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven',
    70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock',
    75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}

class YOLOModel:
    """YOLO model wrapper for object detection with TensorRT and Triton support"""
    
    def __init__(self):
        """Initialize the YOLO model"""
        self.model = None
        self.triton_client = None
        self.model_name = settings.MODEL.MODEL_NAME
        self.device = settings.MODEL.DEVICE
        self.conf_threshold = settings.MODEL.CONFIDENCE_THRESHOLD
        self.iou_threshold = settings.MODEL.IOU_THRESHOLD
        self.max_detections = settings.MODEL.MAX_DETECTIONS
        self.use_optimization = settings.MODEL.USE_TENSORRT  # Flag for TensorRT optimization
        self.use_fp16 = settings.MODEL.TENSORRT_FP16
        self.optimize_dir = settings.MODEL.TENSORRT_CACHE_DIR
        self.use_triton = settings.MODEL.USE_TRITON  # Flag for Triton Inference Server
        self.triton_url = settings.MODEL.TRITON_URL
        self.triton_model_name = settings.MODEL.TRITON_MODEL_NAME
        self.triton_model_version = settings.MODEL.TRITON_MODEL_VERSION
        # Add settings for Triton FP16 model
        self.triton_use_fp16 = getattr(settings.MODEL, 'TRITON_USE_FP16', False)
        self.triton_fp16_model_name = getattr(settings.MODEL, 'TRITON_FP16_MODEL_NAME', 'yolov8trt_fp16')  # Use exact model name
        self.triton_fp16_model_version = getattr(settings.MODEL, 'TRITON_FP16_MODEL_VERSION', self.triton_model_version)
        
        # Set class names for COCO dataset
        self.class_names = COCO_CLASSES
        
        # Check CUDA availability first and adjust device if needed
        try:
            if self.device.startswith('cuda') and not torch.cuda.is_available():
                logger.warning(f"CUDA requested but not available. Falling back to CPU.")
                self.device = 'cpu'
                self.use_optimization = False
        except Exception as e:
            logger.warning(f"Error checking CUDA: {str(e)}. Falling back to CPU.")
            self.device = 'cpu'
            self.use_optimization = False
        
        # Load model
        self.load_model()
    
    def load_model(self):
        """Load the YOLO model with TensorRT or Triton optimization if available"""
        try:
            logger.info(f"Loading YOLO model: {self.model_name}")
            start_time = time.time()
            
            # Set environment variable to bypass the security check
            os.environ["ULTRALYTICS_SKIP_TORCH_WEIGHTS_WARNING"] = "1"
            
            # Check if Triton should be used
            if self.use_triton and TRITON_AVAILABLE:
                logger.info(f"Using Triton Inference Server at {self.triton_url}")
                try:
                    # Initialize Triton client
                    self.triton_client = triton_grpc.InferenceServerClient(
                        url=self.triton_url,
                        verbose=False
                    )
                    
                    # Check if the server is ready
                    if not self.triton_client.is_server_ready():
                        logger.error("Triton server is not ready")
                        raise RuntimeError("Triton server is not ready")
                    
                    # Determine which model to use (FP16 or regular)
                    active_model_name = self.triton_fp16_model_name if self.triton_use_fp16 else self.triton_model_name
                    active_model_version = self.triton_fp16_model_version if self.triton_use_fp16 else self.triton_model_version
                    
                    logger.info(f"Using Triton model: {active_model_name} (FP16: {self.triton_use_fp16})")
                    
                    # Check if the model is ready
                    if not self.triton_client.is_model_ready(
                        active_model_name, active_model_version
                    ):
                        logger.error(f"Model {active_model_name} is not ready on Triton server")
                        
                        # Fall back to other precision model if available
                        if self.triton_use_fp16:
                            logger.warning("FP16 model not ready, trying FP32 model")
                            self.triton_use_fp16 = False
                            active_model_name = self.triton_model_name
                            active_model_version = self.triton_model_version
                            
                            if not self.triton_client.is_model_ready(
                                active_model_name, active_model_version
                            ):
                                logger.error(f"FP32 model {active_model_name} also not ready")
                                raise RuntimeError(f"No available models ready on Triton server")
                            else:
                                logger.info(f"Successfully switched to FP32 model {active_model_name}")
                        else:
                            # If FP32 model not ready, try FP16 model
                            logger.warning("FP32 model not ready, trying FP16 model")
                            self.triton_use_fp16 = True
                            active_model_name = self.triton_fp16_model_name
                            active_model_version = self.triton_fp16_model_version
                            
                            if not self.triton_client.is_model_ready(
                                active_model_name, active_model_version
                            ):
                                logger.error(f"FP16 model {active_model_name} also not ready")
                                raise RuntimeError(f"No available models ready on Triton server")
                            else:
                                logger.info(f"Successfully switched to FP16 model {active_model_name}")
                    
                    # Store the active model names for later use
                    self.active_triton_model_name = active_model_name
                    self.active_triton_model_version = active_model_version
                    
                    # Get model metadata to understand input/output shapes
                    self.model_metadata = self.triton_client.get_model_metadata(
                        active_model_name, active_model_version
                    )
                    self.model_config = self.triton_client.get_model_config(
                        active_model_name, active_model_version
                    )
                    
                    logger.info(f"Successfully connected to Triton server and verified model {active_model_name}")
                    
                except Exception as e:
                    logger.error(f"Failed to initialize Triton client: {str(e)}")
                    logger.warning("Falling back to local model")
                    self.use_triton = False
                    
            # If not using Triton, load the model locally
            if not self.use_triton or not TRITON_AVAILABLE:
                # Check for pre-exported TensorRT engine if optimization is enabled
                if self.use_optimization and self.device.startswith('cuda'):
                    logger.info("TensorRT optimization enabled, checking for TensorRT engine")
                    
                    # Determine path to optimized model
                    model_basename = Path(self.model_name).stem
                    engine_path = Path(self.optimize_dir) / f"{model_basename}.engine"
                    
                    if engine_path.exists():
                        logger.info(f"Loading TensorRT engine: {engine_path}")
                        try:
                            # Load TensorRT engine directly
                            self.model = YOLO(str(engine_path))
                            logger.info("TensorRT engine loaded successfully")
                        except Exception as e:
                            logger.error(f"Failed to load TensorRT engine: {str(e)}")
                            logger.warning("Falling back to standard PyTorch model")
                            self.model = YOLO(self.model_name)
                            self.model.to(self.device)
                    else:
                        logger.info(f"TensorRT engine not found at {engine_path}")
                        logger.info("Loading standard PyTorch model")
                        self.model = YOLO(self.model_name)
                        self.model.to(self.device)
                else:
                    # Load standard PyTorch model
                    logger.info("Loading standard PyTorch model")
                    self.model = YOLO(self.model_name)
                    logger.info(f"Moving model to device: {self.device}")
                    self.model.to(self.device)
            
            load_time = time.time() - start_time
            logger.info(f"Model initialization completed in {load_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            
            # Attempt to fall back to CPU if GPU loading failed
            if not self.use_triton and self.device != 'cpu':
                logger.info("Attempting to fall back to CPU")
                try:
                    self.device = 'cpu'
                    self.model = YOLO(self.model_name)
                    self.model.to(self.device)
                    logger.info("Successfully loaded model on CPU")
                except Exception as e2:
                    logger.error(f"CPU fallback also failed: {str(e2)}")
                    raise RuntimeError(f"Failed to load YOLO model: {str(e)}, CPU fallback also failed: {str(e2)}")
            else:
                raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def predict(self, image: np.ndarray) -> List[DetectionResult]:
        """
        Run inference on an image
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List[DetectionResult]: Detection results
        """
        # Measure inference time
        start_time = time.time()
        
        try:
            # Check if using Triton
            if self.use_triton and self.triton_client is not None:
                # Process with Triton
                detections = self._predict_with_triton(image)
            else:
                # Process with local model
                if self.model is None:
                    logger.error("Model not loaded")
                    raise RuntimeError("Model not loaded")
                
                # Run inference with local model
                with torch.no_grad():
                    # Check if model is TensorRT engine
                    model_path = getattr(self.model, "ckpt_path", "")
                    is_tensorrt = str(model_path).endswith(".engine")
                    
                    # Prediction args
                    pred_args = {
                        'conf': self.conf_threshold,
                        'iou': self.iou_threshold,
                        'max_det': self.max_detections,
                        'verbose': False
                    }
                    
                    # For TensorRT engines, specify device
                    if is_tensorrt:
                        pred_args['device'] = self.device
                    
                    # Run prediction
                    results = self.model.predict(image, **pred_args)
                
                # Process results
                detections = self._process_results(results[0])
            
            # Log performance
            inference_time = time.time() - start_time
            logger.debug(f"Inference completed in {inference_time:.4f} seconds with {len(detections)} detections")
            
            return detections
            
        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            raise RuntimeError(f"Inference failed: {str(e)}")
    
    def _predict_with_triton(self, image: np.ndarray) -> List[DetectionResult]:
        """
        Run inference using Triton Inference Server for yolov8trt
        Modified to provide exact fixed input shape required by Triton
        
        Args:
            image: Input image as numpy array (in BGR format from OpenCV)
            
        Returns:
            List[DetectionResult]: Detection results
        """
        try:
            # Get original image dimensions
            orig_h, orig_w = image.shape[:2]
            
            # Force exact input shape required by Triton (1, 3, 640, 640)
            input_shape = (640, 640)
            
            # Option 1: Letterbox but ensure exact shape
            img, ratio, (dw, dh) = self.letterbox(image, new_shape=input_shape, auto=False)
            
            # Alternative Option 2 (if letterbox with auto=False doesn't work): 
            # Direct resize to 640x640 (distorts aspect ratio but ensures fixed shape)
            # img = cv2.resize(image, (640, 640))
            # ratio = (orig_w / 640, orig_h / 640)
            # dw, dh = 0, 0
            
            # Step 2: Convert BGR to RGB (yolov8trt expects RGB)
            img = img[:, :, ::-1]  # BGR to RGB
            
            # Step 3: Normalize to 0-1 and convert to CHW format (channels, height, width)
            img = img.transpose(2, 0, 1)  # HWC to CHW
            img = np.ascontiguousarray(img) / 255.0  # Normalize 0-1
            
            # Step 4: Ensure correct datatype based on the model precision
            if self.triton_use_fp16:
                img = img.astype(np.float16)  # Use float16 for FP16 model
                logger.debug("Using FP16 precision for Triton inference")
            else:
                img = img.astype(np.float32)  # Use float32 for FP32 model
                logger.debug("Using FP32 precision for Triton inference")
            
            # Add batch dimension
            img = np.expand_dims(img, axis=0)
            
            # Log input shape for debugging
            logger.info(f"Triton input shape: {img.shape}, dtype: {img.dtype}")
            
            # Verify shape matches exactly what Triton expects
            if img.shape != (1, 3, 640, 640):
                # Force reshape if needed (should not happen with auto=False in letterbox)
                logger.warning(f"Reshaping input to exact (1, 3, 640, 640) shape, was: {img.shape}")
                img_resized = cv2.resize(image, (640, 640))
                img_resized = img_resized[:, :, ::-1]  # BGR to RGB
                img_resized = img_resized.transpose(2, 0, 1)  # HWC to CHW
                img_resized = np.ascontiguousarray(img_resized) / 255.0
                
                if self.triton_use_fp16:
                    img = img_resized.astype(np.float16)
                else:
                    img = img_resized.astype(np.float32)
                    
                img = np.expand_dims(img, axis=0)
                
                # Update aspect ratio info for this fallback case
                ratio = (orig_w / 640, orig_h / 640)
                dw, dh = 0, 0
            
            # Get the active model names for Triton
            active_model_name = self.active_triton_model_name
            active_model_version = self.active_triton_model_version
            
            # Create inference input with fixed shape
            input_tensor = triton_grpc.InferInput(
                "images",  # Must match the input name in config.pbtxt
                [1, 3, 640, 640],  # Fixed shape expected by Triton
                triton_utils.np_to_triton_dtype(img.dtype)
            )
            input_tensor.set_data_from_numpy(img)
            
            # Define the output
            output = triton_grpc.InferRequestedOutput("output0")  # Must match the output name in config.pbtxt
            
            # Run inference
            response = self.triton_client.infer(
                model_name=active_model_name,
                inputs=[input_tensor],
                outputs=[output],
                client_timeout=30.0
            )
            
            # Get output data from response
            output_data = response.as_numpy("output0")
            
            # Log output shape for debugging
            logger.info(f"Triton output shape: {output_data.shape}, dtype: {output_data.dtype}")
            
            # Process detections based on output format
            detections = []
            
            # yolov8trt output format - transpose to easier format for processing
            if len(output_data.shape) == 3 and output_data.shape[1] > 4:
                # Transpose from (1, 84, N) to (1, N, 84)
                output = output_data.transpose(0, 2, 1)
                
                # Extract dimensions
                batch_size, num_anchors, num_values = output.shape
                
                # Number of classes is num_values - 4 (x,y,w,h)
                num_classes = num_values - 4
                
                # Extract boxes and class scores for first batch
                boxes = output[0, :, :4]  # [num_anchors, 4] - (x, y, w, h) in input_shape coordinates
                scores = output[0, :, 4:]  # [num_anchors, num_classes]
                
                # Find best class and confidence for each anchor
                class_scores, class_ids = scores.max(axis=1), scores.argmax(axis=1)
                
                # Filter by confidence threshold
                mask = class_scores > self.conf_threshold
                
                if mask.sum() > 0:
                    # Apply confidence filter
                    filtered_boxes = boxes[mask]
                    filtered_scores = class_scores[mask]
                    filtered_class_ids = class_ids[mask]
                    
                    # Log detection count
                    logger.info(f"Found {len(filtered_boxes)} detections above threshold")
                    if len(filtered_boxes) > 0:
                        logger.debug(f"First detection: box={filtered_boxes[0]}, score={filtered_scores[0]}, class={filtered_class_ids[0]}")
                    
                    # Convert boxes from (x,y,w,h) to (x1,y1,x2,y2)
                    x, y, w, h = filtered_boxes[:, 0], filtered_boxes[:, 1], filtered_boxes[:, 2], filtered_boxes[:, 3]
                    
                    # Convert to corner format
                    x1 = x - w / 2
                    y1 = y - h / 2
                    x2 = x + w / 2
                    y2 = y + h / 2
                    
                    # Stack back
                    filtered_boxes = np.stack([x1, y1, x2, y2], axis=1)
                    
                    # Scale coordinates back to original image
                    # First, account for letterbox padding
                    filtered_boxes[:, 0] = (filtered_boxes[:, 0] - dw)  # x1
                    filtered_boxes[:, 1] = (filtered_boxes[:, 1] - dh)  # y1
                    filtered_boxes[:, 2] = (filtered_boxes[:, 2] - dw)  # x2
                    filtered_boxes[:, 3] = (filtered_boxes[:, 3] - dh)  # y2
                    
                    # Then scale back based on ratio
                    filtered_boxes[:, 0] = filtered_boxes[:, 0] / ratio[0]  # x1
                    filtered_boxes[:, 1] = filtered_boxes[:, 1] / ratio[1]  # y1
                    filtered_boxes[:, 2] = filtered_boxes[:, 2] / ratio[0]  # x2
                    filtered_boxes[:, 3] = filtered_boxes[:, 3] / ratio[1]  # y2
                    
                    # Clip to image boundaries
                    filtered_boxes[:, 0] = np.clip(filtered_boxes[:, 0], 0, orig_w)
                    filtered_boxes[:, 1] = np.clip(filtered_boxes[:, 1], 0, orig_h)
                    filtered_boxes[:, 2] = np.clip(filtered_boxes[:, 2], 0, orig_w)
                    filtered_boxes[:, 3] = np.clip(filtered_boxes[:, 3], 0, orig_h)
                    
                    # Ensure boxes have valid dimensions (h, w > 0)
                    valid_boxes = (filtered_boxes[:, 2] > filtered_boxes[:, 0]) & (filtered_boxes[:, 3] > filtered_boxes[:, 1])
                    
                    if valid_boxes.sum() > 0:
                        # Apply dimension filter
                        valid_box_indices = np.where(valid_boxes)[0]
                        
                        filtered_boxes = filtered_boxes[valid_box_indices]
                        filtered_scores = filtered_scores[valid_box_indices]
                        filtered_class_ids = filtered_class_ids[valid_box_indices]
                        
                        # Apply NMS
                        try:
                            # Use PyTorch NMS if available
                            import torch
                            
                            boxes_torch = torch.from_numpy(filtered_boxes).to(torch.float32)
                            scores_torch = torch.from_numpy(filtered_scores).to(torch.float32)
                            
                            keep_indices = torch.ops.torchvision.nms(
                                boxes_torch, scores_torch, self.iou_threshold
                            ).cpu().numpy()
                        except (ImportError, AttributeError):
                            # Fallback to custom NMS
                            keep_indices = self._custom_nms(filtered_boxes, filtered_scores, self.iou_threshold)
                        
                        # Create detection objects
                        for idx in keep_indices:
                            box = filtered_boxes[idx]
                            score = float(filtered_scores[idx])
                            class_id = int(filtered_class_ids[idx])
                            
                            # Get class name from COCO classes
                            class_name = self.class_names.get(class_id, f"class_{class_id}")
                            
                            detections.append(
                                DetectionResult(
                                    class_id=class_id,
                                    class_name=class_name,
                                    confidence=score,
                                    bbox=[float(box[0]), float(box[1]), float(box[2]), float(box[3])]
                                )
                            )
                        
                        # Limit to max detections if needed
                        if len(detections) > self.max_detections:
                            detections = sorted(detections, key=lambda x: x.confidence, reverse=True)
                            detections = detections[:self.max_detections]
                
                logger.info(f"Triton inference found {len(detections)} valid detections")
                return detections
                
            else:
                logger.warning(f"Unexpected output format: {output_data.shape}")
                return []
                
        except Exception as e:
            logger.error(f"Error during Triton inference: {str(e)}")
            logger.error(f"  Model name: {self.active_triton_model_name}")
            logger.error(f"  Model precision: {'FP16' if self.triton_use_fp16 else 'FP32'}")
            logger.error(f"  Triton URL: {self.triton_url}")
            if 'image' in locals() and image is not None:
                logger.error(f"  Original image shape: {image.shape}")
            if 'img' in locals() and img is not None:
                logger.error(f"  Preprocessed image shape: {img.shape}, dtype: {img.dtype}")
                
            # Try to fall back to the other precision model if one fails
            # Don't attempt to fall back if we've already tried both models
            if hasattr(self, '_fallback_attempted') and self._fallback_attempted:
                logger.error("Both FP16 and FP32 models failed, no further fallbacks available")
                raise RuntimeError(f"Triton inference failed with both FP16 and FP32 models: {str(e)}")
                
            self._fallback_attempted = True  # Mark that we've attempted a fallback
            
            if self.triton_use_fp16:
                logger.warning("FP16 inference failed, attempting to fall back to FP32")
                try:
                    self.triton_use_fp16 = False
                    self.active_triton_model_name = self.triton_model_name
                    self.active_triton_model_version = self.triton_model_version
                    result = self._predict_with_triton(image)
                    self._fallback_attempted = False  # Reset for next call
                    return result
                except Exception as fallback_error:
                    logger.error(f"FP32 fallback also failed: {str(fallback_error)}")
            else:
                logger.warning("FP32 inference failed, attempting to fall back to FP16")
                try:
                    self.triton_use_fp16 = True
                    self.active_triton_model_name = self.triton_fp16_model_name
                    self.active_triton_model_version = self.triton_fp16_model_version
                    result = self._predict_with_triton(image)
                    self._fallback_attempted = False  # Reset for next call
                    return result
                except Exception as fallback_error:
                    logger.error(f"FP16 fallback also failed: {str(fallback_error)}")
                    
            self._fallback_attempted = False  # Reset for next call
            raise RuntimeError(f"Triton inference failed: {str(e)}")
            
    def _custom_nms(self, boxes, scores, iou_threshold):
        """
        Custom implementation of non-maximum suppression.
        Used as fallback if PyTorch is not available.
        
        Args:
            boxes: Boxes in xyxy format as numpy array
            scores: Confidence scores as numpy array
            iou_threshold: IoU threshold for suppression
            
        Returns:
            indices: Indices of boxes to keep
        """
        # Sort by score
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            # Calculate IoU of the kept box with the rest
            xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            intersection = w * h
            
            # Calculate union
            area1 = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area2 = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
            union = area1 + area2 - intersection
            
            # Calculate IoU
            iou = intersection / (union + 1e-16)
            
            # Find indices of boxes with IoU <= threshold
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return keep

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
        """
        Resize and pad image while meeting stride-multiple constraints.
        Identical to yolov8trt preprocessing.
        
        Args:
            img: Input image
            new_shape: Image size after resize
            color: Pad color
            auto: Minimum rectangle
            scaleFill: Stretch, no pad
            scaleup: Allow scale up
            stride: Stride multiple
            
        Returns:
            Resized and padded image, scale factors, and padding amounts
        """
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, ratio, (dw, dh)
    
    def _process_results(self, result) -> List[DetectionResult]:
        """
        Process YOLO results into standardized schema
        
        Args:
            result: YOLO result object
            
        Returns:
            List[DetectionResult]: Standardized detection results
        """
        detections = []
        
        # Extract boxes, confidence scores and class IDs
        boxes = result.boxes.xyxy.cpu().numpy() if hasattr(result.boxes, 'xyxy') else []
        confs = result.boxes.conf.cpu().numpy() if hasattr(result.boxes, 'conf') else []
        class_ids = result.boxes.cls.cpu().numpy().astype(int) if hasattr(result.boxes, 'cls') else []
        
        # Get class names mapping
        names = result.names
        
        # Create detection results
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            conf = float(confs[i])
            class_id = int(class_ids[i])
            class_name = names[class_id]
            
            detections.append(
                DetectionResult(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=conf,
                    bbox=[float(x1), float(y1), float(x2), float(y2)]
                )
            )
        
        return detections
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model
        
        Returns:
            Dict[str, Any]: Model information
        """
        gpu_available = False
        gpu_name = None
        
        try:
            gpu_available = torch.cuda.is_available()
            if gpu_available:
                gpu_name = torch.cuda.get_device_name(0)
        except:
            pass
            
        info = {
            "model_name": self.model_name,
            "device": self.device,
            "gpu_available": gpu_available,
            "gpu_name": gpu_name,
            "optimization_enabled": self.use_optimization,
            "fp16_enabled": self.use_fp16 if self.use_optimization else None,
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "max_detections": self.max_detections,
            "using_triton": self.use_triton and self.triton_client is not None,
        }
        
        # Add more detailed info if model is loaded locally
        if self.model is not None:
            info["task"] = getattr(self.model, "task", "unknown")
            
            # Check if model is TensorRT
            model_path = getattr(self.model, "ckpt_path", "")
            if str(model_path).endswith(".engine"):
                info["model_format"] = "TensorRT"
            else:
                info["model_format"] = "PyTorch"
            
            # Add class names if available
            if hasattr(self.model, "names"):
                info["classes"] = self.model.names
        
        # Add Triton-specific info
        if self.use_triton and self.triton_client is not None:
            info["triton_url"] = self.triton_url
            info["triton_model_name"] = self.active_triton_model_name  # Use the active model name
            info["triton_model_version"] = self.active_triton_model_version  # Use the active model version
            info["model_format"] = "Triton"
            info["triton_using_fp16"] = self.triton_use_fp16  # Add FP16 flag
            
            # Add class names for Triton
            info["classes"] = self.class_names
        
        return info

def get_model() -> YOLOModel:
    """
    Get or create the model instance per worker
    
    Returns:
        YOLOModel: The YOLO model instance
    """
    # Remove the global singleton pattern to allow each worker to have its own model instance
    # This is especially important with Triton, as each worker just needs a client
    return YOLOModel()