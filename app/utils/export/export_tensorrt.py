#!/usr/bin/env python3
"""
Script to export YOLOv8 models directly to TensorRT format
"""
import os
import argparse
import torch
import tensorrt as trt
from ultralytics import YOLO
from pathlib import Path

def export_to_tensorrt_v10(model_path, output_dir, device='0', fp16=True, workspace=4):
    """
    Export YOLOv8 model to TensorRT using our custom approach for TensorRT 10.x
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # First export to ONNX
    print(f"Step 1: Exporting to ONNX first")
    model = YOLO(model_path)
    
    # Define ONNX export arguments
    onnx_args = {
        'format': 'onnx',
        'dynamic': False,  # Static shapes for better performance
        'half': fp16,      # Use FP16 precision
        'device': device,  # Use specified device
        'simplify': True,  # Simplify model
        'opset': 12,       # ONNX opset version
        'verbose': True,   # Show verbose output
    }
    
    # Export to ONNX
    onnx_path = model.export(**onnx_args)
    print(f"✅ ONNX model exported to: {onnx_path}")
    
    # Now convert ONNX to TensorRT
    print(f"Step 2: Converting ONNX to TensorRT")
    model_basename = Path(model_path).stem
    output_file = Path(output_dir) / f"{model_basename}_fp{'16' if fp16 else '32'}.engine"
    
    # Initialize TensorRT resources
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    
    # Create network definition with explicit batch
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    
    # Create ONNX parser
    parser = trt.OnnxParser(network, logger)
    
    # Parse ONNX model
    with open(onnx_path, 'rb') as model_file:
        model_content = model_file.read()
        if not parser.parse(model_content):
            for error in range(parser.num_errors):
                print(f"❌ ONNX parsing error: {parser.get_error(error)}")
            raise RuntimeError("Failed to parse ONNX model")
    
    # Create builder config
    config = builder.create_builder_config()
    
    # In TensorRT 10.x, max_workspace_size is replaced with memory pool limits
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace * (1 << 30))  # GB to bytes
    
    # Set FP16 mode if requested and supported
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("✅ FP16 mode enabled")
    
    # Build engine
    print("⏳ Building TensorRT engine (this may take several minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)
    if not serialized_engine:
        raise RuntimeError("Failed to build TensorRT engine")
    
    # Save engine to file
    with open(output_file, 'wb') as f:
        f.write(serialized_engine)
    
    print(f"✅ TensorRT engine saved to: {output_file}")
    return str(output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLOv8 model to TensorRT")
    parser.add_argument("--model", type=str, default="yolov8l.pt", help="Path to YOLO model")
    parser.add_argument("--output-dir", type=str, default="/tmp/tensorrt", help="Output directory")
    parser.add_argument("--device", type=str, default="0", help="Device ID (e.g., 0 for first GPU)")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 precision")
    parser.add_argument("--workspace", type=int, default=4, help="Workspace size in GB")
    
    args = parser.parse_args()
    export_to_tensorrt_v10(args.model, args.output_dir, args.device, args.fp16, args.workspace)