
#!/usr/bin/env python3
"""
Improved YOLO model conversion script with auto-detection of model properties.
This handles YOLOv8 and should adapt to future versions with minimal changes.
"""

import os
import sys
import subprocess
import json
import argparse
import shutil
import re
import onnx
import numpy as np
from pathlib import Path


def parse_args():
    """Parse command line arguments for model conversion."""
    parser = argparse.ArgumentParser(description='Convert YOLO model to TensorRT format for Triton')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='yolov8l.pt',
                      help='YOLO model path or name')
    parser.add_argument('--model-name', type=str, default='yolov8trt_fp16',
                      help='Name for the Triton model')
    parser.add_argument('--model-version', type=str, choices=['v8', 'v9', 'v10', 'auto'], default='auto',
                      help='YOLO model version (v8, v9, v10, or auto for auto-detection)')
    
    # Export parameters
    parser.add_argument('--dynamic', action='store_true', default=True,
                      help='Enable dynamic input shapes')
    parser.add_argument('--opset', type=int, default=16,
                      help='ONNX opset version')
    parser.add_argument('--half', action='store_true', default=True,
                      help='Enable FP16 precision')
    parser.add_argument('--device', type=str, default='0',
                      help='CUDA device to use')
    
    # TensorRT parameters
    parser.add_argument('--min-shape', type=str, default='1x3x640x640',
                      help='Minimum input shape (batch x channels x height x width)')
    parser.add_argument('--opt-shape', type=str, default='4x3x640x640',
                      help='Optimal input shape')
    parser.add_argument('--max-shape', type=str, default='8x3x640x640',
                      help='Maximum input shape')
    parser.add_argument('--fp16', action='store_true', default=True,
                      help='Enable FP16 precision for TensorRT')
    
    # Output parameters
    parser.add_argument('--model-repository', type=str, default='/model_repository',
                      help='Triton model repository directory')
    parser.add_argument('--work-dir', type=str, default='/tmp/conversion',
                      help='Working directory for conversion')
    
    # Triton config parameters
    parser.add_argument('--output-dims', type=str, default='auto',
                      help='Output dimensions for YOLO model (default: auto-detect)')
    parser.add_argument('--output-name', type=str, default='auto',
                      help='Output name for YOLO model (default: auto-detect)')
    parser.add_argument('--triton-batch-size', type=int, default=1,
                      help='Max batch size for Triton (default: 1)')
    
    return parser.parse_args()


def run_command(cmd, cwd=None):
    """Execute a command and handle errors."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)
    print(result.stdout)
    return result


def detect_model_version(model_name):
    """Auto-detect YOLO model version from name or file."""
    # Check for specific version numbers
    if 'v10' in model_name.lower():
        return 'v10'
    if 'v9' in model_name.lower():
        return 'v9'
    if 'v8' in model_name.lower():
        return 'v8'
    
    # Use regex to match version numbers (handles future versions like v11, v12, etc.)
    version_match = re.search(r'v(\d+)', model_name.lower())
    if version_match:
        version = f"v{version_match.group(1)}"
        print(f"Detected YOLO version: {version}")
        return version
        
    # Default to most recent stable version
    print(f"Could not detect model version from name: {model_name}, assuming YOLOv8")
    return 'v8'


def export_yolo_model(args, work_dir):
    """Export YOLO model to ONNX format."""
    print(f"Exporting YOLO model {args.model} to ONNX...")
    
    # Auto-detect model version if not specified
    if args.model_version == 'auto':
        args.model_version = detect_model_version(args.model)
    
    print(f"Using YOLO version: {args.model_version}")
    
    # Construct export command
    export_cmd = [
        "yolo", "export",
        f"model={args.model}",
        "format=onnx",
        f"dynamic={str(args.dynamic).lower()}",
        f"opset={args.opset}",
        f"device={args.device}"
    ]
    
    if args.half:
        export_cmd.append("half=True")
    
    run_command(export_cmd, cwd=work_dir)
    
    # Get the ONNX file name
    model_stem = Path(args.model).stem
    onnx_path = work_dir / f"{model_stem}.onnx"
    
    if not onnx_path.exists():
        print(f"Error: ONNX file not found at {onnx_path}")
        sys.exit(1)
    
    return onnx_path


def analyze_onnx_model(onnx_path):
    """Analyze ONNX model to extract input/output information."""
    print(f"Analyzing ONNX model: {onnx_path}")
    
    try:
        model = onnx.load(str(onnx_path))
        
        # Get model inputs
        inputs = []
        for input in model.graph.input:
            input_name = input.name
            input_shape = []
            for dim in input.type.tensor_type.shape.dim:
                if dim.dim_value:
                    input_shape.append(dim.dim_value)
                else:
                    # Dynamic dimension
                    input_shape.append(-1)
            inputs.append({"name": input_name, "shape": input_shape})
        
        # Get model outputs
        outputs = []
        for output in model.graph.output:
            output_name = output.name
            output_shape = []
            for dim in output.type.tensor_type.shape.dim:
                if dim.dim_value:
                    output_shape.append(dim.dim_value)
                else:
                    # Dynamic dimension
                    output_shape.append(-1)
            outputs.append({"name": output_name, "shape": output_shape})
        
        print(f"Model Inputs: {inputs}")
        print(f"Model Outputs: {outputs}")
        
        return inputs, outputs
    
    except Exception as e:
        print(f"Error analyzing ONNX model: {e}")
        return None, None


def convert_to_tensorrt(onnx_path, args, work_dir):
    """Convert ONNX model to TensorRT engine."""
    print(f"Converting ONNX model to TensorRT engine...")
    
    engine_path = work_dir / "model.engine"
    
    # Construct trtexec command
    trt_cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--minShapes=images:{args.min_shape}",
        f"--optShapes=images:{args.opt_shape}",
        f"--maxShapes=images:{args.max_shape}",
        f"--device={args.device}"
    ]
    
    if args.fp16:
        trt_cmd.append("--fp16")
    
    run_command(trt_cmd, cwd=work_dir)
    
    if not engine_path.exists():
        print(f"Error: TensorRT engine not found at {engine_path}")
        sys.exit(1)
    
    return engine_path


def inspect_tensorrt_engine(engine_path):
    """Inspect TensorRT engine to get layer information."""
    print(f"Inspecting TensorRT engine: {engine_path}")
    
    try:
        # Use trtexec to inspect the engine
        cmd = ["trtexec", f"--loadEngine={engine_path}"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Parse output to find layer info
        output = result.stdout + result.stderr
        
        # Extract input and output information
        input_match = re.search(r"Input\s+\"(\w+)\"\s+with shape\s+([\dx]+)", output)
        output_match = re.search(r"Output\s+\"(\w+)\"\s+with shape\s+([\dx]+)", output)
        
        input_info = None
        output_info = None
        
        if input_match:
            name = input_match.group(1)
            shape_str = input_match.group(2)
            shape = [int(dim) for dim in shape_str.split('×')]
            input_info = {"name": name, "shape": shape}
        
        if output_match:
            name = output_match.group(1)
            shape_str = output_match.group(2)
            shape = [int(dim) for dim in shape_str.split('×')]
            output_info = {"name": name, "shape": shape}
        
        print(f"TensorRT Engine Input: {input_info}")
        print(f"TensorRT Engine Output: {output_info}")
        
        return input_info, output_info
    
    except Exception as e:
        print(f"Error inspecting TensorRT engine: {e}")
        print("Falling back to default settings")
        return None, None


def get_output_info(args, onnx_path, engine_path):
    """Determine output tensor information based on model analysis or user settings."""
    
    # Try to get info from ONNX model first
    inputs, outputs = analyze_onnx_model(onnx_path)
    if not inputs or not outputs:
        # Fallback to TensorRT engine inspection
        inputs, outputs = inspect_tensorrt_engine(engine_path)
    
    # Output name
    output_name = "output0"  # Default
    if args.output_name != "auto":
        output_name = args.output_name
    elif outputs and outputs[0]["name"]:
        output_name = outputs[0]["name"]
    
    # Output dimensions
    output_dims = []
    
    if args.output_dims != "auto":
        # User provided output dimensions
        output_dims = [int(dim) for dim in args.output_dims.split(',')]
    elif outputs and len(outputs[0]["shape"]) > 0 and all(dim > 0 for dim in outputs[0]["shape"]):
        # Use detected dimensions
        output_dims = outputs[0]["shape"]
    else:
        # Guess based on model version and common settings
        if args.model_version == 'v8':
            # YOLOv8 standard output format is [84, 8400] for 640x640 input
            # 84 = 4 (bbox) + 80 (COCO classes)
            output_dims = [84, 8400]
        elif args.model_version == 'v9':
            # YOLOv9 has similar output structure (adjust if needed)
            output_dims = [84, 8400]
        elif args.model_version == 'v10':
            # YOLOv10 has similar output structure (adjust if needed)
            output_dims = [84, 8400]
        elif args.model_version.startswith('v'):
            # Future versions - assume similar structure to v8/v9/v10
            print(f"Using standard dimensions for {args.model_version}")
            output_dims = [84, 8400]
        else:
            # Default fallback
            output_dims = [84, 8400]
            print("WARNING: Could not determine output dimensions, using default [84, 8400]")
    
    print(f"Using output name: {output_name}")
    print(f"Using output dimensions: {output_dims}")
    
    return output_name, output_dims


def create_triton_config(args, model_dir, output_name, output_dims):
    """Create Triton Inference Server configuration file."""
    print(f"Creating Triton model configuration...")
    
    # Parse dimensions
    input_shape = [int(x) for x in args.min_shape.split('x')[1:]]  # Remove batch dimension
    
    # Use FP16 or FP32 based on args
    data_type = "TYPE_FP16" if args.fp16 else "TYPE_FP32"
    
    config = f"""name: "{args.model_name}"
platform: "tensorrt_plan"
max_batch_size: {args.triton_batch_size}
input [
  {{
    name: "images"
    data_type: {data_type}
    format: FORMAT_NCHW
    dims: {input_shape}
  }}
]
output [
  {{
    name: "{output_name}"
    data_type: {data_type}
    dims: {output_dims}
  }}
]
"""
    
    config_path = model_dir / "config.pbtxt"
    with open(config_path, 'w') as f:
        f.write(config)
    
    print(f"Configuration written to {config_path}")


def setup_model_directory(args, engine_path, onnx_path):
    """Setup Triton model repository structure."""
    print(f"Setting up Triton model directory...")
    
    # Create model directory structure
    model_dir = Path(args.model_repository) / args.model_name
    version_dir = model_dir / "1"
    version_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine output tensor information
    output_name, output_dims = get_output_info(args, onnx_path, engine_path)
    
    # Copy engine file
    final_engine_path = version_dir / "model.plan"
    shutil.copy2(engine_path, final_engine_path)
    print(f"Copied TensorRT engine to {final_engine_path}")
    
    # Create config file
    create_triton_config(args, model_dir, output_name, output_dims)
    
    # Create labels file if it doesn't exist
    labels_path = model_dir / "labels.txt"
    if not labels_path.exists():
        # COCO labels for YOLO models
        coco_labels = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
            "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
            "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        ]
        
        with open(labels_path, 'w') as f:
            for label in coco_labels:
                f.write(f"{label}\n")
        print(f"Created labels file at {labels_path}")
    
    print(f"Model setup complete at {model_dir}")


def main():
    """Main conversion pipeline."""
    args = parse_args()
    
    # Create working directory
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting model conversion...")
    print(f"Model: {args.model}")
    print(f"Target name: {args.model_name}")
    print(f"Work directory: {work_dir}")
    
    # Export YOLO model to ONNX
    onnx_path = export_yolo_model(args, work_dir)
    
    # Convert to TensorRT
    engine_path = convert_to_tensorrt(onnx_path, args, work_dir)
    
    # Setup Triton model directory
    setup_model_directory(args, engine_path, onnx_path)
    
    print("\nConversion complete!")
    print(f"Model available at: {args.model_repository}/{args.model_name}/")
    
    # Cleanup working directory
    # shutil.rmtree(work_dir)
    # print("Cleaned up temporary files.")


if __name__ == "__main__":
    main()