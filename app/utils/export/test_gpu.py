#!/usr/bin/env python3
"""
Script to verify GPU availability for PyTorch and TensorRT
"""
import torch
import os

print("\n===== GPU Detection Test =====")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    device_count = torch.cuda.device_count()
    print(f"GPU count: {device_count}")
    
    for i in range(device_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"GPU {i} memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
        
    # Test a basic operation on GPU
    print("\n--- Testing basic GPU operation ---")
    a = torch.ones(1000, 1000).cuda()
    b = torch.ones(1000, 1000).cuda()
    
    # Measure time
    import time
    start = time.time()
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    end = time.time()
    
    print(f"Matrix multiplication time: {(end-start)*1000:.2f} ms")
    print("GPU operation successful!")
    
    # Check for TensorRT
    print("\n--- Checking TensorRT availability ---")
    try:
        import tensorrt as trt
        print(f"TensorRT version: {trt.__version__}")
        logger = trt.Logger(trt.Logger.ERROR)
        builder = trt.Builder(logger)
        print("TensorRT initialization successful!")
    except ImportError:
        print("TensorRT Python package not found.")
    except Exception as e:
        print(f"TensorRT error: {e}")
    
    # Check NVIDIA environment
    print("\n--- NVIDIA Environment Variables ---")
    for k, v in os.environ.items():
        if 'NVIDIA' in k or 'CUDA' in k:
            print(f"{k}={v}")
else:
    print("No CUDA-capable GPU detected. Please check your NVIDIA drivers and installation.")

print("\n===== Test Complete =====")