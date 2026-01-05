"""
================================================================================
QUANTIZATION MODULE
================================================================================

This module implements quantization to reduce model precision and size.

WHAT IS QUANTIZATION?
─────────────────────
Quantization reduces the number of bits used to represent weights and activations.

Before (FP32): 0.12345678901234567890... stored in 32 bits (4 bytes)
After (INT8):  0.12 stored in 8 bits (1 byte) → 4× smaller!

WHY QUANTIZE?
─────────────
1. Smaller model size (4× for INT8)
2. Faster inference (integer math is faster than floating point)
3. Lower memory bandwidth (important for mobile/edge devices)

QUANTIZATION METHODS:
─────────────────────
1. Dynamic Quantization (easiest):
   - Weights quantized ahead of time
   - Activations quantized at runtime
   - Good balance of simplicity and performance

2. Static Post-Training Quantization (PTQ):
   - Both weights and activations pre-quantized
   - Requires calibration data
   - Best performance but needs more setup

3. Quantization-Aware Training (QAT):
   - Simulates quantization during training
   - Model learns to be robust to quantization
   - Best accuracy, but requires retraining

4. FP16 (Half Precision):
   - Simple: just convert to 16-bit floats
   - Works on GPU (unlike INT8 in PyTorch)
   - 2× compression with ~0% accuracy loss

CRITICAL: WHY INT8 ONLY WORKS ON CPU
────────────────────────────────────
PyTorch's quantization uses FBGEMM (x86 CPU) or QNNPACK (ARM CPU).
These libraries don't have GPU backends in PyTorch!

For GPU quantization, you need:
- NVIDIA TensorRT (converts to TRT format)
- ONNX Runtime (can use INT8 on GPU)
- FP16 (works natively in PyTorch on GPU)

For your Kaggle research:
- Use FP16 for GPU benchmarks (fast, accurate)
- Use INT8 for CPU deployment metrics
- Report both in your paper!
"""

import torch
import torch.nn as nn
import torch.quantization as quant
from typing import Dict, Optional, List
import numpy as np
from tqdm import tqdm
import copy
import warnings
import time


# =============================================================================
# QUANTIZATION MANAGER
# =============================================================================

class QuantizationManager:
    """
    Manages quantization operations for transformer models.
    
    Usage:
        manager = QuantizationManager(model, method='dynamic')
        quantized = manager.apply_dynamic_quantization()
        # or
        manager = QuantizationManager(model, method='fp16')
        quantized = manager.apply_fp16_quantization(device='cuda')
    """
    
    def __init__(self, model: nn.Module, method: str = 'dynamic', dtype: str = 'int8'):
        """
        Initialize quantization manager.
        
        Args:
            model: Model to quantize
            method: 'dynamic', 'static', 'qat', or 'fp16'
            dtype: 'int8' or 'fp16'
        """
        self.original_model = model
        self.method = method
        self.dtype = dtype
        self.quantized_model = None
        self.calibrated = False
        
        print(f"\n📉 QuantizationManager initialized:")
        print(f"   Method: {method}")
        print(f"   Data type: {dtype}")
        
        if method in ['dynamic', 'static'] and dtype == 'int8':
            print(f"   ⚠️  Note: INT8 quantization only works on CPU")
    
    def apply_dynamic_quantization(self) -> nn.Module:
        """
        Apply dynamic quantization.
        
        WHAT: Quantizes weights ahead of time; activations quantized at runtime
        WHY: Simple to apply, no calibration needed
        HOW: One function call!
        
        Pros:
        - Very easy to apply (one line!)
        - Works well for LSTM, Transformer layers
        - No calibration data needed
        
        Cons:
        - Activations quantized at runtime (some overhead)
        - Only works on CPU
        """
        print("\n   Applying dynamic quantization...")
        
        # Create a copy to avoid modifying original
        model_copy = copy.deepcopy(self.original_model)
        
        # Apply dynamic quantization to Linear layers
        # These are the main compute bottleneck in transformers
        self.quantized_model = torch.quantization.quantize_dynamic(
            model_copy,
            {nn.Linear},  # Only quantize Linear layers
            dtype=torch.qint8
        )
        
        print("   ✅ Dynamic quantization applied")
        return self.quantized_model
    
    def prepare_static_quantization(self, backend: str = 'fbgemm'):
        """
        Prepare model for static quantization.
        
        WHAT: Inserts quantization observers into the model
        WHY: Need to collect statistics for calibration
        HOW: Replace modules with quantization-aware versions
        
        Args:
            backend: 'fbgemm' for x86 CPU, 'qnnpack' for ARM
        
        After calling this, you must:
        1. Call calibrate() with representative data
        2. Call convert_static_quantization() to get quantized model
        """
        print(f"\n   Preparing static quantization (backend: {backend})...")
        
        # Set backend
        torch.backends.quantized.engine = backend
        
        # Get quantization config
        self.original_model.qconfig = quant.get_default_qconfig(backend)
        
        # Prepare model (inserts observers)
        self.quantized_model = quant.prepare(
            copy.deepcopy(self.original_model),
            inplace=False
        )
        
        print("   ✅ Model prepared for static quantization")
        print("   Next: Call calibrate() with representative data")
    
    def calibrate(self, dataloader, device: str, num_batches: int = 100):
        """
        Calibrate static quantization with representative data.
        
        WHAT: Run forward passes to collect activation statistics
        WHY: Need to know the range of activations for quantization
        HOW: Run inference on calibration data, observers record min/max values
        
        Args:
            dataloader: DataLoader with calibration data
            device: Must be 'cpu' for static quantization
            num_batches: Number of batches for calibration (more = better but slower)
        """
        if self.quantized_model is None:
            raise ValueError("Must call prepare_static_quantization() first!")
        
        print(f"\n   Calibrating on {num_batches} batches...")
        
        # MUST be on CPU for calibration
        self.quantized_model.to('cpu')
        self.quantized_model.eval()
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, total=num_batches)):
                if i >= num_batches:
                    break
                
                # Move to CPU for calibration
                input_ids = batch['input_ids'].to('cpu')
                attention_mask = batch['attention_mask'].to('cpu')
                
                try:
                    self.quantized_model(input_ids, attention_mask)
                except Exception as e:
                    warnings.warn(f"Calibration error: {e}")
                    break
        
        self.calibrated = True
        print("   ✅ Calibration complete")
    
    def convert_static_quantization(self) -> nn.Module:
        """
        Convert prepared model to quantized model.
        
        Must call after calibrate()!
        
        WHAT: Replaces observers with actual quantization
        WHY: Now we know the ranges, can do fixed-point quantization
        HOW: Convert floating-point ops to integer ops
        """
        if not self.calibrated:
            raise ValueError("Must call calibrate() first!")
        
        print("\n   Converting to quantized model...")
        
        self.quantized_model = quant.convert(
            self.quantized_model,
            inplace=False
        )
        
        print("   ✅ Static quantization complete")
        return self.quantized_model
    
    def apply_fp16_quantization(self, device: str = 'cuda') -> nn.Module:
        """
        Convert model to FP16 (half precision).
        
        WHAT: Convert all parameters from 32-bit to 16-bit floats
        WHY: 2× smaller, 2× faster on GPU with Tensor Cores
        HOW: Just call model.half()!
        
        Pros:
        - Works on GPU (unlike INT8)
        - Almost no accuracy loss
        - Very simple to apply
        - 2× speedup on modern GPUs
        
        Cons:
        - Only 2× compression (not 4× like INT8)
        - Needs GPU with FP16 support (most modern GPUs)
        
        Args:
            device: Device to put model on (should be 'cuda' for speedup)
        
        Returns:
            Model in FP16 precision
        """
        if device == 'cpu':
            warnings.warn("FP16 on CPU is slow! Use GPU for FP16 benefits.")
        
        print("\n   Applying FP16 (half precision) conversion...")
        
        # Simple! Just convert to half precision
        self.quantized_model = copy.deepcopy(self.original_model)
        self.quantized_model = self.quantized_model.half().to(device)
        
        print("   ✅ FP16 conversion complete")
        return self.quantized_model
    
    def get_model_size(self, model: Optional[nn.Module] = None) -> Dict:
        """
        Calculate model size in memory.
        
        Returns:
            Dict with size information
        """
        model = model or self.quantized_model or self.original_model
        
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        total_mb = (param_size + buffer_size) / (1024 ** 2)
        
        return {
            'param_size_mb': param_size / (1024 ** 2),
            'buffer_size_mb': buffer_size / (1024 ** 2),
            'total_size_mb': total_mb,
            'num_parameters': sum(p.numel() for p in model.parameters())
        }
    
    def compare_sizes(self) -> Dict:
        """Compare original and quantized model sizes."""
        original_size = self.get_model_size(self.original_model)
        quantized_size = self.get_model_size(self.quantized_model)
        
        compression = original_size['total_size_mb'] / quantized_size['total_size_mb']
        
        print(f"\n📊 Quantization Results:")
        print(f"   Original size: {original_size['total_size_mb']:.2f} MB")
        print(f"   Quantized size: {quantized_size['total_size_mb']:.2f} MB")
        print(f"   Compression ratio: {compression:.2f}×")
        print(f"   Size reduction: {(1 - 1/compression) * 100:.1f}%")
        
        return {
            'original_size_mb': original_size['total_size_mb'],
            'quantized_size_mb': quantized_size['total_size_mb'],
            'compression_ratio': compression,
            'size_reduction_pct': (1 - 1/compression) * 100
        }


# =============================================================================
# HIGH-LEVEL QUANTIZATION FUNCTION
# =============================================================================

def quantize_model(
    model: nn.Module,
    method: str,
    config,
    calibration_loader=None,
    device: str = 'cpu'
) -> nn.Module:
    """
    High-level function to quantize a model.
    
    WHAT: One-stop function for any quantization method
    WHY: Simplifies the main script
    HOW: Dispatches to appropriate method
    
    Args:
        model: Model to quantize
        method: 'dynamic', 'static', 'qat', or 'fp16'
        config: Configuration with quantization parameters
        calibration_loader: DataLoader for calibration (static only)
        device: Device to put quantized model on
    
    Returns:
        Quantized model
    """
    manager = QuantizationManager(model, method=method, dtype=config.quant_dtype)
    
    if method == 'dynamic':
        quantized_model = manager.apply_dynamic_quantization()
        
    elif method == 'static':
        manager.prepare_static_quantization()
        if calibration_loader is not None:
            manager.calibrate(
                calibration_loader,
                device='cpu',  # Static quant calibration must be on CPU
                num_batches=config.quant_calibration_batches
            )
        else:
            raise ValueError("Static quantization requires calibration_loader!")
        quantized_model = manager.convert_static_quantization()
        
    elif method == 'fp16':
        quantized_model = manager.apply_fp16_quantization(device)
        
    else:
        raise ValueError(f"Unknown quantization method: {method}")
    
    # Print comparison
    manager.compare_sizes()
    
    return quantized_model


# =============================================================================
# INFERENCE BENCHMARKING
# =============================================================================

def benchmark_inference_speed(
    model: nn.Module,
    dataloader,
    device: str,
    num_iterations: int = 100,
    warmup: int = 10
) -> Dict:
    """
    Benchmark inference speed of a model.
    
    WHAT: Measure how fast the model runs inference
    WHY: Need to quantify speedup from compression
    HOW: Run many iterations, measure time, compute statistics
    
    Args:
        model: Model to benchmark
        dataloader: DataLoader with test data
        device: Device to run on
        num_iterations: Number of iterations for timing
        warmup: Warmup iterations (not timed, just to "warm up" GPU)
    
    Returns:
        Dict with timing statistics
    """
    model.eval()
    model.to(device)
    
    # Get a batch for benchmarking
    batch = next(iter(dataloader))
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    batch_size = input_ids.shape[0]
    
    # Warmup (important for GPU!)
    print(f"   Warming up ({warmup} iterations)...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_ids, attention_mask)
    
    # Synchronize GPU before timing
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Timed runs
    print(f"   Benchmarking ({num_iterations} iterations)...")
    latencies = []
    
    with torch.no_grad():
        for _ in range(num_iterations):
            if device == 'cuda':
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            _ = model(input_ids, attention_mask)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms
    
    latencies = np.array(latencies)
    
    results = {
        'mean_latency_ms': np.mean(latencies),
        'std_latency_ms': np.std(latencies),
        'p50_latency_ms': np.percentile(latencies, 50),
        'p95_latency_ms': np.percentile(latencies, 95),
        'p99_latency_ms': np.percentile(latencies, 99),
        'throughput_samples_per_sec': (batch_size / np.mean(latencies)) * 1000,
        'batch_size': batch_size,
        'num_iterations': num_iterations,
        'device': device
    }
    
    print(f"   Mean latency: {results['mean_latency_ms']:.2f} ms")
    print(f"   Throughput: {results['throughput_samples_per_sec']:.1f} samples/sec")
    
    return results


def profile_memory(model: nn.Module, dataloader, device: str) -> Dict:
    """
    Profile memory usage of a model.
    
    Args:
        model: Model to profile
        dataloader: DataLoader with test data
        device: Device to profile on
    
    Returns:
        Dict with memory statistics
    """
    if device != 'cuda':
        return {'peak_memory_mb': 0}  # Can't profile CPU memory easily
    
    model.eval()
    model.to(device)
    
    # Clear cache and reset stats
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Run inference
    batch = next(iter(dataloader))
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    
    with torch.no_grad():
        _ = model(input_ids, attention_mask)
    
    # Get peak memory
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
    
    return {
        'peak_memory_mb': peak_memory
    }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing quantization module...")
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(768, 256),
        nn.ReLU(),
        nn.Linear(256, 5)
    )
    
    # Test FP16 quantization (works on CPU too, just slower)
    manager = QuantizationManager(model, method='fp16')
    fp16_model = manager.apply_fp16_quantization(device='cpu')
    manager.compare_sizes()
    
    # Test dynamic quantization
    manager2 = QuantizationManager(model, method='dynamic')
    int8_model = manager2.apply_dynamic_quantization()
    manager2.compare_sizes()
    
    print("\n✅ Quantization module tests passed!")
