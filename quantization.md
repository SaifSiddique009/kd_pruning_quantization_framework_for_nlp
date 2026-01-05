# 📘 Script 5: `research_quantization.py`

## Overview

This script implements **quantization** - the technique of reducing the numerical precision of model weights and activations. Instead of using 32-bit floating-point numbers, we use smaller representations like 16-bit, 8-bit, or even 4-bit numbers.

**Why quantization is the "last mile" of compression:**
- Provides 2-8× size reduction with minimal code changes
- Often the easiest compression technique to apply
- Works well combined with KD and pruning for maximum compression
- Critical for deploying to mobile devices, CPUs, and edge hardware

---

## The Big Picture: What Quantization Does

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ QUANTIZATION: REDUCING NUMERICAL PRECISION                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ ORIGINAL WEIGHT (FP32 - 32 bits):                                          │
│                                                                             │
│   0.123456789012345678901234567890123                                      │
│   └──────────────── 32 bits ─────────────────┘                             │
│                                                                             │
│   Range: ±3.4 × 10³⁸                                                        │
│   Precision: ~7 decimal digits                                             │
│   Memory: 4 bytes per weight                                               │
│                                                                             │
│ ═══════════════════════════════════════════════════════════════════════════│
│                                                                             │
│ QUANTIZED TO FP16 (16 bits):                                               │
│                                                                             │
│   0.1235                                                                    │
│   └── 16 bits ──┘                                                           │
│                                                                             │
│   Range: ±65,504                                                            │
│   Precision: ~3-4 decimal digits                                           │
│   Memory: 2 bytes per weight (2× compression)                              │
│                                                                             │
│ ═══════════════════════════════════════════════════════════════════════════│
│                                                                             │
│ QUANTIZED TO INT8 (8 bits):                                                │
│                                                                             │
│   31 (representing ~0.12 after scaling)                                    │
│   └ 8 bits ┘                                                                │
│                                                                             │
│   Range: -128 to 127 (mapped to original weight range)                     │
│   Memory: 1 byte per weight (4× compression)                               │
│                                                                             │
│ ═══════════════════════════════════════════════════════════════════════════│
│                                                                             │
│ QUANTIZED TO INT4 (4 bits):                                                │
│                                                                             │
│   7 (representing ~0.12 after scaling)                                     │
│   └4b┘                                                                      │
│                                                                             │
│   Range: -8 to 7 (or 0 to 15 unsigned)                                     │
│   Memory: 0.5 bytes per weight (8× compression!)                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Section 1: Imports and Setup

```python
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.quantization as quant
from torch.quantization import (
    quantize_dynamic,
    prepare,
    convert,
    get_default_qconfig,
    QConfig,
    default_observer,
    default_weight_observer
)
import numpy as np
from typing import Dict, Optional, Tuple, List, Union
from tqdm import tqdm
import time
import copy
```

### Key Imports Explained:

| Import | Purpose |
|--------|---------|
| `quantize_dynamic` | Dynamic quantization (easiest method) |
| `prepare`, `convert` | Static quantization workflow |
| `QConfig` | Configuration for quantization observers |
| `default_observer` | Tracks activation statistics |
| `default_weight_observer` | Tracks weight statistics |

### Understanding PyTorch Quantization Architecture:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ PYTORCH QUANTIZATION SYSTEM                                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ OBSERVERS: Track statistics to determine quantization parameters           │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ MinMaxObserver                                                      │  │
│   │   Tracks: min and max values                                        │  │
│   │   Use: Simple, works for most cases                                 │  │
│   │   Formula: scale = (max - min) / 255                                │  │
│   │            zero_point = -min / scale                                │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ MovingAverageMinMaxObserver                                         │  │
│   │   Tracks: Exponential moving average of min/max                     │  │
│   │   Use: When values change over time (training)                      │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ HistogramObserver                                                   │  │
│   │   Tracks: Full histogram of values                                  │  │
│   │   Use: Most accurate, but slower                                    │  │
│   │   Finds optimal scale by minimizing quantization error              │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│ QCONFIG: Combines observers for weights and activations                    │
│                                                                             │
│   qconfig = QConfig(                                                        │
│       activation=default_observer,    # How to observe activations         │
│       weight=default_weight_observer  # How to observe weights             │
│   )                                                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Section 2: The Quantization Manager Class

```python
class QuantizationManager:
    """
    Manages all quantization operations for a model.
    
    Supports multiple quantization methods:
        - dynamic: Easiest, weights quantized, activations at runtime
        - static: Both pre-quantized, needs calibration
        - qat: Quantization-aware training
        - fp16: Half precision (GPU friendly)
        - int4: 4-bit quantization (maximum compression)
    """
    
    def __init__(
        self,
        model: nn.Module,
        method: str = 'dynamic',
        dtype: str = 'int8'
    ):
        """
        Args:
            model: Model to quantize
            method: Quantization method ('dynamic', 'static', 'qat', 'fp16', 'int4')
            dtype: Target data type ('int8', 'int4', 'fp16')
        """
        self.model = model
        self.method = method
        self.dtype = dtype
        self.original_size = self._get_model_size(model)
        self.quantized_model = None
```

### Getting Model Size:

```python
def _get_model_size(self, model: nn.Module) -> float:
    """
    Calculate model size in megabytes.
    
    Two approaches:
    1. Count parameters × bytes per parameter
    2. Save model and check file size (more accurate for quantized)
    """
    # Method 1: Parameter counting
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    total_size_mb = (param_size + buffer_size) / (1024 ** 2)
    
    return total_size_mb
```

### Understanding `element_size()`:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ DATA TYPES AND THEIR SIZES                                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ Data Type      │ Bits │ Bytes │ element_size() │ Range                     │
│ ───────────────┼──────┼───────┼────────────────┼─────────────────────────  │
│ torch.float32  │  32  │   4   │       4        │ ±3.4 × 10³⁸              │
│ torch.float16  │  16  │   2   │       2        │ ±65,504                   │
│ torch.bfloat16 │  16  │   2   │       2        │ ±3.4 × 10³⁸ (less precise)│
│ torch.int8     │   8  │   1   │       1        │ -128 to 127               │
│ torch.qint8    │   8  │   1   │       1        │ -128 to 127 (quantized)   │
│ torch.quint8   │   8  │   1   │       1        │ 0 to 255 (unsigned)       │
│                                                                             │
│ Example calculation:                                                        │
│   BanglaBERT: 110M parameters × 4 bytes = 440 MB                           │
│   After INT8: 110M parameters × 1 byte = 110 MB (4× smaller!)              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Section 3: Dynamic Quantization (The Easiest Method)

Dynamic quantization is the simplest approach. Weights are quantized ahead of time, but activations are quantized dynamically during inference.

```python
def apply_dynamic_quantization(self) -> nn.Module:
    """
    Apply dynamic quantization to the model.
    
    HOW IT WORKS:
    ─────────────
    1. Weights are quantized ONCE when you call this function
    2. Activations are quantized ON-THE-FLY during each forward pass
    3. No calibration data needed!
    
    WHY "DYNAMIC"?
    ──────────────
    The quantization parameters for activations are computed dynamically
    based on the actual values seen during inference. This means:
    - No need for representative calibration data
    - Works well when activation ranges vary
    - Slightly slower than static (computes stats at runtime)
    
    LIMITATIONS:
    ────────────
    - Only works on CPU (PyTorch limitation)
    - Cannot run on GPU after quantization
    - Good for: server deployment, batch processing
    """
    print(f"\n📉 Applying dynamic quantization (INT8)...")
    print(f"   ⚠️  Note: Dynamic quantization runs on CPU only")
    
    # Move model to CPU first
    model_cpu = copy.deepcopy(self.model).cpu()
    model_cpu.eval()
    
    # Apply dynamic quantization
    # Only quantize Linear layers (where most computation happens)
    quantized_model = quantize_dynamic(
        model_cpu,
        {nn.Linear},          # Which layer types to quantize
        dtype=torch.qint8     # Target data type
    )
    
    self.quantized_model = quantized_model
    
    # Report compression
    new_size = self._get_model_size(quantized_model)
    compression = self.original_size / new_size
    print(f"   Size: {self.original_size:.1f} MB → {new_size:.1f} MB")
    print(f"   Compression: {compression:.2f}×")
    
    return quantized_model
```

### Visual: How Dynamic Quantization Works

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ DYNAMIC QUANTIZATION WORKFLOW                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ STEP 1: QUANTIZE WEIGHTS (done once, at model load time)                   │
│                                                                             │
│   Original weight (FP32):  [0.123, -0.456, 0.789, ...]                     │
│                                                                             │
│   Compute scale and zero_point:                                            │
│     min_val = -0.456, max_val = 0.789                                      │
│     scale = (0.789 - (-0.456)) / 255 = 0.00488                            │
│     zero_point = round(-(-0.456) / 0.00488) = 93                          │
│                                                                             │
│   Quantize:                                                                 │
│     q_weight = round(weight / scale) + zero_point                          │
│     q_weight = [118, 0, 255, ...]  (INT8 values)                          │
│                                                                             │
│   Store: q_weight (INT8) + scale (FP32) + zero_point (INT32)              │
│                                                                             │
│ ═══════════════════════════════════════════════════════════════════════════│
│                                                                             │
│ STEP 2: INFERENCE (happens for each input)                                 │
│                                                                             │
│   Input activation (FP32):  [1.5, -0.3, 2.1, ...]                         │
│                                                                             │
│   Dynamically quantize activation:                                         │
│     Compute min/max of THIS batch                                          │
│     Compute scale and zero_point                                           │
│     q_activation = [...]  (INT8)                                           │
│                                                                             │
│   Matrix multiply in INT8:                                                  │
│     q_output = q_activation @ q_weight                                     │
│     (This is the fast part! INT8 ops are 2-4× faster than FP32)           │
│                                                                             │
│   Dequantize output back to FP32:                                          │
│     output = (q_output - zero_point) × scale                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### What You Can Modify:

| Modification | Effect | When to Use |
|--------------|--------|-------------|
| `{nn.Linear, nn.Conv2d}` | Quantize Conv layers too | If model has convolutions |
| `dtype=torch.quint8` | Unsigned quantization | When values are always positive |

---

## Section 4: Static Quantization (Better Accuracy)

Static quantization pre-computes quantization parameters for both weights AND activations using calibration data. This gives better accuracy but requires representative data.

```python
def prepare_static_quantization(self) -> nn.Module:
    """
    Prepare model for static quantization.
    
    This inserts "observer" modules that will track statistics
    during calibration. The model can still run in FP32 mode.
    
    WORKFLOW:
    1. prepare_static_quantization()  ← You are here
    2. calibrate()                    ← Run representative data
    3. convert_static_quantization()  ← Finalize quantization
    """
    print(f"\n📉 Preparing for static quantization...")
    
    # Move to CPU (required for PyTorch quantization)
    model_cpu = copy.deepcopy(self.model).cpu()
    model_cpu.eval()
    
    # Set quantization configuration
    # This tells PyTorch how to observe and quantize
    model_cpu.qconfig = get_default_qconfig('fbgemm')
    # 'fbgemm' is optimized for server CPUs (x86)
    # 'qnnpack' is optimized for mobile CPUs (ARM)
    
    # Prepare the model
    # This inserts observer modules throughout the model
    prepared_model = prepare(model_cpu, inplace=False)
    
    self.prepared_model = prepared_model
    print(f"   ✓ Model prepared with observers")
    print(f"   Next step: Run calibration data through the model")
    
    return prepared_model
```

### Calibration Step:

```python
def calibrate(
    self,
    dataloader,
    device: str = 'cpu',
    num_batches: int = 100
) -> None:
    """
    Run calibration data through the prepared model.
    
    WHY CALIBRATE?
    ──────────────
    Static quantization needs to know the typical range of activations
    BEFORE inference. We run representative data and observers track:
    - Minimum activation value seen
    - Maximum activation value seen
    - (Optionally) Full histogram of values
    
    These statistics are used to compute scale and zero_point.
    
    HOW MUCH DATA?
    ──────────────
    - Too little: Might miss extreme values, causing clipping
    - Too much: Diminishing returns, wastes time
    - Rule of thumb: 100-1000 samples is usually sufficient
    """
    print(f"   Calibrating with {num_batches} batches...")
    
    self.prepared_model.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, total=num_batches)):
            if i >= num_batches:
                break
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass - observers record statistics
            _ = self.prepared_model(input_ids, attention_mask)
    
    print(f"   ✓ Calibration complete")
```

### Converting to Quantized Model:

```python
def convert_static_quantization(self) -> nn.Module:
    """
    Convert the prepared+calibrated model to a quantized model.
    
    This is where the actual quantization happens:
    1. Observers are removed
    2. Quantization parameters are computed from observed stats
    3. Weights are quantized
    4. Quantize/Dequantize operations are inserted
    """
    print(f"   Converting to quantized model...")
    
    # Convert the model
    quantized_model = convert(self.prepared_model, inplace=False)
    
    self.quantized_model = quantized_model
    
    # Report results
    new_size = self._get_model_size(quantized_model)
    compression = self.original_size / new_size
    print(f"   Size: {self.original_size:.1f} MB → {new_size:.1f} MB")
    print(f"   Compression: {compression:.2f}×")
    
    return quantized_model
```

### Visual: Static vs Dynamic Quantization

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ STATIC VS DYNAMIC QUANTIZATION                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ DYNAMIC QUANTIZATION:                                                       │
│                                                                             │
│   Preparation:  None needed                                                │
│   Calibration:  None needed                                                │
│                                                                             │
│   Inference (each batch):                                                  │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ Input → [Compute activation stats] → [Quantize activation] →        │  │
│   │         [INT8 matmul with pre-quantized weights] → [Dequantize] → Out│  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                    ↑                                                        │
│                    Extra computation at runtime                             │
│                                                                             │
│ ═══════════════════════════════════════════════════════════════════════════│
│                                                                             │
│ STATIC QUANTIZATION:                                                        │
│                                                                             │
│   Preparation:  Run calibration data once                                  │
│   Calibration:  Observers track min/max of all activations                 │
│                                                                             │
│   Inference (each batch):                                                  │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ Input → [Quantize with PRE-COMPUTED params] →                        │  │
│   │         [INT8 matmul] → [Dequantize] → Output                        │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│              ↑                                                              │
│              No runtime stat computation (faster!)                         │
│                                                                             │
│ ═══════════════════════════════════════════════════════════════════════════│
│                                                                             │
│ COMPARISON:                                                                 │
│                                                                             │
│                    │ Dynamic        │ Static                               │
│   ─────────────────┼────────────────┼───────────────────────────────────── │
│   Calibration      │ Not needed     │ Required (100+ samples)              │
│   Accuracy         │ Good           │ Better (more accurate params)        │
│   Speed            │ Fast           │ Faster (no runtime stats)            │
│   Flexibility      │ High           │ Lower (fixed activation range)       │
│   Best for         │ Variable data  │ Consistent data distribution         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Section 5: Quantization-Aware Training (QAT)

QAT simulates quantization during training, allowing the model to learn to be robust to quantization errors.

```python
def prepare_qat(self) -> nn.Module:
    """
    Prepare model for Quantization-Aware Training.
    
    WHY QAT?
    ────────
    Normal quantization: Train in FP32 → Quantize → Hope it works
    QAT: Train WITH simulated quantization → Model learns to handle it
    
    HOW IT WORKS:
    ─────────────
    During forward pass:
        1. Weights are fake-quantized (quantize then dequantize)
        2. This introduces quantization noise
        3. Model learns to be robust to this noise
    
    During backward pass:
        1. Gradients flow through as if no quantization
        2. This is called "Straight-Through Estimator" (STE)
    
    Result: Better accuracy after final quantization!
    
    TRADEOFF:
    ─────────
    - Requires full training (expensive!)
    - Best accuracy of all methods
    - Use when accuracy is critical
    """
    print(f"\n📉 Preparing for Quantization-Aware Training...")
    
    # Move to CPU for quantization setup, but can train on GPU
    model_copy = copy.deepcopy(self.model)
    model_copy.train()  # Must be in training mode
    
    # Set QAT configuration
    model_copy.qconfig = get_default_qconfig('fbgemm')
    
    # Prepare for QAT
    # This inserts FakeQuantize modules
    prepared_model = prepare(model_copy, inplace=False)
    
    self.qat_model = prepared_model
    print(f"   ✓ Model prepared for QAT")
    print(f"   Next: Train this model, then convert with convert_qat()")
    
    return prepared_model
```

### Visual: How QAT Works

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ QUANTIZATION-AWARE TRAINING (QAT)                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ NORMAL TRAINING (FP32):                                                     │
│                                                                             │
│   Weight: 0.12345678                                                        │
│      ↓                                                                      │
│   Multiply with input                                                       │
│      ↓                                                                      │
│   Output: Perfect precision                                                 │
│                                                                             │
│   After training → Quantize → Weight becomes 0.12 → ACCURACY DROP!         │
│                                                                             │
│ ═══════════════════════════════════════════════════════════════════════════│
│                                                                             │
│ QAT (Simulated Quantization):                                              │
│                                                                             │
│   Weight: 0.12345678                                                        │
│      ↓                                                                      │
│   FAKE QUANTIZE: round to INT8, then back to FP32                          │
│      ↓                                                                      │
│   Fake quantized weight: 0.12 (in FP32 format, but limited precision)      │
│      ↓                                                                      │
│   Multiply with input (using imprecise weight)                             │
│      ↓                                                                      │
│   Output: Includes quantization noise!                                      │
│      ↓                                                                      │
│   Loss computed with noisy output                                          │
│      ↓                                                                      │
│   Model learns to be robust to noise                                        │
│                                                                             │
│   After training → Quantize → Weight becomes 0.12 → MINIMAL DROP!          │
│                                                                             │
│ ═══════════════════════════════════════════════════════════════════════════│
│                                                                             │
│ THE STRAIGHT-THROUGH ESTIMATOR (STE):                                       │
│                                                                             │
│   Problem: round() has zero gradient almost everywhere!                     │
│            d/dx round(x) = 0 (except at integers where undefined)          │
│            Gradient descent would not work.                                 │
│                                                                             │
│   Solution: Pretend round() is the identity function during backward:       │
│                                                                             │
│   Forward:  x → round(x/s)*s → y   (includes rounding)                     │
│   Backward: dy/dx = 1              (ignores rounding)                       │
│                                                                             │
│   This lets gradients flow through quantization operations!                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### QAT Training Loop:

```python
def train_qat(
    self,
    train_loader,
    val_loader,
    optimizer,
    num_epochs: int = 5,
    device: str = 'cpu'
) -> nn.Module:
    """
    Train the QAT-prepared model.
    
    The training loop is almost identical to normal training!
    The FakeQuantize modules handle the quantization simulation.
    """
    print(f"\n🔧 Training with QAT for {num_epochs} epochs...")
    
    model = self.qat_model.to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f"QAT Epoch {epoch+1}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with fake quantization
            outputs = model(input_ids, attention_mask)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            
            loss = loss_fn(logits, labels)
            loss.backward()  # Gradients flow through FakeQuantize via STE
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"   Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}")
    
    # Convert to final quantized model
    model.eval()
    model.cpu()
    quantized_model = convert(model, inplace=False)
    
    self.quantized_model = quantized_model
    return quantized_model
```

---

## Section 6: FP16 Quantization (GPU-Friendly)

Unlike INT8 which only works on CPU in PyTorch, FP16 works on GPU and is much simpler.

```python
def apply_fp16_quantization(self, device: str = 'cuda') -> nn.Module:
    """
    Convert model to half precision (FP16).
    
    WHY FP16?
    ─────────
    - Works on GPU! (Unlike INT8 in standard PyTorch)
    - 2× memory reduction
    - Often FASTER on modern GPUs (tensor cores)
    - Minimal accuracy loss (~0.1% typically)
    
    HOW IT WORKS:
    ─────────────
    Simply convert all FP32 parameters to FP16.
    No calibration, no training, just type conversion.
    
    CAUTION:
    ────────
    - Some operations may need FP32 for stability (loss computation)
    - Very small/large values may overflow/underflow
    - Use torch.cuda.amp for automatic mixed precision if concerned
    """
    print(f"\n📉 Applying FP16 quantization...")
    print(f"   ✓ Works on GPU!")
    
    # Simply convert to half precision
    model_fp16 = copy.deepcopy(self.model).half().to(device)
    
    self.quantized_model = model_fp16
    
    # Report compression (always 2× for FP16)
    print(f"   Size: {self.original_size:.1f} MB → {self.original_size/2:.1f} MB")
    print(f"   Compression: 2.00×")
    
    return model_fp16
```

### FP32 vs FP16 vs BF16:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ FLOATING POINT FORMATS                                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ FP32 (Single Precision):                                                   │
│ ┌─────┬────────────────────┬───────────────────────────────────────────┐   │
│ │Sign │    Exponent (8)    │           Mantissa (23)                   │   │
│ │  1  │    8 bits          │           23 bits                         │   │
│ └─────┴────────────────────┴───────────────────────────────────────────┘   │
│   Range: ±3.4 × 10³⁸                                                        │
│   Precision: ~7 decimal digits                                             │
│                                                                             │
│ ═══════════════════════════════════════════════════════════════════════════│
│                                                                             │
│ FP16 (Half Precision):                                                     │
│ ┌─────┬───────────┬───────────────────┐                                    │
│ │Sign │ Exp (5)   │   Mantissa (10)   │                                    │
│ │  1  │  5 bits   │    10 bits        │                                    │
│ └─────┴───────────┴───────────────────┘                                    │
│   Range: ±65,504 (MUCH smaller!)                                           │
│   Precision: ~3 decimal digits                                             │
│   Good for: Most neural network computations                               │
│   Bad for: Loss computation, gradient accumulation                         │
│                                                                             │
│ ═══════════════════════════════════════════════════════════════════════════│
│                                                                             │
│ BF16 (Brain Float):                                                         │
│ ┌─────┬────────────────────┬───────────┐                                   │
│ │Sign │    Exponent (8)    │ Mant (7)  │                                   │
│ │  1  │    8 bits          │  7 bits   │                                   │
│ └─────┴────────────────────┴───────────┘                                   │
│   Range: ±3.4 × 10³⁸ (same as FP32!)                                       │
│   Precision: ~2 decimal digits (less than FP16)                            │
│   Good for: Training (same range as FP32, less overflow risk)              │
│                                                                             │
│ ═══════════════════════════════════════════════════════════════════════════│
│                                                                             │
│ PRACTICAL ADVICE:                                                           │
│                                                                             │
│   Training:   Use BF16 or mixed precision (FP16 + FP32 where needed)       │
│   Inference:  Use FP16 (best speed/accuracy tradeoff)                      │
│   Edge/Mobile: Use INT8 (maximum compression)                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Section 7: INT4 Quantization (Maximum Compression)

INT4 provides 8× compression but requires special libraries (bitsandbytes) and careful implementation.

```python
def apply_int4_quantization(self, device: str = 'cuda') -> nn.Module:
    """
    Apply 4-bit quantization using bitsandbytes library.
    
    WHY INT4?
    ─────────
    - 8× compression (vs 4× for INT8)
    - Still maintains reasonable accuracy
    - Used by QLoRA for efficient LLM fine-tuning
    
    HOW IT WORKS:
    ─────────────
    Uses NF4 (Normalized Float 4-bit) from bitsandbytes:
    1. Weights are normalized to have zero mean and unit variance
    2. Quantized to 4 bits using a learned code book
    3. Double quantization: Even the scaling factors are quantized!
    
    REQUIREMENTS:
    ─────────────
    - pip install bitsandbytes
    - CUDA GPU required
    """
    print(f"\n📉 Applying INT4 quantization...")
    
    try:
        import bitsandbytes as bnb
        from transformers import BitsAndBytesConfig
    except ImportError:
        print("   ❌ bitsandbytes not installed!")
        print("   Run: pip install bitsandbytes")
        print("   Falling back to INT8...")
        return self.apply_dynamic_quantization()
    
    # Create quantization config
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,  # Compute in FP16 for speed
        bnb_4bit_use_double_quant=True,         # Quantize the quantization constants!
        bnb_4bit_quant_type="nf4"               # Normalized Float 4-bit
    )
```

### Converting Linear Layers to 4-bit:

```python
    def replace_linear_with_4bit(model):
        """
        Replace all Linear layers with 4-bit quantized versions.
        """
        for name, module in model.named_children():
            if isinstance(module, nn.Linear):
                # Create 4-bit linear layer
                new_layer = bnb.nn.Linear4bit(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    compute_dtype=torch.float16,
                    quant_type="nf4"
                )
                
                # Quantize and copy weights
                new_layer.weight = bnb.nn.Params4bit(
                    module.weight.data,
                    requires_grad=False,
                    quant_type="nf4"
                )
                
                if module.bias is not None:
                    new_layer.bias = nn.Parameter(module.bias.data)
                
                setattr(model, name, new_layer)
            else:
                # Recursively process child modules
                replace_linear_with_4bit(module)
        
        return model
    
    # Apply 4-bit quantization
    model_4bit = replace_linear_with_4bit(copy.deepcopy(self.model))
    model_4bit = model_4bit.to(device)
    
    self.quantized_model = model_4bit
    
    # Report compression
    new_size = self.original_size / 8  # Approximate
    print(f"   Size: {self.original_size:.1f} MB → ~{new_size:.1f} MB")
    print(f"   Compression: ~8.00×")
    
    return model_4bit
```

### Visual: NF4 Quantization

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ NF4 (NORMALIZED FLOAT 4-BIT) QUANTIZATION                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ PROBLEM WITH UNIFORM INT4:                                                  │
│                                                                             │
│   Weight distribution is typically GAUSSIAN, not uniform:                  │
│                                                                             │
│         │        ▄▄                                                         │
│   Count │       ████                                                        │
│         │      ██████                                                       │
│         │     ████████                                                      │
│         │    ██████████                                                     │
│         │   ████████████                                                    │
│         │  ██████████████                                                   │
│         │ ████████████████                                                  │
│         └──────────────────────                                             │
│           -1.0       0       +1.0                                           │
│                                                                             │
│   Uniform INT4: 16 evenly spaced values across range                       │
│   Problem: Most weights are near 0, wasting precision on extremes!         │
│                                                                             │
│ ═══════════════════════════════════════════════════════════════════════════│
│                                                                             │
│ NF4 SOLUTION:                                                               │
│                                                                             │
│   Use quantization levels that match the distribution:                     │
│                                                                             │
│   NF4 code book (16 values, not evenly spaced):                            │
│   [-1.0, -0.7, -0.5, -0.4, -0.3, -0.2, -0.1, 0,                           │
│     0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]                                  │
│                                                                             │
│   More levels near 0 (where most weights are) = better accuracy!           │
│                                                                             │
│ ═══════════════════════════════════════════════════════════════════════════│
│                                                                             │
│ DOUBLE QUANTIZATION:                                                        │
│                                                                             │
│   Normal quantization stores:                                              │
│     - Quantized weights (INT4)                                             │
│     - Scale factors (FP32) ← Still 32 bits!                                │
│                                                                             │
│   Double quantization:                                                      │
│     - Quantized weights (INT4)                                             │
│     - Quantized scale factors (INT8) ← Even smaller!                       │
│     - Second-level scale (FP32, shared across many weights)                │
│                                                                             │
│   Extra compression at minimal accuracy cost!                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Section 8: Benchmarking and Comparison

```python
def benchmark_inference_speed(
    self,
    model: nn.Module,
    dataloader,
    device: str,
    num_batches: int = 50
) -> Dict[str, float]:
    """
    Measure inference speed and latency.
    
    Returns:
        - latency_mean: Average time per batch (ms)
        - latency_p95: 95th percentile latency (ms)
        - throughput: Samples per second
    """
    model.eval()
    latencies = []
    total_samples = 0
    
    # Warm-up (first few runs are often slower)
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= 3:
                break
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            _ = model(input_ids, attention_mask)
    
    # Actual benchmarking
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Synchronize GPU before timing
            if device == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.time()
            
            _ = model(input_ids, attention_mask)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            
            latency = (end_time - start_time) * 1000  # Convert to ms
            latencies.append(latency)
            total_samples += len(input_ids)
    
    total_time = sum(latencies) / 1000  # Total time in seconds
    
    return {
        'latency_mean': np.mean(latencies),
        'latency_std': np.std(latencies),
        'latency_p50': np.percentile(latencies, 50),
        'latency_p95': np.percentile(latencies, 95),
        'latency_p99': np.percentile(latencies, 99),
        'throughput': total_samples / total_time
    }
```

### Compare Sizes:

```python
def compare_sizes(self) -> Dict[str, float]:
    """
    Compare original and quantized model sizes.
    """
    if self.quantized_model is None:
        return {'error': 'No quantized model available'}
    
    original_size = self.original_size
    quantized_size = self._get_model_size(self.quantized_model)
    
    return {
        'original_size_mb': original_size,
        'quantized_size_mb': quantized_size,
        'compression_ratio': original_size / quantized_size,
        'size_reduction_percent': (1 - quantized_size / original_size) * 100
    }
```

---

## Section 9: The Main Quantize Function

```python
def quantize_model(
    model: nn.Module,
    method: str = 'dynamic',
    dataloader = None,
    device: str = 'cpu',
    num_calibration_batches: int = 100,
    qat_epochs: int = 3,
    optimizer = None
) -> nn.Module:
    """
    Main entry point for quantizing a model.
    
    This is what research_main.py calls.
    """
    manager = QuantizationManager(model, method=method)
    
    if method == 'dynamic':
        return manager.apply_dynamic_quantization()
    
    elif method == 'static':
        manager.prepare_static_quantization()
        manager.calibrate(dataloader, device, num_calibration_batches)
        return manager.convert_static_quantization()
    
    elif method == 'qat':
        if optimizer is None:
            raise ValueError("QAT requires an optimizer")
        manager.prepare_qat()
        return manager.train_qat(dataloader, None, optimizer, qat_epochs, device)
    
    elif method == 'fp16':
        return manager.apply_fp16_quantization(device)
    
    elif method == 'int4':
        return manager.apply_int4_quantization(device)
    
    else:
        raise ValueError(f"Unknown quantization method: {method}")
```

---

## Complete Method Comparison

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ QUANTIZATION METHOD COMPARISON                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ Method   │ Bits │ Compress │ Device │ Calibration │ Accuracy │ Speed       │
│ ─────────┼──────┼──────────┼────────┼─────────────┼──────────┼───────────  │
│ None     │  32  │    1×    │ GPU    │     -       │ Baseline │ 1.0×        │
│ FP16     │  16  │    2×    │ GPU ✓  │    None     │ ~0% loss │ 1.0-1.5×    │
│ Dynamic  │   8  │    4×    │ CPU ⚠  │    None     │ 1-2% loss│ 1.5-2×      │
│ Static   │   8  │    4×    │ CPU ⚠  │   Required  │ 0.5-1% ↓ │ 2-3×        │
│ QAT      │   8  │    4×    │ CPU ⚠  │  Training   │ 0-0.5% ↓ │ 2-3×        │
│ INT4     │   4  │    8×    │ GPU ✓  │   Required  │ 2-4% loss│ 1.5-2×      │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ DECISION FLOWCHART:                                                         │
│                                                                             │
│   Need GPU inference?                                                       │
│       │                                                                     │
│       ├── YES → Need maximum compression?                                   │
│       │          │                                                          │
│       │          ├── YES → INT4 (8× compression)                           │
│       │          │                                                          │
│       │          └── NO → FP16 (2× compression, minimal loss)              │
│       │                                                                     │
│       └── NO (CPU is fine) → Accuracy critical?                            │
│                              │                                              │
│                              ├── YES → Have training time?                  │
│                              │          │                                   │
│                              │          ├── YES → QAT                       │
│                              │          │                                   │
│                              │          └── NO → Static                     │
│                              │                                              │
│                              └── NO → Dynamic (simplest)                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Section 10: ONNX Export (Alternative Quantization Path)

Sometimes PyTorch quantization is limiting. ONNX provides an alternative path that works on more platforms.

```python
def export_to_onnx(
    model: nn.Module,
    save_path: str,
    input_shape: Tuple[int, int] = (1, 128),
    quantize: bool = True
) -> str:
    """
    Export model to ONNX format with optional quantization.
    
    WHY ONNX?
    ─────────
    - Platform independent (runs on Windows, Linux, Mac, mobile)
    - Supports more quantization options
    - Often faster than PyTorch inference
    - Required for some deployment targets (Edge devices, TensorRT)
    
    ONNX QUANTIZATION OPTIONS:
    ──────────────────────────
    - QOperator: Quantize operators (like PyTorch static)
    - QDQ: Insert Quantize/Dequantize nodes (more flexible)
    """
    import torch.onnx
    
    model.eval()
    model.cpu()
    
    # Create dummy input
    dummy_input = (
        torch.zeros(input_shape, dtype=torch.long),  # input_ids
        torch.ones(input_shape, dtype=torch.long)    # attention_mask
    )
    
    # Export to ONNX
    onnx_path = save_path if save_path.endswith('.onnx') else f"{save_path}.onnx"
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=['input_ids', 'attention_mask'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch', 1: 'sequence'},
            'attention_mask': {0: 'batch', 1: 'sequence'},
            'logits': {0: 'batch'}
        },
        opset_version=14
    )
    
    print(f"   ✓ Exported to: {onnx_path}")
    
    # Optionally quantize the ONNX model
    if quantize:
        try:
            from onnxruntime.quantization import quantize_dynamic as onnx_quantize
            
            quantized_path = onnx_path.replace('.onnx', '_quantized.onnx')
            onnx_quantize(onnx_path, quantized_path)
            
            print(f"   ✓ Quantized ONNX saved to: {quantized_path}")
            return quantized_path
            
        except ImportError:
            print("   ⚠️  onnxruntime not installed, skipping ONNX quantization")
    
    return onnx_path
```

---

## Summary: What You Can Modify in This Script

| Category | What to Modify | Research Question |
|----------|----------------|-------------------|
| **Method** | dynamic, static, qat, fp16, int4 | Which method gives best accuracy/compression? |
| **Calibration** | Number of batches | How much calibration data is needed? |
| **QAT** | Number of epochs | How much training improves accuracy? |
| **Layers** | Which layers to quantize | Are some layers more sensitive? |
| **Backend** | fbgemm vs qnnpack | Best backend for your hardware? |
| **Observer** | MinMax vs Histogram | Which observer works better? |

---

## Experiments You Can Run

```bash
# Experiment 1: Compare all quantization methods
for method in dynamic static fp16; do
    python research_main.py --pipeline quant_only \
        --quant_method $method \
        --output_dir results/quant_$method
done

# Experiment 2: Calibration size study (for static quantization)
for batches in 10 50 100 500; do
    python research_main.py --pipeline quant_only \
        --quant_method static \
        --quant_calibration_batches $batches \
        --output_dir results/calib_$batches
done

# Experiment 3: Combined compression (KD + Quantization)
python research_main.py --pipeline kd_quant \
    --kd_method logit \
    --quant_method fp16 \
    --output_dir results/kd_fp16

# Experiment 4: Full compression pipeline
python research_main.py --pipeline kd_prune_quant \
    --kd_method multi_level \
    --prune_method wanda \
    --prune_sparsity 0.5 \
    --quant_method fp16 \
    --output_dir results/full_compression
```

---

## Key Takeaways

1. **FP16 is the safest choice** - works on GPU, minimal accuracy loss, simple to apply
2. **INT8 is for CPU deployment** - use dynamic if no calibration data, static otherwise
3. **INT4 is for maximum compression** - requires bitsandbytes, best for very large models
4. **QAT is for best accuracy** - requires full training, use when every 0.1% matters
5. **Always benchmark** - speed gains vary by hardware and model

---

## Practice Exercise

Before moving to the next script:

1. **Calculate the theoretical compression**: If a model has 110M FP32 parameters, how big is it in MB? After INT8? After INT4?
2. **Think about trade-offs**: Why might FP16 be faster than FP32 on modern GPUs?
3. **Consider the deployment**: If you're deploying to a phone, which method would you choose and why?

---

**Ready for the next script? The next one is `research_evaluation.py` which implements comprehensive metrics calculation and comparison across all compression stages.**

Would you like me to continue?