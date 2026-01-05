# 📘 Script 4: `research_pruning.py`

## Overview

This script implements **pruning** - the technique of removing unimportant weights from a neural network. Think of it like trimming a tree: you remove branches that don't contribute much, making the tree lighter while keeping its essential structure.

**Why pruning matters:**
- Reduces model size by 30-70%
- Can speed up inference (with proper hardware/libraries)
- Often combined with KD and quantization for maximum compression
- Research shows neural networks are heavily over-parameterized

---

## The Big Picture: What Pruning Does

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ THE LOTTERY TICKET HYPOTHESIS (Frankle & Carlin, 2019)                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ Key insight: Large neural networks contain smaller subnetworks              │
│ ("winning tickets") that can achieve similar accuracy alone.               │
│                                                                             │
│ Original Network (110M parameters):                                        │
│                                                                             │
│   ●━━━●━━━●━━━●━━━●                                                        │
│   ┃ ╲ ┃ ╱ ┃ ╲ ┃ ╱ ┃                                                        │
│   ●━━━●━━━●━━━●━━━●     All connections present                            │
│   ┃ ╱ ┃ ╲ ┃ ╱ ┃ ╲ ┃     Many are redundant or near-zero                    │
│   ●━━━●━━━●━━━●━━━●                                                        │
│                                                                             │
│ After 50% Pruning (55M parameters):                                        │
│                                                                             │
│   ●━━━●   ●━━━●   ●                                                        │
│   ┃   ┃ ╱     ┃ ╱ ┃                                                        │
│   ●   ●━━━●━━━●   ●     Removed weak connections                           │
│   ┃ ╱     ┃       ┃     Kept important pathways                            │
│   ●━━━●━━━●   ●━━━●                                                        │
│                                                                             │
│ Result: Similar accuracy with half the parameters!                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Section 1: Imports and Constants

```python
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm
from collections import defaultdict
import copy
```

### Key Import: `torch.nn.utils.prune`

PyTorch provides a built-in pruning module. Understanding its design helps you understand all pruning:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ HOW PYTORCH PRUNING WORKS                                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ STEP 1: Before pruning                                                     │
│                                                                             │
│   layer.weight = tensor([0.5, -0.02, 0.3, 0.01, -0.4])                    │
│                                                                             │
│ STEP 2: Apply pruning mask                                                 │
│                                                                             │
│   prune.l1_unstructured(layer, 'weight', amount=0.4)                       │
│                                                                             │
│   What happens internally:                                                 │
│   - Original weight is saved as 'weight_orig'                              │
│   - A mask is created: mask = [1, 0, 1, 0, 1]  (0 = pruned)               │
│   - 'weight' becomes a property: weight = weight_orig * mask               │
│                                                                             │
│   layer.weight_orig = tensor([0.5, -0.02, 0.3, 0.01, -0.4])               │
│   layer.weight_mask = tensor([1,   0,     1,   0,    1   ])               │
│   layer.weight      = tensor([0.5, 0,     0.3, 0,   -0.4])  # computed    │
│                                                                             │
│ STEP 3: Make permanent (optional)                                          │
│                                                                             │
│   prune.remove(layer, 'weight')                                            │
│                                                                             │
│   Now:                                                                      │
│   layer.weight = tensor([0.5, 0, 0.3, 0, -0.4])  # actual zeros           │
│   (weight_orig and weight_mask are removed)                                │
│                                                                             │
│ WHY THIS DESIGN?                                                            │
│   - Masks allow pruning to be reversible                                   │
│   - Multiple pruning rounds can be combined                                │
│   - Gradual pruning can adjust masks during training                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Section 2: Base Pruning Manager Class

```python
class PruningManager:
    """
    Base class for all pruning operations.
    
    Provides:
        - Layer identification (which layers to prune)
        - Sparsity tracking (how many zeros)
        - Common utilities
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_sparsity: float = 0.5,
        prune_layers: str = 'all',
        global_pruning: bool = True
    ):
        """
        Args:
            model: Model to prune
            target_sparsity: Fraction of weights to remove (0.5 = 50%)
            prune_layers: Which layers to prune ('all', 'attention', 'ffn', 'encoder')
            global_pruning: If True, prune globally; if False, prune each layer independently
        """
        self.model = model
        self.target_sparsity = target_sparsity
        self.prune_layers = prune_layers
        self.global_pruning = global_pruning
        
        # Find layers to prune
        self.prunable_layers = self._identify_prunable_layers()
```

### Understanding `prune_layers` Options:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ WHICH LAYERS TO PRUNE?                                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ TRANSFORMER ARCHITECTURE:                                                   │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ ENCODER BLOCK (×12 for BERT-base)                                   │  │
│   │                                                                     │  │
│   │   ┌─────────────────────────────────────────────────────────────┐  │  │
│   │   │ ATTENTION LAYER                                             │  │  │
│   │   │   Query:  Linear(768 → 768)   ← prune_layers='attention'   │  │  │
│   │   │   Key:    Linear(768 → 768)                                 │  │  │
│   │   │   Value:  Linear(768 → 768)                                 │  │  │
│   │   │   Output: Linear(768 → 768)                                 │  │  │
│   │   └─────────────────────────────────────────────────────────────┘  │  │
│   │                           ↓                                        │  │
│   │   ┌─────────────────────────────────────────────────────────────┐  │  │
│   │   │ FFN LAYER (Feed-Forward Network)                            │  │  │
│   │   │   Intermediate: Linear(768 → 3072)  ← prune_layers='ffn'   │  │  │
│   │   │   Output:       Linear(3072 → 768)                          │  │  │
│   │   └─────────────────────────────────────────────────────────────┘  │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   prune_layers='all'       → Prune everything (attention + FFN)            │
│   prune_layers='attention' → Prune only Q, K, V, O projections             │
│   prune_layers='ffn'       → Prune only feed-forward layers                │
│   prune_layers='encoder'   → Prune encoder only, not classifier            │
│                                                                             │
│ RESEARCH INSIGHT:                                                           │
│   FFN layers have 2/3 of transformer parameters!                           │
│   Pruning FFN aggressively often works well.                               │
│   Pruning attention too much hurts model's ability to "pay attention"      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Layer Identification Method:

```python
def _identify_prunable_layers(self) -> List[Tuple[nn.Module, str]]:
    """
    Find all layers that can be pruned based on prune_layers setting.
    
    Returns list of (module, parameter_name) tuples.
    """
    prunable = []
    
    for name, module in self.model.named_modules():
        # Only prune Linear layers (have weight matrices)
        if not isinstance(module, nn.Linear):
            continue
        
        # Filter based on prune_layers setting
        if self.prune_layers == 'all':
            prunable.append((module, 'weight'))
            
        elif self.prune_layers == 'attention':
            # Attention layers have 'query', 'key', 'value', or 'attention' in name
            if any(x in name.lower() for x in ['query', 'key', 'value', 'attention']):
                prunable.append((module, 'weight'))
                
        elif self.prune_layers == 'ffn':
            # FFN layers have 'intermediate' or 'output' (but not attention output)
            if 'intermediate' in name.lower() or ('output' in name.lower() and 'attention' not in name.lower()):
                prunable.append((module, 'weight'))
                
        elif self.prune_layers == 'encoder':
            # Everything except classifier
            if 'classifier' not in name.lower():
                prunable.append((module, 'weight'))
    
    print(f"   Found {len(prunable)} prunable layers")
    return prunable
```

### Global vs Local Pruning:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ GLOBAL VS LOCAL PRUNING                                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ LOCAL PRUNING (prune each layer independently):                            │
│ ────────────────────────────────────────────────                            │
│                                                                             │
│   Layer 1 weights: [0.5, 0.4, 0.3, 0.2, 0.1]   → Remove 2 smallest        │
│   After pruning:   [0.5, 0.4, 0.3, 0,   0  ]   → 40% sparse               │
│                                                                             │
│   Layer 2 weights: [0.9, 0.8, 0.7, 0.6, 0.5]   → Remove 2 smallest        │
│   After pruning:   [0.9, 0.8, 0.7, 0,   0  ]   → 40% sparse               │
│                                                                             │
│   Problem: Layer 2's smallest (0.5, 0.6) are LARGER than Layer 1's        │
│            largest (0.3)! We removed important weights!                    │
│                                                                             │
│ ═══════════════════════════════════════════════════════════════════════════│
│                                                                             │
│ GLOBAL PRUNING (consider all layers together):                             │
│ ──────────────────────────────────────────────                              │
│                                                                             │
│   All weights: [0.5, 0.4, 0.3, 0.2, 0.1, 0.9, 0.8, 0.7, 0.6, 0.5]         │
│   Sorted:      [0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.6, 0.7, 0.8, 0.9]         │
│                                                                             │
│   Remove 40% (4 smallest): 0.1, 0.2, 0.3, 0.4                              │
│                                                                             │
│   Layer 1: [0.5, 0,   0,   0,   0  ]  → 80% sparse (had small weights)    │
│   Layer 2: [0.9, 0.8, 0.7, 0.6, 0.5]  → 0% sparse (had large weights)     │
│                                                                             │
│   Better! Globally, we removed the truly unimportant weights.              │
│                                                                             │
│ RECOMMENDATION: Use global_pruning=True (default)                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Section 3: Magnitude Pruning (Simplest Method)

```python
def apply_magnitude_pruning(self):
    """
    Prune weights with smallest absolute magnitude.
    
    The simplest and most common pruning method.
    Intuition: Small weights contribute little to output.
    
    Formula:
        importance(w) = |w|
        prune if |w| < threshold
    """
    print(f"\n✂️  Applying magnitude pruning (sparsity={self.target_sparsity*100:.0f}%)")
    
    if self.global_pruning:
        # Gather all weights into a single list
        parameters_to_prune = [
            (module, name) for module, name in self.prunable_layers
        ]
        
        # Apply global unstructured pruning
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=self.target_sparsity
        )
    else:
        # Prune each layer independently
        for module, name in self.prunable_layers:
            prune.l1_unstructured(module, name, amount=self.target_sparsity)
    
    sparsity_info = self.get_sparsity()
    print(f"   Achieved sparsity: {sparsity_info['overall']*100:.2f}%")
```

### How L1 Unstructured Pruning Works:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ L1 (MAGNITUDE) PRUNING ALGORITHM                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ STEP 1: Collect all weights                                                │
│                                                                             │
│   weights = [0.8, -0.05, 0.3, 0.01, -0.6, 0.02, 0.9, -0.03, 0.15, -0.7]   │
│                                                                             │
│ STEP 2: Take absolute values                                               │
│                                                                             │
│   |weights| = [0.8, 0.05, 0.3, 0.01, 0.6, 0.02, 0.9, 0.03, 0.15, 0.7]     │
│                                                                             │
│ STEP 3: Find threshold for target sparsity (50%)                           │
│                                                                             │
│   Sorted: [0.01, 0.02, 0.03, 0.05, 0.15, 0.3, 0.6, 0.7, 0.8, 0.9]         │
│                                    ↑                                        │
│                              50th percentile                                │
│   threshold = 0.15                                                          │
│                                                                             │
│ STEP 4: Create mask                                                        │
│                                                                             │
│   mask = |weight| > threshold                                              │
│   mask = [1, 0, 1, 0, 1, 0, 1, 0, 1, 1]                                    │
│                                                                             │
│ STEP 5: Apply mask                                                         │
│                                                                             │
│   pruned_weights = weights * mask                                          │
│   pruned_weights = [0.8, 0, 0.3, 0, -0.6, 0, 0.9, 0, 0.15, -0.7]          │
│                                                                             │
│   50% of weights are now zero!                                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### What You Can Modify:

| Modification | Effect | When to Use |
|--------------|--------|-------------|
| `target_sparsity=0.3` | Conservative, 30% removed | When accuracy is critical |
| `target_sparsity=0.7` | Aggressive, 70% removed | When compression is critical |
| `global_pruning=False` | Per-layer pruning | Ensure uniform sparsity |
| `prune_layers='ffn'` | Only prune FFN | Preserve attention capability |

---

## Section 4: Gradual Pruning (Better Accuracy)

Gradual pruning is more sophisticated. Instead of removing all weights at once, it slowly increases sparsity during training, allowing the model to adapt.

```python
class GradualPruner:
    """
    Gradually increase sparsity during training.
    
    WHY GRADUAL?
    ────────────
    One-shot pruning: Remove 50% instantly → model breaks → hard to recover
    Gradual pruning:  Remove 5%, adapt, remove 5%, adapt... → model survives
    
    Research shows gradual pruning achieves 2-5% better accuracy
    at the same sparsity level compared to one-shot.
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_sparsity: float = 0.5,
        start_epoch: int = 0,
        end_epoch: int = 10,
        schedule: str = 'cubic',
        prune_frequency: int = 100,
        prune_layers: str = 'all'
    ):
        """
        Args:
            model: Model to prune
            target_sparsity: Final sparsity to achieve
            start_epoch: When to start pruning
            end_epoch: When to reach target sparsity
            schedule: How to increase sparsity ('linear', 'cubic', 'exponential')
            prune_frequency: Prune every N training steps
            prune_layers: Which layers to prune
        """
        self.model = model
        self.target_sparsity = target_sparsity
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.schedule = schedule
        self.prune_frequency = prune_frequency
        self.current_sparsity = 0.0
        
        # Initialize pruning masks (all ones = nothing pruned yet)
        self.prunable_layers = self._identify_prunable_layers()
        self._initialize_masks()
```

### Understanding Pruning Schedules:

```python
def _compute_sparsity_for_step(self, current_step: int, total_steps: int) -> float:
    """
    Compute target sparsity for the current training step.
    
    Different schedules provide different pruning trajectories.
    """
    # Compute progress (0 to 1)
    progress = min(1.0, current_step / total_steps)
    
    if self.schedule == 'linear':
        # Linear: Constant pruning rate
        # sparsity = target × progress
        sparsity = self.target_sparsity * progress
        
    elif self.schedule == 'cubic':
        # Cubic: Slow start, fast middle, slow end (RECOMMENDED)
        # sparsity = target × (1 - (1 - progress)³)
        sparsity = self.target_sparsity * (1 - (1 - progress) ** 3)
        
    elif self.schedule == 'exponential':
        # Exponential: Slow start, increasingly fast
        # sparsity = target × (1 - exp(-5 × progress))
        sparsity = self.target_sparsity * (1 - np.exp(-5 * progress))
    
    return sparsity
```

### Visual Comparison of Schedules:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ PRUNING SCHEDULE COMPARISON                                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ Sparsity (%)                                                               │
│     │                                                                       │
│ 50% ├─────────────────────────────────────────────●●●●● ─── Target        │
│     │                                         ●●●●                          │
│     │                                     ●●●●     ●●●●                     │
│ 40% ├─────────────────────────────────●●●●────────────●●●                  │
│     │                              ●●●                   ●●                 │
│     │                          ●●●●                       ●                 │
│ 30% ├───────────────────────●●●●──────────────────────────●●               │
│     │                    ●●●                                ●               │
│     │                 ●●●                                                   │
│ 20% ├──────────────●●●────────────────────────────────────────             │
│     │           ●●●                                                         │
│     │        ●●●                                                            │
│ 10% ├─────●●●──────────────────────────────────────────────────            │
│     │   ●●                                                                  │
│     │ ●●                                                                    │
│  0% ├●●────────────────────────────────────────────────────────            │
│     └──────────────────────────────────────────────────────────            │
│       0    1    2    3    4    5    6    7    8    9    10  Epoch          │
│                                                                             │
│     ●●●●● LINEAR:       Constant rate (simplest)                           │
│     ───── CUBIC:        Slow-fast-slow (best for accuracy)                 │
│     ····· EXPONENTIAL:  Slow start, fast end                               │
│                                                                             │
│ WHY CUBIC IS BEST:                                                          │
│   - Slow start: Model has time to identify important weights               │
│   - Fast middle: Remove weights efficiently                                 │
│   - Slow end: Fine-tune remaining structure                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### The Critical Step Method:

```python
def step(self, current_step: int, total_steps: int):
    """
    Called every training step. May or may not prune depending on frequency.
    
    This is the heart of gradual pruning!
    """
    # Only prune every prune_frequency steps
    if current_step % self.prune_frequency != 0:
        return
    
    # Compute target sparsity for this step
    target = self._compute_sparsity_for_step(current_step, total_steps)
    
    # If target hasn't increased, nothing to do
    if target <= self.current_sparsity:
        return
    
    # CRITICAL: Compute INCREMENTAL amount to prune
    # This is where many implementations go wrong!
    incremental_sparsity = self._compute_incremental_sparsity(target)
    
    # Apply incremental pruning
    self._apply_incremental_pruning(incremental_sparsity)
    
    self.current_sparsity = target
```

### The Incremental Sparsity Bug (And Fix):

```python
def _compute_incremental_sparsity(self, new_target: float) -> float:
    """
    Compute how much MORE to prune to reach new target.
    
    CRITICAL BUG in naive implementations:
    ─────────────────────────────────────
    
    WRONG approach:
        Current: 30% sparse
        Target: 40% sparse
        Naive: Prune 10% of ORIGINAL weights
        
        But 30% are already zero! So we're pruning 10% of 70% = 7%
        Result: 30% + 7% = 37% (not 40%!)
        
    CORRECT approach:
        Current: 30% sparse (70% remain)
        Target: 40% sparse (60% should remain)
        Need to remove: (70% - 60%) / 70% = 14.3% of REMAINING weights
        
    Formula:
        incremental = (target - current) / (1 - current)
    """
    if self.current_sparsity >= 1.0:
        return 0.0
    
    # Correct incremental calculation
    incremental = (new_target - self.current_sparsity) / (1 - self.current_sparsity)
    
    return min(incremental, 1.0)  # Cap at 100%
```

### Visual Explanation of the Bug:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ WHY INCREMENTAL CALCULATION MATTERS                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ SCENARIO: Go from 30% to 40% sparsity                                      │
│                                                                             │
│ Original weights: [w1, w2, w3, w4, w5, w6, w7, w8, w9, w10]                │
│                                                                             │
│ After 30% pruning:                                                          │
│   [w1, 0, 0, w4, w5, w6, 0, w8, w9, w10]                                   │
│   3 zeros out of 10 = 30% sparse                                           │
│   7 non-zero weights remain                                                │
│                                                                             │
│ ═══════════════════════════════════════════════════════════════════════════│
│                                                                             │
│ WRONG: Prune 10% of original (1 weight)                                    │
│   Remove 1 from [w1, w4, w5, w6, w8, w9, w10]                              │
│   Result: 4 zeros → 40% ✓ (got lucky with 10 weights!)                    │
│                                                                             │
│   But with 1000 weights:                                                   │
│   30% sparse = 300 zeros, 700 remain                                       │
│   Prune 10% of 1000 = 100 weights                                          │
│   Result: 300 + 100 = 400 zeros = 40% ✓                                   │
│                                                                             │
│   Wait, that worked? Let's try 50% → 60%:                                  │
│   50% sparse = 500 zeros, 500 remain                                       │
│   Prune 10% of 1000 = 100 weights                                          │
│   Result: 500 + 100 = 600 zeros = 60% ✓                                   │
│                                                                             │
│   Now 60% → 70%:                                                           │
│   60% sparse = 600 zeros, 400 remain                                       │
│   Prune 10% of 1000 = 100 weights                                          │
│   Result: 600 + 100 = 700 zeros = 70% ✓                                   │
│                                                                             │
│   Now 70% → 80%:                                                           │
│   70% sparse = 700 zeros, 300 remain                                       │
│   Prune 10% of 1000 = 100 weights                                          │
│   BUT we only have 300 non-zero! Can't prune 100!                         │
│   ❌ BREAKS when remaining < increment                                     │
│                                                                             │
│ ═══════════════════════════════════════════════════════════════════════════│
│                                                                             │
│ CORRECT: Prune fraction of REMAINING weights                               │
│                                                                             │
│   30% → 40%:                                                               │
│   Remaining = 70%                                                          │
│   Need to reach: 60% remaining                                             │
│   Prune: (70% - 60%) / 70% = 14.3% of remaining                           │
│                                                                             │
│   50% → 60%:                                                               │
│   Remaining = 50%                                                          │
│   Need to reach: 40% remaining                                             │
│   Prune: (50% - 40%) / 50% = 20% of remaining                             │
│                                                                             │
│   70% → 80%:                                                               │
│   Remaining = 30%                                                          │
│   Need to reach: 20% remaining                                             │
│   Prune: (30% - 20%) / 30% = 33.3% of remaining                           │
│   ✓ Always possible!                                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Section 5: Wanda Pruning (State-of-the-Art)

Wanda (Weights AND Activations) is a 2023 method that considers not just weight magnitude but also how much each weight is actually used during inference.

```python
class WandaPruner:
    """
    Wanda Pruning: Pruning by Weights AND Activations
    
    Paper: "A Simple and Effective Pruning Approach for Large Language Models"
           (Sun et al., 2023)
    
    KEY INSIGHT:
    ────────────
    A small weight might still be important if it's multiplied by large activations!
    
    Example:
        weight = 0.1 (small)
        activation = 10.0 (large)
        contribution = 0.1 × 10.0 = 1.0 (significant!)
        
    Magnitude pruning would remove this weight. Wanda keeps it.
    
    FORMULA:
        importance(w) = |w| × ||activation||
        
    where ||activation|| is the L2 norm of activations that multiply with w.
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_sparsity: float = 0.5,
        prune_layers: str = 'all'
    ):
        self.model = model
        self.target_sparsity = target_sparsity
        self.prune_layers = prune_layers
        
        # Will store activation statistics
        self.activation_norms = {}
        self.hooks = []
```

### Collecting Activation Statistics:

```python
def collect_activations(
    self,
    dataloader,
    device: str,
    num_samples: int = 512
) -> None:
    """
    Run calibration data through model to collect activation statistics.
    
    This is what makes Wanda "data-aware" - it sees how the model
    actually uses each weight on real data.
    """
    print(f"   Collecting activations from {num_samples} samples...")
    
    self.model.eval()
    self.activation_norms = defaultdict(list)
    
    # Register hooks to capture activations
    def make_hook(name):
        def hook(module, input, output):
            # input[0] is the activation entering this layer
            # Shape: (batch, seq_len, hidden_size)
            activation = input[0].detach()
            
            # Compute L2 norm across sequence dimension
            # Shape: (batch, hidden_size)
            norm = torch.norm(activation, p=2, dim=1)
            
            # Average across batch
            # Shape: (hidden_size,)
            mean_norm = norm.mean(dim=0)
            
            self.activation_norms[name].append(mean_norm.cpu())
        
        return hook
    
    # Register hooks on all prunable layers
    for name, module in self.model.named_modules():
        if isinstance(module, nn.Linear) and self._should_prune(name):
            hook = module.register_forward_hook(make_hook(name))
            self.hooks.append(hook)
    
    # Run calibration data through model
    samples_seen = 0
    with torch.no_grad():
        for batch in dataloader:
            if samples_seen >= num_samples:
                break
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            _ = self.model(input_ids, attention_mask)
            samples_seen += len(input_ids)
    
    # Remove hooks
    for hook in self.hooks:
        hook.remove()
    self.hooks = []
    
    # Average activation norms across all batches
    for name in self.activation_norms:
        norms = torch.stack(self.activation_norms[name])
        self.activation_norms[name] = norms.mean(dim=0)
    
    print(f"   ✓ Collected activations for {len(self.activation_norms)} layers")
```

### Applying Wanda Pruning:

```python
def apply_wanda_pruning(self):
    """
    Apply Wanda pruning using collected activation statistics.
    """
    print(f"\n✂️  Applying Wanda pruning (sparsity={self.target_sparsity*100:.0f}%)")
    
    for name, module in self.model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name not in self.activation_norms:
            continue
        
        weight = module.weight.data  # Shape: (out_features, in_features)
        activation_norm = self.activation_norms[name].to(weight.device)
        
        # Compute Wanda importance scores
        # importance = |weight| × activation_norm
        # Broadcasting: (out, in) × (in,) = (out, in)
        importance = weight.abs() * activation_norm.unsqueeze(0)
        
        # Find threshold for target sparsity
        threshold = torch.quantile(importance.flatten(), self.target_sparsity)
        
        # Create mask
        mask = (importance > threshold).float()
        
        # Apply mask
        module.weight.data *= mask
    
    print(f"   ✓ Wanda pruning complete")
```

### Visual: Magnitude vs Wanda:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ MAGNITUDE VS WANDA PRUNING                                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ Example layer with 8 weights:                                              │
│                                                                             │
│   Weight:     [0.8,  0.05, 0.3,  0.02, 0.6,  0.01, 0.9,  0.1 ]            │
│   Activation: [1.0,  5.0,  1.0,  1.0,  1.0,  1.0,  1.0,  8.0 ]            │
│                                                                             │
│ ═══════════════════════════════════════════════════════════════════════════│
│                                                                             │
│ MAGNITUDE PRUNING (|weight| only):                                         │
│                                                                             │
│   Importance: [0.8,  0.05, 0.3,  0.02, 0.6,  0.01, 0.9,  0.1 ]            │
│   Sorted:     [0.01, 0.02, 0.05, 0.1,  0.3,  0.6,  0.8,  0.9 ]            │
│                                                                             │
│   50% pruning removes: 0.01, 0.02, 0.05, 0.1                               │
│                                                                             │
│   Result:     [0.8,  0,    0.3,  0,    0.6,  0,    0.9,  0   ]            │
│               Kept   ❌    Kept  ❌    Kept  ❌    Kept  ❌               │
│                                                                             │
│ ═══════════════════════════════════════════════════════════════════════════│
│                                                                             │
│ WANDA PRUNING (|weight| × activation):                                     │
│                                                                             │
│   Importance: [0.8,  0.25, 0.3,  0.02, 0.6,  0.01, 0.9,  0.8 ]            │
│                 │     │                                   │                 │
│                 │     └── 0.05 × 5.0 = 0.25 (boosted!)    │                 │
│                 │                                         │                 │
│                 └── 0.8 × 1.0 = 0.8                      └── 0.1 × 8.0 = 0.8│
│                                                                             │
│   Sorted:     [0.01, 0.02, 0.25, 0.3,  0.6,  0.8,  0.8,  0.9 ]            │
│                                                                             │
│   50% pruning removes: 0.01, 0.02, 0.25, 0.3                               │
│                                                                             │
│   Result:     [0.8,  0,    0,    0,    0.6,  0,    0.9,  0.1 ]            │
│               Kept   ❌    ❌    ❌    Kept  ❌    Kept  Kept!             │
│                                                                             │
│ KEY DIFFERENCE: Wanda kept weight[7]=0.1 because activation=8.0            │
│                 Magnitude pruning removed it (small weight)                │
│                 Wanda removed weight[2]=0.3 instead (low activation)       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### What You Can Modify:

| Modification | Effect | Research Use |
|--------------|--------|--------------|
| `num_samples` | More samples = more accurate statistics | Trade off speed vs accuracy |
| Use different norm | L1 instead of L2 | Different importance measure |
| Per-output pruning | Prune within each output neuron | More uniform sparsity |

**Example - Per-Output Wanda:**
```python
# Instead of global threshold, prune within each output row
def apply_wanda_per_output(self):
    for name, module in self.model.named_modules():
        if name not in self.activation_norms:
            continue
        
        weight = module.weight.data  # (out, in)
        importance = weight.abs() * self.activation_norms[name]
        
        # Prune within each output neuron (row)
        for i in range(weight.shape[0]):
            row_importance = importance[i]
            threshold = torch.quantile(row_importance, self.target_sparsity)
            mask = (row_importance > threshold).float()
            weight[i] *= mask
```

---

## Section 6: Structured Pruning

Unlike unstructured pruning (individual weights), structured pruning removes entire neurons, attention heads, or layers. This provides **actual speedup** without special sparse libraries.

```python
class StructuredPruner:
    """
    Remove entire neurons, attention heads, or layers.
    
    WHY STRUCTURED?
    ───────────────
    Unstructured pruning creates sparse matrices:
        [0.5, 0, 0.3, 0, 0, 0.8, 0, 0.2]
        
    Standard GPUs can't skip zeros efficiently.
    You need special sparse libraries to get speedup.
    
    Structured pruning removes whole rows/columns:
        Original:  (768, 768) matrix
        Pruned:    (768, 512) matrix  ← Actually smaller!
        
    ANY hardware can benefit from smaller matrices.
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_sparsity: float = 0.5,
        structure: str = 'neuron'  # 'neuron', 'head', 'layer'
    ):
        self.model = model
        self.target_sparsity = target_sparsity
        self.structure = structure
```

### Attention Head Pruning:

```python
def prune_attention_heads(self):
    """
    Remove entire attention heads based on importance.
    
    Each attention head learns different patterns:
    - Some heads focus on nearby words
    - Some heads focus on specific relationships
    - Some heads are redundant
    
    We can remove redundant heads with minimal accuracy loss.
    """
    print(f"\n✂️  Pruning attention heads (target: {self.target_sparsity*100:.0f}%)")
    
    # Find all attention layers
    for name, module in self.model.named_modules():
        if not self._is_attention_layer(module):
            continue
        
        num_heads = module.num_attention_heads
        head_dim = module.attention_head_size
        
        # Compute importance of each head
        # Method: L2 norm of each head's query weights
        query_weight = module.query.weight  # (hidden, hidden)
        
        head_importance = []
        for h in range(num_heads):
            start = h * head_dim
            end = (h + 1) * head_dim
            head_weights = query_weight[start:end, :]
            importance = torch.norm(head_weights, p=2)
            head_importance.append(importance.item())
        
        # Determine which heads to prune
        num_to_prune = int(num_heads * self.target_sparsity)
        heads_to_prune = np.argsort(head_importance)[:num_to_prune]
        
        # Zero out pruned heads
        for h in heads_to_prune:
            start = h * head_dim
            end = (h + 1) * head_dim
            module.query.weight.data[start:end, :] = 0
            module.key.weight.data[start:end, :] = 0
            module.value.weight.data[start:end, :] = 0
        
        print(f"   {name}: Pruned {len(heads_to_prune)}/{num_heads} heads")
```

### Visualization of Structured Pruning:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ STRUCTURED VS UNSTRUCTURED PRUNING                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ UNSTRUCTURED (individual weights):                                         │
│                                                                             │
│   Original matrix (4×4):          After 50% unstructured:                  │
│   ┌───┬───┬───┬───┐               ┌───┬───┬───┬───┐                        │
│   │0.5│0.3│0.8│0.2│               │0.5│ 0 │0.8│ 0 │                        │
│   ├───┼───┼───┼───┤               ├───┼───┼───┼───┤                        │
│   │0.1│0.7│0.4│0.9│       →       │ 0 │0.7│ 0 │0.9│                        │
│   ├───┼───┼───┼───┤               ├───┼───┼───┼───┤                        │
│   │0.6│0.2│0.5│0.3│               │0.6│ 0 │0.5│ 0 │                        │
│   ├───┼───┼───┼───┤               ├───┼───┼───┼───┤                        │
│   │0.8│0.4│0.1│0.7│               │0.8│ 0 │ 0 │0.7│                        │
│   └───┴───┴───┴───┘               └───┴───┴───┴───┘                        │
│                                                                             │
│   Still 4×4 matrix! GPU processes same number of elements.                │
│   Speedup requires sparse matrix libraries.                                │
│                                                                             │
│ ═══════════════════════════════════════════════════════════════════════════│
│                                                                             │
│ STRUCTURED (entire columns/neurons):                                       │
│                                                                             │
│   Original matrix (4×4):          After removing 2 neurons:                │
│   ┌───┬───┬───┬───┐               ┌───┬───┐                                │
│   │0.5│0.3│0.8│0.2│               │0.5│0.8│                                │
│   ├───┼───┼───┼───┤               ├───┼───┤                                │
│   │0.1│0.7│0.4│0.9│       →       │0.1│0.4│    Matrix is now 4×2!         │
│   ├───┼───┼───┼───┤               ├───┼───┤                                │
│   │0.6│0.2│0.5│0.3│               │0.6│0.5│    50% fewer multiplications  │
│   ├───┼───┼───┼───┤               ├───┼───┤                                │
│   │0.8│0.4│0.1│0.7│               │0.8│0.1│    Works on ANY hardware!      │
│   └───┴───┴───┴───┘               └───┴───┘                                │
│                                                                             │
│ TRADEOFF:                                                                   │
│   Unstructured: Higher compression possible, needs sparse libraries        │
│   Structured:   Lower compression, but real speedup on all hardware        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Section 7: Fine-Tuning After Pruning

After pruning, the model's accuracy drops. Fine-tuning allows remaining weights to compensate.

```python
def fine_tune_after_pruning(
    model: nn.Module,
    train_loader,
    val_loader,
    config,
    device: str
) -> nn.Module:
    """
    Fine-tune the model after pruning to recover accuracy.
    
    WHY FINE-TUNE?
    ──────────────
    Pruning removes weights that were contributing (even if small).
    The remaining weights need to adjust to compensate.
    
    Without fine-tuning: 70% F1 → 60% F1 (10% drop)
    With fine-tuning:    70% F1 → 67% F1 (3% drop)
    
    Fine-tuning is CRITICAL for good pruning results!
    """
    print(f"\n🔧 Fine-tuning pruned model for {config.fine_tune_epochs} epochs...")
    
    model.train()
    
    # Lower learning rate than initial training
    # We're fine-tuning, not training from scratch
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr * 0.1,  # 10% of original LR
        weight_decay=config.weight_decay
    )
    
    loss_fn = nn.BCEWithLogitsLoss()
    
    best_f1 = 0
    
    for epoch in range(config.fine_tune_epochs):
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Fine-tune Epoch {epoch+1}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(input_ids, attention_mask)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            
            loss = loss_fn(logits, labels)
            loss.backward()
            
            # IMPORTANT: Zero out gradients for pruned weights
            # This keeps pruned weights at zero!
            _zero_pruned_gradients(model)
            
            optimizer.step()
            total_loss += loss.item()
        
        # Evaluate
        f1 = evaluate_model(model, val_loader, device)
        print(f"   Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, F1={f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
    
    print(f"   ✓ Fine-tuning complete. Best F1: {best_f1:.4f}")
    return model


def _zero_pruned_gradients(model: nn.Module):
    """
    Zero out gradients for pruned (zero) weights.
    
    WHY?
    Without this, pruned weights could become non-zero again during training!
    
    gradient × lr = weight_update
    If pruned_weight = 0 but gradient ≠ 0:
        new_weight = 0 + gradient × lr ≠ 0  ← Weight comes back!
    
    By zeroing gradients for pruned weights:
        new_weight = 0 + 0 × lr = 0  ← Weight stays pruned!
    """
    for name, param in model.named_parameters():
        if param.grad is not None:
            # Create mask of non-zero weights
            mask = (param.data != 0).float()
            # Zero gradient where weight is zero
            param.grad.data *= mask
```

### Visualization of Fine-Tuning:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ WHY FINE-TUNING HELPS                                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ Before pruning:                                                            │
│                                                                             │
│   Input → [w1=0.5] → [w2=0.1] → [w3=0.8] → Output                         │
│                ↘     ↗                                                      │
│                [w4=0.05]                                                    │
│                                                                             │
│   All weights work together to produce correct output.                     │
│                                                                             │
│ ═══════════════════════════════════════════════════════════════════════════│
│                                                                             │
│ After pruning w4 (smallest):                                               │
│                                                                             │
│   Input → [w1=0.5] → [w2=0.1] → [w3=0.8] → Output (slightly wrong!)       │
│                ↘     ↗                                                      │
│                [w4=0] ← pruned                                             │
│                                                                             │
│   w4's contribution is missing. Output is less accurate.                   │
│                                                                             │
│ ═══════════════════════════════════════════════════════════════════════════│
│                                                                             │
│ After fine-tuning:                                                         │
│                                                                             │
│   Input → [w1=0.52] → [w2=0.12] → [w3=0.82] → Output (mostly correct!)    │
│                 ↘     ↗                                                     │
│                 [w4=0] ← stays pruned                                      │
│                                                                             │
│   Remaining weights adjusted to compensate for w4's absence.              │
│   w1: 0.5 → 0.52 (+4%)                                                     │
│   w2: 0.1 → 0.12 (+20%)                                                    │
│   w3: 0.8 → 0.82 (+2.5%)                                                   │
│                                                                             │
│   These small adjustments recover most of the lost accuracy!               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Section 8: Sparsity Tracking

```python
def get_sparsity(self) -> Dict[str, float]:
    """
    Calculate current sparsity statistics.
    
    Returns dict with:
        - overall: Total sparsity across all prunable layers
        - per_layer: Sparsity for each layer
        - zero_params: Count of zero parameters
        - nonzero_params: Count of non-zero parameters
    """
    total_params = 0
    zero_params = 0
    per_layer = {}
    
    for name, module in self.model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        
        weight = module.weight.data
        layer_total = weight.numel()
        layer_zeros = (weight == 0).sum().item()
        
        total_params += layer_total
        zero_params += layer_zeros
        
        per_layer[name] = layer_zeros / layer_total
    
    overall_sparsity = zero_params / total_params if total_params > 0 else 0
    
    return {
        'overall': overall_sparsity,
        'per_layer': per_layer,
        'zero_params': zero_params,
        'nonzero_params': total_params - zero_params,
        'total_params': total_params
    }
```

### Making Pruning Permanent:

```python
def make_pruning_permanent(self):
    """
    Convert pruning masks to actual zeros.
    
    PyTorch pruning uses masks: weight = weight_orig × mask
    This removes the mask and bakes zeros into the weight.
    
    WHEN TO DO THIS:
        - After all pruning is complete
        - Before saving the model
        - Before quantization (quantization doesn't understand masks)
    """
    for module, name in self.prunable_layers:
        if prune.is_pruned(module):
            prune.remove(module, name)
    
    print("   ✓ Pruning made permanent")
```

---

## Section 9: Utility Function to Get Any Pruner

```python
def get_pruner(
    model: nn.Module,
    method: str,
    target_sparsity: float,
    **kwargs
) -> Union[PruningManager, GradualPruner, WandaPruner, StructuredPruner]:
    """
    Factory function to get the right pruner based on method name.
    
    This is what research_main.py calls to get a pruner.
    """
    if method == 'magnitude':
        return PruningManager(model, target_sparsity, **kwargs)
    
    elif method == 'gradual':
        return GradualPruner(model, target_sparsity, **kwargs)
    
    elif method == 'wanda':
        return WandaPruner(model, target_sparsity, **kwargs)
    
    elif method == 'structured':
        return StructuredPruner(model, target_sparsity, **kwargs)
    
    else:
        raise ValueError(f"Unknown pruning method: {method}")
```

---

## Summary: Complete Pruning Method Comparison

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ PRUNING METHOD COMPARISON                                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ Method      │ Complexity │ Accuracy │ Speed    │ When to Use               │
│ ────────────┼────────────┼──────────┼──────────┼─────────────────────────── │
│ magnitude   │ Simple     │ Good     │ Fast     │ Quick experiments         │
│             │            │          │          │ Baseline comparison       │
│             │            │          │          │                           │
│ gradual     │ Medium     │ Better   │ Slower   │ Training time available   │
│             │ (needs     │ (+2-5%)  │          │ Want best accuracy        │
│             │ training)  │          │          │                           │
│             │            │          │          │                           │
│ wanda       │ Medium     │ Best     │ Medium   │ State-of-the-art results  │
│             │ (needs     │ (+3-7%)  │          │ Paper submissions         │
│             │ calibration)│         │          │                           │
│             │            │          │          │                           │
│ structured  │ Complex    │ Moderate │ Real     │ Need actual speedup       │
│             │            │ (-2-5%)  │ speedup! │ Deployment constraints    │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ RECOMMENDED WORKFLOW:                                                       │
│                                                                             │
│ 1. Start with 'magnitude' at 50% for quick experiments                     │
│ 2. Try 'wanda' for better accuracy (if you have calibration data)          │
│ 3. Use 'gradual' if you have time for full training                        │
│ 4. Use 'structured' only if you need real speedup without sparse libs      │
│                                                                             │
│ Always fine-tune after pruning!                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## What You Can Modify for Research

| Category | What to Modify | Research Question |
|----------|----------------|-------------------|
| **Sparsity** | 0.3, 0.5, 0.7 | Compression-accuracy tradeoff |
| **Method** | magnitude, wanda, gradual | Which method works best? |
| **Schedule** | linear, cubic, exponential | Best gradual pruning schedule? |
| **Layers** | all, attention, ffn | Which layers are most compressible? |
| **Global** | True/False | Global vs per-layer pruning? |
| **Fine-tune** | epochs, learning rate | How much recovery is possible? |

---

## Experiments You Can Run

```bash
# Experiment 1: Sparsity sweep
for sparsity in 0.3 0.4 0.5 0.6 0.7; do
    python research_main.py --pipeline prune_only \
        --prune_method magnitude \
        --prune_sparsity $sparsity \
        --output_dir results/sparsity_$sparsity
done

# Experiment 2: Method comparison at 50% sparsity
for method in magnitude wanda gradual; do
    python research_main.py --pipeline prune_only \
        --prune_method $method \
        --prune_sparsity 0.5 \
        --output_dir results/method_$method
done

# Experiment 3: Layer-specific pruning
for layer in all attention ffn encoder; do
    python research_main.py --pipeline prune_only \
        --prune_method magnitude \
        --prune_sparsity 0.5 \
        --prune_layers $layer \
        --output_dir results/layer_$layer
done

# Experiment 4: Fine-tuning epochs
for epochs in 1 3 5 10; do
    python research_main.py --pipeline prune_only \
        --prune_method magnitude \
        --prune_sparsity 0.5 \
        --fine_tune_after_prune \
        --fine_tune_epochs $epochs \
        --output_dir results/finetune_$epochs
done
```

---

## Practice Exercise

Before moving to the next script:

1. **Calculate incremental sparsity**: If current is 40% and target is 60%, what fraction of remaining weights should be pruned?
2. **Compare methods mentally**: Why would Wanda keep a weight that magnitude pruning removes?
3. **Think about trade-offs**: When would you prefer structured over unstructured pruning?

---

**Ready for the next script? The next one is `research_quantization.py` which implements all quantization methods (dynamic, static, QAT, FP16, INT4).**

Would you like me to continue?