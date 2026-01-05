# 📘 Script 6: `research_evaluation.py`

## Overview

This script is the **measurement backbone** of your entire research framework. It answers the crucial question: "Did our compression actually work?" by computing comprehensive metrics at every stage of the pipeline.

**Why evaluation is critical for research:**
- Compression is useless if accuracy drops too much
- You need to quantify the trade-offs (size vs accuracy vs speed)
- Reproducible metrics are essential for papers and comparisons
- Per-label metrics reveal if compression hurts specific categories (like threat detection)

---

## The Big Picture: What We Measure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ COMPREHENSIVE EVALUATION FRAMEWORK                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ ACCURACY METRICS (How correct are predictions?)                            │
│ ─────────────────────────────────────────────────                           │
│   • F1 Score (macro, weighted, micro)                                      │
│   • Precision and Recall                                                   │
│   • Exact Match Accuracy                                                   │
│   • Per-label Accuracy                                                     │
│   • Hamming Loss                                                           │
│   • ROC-AUC                                                                │
│                                                                             │
│ EFFICIENCY METRICS (How fast and small is the model?)                      │
│ ───────────────────────────────────────────────────────                     │
│   • Inference Latency (mean, P50, P95, P99)                               │
│   • Throughput (samples/second)                                            │
│   • Model Size (MB)                                                        │
│   • Parameter Count                                                        │
│   • Peak Memory Usage                                                      │
│                                                                             │
│ COMPRESSION METRICS (How much did we compress?)                            │
│ ─────────────────────────────────────────────────                           │
│   • Size Compression Ratio                                                 │
│   • Sparsity Percentage                                                    │
│   • Speedup Factor                                                         │
│                                                                             │
│ SAFETY METRICS (Is the model still safe for deployment?)                   │
│ ──────────────────────────────────────────────────────────                  │
│   • Priority-Weighted F1 (threat weighted 5×)                              │
│   • Per-label F1 for critical categories                                   │
│   • Threat Detection Accuracy                                              │
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
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, hamming_loss, roc_auc_score,
    classification_report, multilabel_confusion_matrix
)
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
import time
import json
from collections import defaultdict
```

### Key Imports Explained:

| Import | Purpose |
|--------|---------|
| `f1_score` | Harmonic mean of precision and recall |
| `hamming_loss` | Fraction of wrong labels |
| `roc_auc_score` | Area under ROC curve |
| `multilabel_confusion_matrix` | Per-label confusion matrices |
| `dataclass` | Clean data structure for metrics |

### The Label Configuration:

```python
LABEL_COLUMNS = ['bully', 'sexual', 'religious', 'threat', 'spam']

# Priority weights for safety-critical evaluation
# Higher weight = more important to get right
LABEL_PRIORITIES = {
    'threat': 5.0,      # Most critical - could indicate violence
    'sexual': 4.0,      # Very important - protect minors
    'religious': 3.0,   # Important - prevent hate speech
    'bully': 2.0,       # Standard importance
    'spam': 1.0         # Least critical
}
```

### Why Priority Weights?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ WHY PRIORITY-WEIGHTED EVALUATION?                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ SCENARIO: Two compressed models, same overall F1 = 0.70                    │
│                                                                             │
│ Model A per-label F1:                                                      │
│   bully: 0.75, sexual: 0.72, religious: 0.70, threat: 0.55, spam: 0.78    │
│                                                 ↑                           │
│                                         Threat detection broken!            │
│                                                                             │
│ Model B per-label F1:                                                      │
│   bully: 0.68, sexual: 0.70, religious: 0.68, threat: 0.72, spam: 0.72    │
│                                                 ↑                           │
│                                         Threat detection preserved!         │
│                                                                             │
│ STANDARD F1: Both models look equal (0.70)                                 │
│                                                                             │
│ PRIORITY-WEIGHTED F1:                                                       │
│   Model A: (2×0.75 + 4×0.72 + 3×0.70 + 5×0.55 + 1×0.78) / 15 = 0.67       │
│   Model B: (2×0.68 + 4×0.70 + 3×0.68 + 5×0.72 + 1×0.72) / 15 = 0.70       │
│                                                                             │
│ Model B is BETTER for deployment because it protects threat detection!     │
│                                                                             │
│ In cyberbullying detection, missing a THREAT is much worse than            │
│ missing spam. Priority weighting captures this real-world importance.      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Section 2: The Metrics Data Class

```python
@dataclass
class CompressionStageMetrics:
    """
    Comprehensive metrics for a single compression stage.
    
    WHY A DATACLASS?
    ────────────────
    - Clean, typed structure for 40+ metrics
    - Easy to convert to dict/JSON/CSV
    - IDE autocomplete for metric names
    - Immutable after creation (prevents bugs)
    
    This is the "report card" for each compression stage.
    """
    
    # Identification
    stage: str = ""                          # 'baseline', 'after_kd', 'after_pruning', etc.
    timestamp: str = ""                      # When evaluation was run
    
    # Model characteristics
    model_size_mb: float = 0.0               # Size in megabytes
    num_parameters: int = 0                  # Total parameter count
    num_trainable_params: int = 0            # Trainable parameters
    sparsity_percent: float = 0.0            # Percentage of zero weights
    
    # Classification metrics (overall)
    accuracy_exact: float = 0.0              # Exact match (all labels correct)
    accuracy_per_label: float = 0.0          # Average per-label accuracy
    f1_macro: float = 0.0                    # Unweighted mean of per-label F1
    f1_weighted: float = 0.0                 # Weighted by support
    f1_micro: float = 0.0                    # Global F1
    precision_macro: float = 0.0             # Unweighted mean precision
    recall_macro: float = 0.0                # Unweighted mean recall
    hamming: float = 0.0                     # Fraction of wrong labels
    roc_auc: float = 0.0                     # Area under ROC curve
    
    # Per-label metrics (stored as dicts)
    per_label_f1: Dict[str, float] = field(default_factory=dict)
    per_label_precision: Dict[str, float] = field(default_factory=dict)
    per_label_recall: Dict[str, float] = field(default_factory=dict)
    per_label_accuracy: Dict[str, float] = field(default_factory=dict)
    
    # Safety-critical metrics
    priority_weighted_f1: float = 0.0        # F1 weighted by label importance
    threat_f1: float = 0.0                   # Specific F1 for threat detection
    threat_recall: float = 0.0               # Recall for threats (don't miss any!)
    
    # Efficiency metrics
    latency_mean_ms: float = 0.0             # Average inference time
    latency_p50_ms: float = 0.0              # Median latency
    latency_p95_ms: float = 0.0              # 95th percentile
    latency_p99_ms: float = 0.0              # 99th percentile
    throughput_samples_per_sec: float = 0.0  # Processing speed
    peak_memory_mb: float = 0.0              # Maximum GPU memory used
    
    # Compression metrics (relative to baseline)
    size_compression_ratio: float = 1.0      # original_size / current_size
    speedup_factor: float = 1.0              # original_latency / current_latency
    
    # Confusion matrix data (for detailed analysis)
    confusion_matrices: Dict[str, List[List[int]]] = field(default_factory=dict)
```

### Understanding Multi-Label Metrics

Before diving into the code, let's understand what these metrics mean for multi-label classification:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ MULTI-LABEL CLASSIFICATION METRICS                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ EXAMPLE PREDICTIONS:                                                        │
│                                                                             │
│   Ground Truth:  [bully=1, sexual=0, religious=0, threat=1, spam=0]        │
│   Prediction:    [bully=1, sexual=0, religious=1, threat=0, spam=0]        │
│                         ✓        ✓          ✗         ✗        ✓           │
│                                                                             │
│ ═══════════════════════════════════════════════════════════════════════════│
│                                                                             │
│ EXACT MATCH ACCURACY:                                                       │
│   "Are ALL labels correct?"                                                │
│   This sample: 0 (not all correct)                                         │
│   Very strict! Often low for multi-label.                                  │
│                                                                             │
│ HAMMING LOSS:                                                               │
│   "What fraction of labels are wrong?"                                     │
│   This sample: 2/5 = 0.4 (religious and threat wrong)                      │
│   Lower is better. Good for multi-label.                                   │
│                                                                             │
│ ═══════════════════════════════════════════════════════════════════════════│
│                                                                             │
│ PER-LABEL METRICS (computed separately for each label):                    │
│                                                                             │
│   For "bully" label across all samples:                                    │
│       True Positives (TP): Predicted 1, Actual 1                           │
│       False Positives (FP): Predicted 1, Actual 0                          │
│       False Negatives (FN): Predicted 0, Actual 1                          │
│       True Negatives (TN): Predicted 0, Actual 0                           │
│                                                                             │
│       Precision = TP / (TP + FP)   "Of predicted positives, how many right?"│
│       Recall = TP / (TP + FN)      "Of actual positives, how many found?" │
│       F1 = 2 × (Precision × Recall) / (Precision + Recall)                │
│                                                                             │
│ ═══════════════════════════════════════════════════════════════════════════│
│                                                                             │
│ MACRO VS MICRO VS WEIGHTED:                                                 │
│                                                                             │
│   Per-label F1: [bully=0.8, sexual=0.7, religious=0.6, threat=0.5, spam=0.9]│
│   Per-label support: [1000, 500, 300, 100, 800]  (number of positive samples)│
│                                                                             │
│   MACRO F1: Simple average                                                 │
│       = (0.8 + 0.7 + 0.6 + 0.5 + 0.9) / 5 = 0.70                          │
│       Treats all labels equally (good for imbalanced data)                 │
│                                                                             │
│   WEIGHTED F1: Average weighted by support                                 │
│       = (0.8×1000 + 0.7×500 + 0.6×300 + 0.5×100 + 0.9×800) / 2700 = 0.77 │
│       Weights by frequency (dominated by common labels)                    │
│                                                                             │
│   MICRO F1: Aggregate TP, FP, FN across all labels, then compute          │
│       = 2 × (total_TP) / (2×total_TP + total_FP + total_FN)               │
│       Treats each prediction equally (not each label)                      │
│                                                                             │
│ FOR CYBERBULLYING: Use MACRO F1 as primary metric                         │
│   - Treats rare labels (threat) as important as common ones (bully)        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Conversion Methods:

```python
def to_dict(self) -> Dict[str, Any]:
    """Convert metrics to dictionary for JSON export."""
    return asdict(self)

def to_flat_dict(self) -> Dict[str, Any]:
    """
    Flatten nested dicts for CSV export.
    
    Before: {'per_label_f1': {'bully': 0.8, 'threat': 0.5}}
    After:  {'per_label_f1_bully': 0.8, 'per_label_f1_threat': 0.5}
    """
    result = {}
    for key, value in asdict(self).items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                result[f"{key}_{sub_key}"] = sub_value
        else:
            result[key] = value
    return result

def print_summary(self):
    """Print a human-readable summary of metrics."""
    print(f"\n{'='*60}")
    print(f"📊 METRICS: {self.stage.upper()}")
    print(f"{'='*60}")
    
    print(f"\n📦 Model Size:")
    print(f"   Size: {self.model_size_mb:.2f} MB")
    print(f"   Parameters: {self.num_parameters/1e6:.2f}M")
    print(f"   Sparsity: {self.sparsity_percent:.1f}%")
    
    print(f"\n📈 Classification Metrics:")
    print(f"   F1 Macro: {self.f1_macro:.4f}")
    print(f"   F1 Weighted: {self.f1_weighted:.4f}")
    print(f"   Precision: {self.precision_macro:.4f}")
    print(f"   Recall: {self.recall_macro:.4f}")
    print(f"   Exact Accuracy: {self.accuracy_exact:.4f}")
    print(f"   Hamming Loss: {self.hamming:.4f}")
    
    print(f"\n🏷️  Per-Label F1:")
    for label, f1 in self.per_label_f1.items():
        priority = LABEL_PRIORITIES.get(label, 1)
        print(f"   {label}: {f1:.4f} (priority: {priority})")
    
    print(f"\n🛡️  Safety Metrics:")
    print(f"   Priority-Weighted F1: {self.priority_weighted_f1:.4f}")
    print(f"   Threat F1: {self.threat_f1:.4f}")
    print(f"   Threat Recall: {self.threat_recall:.4f}")
    
    print(f"\n⚡ Efficiency Metrics:")
    print(f"   Latency (mean): {self.latency_mean_ms:.2f} ms")
    print(f"   Latency (P95): {self.latency_p95_ms:.2f} ms")
    print(f"   Throughput: {self.throughput_samples_per_sec:.1f} samples/sec")
    
    print(f"\n📉 Compression:")
    print(f"   Size Compression: {self.size_compression_ratio:.2f}×")
    print(f"   Speedup: {self.speedup_factor:.2f}×")
```

---

## Section 3: The Compression Evaluator Class

```python
class CompressionEvaluator:
    """
    Main class for evaluating compressed models.
    
    RESPONSIBILITIES:
    ─────────────────
    1. Run inference on validation data
    2. Compute all classification metrics
    3. Measure latency and throughput
    4. Track model size and sparsity
    5. Compare across compression stages
    
    USAGE:
    ──────
    evaluator = CompressionEvaluator()
    
    baseline_metrics = evaluator.evaluate_model(teacher, val_loader, 'cuda', 'baseline')
    kd_metrics = evaluator.evaluate_model(student, val_loader, 'cuda', 'after_kd')
    
    comparison = evaluator.compare_stages([baseline_metrics, kd_metrics])
    """
    
    def __init__(self, label_columns: List[str] = LABEL_COLUMNS):
        self.label_columns = label_columns
        self.baseline_metrics: Optional[CompressionStageMetrics] = None
```

### The Main Evaluation Method:

```python
def evaluate_model(
    self,
    model: nn.Module,
    dataloader,
    device: str,
    stage: str = 'unknown',
    threshold: float = 0.5
) -> CompressionStageMetrics:
    """
    Comprehensively evaluate a model.
    
    Args:
        model: Model to evaluate
        dataloader: Validation data
        device: 'cuda' or 'cpu'
        stage: Name of this stage ('baseline', 'after_kd', etc.)
        threshold: Classification threshold (default 0.5)
    
    Returns:
        CompressionStageMetrics with all computed metrics
    """
    print(f"\n🔍 Evaluating: {stage}")
    
    model.eval()
    model.to(device)
    
    # Collect predictions
    all_predictions = []
    all_probabilities = []
    all_labels = []
    latencies = []
    
    # Warm-up run (first inference is often slower)
    self._warmup(model, dataloader, device)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].numpy()
            
            # Measure latency
            if device == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.time()
            
            outputs = model(input_ids, attention_mask)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            latencies.append((end_time - start_time) * 1000)  # ms
            
            # Get predictions
            probabilities = torch.sigmoid(logits).cpu().numpy()
            predictions = (probabilities > threshold).astype(int)
            
            all_probabilities.extend(probabilities)
            all_predictions.extend(predictions)
            all_labels.extend(labels)
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    all_labels = np.array(all_labels)
    
    # Compute all metrics
    metrics = self._compute_all_metrics(
        all_predictions, all_probabilities, all_labels,
        latencies, model, stage
    )
    
    # Store baseline for comparison
    if stage == 'baseline':
        self.baseline_metrics = metrics
    
    # Compute relative metrics if baseline exists
    if self.baseline_metrics is not None and stage != 'baseline':
        metrics.size_compression_ratio = (
            self.baseline_metrics.model_size_mb / metrics.model_size_mb
        )
        metrics.speedup_factor = (
            self.baseline_metrics.latency_mean_ms / metrics.latency_mean_ms
        )
    
    return metrics
```

### Understanding the Evaluation Flow:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ EVALUATION PIPELINE                                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ STEP 1: WARM-UP                                                            │
│ ─────────────────                                                           │
│   Run 3 batches without timing                                             │
│   Why? First inference often includes:                                      │
│   - CUDA kernel compilation                                                 │
│   - Memory allocation                                                       │
│   - Cache warming                                                           │
│   These would skew latency measurements                                    │
│                                                                             │
│ STEP 2: INFERENCE WITH TIMING                                              │
│ ───────────────────────────────                                             │
│   For each batch:                                                          │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ 1. torch.cuda.synchronize()  ← Ensure GPU is idle                   │  │
│   │ 2. start_time = time.time()                                         │  │
│   │ 3. outputs = model(batch)                                           │  │
│   │ 4. torch.cuda.synchronize()  ← Wait for GPU to finish               │  │
│   │ 5. end_time = time.time()                                           │  │
│   │ 6. latency = end_time - start_time                                  │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   Why synchronize?                                                          │
│   CUDA operations are asynchronous by default.                             │
│   Without sync, we'd measure time to LAUNCH operations, not COMPLETE them. │
│                                                                             │
│ STEP 3: COLLECT OUTPUTS                                                    │
│ ─────────────────────────                                                   │
│   logits → sigmoid → probabilities → threshold → predictions              │
│                                                                             │
│   Store:                                                                    │
│   - predictions: Binary [0,1] for each label                              │
│   - probabilities: Float [0,1] for ROC-AUC computation                    │
│   - labels: Ground truth                                                   │
│                                                                             │
│ STEP 4: COMPUTE METRICS                                                    │
│ ─────────────────────────                                                   │
│   Feed collected data to sklearn metrics functions                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Section 4: Computing Classification Metrics

```python
def _compute_all_metrics(
    self,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    labels: np.ndarray,
    latencies: List[float],
    model: nn.Module,
    stage: str
) -> CompressionStageMetrics:
    """
    Compute all classification, efficiency, and compression metrics.
    """
    metrics = CompressionStageMetrics()
    metrics.stage = stage
    metrics.timestamp = datetime.now().isoformat()
    
    # =========================================================================
    # CLASSIFICATION METRICS
    # =========================================================================
    
    # Overall metrics
    metrics.f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
    metrics.f1_weighted = f1_score(labels, predictions, average='weighted', zero_division=0)
    metrics.f1_micro = f1_score(labels, predictions, average='micro', zero_division=0)
    metrics.precision_macro = precision_score(labels, predictions, average='macro', zero_division=0)
    metrics.recall_macro = recall_score(labels, predictions, average='macro', zero_division=0)
    
    # Exact match accuracy (all labels must be correct)
    metrics.accuracy_exact = accuracy_score(labels, predictions)
    
    # Hamming loss (fraction of wrong labels)
    metrics.hamming = hamming_loss(labels, predictions)
    
    # Per-label accuracy
    per_label_acc = (predictions == labels).mean(axis=0)
    metrics.accuracy_per_label = per_label_acc.mean()
    
    # ROC-AUC (needs probabilities, not binary predictions)
    try:
        metrics.roc_auc = roc_auc_score(labels, probabilities, average='macro')
    except ValueError:
        # Can fail if a label has no positive samples
        metrics.roc_auc = 0.0
```

### Computing Per-Label Metrics:

```python
    # =========================================================================
    # PER-LABEL METRICS
    # =========================================================================
    
    # Compute metrics for each label separately
    for i, label in enumerate(self.label_columns):
        label_preds = predictions[:, i]
        label_true = labels[:, i]
        label_probs = probabilities[:, i]
        
        # F1, Precision, Recall for this label
        metrics.per_label_f1[label] = f1_score(
            label_true, label_preds, zero_division=0
        )
        metrics.per_label_precision[label] = precision_score(
            label_true, label_preds, zero_division=0
        )
        metrics.per_label_recall[label] = recall_score(
            label_true, label_preds, zero_division=0
        )
        metrics.per_label_accuracy[label] = accuracy_score(
            label_true, label_preds
        )
        
        # Store confusion matrix for detailed analysis
        # [[TN, FP], [FN, TP]]
        cm = multilabel_confusion_matrix(labels, predictions)[i]
        metrics.confusion_matrices[label] = cm.tolist()
    
    # Special handling for critical labels
    metrics.threat_f1 = metrics.per_label_f1.get('threat', 0)
    metrics.threat_recall = metrics.per_label_recall.get('threat', 0)
```

### Computing Priority-Weighted F1:

```python
    # =========================================================================
    # PRIORITY-WEIGHTED F1
    # =========================================================================
    
    # This is our custom metric that weights by label importance
    total_weighted_f1 = 0
    total_weight = 0
    
    for label, f1 in metrics.per_label_f1.items():
        weight = LABEL_PRIORITIES.get(label, 1.0)
        total_weighted_f1 += f1 * weight
        total_weight += weight
    
    metrics.priority_weighted_f1 = total_weighted_f1 / total_weight if total_weight > 0 else 0
```

### Visual: How Priority Weighting Works

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ PRIORITY-WEIGHTED F1 CALCULATION                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ Per-label F1 scores:                                                       │
│   bully: 0.72                                                              │
│   sexual: 0.68                                                             │
│   religious: 0.65                                                          │
│   threat: 0.58                                                             │
│   spam: 0.80                                                               │
│                                                                             │
│ Label priorities:                                                          │
│   bully: 2, sexual: 4, religious: 3, threat: 5, spam: 1                   │
│                                                                             │
│ STANDARD MACRO F1:                                                          │
│   = (0.72 + 0.68 + 0.65 + 0.58 + 0.80) / 5                                │
│   = 3.43 / 5                                                               │
│   = 0.686                                                                   │
│                                                                             │
│ PRIORITY-WEIGHTED F1:                                                       │
│   = (2×0.72 + 4×0.68 + 3×0.65 + 5×0.58 + 1×0.80) / (2+4+3+5+1)           │
│   = (1.44 + 2.72 + 1.95 + 2.90 + 0.80) / 15                               │
│   = 9.81 / 15                                                              │
│   = 0.654                                                                   │
│          ↑                                                                  │
│   Lower! Because threat (priority=5) has low F1 (0.58)                     │
│   This metric properly penalizes poor threat detection                     │
│                                                                             │
│ If we improved threat F1 from 0.58 to 0.70:                               │
│   Standard Macro F1: 0.686 → 0.710 (+0.024)                               │
│   Priority-Weighted: 0.654 → 0.694 (+0.040)                               │
│   Priority-weighted shows bigger improvement!                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Section 5: Computing Efficiency Metrics

```python
    # =========================================================================
    # EFFICIENCY METRICS
    # =========================================================================
    
    # Latency statistics
    latencies_np = np.array(latencies)
    metrics.latency_mean_ms = np.mean(latencies_np)
    metrics.latency_p50_ms = np.percentile(latencies_np, 50)
    metrics.latency_p95_ms = np.percentile(latencies_np, 95)
    metrics.latency_p99_ms = np.percentile(latencies_np, 99)
    
    # Throughput
    total_samples = len(labels)
    total_time_sec = sum(latencies) / 1000
    metrics.throughput_samples_per_sec = total_samples / total_time_sec
    
    # Peak memory (CUDA only)
    if next(model.parameters()).is_cuda:
        metrics.peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        torch.cuda.reset_peak_memory_stats()
    else:
        metrics.peak_memory_mb = 0
```

### Understanding Latency Percentiles:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ WHY DIFFERENT LATENCY PERCENTILES?                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ Imagine 100 inference runs with these latencies (ms):                      │
│                                                                             │
│   [10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13,    │
│    13, 14, 14, 14, 14, 15, 15, 15, 15, 16, 16, 16, 17, 17, 17, 18, 18,    │
│    18, 19, 19, 20, 20, 21, 21, 22, 23, 24, 25, 26, 28, 30, 35, 50]        │
│                                                                             │
│   MEAN:      17.8 ms   ← Average (affected by outliers!)                   │
│   P50 (Median): 15 ms  ← "Typical" latency                                 │
│   P95:       30 ms     ← 95% of requests are faster than this              │
│   P99:       50 ms     ← 99% of requests are faster than this              │
│                                                                             │
│ WHY EACH MATTERS:                                                           │
│                                                                             │
│   MEAN: Easy to understand, but skewed by outliers                         │
│         If one request takes 1000ms, mean jumps significantly              │
│                                                                             │
│   P50:  "What does a typical user experience?"                             │
│         Half of requests are faster, half are slower                       │
│                                                                             │
│   P95:  "What's the worst case for most users?"                            │
│         Important for SLA (Service Level Agreement)                        │
│         "We guarantee 95% of requests complete in <X ms"                   │
│                                                                             │
│   P99:  "What's the tail latency?"                                         │
│         Shows worst-case performance                                       │
│         High P99 might indicate GC pauses, resource contention             │
│                                                                             │
│ FOR YOUR PAPER: Report P50 (typical) and P95 (worst-case guarantee)        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Section 6: Computing Model Size and Sparsity

```python
    # =========================================================================
    # MODEL SIZE AND SPARSITY
    # =========================================================================
    
    # Count parameters
    total_params = 0
    trainable_params = 0
    zero_params = 0
    
    for param in model.parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
        # Count zeros (for sparsity)
        zero_params += (param.data == 0).sum().item()
    
    metrics.num_parameters = total_params
    metrics.num_trainable_params = trainable_params
    metrics.sparsity_percent = (zero_params / total_params) * 100 if total_params > 0 else 0
    
    # Model size in MB
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    metrics.model_size_mb = (param_size + buffer_size) / (1024 ** 2)
    
    return metrics
```

### Understanding Sparsity Calculation:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ SPARSITY CALCULATION                                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ Original model weights:                                                     │
│   [0.5, 0.3, 0.8, 0.1, 0.6, 0.2, 0.9, 0.4, 0.7, 0.15]                     │
│   Total: 10 params, Zero: 0, Sparsity: 0%                                  │
│                                                                             │
│ After 50% magnitude pruning:                                               │
│   [0.5, 0,   0.8, 0,   0.6, 0,   0.9, 0,   0.7, 0  ]                       │
│         ↑         ↑         ↑         ↑         ↑                          │
│   Total: 10 params, Zero: 5, Sparsity: 50%                                 │
│                                                                             │
│ FORMULA:                                                                    │
│   sparsity_percent = (zero_params / total_params) × 100                    │
│                                                                             │
│ WHY TRACK SPARSITY?                                                         │
│   - Verifies pruning worked correctly                                      │
│   - Higher sparsity = more compression potential                           │
│   - Helps debug if sparsity doesn't match target                          │
│                                                                             │
│ NOTE: Sparsity doesn't directly equal size reduction!                      │
│   Unless you use sparse storage formats, zeros still take memory.          │
│   Sparse storage: Only store non-zero values + indices                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Section 7: Comparing Compression Stages

```python
def compare_stages(metrics_list: List[CompressionStageMetrics]) -> pd.DataFrame:
    """
    Create a comparison table across compression stages.
    
    This is what you'll put in your paper!
    
    Example output:
    
    Stage           | F1 Macro | Size (MB) | Compress | Speedup | Threat F1
    ----------------|----------|-----------|----------|---------|----------
    baseline        | 0.7200   | 420.0     | 1.00×    | 1.00×   | 0.6500
    after_kd        | 0.6980   | 250.0     | 1.68×    | 1.20×   | 0.6300
    after_pruning   | 0.6750   | 200.0     | 2.10×    | 1.35×   | 0.6100
    after_quant     | 0.6700   | 50.0      | 8.40×    | 2.50×   | 0.6000
    """
    rows = []
    
    for metrics in metrics_list:
        rows.append({
            'Stage': metrics.stage,
            'F1 Macro': f"{metrics.f1_macro:.4f}",
            'F1 Weighted': f"{metrics.f1_weighted:.4f}",
            'Precision': f"{metrics.precision_macro:.4f}",
            'Recall': f"{metrics.recall_macro:.4f}",
            'Size (MB)': f"{metrics.model_size_mb:.1f}",
            'Sparsity (%)': f"{metrics.sparsity_percent:.1f}",
            'Compression': f"{metrics.size_compression_ratio:.2f}×",
            'Speedup': f"{metrics.speedup_factor:.2f}×",
            'Latency P95 (ms)': f"{metrics.latency_p95_ms:.1f}",
            'Priority F1': f"{metrics.priority_weighted_f1:.4f}",
            'Threat F1': f"{metrics.threat_f1:.4f}",
            'Threat Recall': f"{metrics.threat_recall:.4f}"
        })
    
    df = pd.DataFrame(rows)
    return df
```

### Computing Trade-off Scores:

```python
def compute_tradeoff_score(
    metrics: CompressionStageMetrics,
    baseline: CompressionStageMetrics,
    accuracy_weight: float = 0.5,
    compression_weight: float = 0.3,
    speed_weight: float = 0.2
) -> float:
    """
    Compute a single score balancing accuracy, compression, and speed.
    
    Higher score = better overall trade-off.
    
    FORMULA:
        score = w_a × (accuracy_retention) + 
                w_c × (normalized_compression) + 
                w_s × (normalized_speedup)
    
    WHERE:
        accuracy_retention = current_f1 / baseline_f1 (want close to 1)
        normalized_compression = compression_ratio / 10 (cap at 10×)
        normalized_speedup = speedup / 5 (cap at 5×)
    
    INTERPRETATION:
        - Score ≈ 0.8-0.9: Good trade-off
        - Score < 0.6: Too much accuracy loss
        - Score > 0.95: Excellent compression with minimal loss
    """
    # Accuracy retention (0 to 1, higher is better)
    accuracy_retention = metrics.f1_macro / baseline.f1_macro if baseline.f1_macro > 0 else 0
    
    # Normalized compression (0 to 1, assuming max 10×)
    normalized_compression = min(metrics.size_compression_ratio / 10, 1.0)
    
    # Normalized speedup (0 to 1, assuming max 5×)
    normalized_speedup = min(metrics.speedup_factor / 5, 1.0)
    
    # Weighted score
    score = (
        accuracy_weight * accuracy_retention +
        compression_weight * normalized_compression +
        speed_weight * normalized_speedup
    )
    
    return score
```

### Visual: Trade-off Scoring

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ TRADE-OFF SCORING EXAMPLE                                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ Configuration A (Aggressive compression):                                  │
│   F1: 0.72 → 0.60 (83% retention)                                          │
│   Compression: 8×                                                          │
│   Speedup: 3×                                                               │
│                                                                             │
│   Score = 0.5 × 0.83 + 0.3 × (8/10) + 0.2 × (3/5)                         │
│         = 0.415 + 0.24 + 0.12                                              │
│         = 0.775                                                             │
│                                                                             │
│ Configuration B (Conservative compression):                                │
│   F1: 0.72 → 0.70 (97% retention)                                          │
│   Compression: 2×                                                          │
│   Speedup: 1.2×                                                             │
│                                                                             │
│   Score = 0.5 × 0.97 + 0.3 × (2/10) + 0.2 × (1.2/5)                       │
│         = 0.485 + 0.06 + 0.048                                             │
│         = 0.593                                                             │
│                                                                             │
│ Configuration A has HIGHER score despite lower accuracy!                   │
│ Why? Because it achieves much better compression/speed.                    │
│                                                                             │
│ ADJUSTING WEIGHTS:                                                          │
│   - accuracy_weight=0.8: Prioritize accuracy (for safety-critical apps)    │
│   - compression_weight=0.5: Prioritize size (for mobile deployment)        │
│   - speed_weight=0.5: Prioritize latency (for real-time applications)      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Section 8: Exporting Results

```python
def export_metrics_to_csv(
    metrics_list: List[CompressionStageMetrics],
    output_path: str
) -> None:
    """
    Export metrics to CSV for spreadsheet analysis.
    
    Creates a file with all metrics flattened to columns.
    Perfect for:
    - Loading into Excel/Google Sheets
    - Creating paper tables
    - Statistical analysis
    """
    rows = [m.to_flat_dict() for m in metrics_list]
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"   ✓ Metrics exported to: {output_path}")


def export_metrics_to_json(
    metrics_list: List[CompressionStageMetrics],
    output_path: str
) -> None:
    """
    Export metrics to JSON for programmatic access.
    
    Preserves nested structure (per_label_f1 as dict).
    Perfect for:
    - Loading in Python/JavaScript
    - API responses
    - Automated reporting
    """
    data = [m.to_dict() for m in metrics_list]
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    
    print(f"   ✓ Metrics exported to: {output_path}")
```

### Generating LaTeX Tables:

```python
def generate_latex_table(
    metrics_list: List[CompressionStageMetrics],
    columns: List[str] = None
) -> str:
    """
    Generate a LaTeX table for paper inclusion.
    
    Returns a string that can be copied directly into your paper.
    """
    if columns is None:
        columns = ['stage', 'f1_macro', 'model_size_mb', 'size_compression_ratio', 'threat_f1']
    
    # Header
    latex = "\\begin{table}[h]\n"
    latex += "\\centering\n"
    latex += "\\caption{Compression Results}\n"
    latex += "\\label{tab:compression}\n"
    
    # Column format
    col_format = '|' + 'c|' * len(columns)
    latex += f"\\begin{{tabular}}{{{col_format}}}\n"
    latex += "\\hline\n"
    
    # Header row
    header_names = {
        'stage': 'Stage',
        'f1_macro': 'F1 Macro',
        'model_size_mb': 'Size (MB)',
        'size_compression_ratio': 'Compression',
        'threat_f1': 'Threat F1'
    }
    headers = [header_names.get(c, c) for c in columns]
    latex += ' & '.join(headers) + ' \\\\\n'
    latex += "\\hline\n"
    
    # Data rows
    for metrics in metrics_list:
        flat = metrics.to_flat_dict()
        values = []
        for col in columns:
            val = flat.get(col, '')
            if isinstance(val, float):
                if 'ratio' in col or 'compression' in col.lower():
                    values.append(f"{val:.2f}$\\times$")
                else:
                    values.append(f"{val:.4f}")
            else:
                values.append(str(val))
        latex += ' & '.join(values) + ' \\\\\n'
    
    latex += "\\hline\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"
    
    return latex
```

### Example LaTeX Output:

```latex
\begin{table}[h]
\centering
\caption{Compression Results}
\label{tab:compression}
\begin{tabular}{|c|c|c|c|c|}
\hline
Stage & F1 Macro & Size (MB) & Compression & Threat F1 \\
\hline
baseline & 0.7200 & 420.0 & 1.00$\times$ & 0.6500 \\
after_kd & 0.6980 & 250.0 & 1.68$\times$ & 0.6300 \\
after_pruning & 0.6750 & 200.0 & 2.10$\times$ & 0.6100 \\
after_quant & 0.6700 & 50.0 & 8.40$\times$ & 0.6000 \\
\hline
\end{tabular}
\end{table}
```

---

## Section 9: Safety Checks

```python
def check_safety_thresholds(
    metrics: CompressionStageMetrics,
    baseline: CompressionStageMetrics,
    max_f1_drop: float = 0.05,
    max_threat_f1_drop: float = 0.03
) -> Dict[str, bool]:
    """
    Check if compressed model is safe for deployment.
    
    For cyberbullying detection, we CANNOT deploy a model that:
    1. Has too much overall accuracy drop
    2. Has significant threat detection degradation
    
    These thresholds should be set based on your application requirements.
    """
    checks = {}
    
    # Overall F1 check
    f1_drop = baseline.f1_macro - metrics.f1_macro
    checks['f1_acceptable'] = f1_drop <= max_f1_drop
    checks['f1_drop'] = f1_drop
    
    # Threat F1 check (more strict!)
    threat_drop = baseline.threat_f1 - metrics.threat_f1
    checks['threat_acceptable'] = threat_drop <= max_threat_f1_drop
    checks['threat_drop'] = threat_drop
    
    # Threat recall check (we should NOT miss threats)
    recall_drop = baseline.threat_recall - metrics.threat_recall
    checks['threat_recall_acceptable'] = recall_drop <= max_threat_f1_drop
    checks['threat_recall_drop'] = recall_drop
    
    # Overall safety verdict
    checks['safe_for_deployment'] = all([
        checks['f1_acceptable'],
        checks['threat_acceptable'],
        checks['threat_recall_acceptable']
    ])
    
    if not checks['safe_for_deployment']:
        print(f"\n⚠️  SAFETY WARNING!")
        if not checks['f1_acceptable']:
            print(f"   F1 dropped by {f1_drop:.4f} (max allowed: {max_f1_drop})")
        if not checks['threat_acceptable']:
            print(f"   Threat F1 dropped by {threat_drop:.4f} (max allowed: {max_threat_f1_drop})")
        if not checks['threat_recall_acceptable']:
            print(f"   Threat recall dropped by {recall_drop:.4f}")
        print(f"   Consider less aggressive compression settings.")
    
    return checks
```

### Why Safety Checks Matter:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ WHY SAFETY CHECKS FOR CYBERBULLYING DETECTION?                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ SCENARIO: Deploying compressed model to social media platform              │
│                                                                             │
│ User posts: "I'm going to kill you and your whole family"                  │
│                                                                             │
│ GOOD MODEL (detects threat):                                               │
│   → Content flagged for review                                             │
│   → User warned/suspended                                                   │
│   → Potential violence prevented                                           │
│                                                                             │
│ BAD COMPRESSED MODEL (misses threat):                                      │
│   → Content posted normally                                                 │
│   → No intervention                                                         │
│   → Potential real-world harm                                              │
│                                                                             │
│ ═══════════════════════════════════════════════════════════════════════════│
│                                                                             │
│ THIS IS WHY:                                                                │
│                                                                             │
│ 1. We have stricter thresholds for threat detection                        │
│    max_threat_f1_drop = 0.03 (3%) vs max_f1_drop = 0.05 (5%)              │
│                                                                             │
│ 2. We separately track threat RECALL (not just F1)                         │
│    Recall = "What % of actual threats did we catch?"                       │
│    Missing 20% of threats is UNACCEPTABLE even if precision is high       │
│                                                                             │
│ 3. We fail deployment if ANY safety check fails                            │
│    Better to have a slower model than a dangerous one                      │
│                                                                             │
│ FOR YOUR PAPER: Include safety analysis section showing these checks!      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Section 10: Complete Evaluation Workflow

```python
def run_full_evaluation(
    model: nn.Module,
    val_loader,
    device: str,
    stage: str,
    baseline_metrics: Optional[CompressionStageMetrics] = None,
    output_dir: str = './results'
) -> CompressionStageMetrics:
    """
    Run complete evaluation and save all results.
    
    This is typically called from research_main.py after each compression stage.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create evaluator
    evaluator = CompressionEvaluator()
    if baseline_metrics is not None:
        evaluator.baseline_metrics = baseline_metrics
    
    # Run evaluation
    metrics = evaluator.evaluate_model(model, val_loader, device, stage)
    
    # Print summary
    metrics.print_summary()
    
    # Save results
    export_metrics_to_csv([metrics], os.path.join(output_dir, f'metrics_{stage}.csv'))
    export_metrics_to_json([metrics], os.path.join(output_dir, f'metrics_{stage}.json'))
    
    # Safety check if baseline available
    if baseline_metrics is not None:
        safety = check_safety_thresholds(metrics, baseline_metrics)
        
        # Save safety report
        with open(os.path.join(output_dir, f'safety_{stage}.json'), 'w') as f:
            json.dump(safety, f, indent=2)
    
    return metrics
```

---

## Summary: Complete Evaluation Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ COMPLETE EVALUATION PIPELINE                                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ INPUT: Trained model + Validation data                                     │
│                                                                             │
│ STEP 1: WARM-UP                                                            │
│   └─→ Run 3 batches to initialize CUDA/caches                             │
│                                                                             │
│ STEP 2: INFERENCE                                                          │
│   └─→ For each batch:                                                      │
│       ├─→ Time inference (with CUDA sync)                                  │
│       ├─→ Collect probabilities                                            │
│       └─→ Collect predictions                                              │
│                                                                             │
│ STEP 3: CLASSIFICATION METRICS                                             │
│   ├─→ F1 (macro, weighted, micro)                                         │
│   ├─→ Precision, Recall                                                   │
│   ├─→ Exact match accuracy                                                │
│   ├─→ Hamming loss                                                        │
│   ├─→ ROC-AUC                                                             │
│   └─→ Per-label metrics for all 5 categories                              │
│                                                                             │
│ STEP 4: EFFICIENCY METRICS                                                 │
│   ├─→ Latency (mean, P50, P95, P99)                                       │
│   ├─→ Throughput                                                          │
│   └─→ Peak memory                                                         │
│                                                                             │
│ STEP 5: MODEL METRICS                                                      │
│   ├─→ Size (MB)                                                           │
│   ├─→ Parameter count                                                     │
│   └─→ Sparsity (%)                                                        │
│                                                                             │
│ STEP 6: SAFETY CHECKS                                                      │
│   ├─→ Overall F1 drop check                                               │
│   ├─→ Threat F1 drop check                                                │
│   └─→ Threat recall check                                                 │
│                                                                             │
│ STEP 7: EXPORT                                                             │
│   ├─→ CSV (for spreadsheets)                                              │
│   ├─→ JSON (for programs)                                                 │
│   └─→ LaTeX (for papers)                                                  │
│                                                                             │
│ OUTPUT: CompressionStageMetrics + files in output_dir                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## What You Can Modify for Research

| Category | What to Modify | Research Impact |
|----------|----------------|-----------------|
| **Priorities** | `LABEL_PRIORITIES` | Change which labels matter most |
| **Threshold** | Classification threshold (0.5) | Trade precision vs recall |
| **Safety** | `max_f1_drop`, `max_threat_f1_drop` | Strictness of deployment criteria |
| **Trade-off** | Score weights | Prioritize accuracy vs compression |
| **Metrics** | Add custom metrics | Domain-specific evaluation |

---

## Custom Metric Example

```python
# Add a custom metric: "Safe Compression Score"
def compute_safe_compression_score(metrics, baseline):
    """
    Custom metric that penalizes threat detection degradation.
    
    Score = compression_ratio × threat_retention²
    
    The squared term heavily penalizes threat degradation.
    """
    threat_retention = metrics.threat_f1 / baseline.threat_f1 if baseline.threat_f1 > 0 else 0
    compression = metrics.size_compression_ratio
    
    # Square threat retention to penalize drops heavily
    score = compression * (threat_retention ** 2)
    
    return score

# Example:
# Compression: 4×, Threat retention: 95% → Score = 4 × 0.95² = 3.61
# Compression: 4×, Threat retention: 80% → Score = 4 × 0.80² = 2.56
# The second scenario is penalized despite same compression!
```

---

## Practice Exercise

Before moving to the next script:

1. **Calculate priority-weighted F1**: Given F1 scores [0.8, 0.7, 0.6, 0.5, 0.9] and priorities [2, 4, 3, 5, 1], compute the weighted score.

2. **Interpret latency percentiles**: If mean=15ms, P50=12ms, P95=35ms, P99=100ms, what does this tell you about the latency distribution?

3. **Design a safety threshold**: For a children's social media platform, what `max_threat_f1_drop` would you set and why?

---

**Ready for the final script? The last one is `research_main.py` which orchestrates everything together - loading data, running compression stages, and generating final reports.**

Would you like me to continue?