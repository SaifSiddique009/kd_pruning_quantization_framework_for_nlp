# Model Compression Scenarios Guide

This document describes all available compression scenarios for the Bangla Cyberbullying Detection framework.

## Table of Contents

- [Overview](#overview)
- [Quick Reference](#quick-reference)
- [Scenario 1: Baseline](#scenario-1-baseline)
- [Scenario 2: KD Only](#scenario-2-kd-only-knowledge-distillation)
- [Scenario 3: Prune Only](#scenario-3-prune-only)
- [Scenario 4: Quant Only](#scenario-4-quant-only)
- [Scenario 5: KD + Prune](#scenario-5-kd--prune)
- [Scenario 6: KD + Quant](#scenario-6-kd--quant)
- [Scenario 7: Prune + Quant](#scenario-7-prune--quant)
- [Scenario 8: Full Pipeline](#scenario-8-full-pipeline-kd--prune--quant)
- [Scenario 9: Pruning Methods](#scenario-9-pruning-method-comparison)
- [Scenario 10: Quantization Types](#scenario-10-quantization-type-comparison)
- [Scenario 11: Sparsity Levels](#scenario-11-sparsity-level-comparison)
- [Scenario 12: K-Fold Validation](#scenario-12-k-fold-cross-validation)
- [Scenario 13: Label Priority](#scenario-13-label-priority-weighted)
- [Scenario 14: Student Models](#scenario-14-student-model-comparison)
- [Scenario 15: Hyperparameter Search](#scenario-15-hyperparameter-grid-search)
- [Recommended Workflow](#recommended-workflow)
- [Comparing Results](#comparing-results)

---

## Overview

This framework supports multiple compression techniques that can be combined:

| Technique | Description | Compression | Speed Gain |
|-----------|-------------|-------------|------------|
| **Knowledge Distillation (KD)** | Transfer knowledge from large teacher to small student | 2-4x | 2-4x |
| **Pruning** | Remove unimportant weights | 1.5-3x | 1.2-2x |
| **Quantization** | Reduce precision (FP32 -> INT8/INT4) | 2-4x | 1.5-3x |

### Your Models

- **Teacher**: `Saif-Siddique/bangla-cyberbully-xlm-roberta-base` (finetuned)
- **Student**: `neuropark/sahajBERT` (pretrained, will be trained via KD)

---

## Quick Reference

| Scenario | Pipeline | Purpose | Estimated Time |
|----------|----------|---------|----------------|
| 1 | `baseline` | Teacher performance reference | 10 min |
| 2 | `kd_only` | Knowledge distillation | 30 min |
| 3 | `prune_only` | Weight pruning | 20 min |
| 4 | `quant_only` | Quantization | 10 min |
| 5 | `kd_prune` | Distill + prune | 45 min |
| 6 | `kd_quant` | Distill + quantize | 35 min |
| 7 | `prune_quant` | Prune + quantize | 25 min |
| 8 | `kd_prune_quant` | Full compression | 1 hour |

---

## Scenario 1: BASELINE

**Purpose:** Establish performance baseline of teacher model (no compression)

**What happens:**
- Loads your finetuned teacher model from HuggingFace
- Evaluates on test set
- Records metrics (F1, latency, model size)
- This is your reference point for comparison

**Command:**
```bash
python main.py \
    --dataset_path ./data/1_Multilablel_Cyberbully_Data.csv \
    --author_name "Saif-Siddique" \
    --pipeline baseline \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-xlm-roberta-base" \
    --output_dir ./results/scenario1_baseline
```

**Expected Output:**
- `final_metrics.json` with teacher performance
- Model size and latency measurements

---

## Scenario 2: KD ONLY (Knowledge Distillation)

**Purpose:** Transfer knowledge from large teacher to smaller student model

**What happens:**
1. Teacher generates soft labels (probability distributions)
2. Student learns from both:
   - **Soft labels** from teacher (weighted by alpha)
   - **Hard labels** from ground truth (weighted by 1-alpha)
3. Result: Smaller model with similar accuracy

**Key Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--kd_epochs` | 5 | Training epochs |
| `--kd_alpha` | 0.7 | Weight for soft loss (0.7 = 70% teacher, 30% ground truth) |
| `--kd_temperature` | 4.0 | Softens probability distribution (higher = softer) |
| `--kd_learning_rate` | 2e-5 | Learning rate |

**Command:**
```bash
python main.py \
    --dataset_path ./data/1_Multilablel_Cyberbully_Data.csv \
    --author_name "Saif-Siddique" \
    --pipeline kd_only \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-xlm-roberta-base" \
    --student_path "neuropark/sahajBERT" \
    --kd_epochs 5 \
    --kd_learning_rate 2e-5 \
    --kd_alpha 0.7 \
    --kd_temperature 4.0 \
    --output_dir ./results/scenario2_kd_only
```

**Expected Output:**
- Distilled student model in `compressed_models/model_hf/`
- Training curves and metrics

---

## Scenario 3: PRUNE ONLY

**Purpose:** Make teacher model sparser by removing unimportant weights

**What happens:**
1. Identifies least important weights (by magnitude)
2. Sets them to zero (creates sparsity)
3. Fine-tunes to recover accuracy
4. Result: Sparser model, potentially faster on specialized hardware

**Key Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--pruning_method` | magnitude | Algorithm: magnitude, gradual, wanda |
| `--target_sparsity` | 0.3 | Fraction of weights to remove (0.3 = 30%) |
| `--prune_epochs` | 3 | Fine-tuning epochs after pruning |

**Command:**
```bash
python main.py \
    --dataset_path ./data/1_Multilablel_Cyberbully_Data.csv \
    --author_name "Saif-Siddique" \
    --pipeline prune_only \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-xlm-roberta-base" \
    --pruning_method magnitude \
    --target_sparsity 0.3 \
    --prune_epochs 3 \
    --output_dir ./results/scenario3_prune_only
```

---

## Scenario 4: QUANT ONLY

**Purpose:** Reduce precision of model weights

**What happens:**
1. Converts FP32 weights to lower precision
2. FP16: 16-bit floating point
3. INT8: 8-bit integer (~4x smaller)
4. INT4: 4-bit integer (~8x smaller, requires bitsandbytes)

**Key Parameters:**
| Parameter | Options | Description |
|-----------|---------|-------------|
| `--quantization_type` | fp16, int8, int4 | Quantization level |

**Command:**
```bash
python main.py \
    --dataset_path ./data/1_Multilablel_Cyberbully_Data.csv \
    --author_name "Saif-Siddique" \
    --pipeline quant_only \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-xlm-roberta-base" \
    --quantization_type int8 \
    --output_dir ./results/scenario4_quant_only
```

---

## Scenario 5: KD + PRUNE

**Purpose:** Two-stage compression - distill first, then prune

**What happens:**
1. Knowledge distillation to student
2. Prune the distilled student
3. Fine-tune to recover accuracy

**Command:**
```bash
python main.py \
    --dataset_path ./data/1_Multilablel_Cyberbully_Data.csv \
    --author_name "Saif-Siddique" \
    --pipeline kd_prune \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-xlm-roberta-base" \
    --student_path "neuropark/sahajBERT" \
    --kd_epochs 5 \
    --kd_alpha 0.7 \
    --pruning_method magnitude \
    --target_sparsity 0.3 \
    --prune_epochs 3 \
    --output_dir ./results/scenario5_kd_prune
```

---

## Scenario 6: KD + QUANT

**Purpose:** Distill to student, then quantize

**What happens:**
1. Knowledge distillation to student
2. Quantize the distilled student

**Command:**
```bash
python main.py \
    --dataset_path ./data/1_Multilablel_Cyberbully_Data.csv \
    --author_name "Saif-Siddique" \
    --pipeline kd_quant \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-xlm-roberta-base" \
    --student_path "neuropark/sahajBERT" \
    --kd_epochs 5 \
    --quantization_type int8 \
    --output_dir ./results/scenario6_kd_quant
```

---

## Scenario 7: PRUNE + QUANT

**Purpose:** Prune teacher, then quantize (no KD)

**What happens:**
1. Prune teacher model
2. Quantize the pruned teacher

**Command:**
```bash
python main.py \
    --dataset_path ./data/1_Multilablel_Cyberbully_Data.csv \
    --author_name "Saif-Siddique" \
    --pipeline prune_quant \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-xlm-roberta-base" \
    --pruning_method magnitude \
    --target_sparsity 0.3 \
    --quantization_type int8 \
    --output_dir ./results/scenario7_prune_quant
```

---

## Scenario 8: FULL PIPELINE (KD + PRUNE + QUANT)

**Purpose:** Maximum compression using all three techniques

**What happens:**
1. Knowledge distillation to student
2. Prune the distilled student
3. Quantize the pruned student
4. Result: Smallest, fastest model

**Command:**
```bash
python main.py \
    --dataset_path ./data/1_Multilablel_Cyberbully_Data.csv \
    --author_name "Saif-Siddique" \
    --pipeline kd_prune_quant \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-xlm-roberta-base" \
    --student_path "neuropark/sahajBERT" \
    --kd_epochs 5 \
    --kd_alpha 0.7 \
    --kd_temperature 4.0 \
    --pruning_method magnitude \
    --target_sparsity 0.3 \
    --prune_epochs 3 \
    --quantization_type int8 \
    --output_dir ./results/scenario8_full_pipeline
```

---

## Scenario 9: PRUNING METHOD COMPARISON

Compare different pruning algorithms:

| Method | Description | Pros | Cons |
|--------|-------------|------|------|
| **magnitude** | Remove smallest absolute weights | Simple, fast | May remove important small weights |
| **gradual** | Prune incrementally during training | Better accuracy | Slower |
| **wanda** | Activation-aware pruning | State-of-the-art | More complex |

### 9A: Magnitude Pruning
```bash
python main.py \
    --dataset_path ./data/1_Multilablel_Cyberbully_Data.csv \
    --author_name "Saif-Siddique" \
    --pipeline kd_prune \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-xlm-roberta-base" \
    --student_path "neuropark/sahajBERT" \
    --pruning_method magnitude \
    --target_sparsity 0.5 \
    --output_dir ./results/scenario9a_magnitude
```

### 9B: Gradual Pruning
```bash
python main.py \
    --dataset_path ./data/1_Multilablel_Cyberbully_Data.csv \
    --author_name "Saif-Siddique" \
    --pipeline kd_prune \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-xlm-roberta-base" \
    --student_path "neuropark/sahajBERT" \
    --pruning_method gradual \
    --target_sparsity 0.5 \
    --output_dir ./results/scenario9b_gradual
```

### 9C: Wanda Pruning
```bash
python main.py \
    --dataset_path ./data/1_Multilablel_Cyberbully_Data.csv \
    --author_name "Saif-Siddique" \
    --pipeline kd_prune \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-xlm-roberta-base" \
    --student_path "neuropark/sahajBERT" \
    --pruning_method wanda \
    --target_sparsity 0.5 \
    --output_dir ./results/scenario9c_wanda
```

---

## Scenario 10: QUANTIZATION TYPE COMPARISON

Compare different quantization levels:

| Type | Bits | Size Reduction | Quality Loss |
|------|------|----------------|--------------|
| **fp16** | 16 | ~2x | Minimal |
| **int8** | 8 | ~4x | Low |
| **int4** | 4 | ~8x | Moderate |

### 10A: FP16 Quantization
```bash
python main.py \
    --dataset_path ./data/1_Multilablel_Cyberbully_Data.csv \
    --author_name "Saif-Siddique" \
    --pipeline kd_quant \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-xlm-roberta-base" \
    --student_path "neuropark/sahajBERT" \
    --quantization_type fp16 \
    --output_dir ./results/scenario10a_fp16
```

### 10B: INT8 Quantization
```bash
python main.py \
    --dataset_path ./data/1_Multilablel_Cyberbully_Data.csv \
    --author_name "Saif-Siddique" \
    --pipeline kd_quant \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-xlm-roberta-base" \
    --student_path "neuropark/sahajBERT" \
    --quantization_type int8 \
    --output_dir ./results/scenario10b_int8
```

### 10C: INT4 Quantization
```bash
python main.py \
    --dataset_path ./data/1_Multilablel_Cyberbully_Data.csv \
    --author_name "Saif-Siddique" \
    --pipeline kd_quant \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-xlm-roberta-base" \
    --student_path "neuropark/sahajBERT" \
    --quantization_type int4 \
    --output_dir ./results/scenario10c_int4
```

---

## Scenario 11: SPARSITY LEVEL COMPARISON

Compare different pruning intensities:

| Sparsity | Weights Removed | Expected F1 Drop |
|----------|-----------------|------------------|
| 20% | 1 in 5 | Minimal |
| 30% | 1 in 3 | Low |
| 50% | 1 in 2 | Moderate |
| 70% | 7 in 10 | Significant |

### Commands
```bash
# 20% Sparsity
python main.py \
    --pipeline kd_prune \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-xlm-roberta-base" \
    --student_path "neuropark/sahajBERT" \
    --target_sparsity 0.2 \
    --output_dir ./results/scenario11_sparsity_20

# 30% Sparsity
python main.py \
    --pipeline kd_prune \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-xlm-roberta-base" \
    --student_path "neuropark/sahajBERT" \
    --target_sparsity 0.3 \
    --output_dir ./results/scenario11_sparsity_30

# 50% Sparsity
python main.py \
    --pipeline kd_prune \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-xlm-roberta-base" \
    --student_path "neuropark/sahajBERT" \
    --target_sparsity 0.5 \
    --output_dir ./results/scenario11_sparsity_50

# 70% Sparsity
python main.py \
    --pipeline kd_prune \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-xlm-roberta-base" \
    --student_path "neuropark/sahajBERT" \
    --target_sparsity 0.7 \
    --output_dir ./results/scenario11_sparsity_70
```

---

## Scenario 12: K-FOLD CROSS VALIDATION

**Purpose:** Robust evaluation across all K folds

**What happens:**
- Runs the full pipeline on all 5 folds
- Aggregates results: mean +/- std
- More reliable performance estimate

**Command:**
```bash
python main.py \
    --dataset_path ./data/1_Multilablel_Cyberbully_Data.csv \
    --author_name "Saif-Siddique" \
    --pipeline kd_prune_quant \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-xlm-roberta-base" \
    --student_path "neuropark/sahajBERT" \
    --run_full_kfold \
    --num_folds 5 \
    --output_dir ./results/scenario12_kfold
```

**Expected Output:**
```
Aggregate Results (5 folds):
  F1 Macro: 0.847 +/- 0.012
  Precision: 0.832 +/- 0.015
  Recall: 0.863 +/- 0.011
```

---

## Scenario 13: LABEL PRIORITY WEIGHTED

**Purpose:** Weight certain labels higher in evaluation

**Use Case:** When detecting threats is more critical than detecting spam

**Priority Format:** JSON dictionary with label weights
```json
{"threat": 3, "sexual": 2, "bully": 1, "religious": 1, "spam": 1}
```

**Command:**
```bash
python main.py \
    --dataset_path ./data/1_Multilablel_Cyberbully_Data.csv \
    --author_name "Saif-Siddique" \
    --pipeline kd_prune_quant \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-xlm-roberta-base" \
    --student_path "neuropark/sahajBERT" \
    --label_priority '{"threat": 3, "sexual": 2, "bully": 1, "religious": 1, "spam": 1}' \
    --output_dir ./results/scenario13_weighted
```

---

## Scenario 14: STUDENT MODEL COMPARISON

Compare different student architectures:

| Student | Parameters | Language | Notes |
|---------|------------|----------|-------|
| sahajBERT | ~110M | Bangla | Optimized for Bangla |
| DistilBERT | ~66M | Multilingual | Smaller, general purpose |
| MobileBERT | ~25M | English | Very small, mobile-friendly |

### 14A: sahajBERT (Bangla Optimized)
```bash
python main.py \
    --pipeline kd_only \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-xlm-roberta-base" \
    --student_path "neuropark/sahajBERT" \
    --output_dir ./results/scenario14a_sahajbert
```

### 14B: DistilBERT Multilingual
```bash
python main.py \
    --pipeline kd_only \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-xlm-roberta-base" \
    --student_path "distilbert-base-multilingual-cased" \
    --output_dir ./results/scenario14b_distilbert
```

### 14C: MobileBERT
```bash
python main.py \
    --pipeline kd_only \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-xlm-roberta-base" \
    --student_path "google/mobilebert-uncased" \
    --output_dir ./results/scenario14c_mobilebert
```

---

## Scenario 15: HYPERPARAMETER GRID SEARCH

**Purpose:** Find optimal KD hyperparameters

**Search Space:**
| Parameter | Values |
|-----------|--------|
| alpha | 0.5, 0.7, 0.9 |
| temperature | 2.0, 4.0, 6.0 |
| learning_rate | 1e-5, 2e-5, 5e-5 |

**Commands:**
```bash
# Alpha=0.5, Temperature=2.0
python main.py \
    --pipeline kd_only \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-xlm-roberta-base" \
    --student_path "neuropark/sahajBERT" \
    --kd_alpha 0.5 \
    --kd_temperature 2.0 \
    --data_fraction 0.2 \
    --output_dir ./results/scenario15_grid/a0.5_t2.0

# Alpha=0.7, Temperature=4.0 (Default)
python main.py \
    --pipeline kd_only \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-xlm-roberta-base" \
    --student_path "neuropark/sahajBERT" \
    --kd_alpha 0.7 \
    --kd_temperature 4.0 \
    --data_fraction 0.2 \
    --output_dir ./results/scenario15_grid/a0.7_t4.0

# Alpha=0.9, Temperature=6.0
python main.py \
    --pipeline kd_only \
    --teacher_checkpoint "Saif-Siddique/bangla-cyberbully-xlm-roberta-base" \
    --student_path "neuropark/sahajBERT" \
    --kd_alpha 0.9 \
    --kd_temperature 6.0 \
    --data_fraction 0.2 \
    --output_dir ./results/scenario15_grid/a0.9_t6.0
```

---

## Recommended Workflow

```
Step 1: Quick Test (5 min)
    python main.py --data_fraction 0.05 --kd_epochs 1 ...
    Verify everything works

Step 2: Baseline (10 min)
    Scenario 1: Get teacher metrics

Step 3: KD Only (30 min)
    Scenario 2: Establish KD performance

Step 4: Compare Compression Methods (1-2 hours)
    Scenario 3: Prune Only
    Scenario 4: Quant Only
    Compare size/accuracy tradeoff

Step 5: Best Combination (1 hour)
    Scenario 8: Full Pipeline

Step 6: Ablation Studies (2-4 hours)
    Scenario 9: Pruning methods
    Scenario 10: Quantization types
    Scenario 11: Sparsity sweep

Step 7: Final Evaluation (1-2 hours)
    Scenario 12: K-Fold on best config
```

---

## Comparing Results

### Collect Results Script

```python
import json
import os
import pandas as pd

results = []
results_dir = './results'

for scenario in os.listdir(results_dir):
    metrics_path = os.path.join(results_dir, scenario, 'final_metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        results.append({
            'scenario': scenario,
            'f1_macro': metrics.get('f1_macro', 0),
            'f1_weighted': metrics.get('f1_weighted', 0),
            'precision': metrics.get('precision_macro', 0),
            'recall': metrics.get('recall_macro', 0),
            'latency_ms': metrics.get('latency_mean_ms', 0),
            'model_size_mb': metrics.get('model_size_mb', 0),
            'sparsity': metrics.get('sparsity', 0),
            'compression_ratio': metrics.get('compression_ratio', 1)
        })

df = pd.DataFrame(results).sort_values('f1_macro', ascending=False)
print(df.to_string(index=False))
df.to_csv('./results/comparison.csv', index=False)
```

### Expected Output Table

| Scenario | F1 Macro | Size (MB) | Latency (ms) | Compression |
|----------|----------|-----------|--------------|-------------|
| baseline | 0.89 | 450 | 45 | 1.0x |
| kd_only | 0.86 | 250 | 25 | 1.8x |
| kd_prune | 0.84 | 180 | 20 | 2.5x |
| kd_quant | 0.85 | 65 | 15 | 6.9x |
| kd_prune_quant | 0.83 | 45 | 12 | 10.0x |

---

## Troubleshooting

### Out of Memory (OOM)
```bash
# Reduce batch size
--kd_batch_size 8

# Use smaller data fraction for testing
--data_fraction 0.1
```

### Slow Training
```bash
# Use fewer epochs for testing
--kd_epochs 2

# Use FP16 for faster training
--mixed_precision
```

### INT4 Not Working
```bash
# Install bitsandbytes
pip install bitsandbytes

# Or fall back to INT8
--quantization_type int8
```

---

## Citation

If you use this framework, please cite:

```bibtex
@software{bangla_cyberbully_compression,
  title = {Bangla Cyberbullying Detection Model Compression Framework},
  author = {Saif-Siddique},
  url = {https://github.com/Saif-Siddique},
  year = {2024}
}
```
