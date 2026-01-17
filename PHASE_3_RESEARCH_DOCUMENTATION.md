# Phase 3 Research Documentation: Knowledge Distillation and Pruning for Bangla Cyberbullying Detection

> **Document Version**: 1.0
> **Last Updated**: January 2025
> **Author**: Saif Siddique
> **Repository**: https://github.com/SaifSiddique009/kd_pruning_quantization_framework_for_nlp

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Research Background](#2-research-background)
3. [Dataset Description](#3-dataset-description)
4. [Model Architecture](#4-model-architecture)
5. [Knowledge Distillation Implementation](#5-knowledge-distillation-implementation)
6. [Pruning Implementation](#6-pruning-implementation)
7. [Experiment Design](#7-experiment-design)
8. [Evaluation Metrics](#8-evaluation-metrics)
9. [Implementation Details](#9-implementation-details)
10. [Critical Research Decisions](#10-critical-research-decisions)
11. [Experiment Execution Plan](#11-experiment-execution-plan)
12. [Results](#12-results-placeholder)
13. [Appendices](#13-appendices)

---

## 1. Executive Summary

### 1.1 Research Objectives

This research investigates model compression techniques for Bangla cyberbullying detection, focusing on:

1. **Knowledge Distillation (KD)**: Transfer knowledge from large teacher models to smaller student models
2. **Pruning**: Remove unnecessary weights to reduce model size while maintaining accuracy
3. **Combined Approaches**: Evaluate KD followed by various pruning methods

### 1.2 Key Contributions

1. **Multi-label KD Adaptation**: Modified standard KD for multi-label classification using sigmoid activation instead of softmax
2. **Comprehensive Pruning Comparison**: Evaluated 4 pruning methods (Magnitude, WANDA, Gradual, Structured)
3. **Fair Evaluation Framework**: Ensured consistent evaluation using original fold splits from Phase 1 & 2
4. **Student-Matched Hyperparameters**: Used optimal hyperparameters discovered in Phase 1 & 2 for KD training

### 1.3 Experiment Summary

| Scenario | Description | Runs |
|----------|-------------|------|
| Scenario 1 | Baseline evaluations (T1, T2, FS1, FS2) | 4 |
| Scenario 2 | Knowledge Distillation only (KD1-KD4) | 4 |
| Scenario 3 | Magnitude Pruning (FS1, FS2, KD1-KD4) | 6 |
| Scenario 4 | Pruning Methods Comparison (WANDA, Gradual, Structured) | 12 |
| **Total** | | **26** |

---

## 2. Research Background

### 2.1 Problem Statement

Cyberbullying detection in Bangla language presents unique challenges:

- **Limited Resources**: Fewer pre-trained models compared to English
- **Multi-label Nature**: Single comment can contain multiple types of harmful content
- **Deployment Constraints**: Need efficient models for real-time moderation

### 2.2 Motivation for Model Compression

Large transformer models (110M+ parameters) achieve high accuracy but face practical limitations:

| Challenge | Impact | Solution |
|-----------|--------|----------|
| High latency | Poor user experience | Smaller models via KD |
| Large memory | Expensive deployment | Pruning reduces parameters |
| Energy consumption | Environmental/cost concerns | Efficient inference |

### 2.3 Research Questions

1. **RQ1**: Can knowledge distillation effectively transfer cyberbullying detection capability from large to small models?
2. **RQ2**: Which pruning method provides the best accuracy-efficiency trade-off?
3. **RQ3**: Does combining KD with pruning outperform direct pruning of finetuned models?
4. **RQ4**: How do different teacher-student combinations affect distillation quality?

---

## 3. Dataset Description

### 3.1 Multilabel Cyberbullying Dataset

The dataset consists of Bangla social media comments annotated for multiple types of harmful content.

**Dataset Characteristics:**
- **Total Samples**: ~12,000 comments
- **Labels**: 5 binary labels (multi-label classification)
- **Source**: Bangla social media platforms

### 3.2 Label Categories

| Label | Description | Approximate Distribution |
|-------|-------------|-------------------------|
| `bully` | General bullying/harassment | ~37.5% |
| `sexual` | Sexual harassment/content | ~16.7% |
| `religious` | Religious hate speech | ~15.0% |
| `threat` | Threatening content | ~10.0% |
| `spam` | Spam/irrelevant content | ~20.8% |

### 3.3 Data Preprocessing Pipeline

```python
# data.py - Key preprocessing steps

def load_and_prepare_data(config):
    """
    1. Load CSV with comment and label columns
    2. Validate label columns exist
    3. Handle class imbalance via weighted loss
    4. Apply tokenization with caching
    """

    # Load dataset
    df = pd.read_csv(config.dataset_path)

    # Validate labels
    label_columns = ['bully', 'sexual', 'religious', 'threat', 'spam']

    # Calculate class weights for imbalanced labels
    class_weights = calculate_class_weights(labels, label_columns)
    # Formula: weight = negative_count / positive_count
```

### 3.4 Tokenization Strategy

**Dual Tokenization for KD**: When teacher and student use different tokenizers:

```python
# Tokenize for both teacher and student
tokenized_data = {
    'input_ids': teacher_tokens,           # For teacher
    'attention_mask': teacher_masks,
    'student_input_ids': student_tokens,   # For student (if different)
    'student_attention_mask': student_masks,
    'labels': labels
}
```

**Caching Optimization**: Tokenization cached to disk for efficiency
- First run: ~90 seconds
- Subsequent runs: ~2 seconds (loaded from cache)

### 3.5 K-Fold Cross-Validation

```python
# MultilabelStratifiedKFold for fair label distribution
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

splitter = MultilabelStratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42  # Reproducibility
)
```

---

## 4. Model Architecture

### 4.1 Teacher Models

Two high-capacity teacher models finetuned in Phase 1:

| ID | Model | Base | Parameters | HuggingFace Path |
|----|-------|------|------------|------------------|
| **T1** | XLM-RoBERTa | Multilingual | ~110M | `Saif-Siddique/bangla-cyberbully-xlm-roberta-base` |
| **T2** | BanglaBERT | Bangla-specific | ~110M | `Saif-Siddique/bangla-cyberbully-sagor-bangla-bert-base` |

### 4.2 Student Models

**Finetuned Students (FS)** - From Phase 2:

| ID | Model | Parameters | HuggingFace Path |
|----|-------|------------|------------------|
| **FS1** | SahajBERT (finetuned) | ~50M | `Saif-Siddique/bangla-cyberbully-neuropark-sahajBERT` |
| **FS2** | BanglaBERT-small (finetuned) | ~50M | `Saif-Siddique/bangla-cyberbully-csebuetnlp-banglabert_small` |

**Raw Students (RS)** - For KD training:

| ID | Model | Parameters | HuggingFace Path |
|----|-------|------------|------------------|
| **RS1** | SahajBERT (raw) | ~50M | `neuropark/sahajBERT` |
| **RS2** | BanglaBERT-small (raw) | ~50M | `csebuetnlp/banglabert_small` |

### 4.3 Model Architecture Details

```
┌─────────────────────────────────────────────────────────┐
│                    TeacherModel / StudentModel          │
├─────────────────────────────────────────────────────────┤
│  Input: [batch_size, max_length=128]                    │
│                         ↓                               │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Encoder (Pre-trained Transformer)               │   │
│  │  - XLM-RoBERTa / BanglaBERT (Teacher)           │   │
│  │  - SahajBERT / BanglaBERT-small (Student)       │   │
│  │  Output: [batch, 128, 768]                       │   │
│  └─────────────────────────────────────────────────┘   │
│                         ↓                               │
│  Extract [CLS] token: [batch, 768]                      │
│                         ↓                               │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Classifier Head                                 │   │
│  │  Linear(768 → 256) + ReLU + Dropout(0.1)        │   │
│  │  Linear(256 → 5)  ← 5 labels                    │   │
│  └─────────────────────────────────────────────────┘   │
│                         ↓                               │
│  Output: logits [batch, 5]                              │
└─────────────────────────────────────────────────────────┘
```

**Code Reference** (`distillation.py`, Lines 132-137):
```python
self.classifier = nn.Sequential(
    nn.Linear(self.hidden_size, classifier_hidden_size),  # 768 → 256
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(classifier_hidden_size, num_labels)  # 256 → 5
)
```

---

## 5. Knowledge Distillation Implementation

### 5.1 KD Theory

Knowledge Distillation transfers knowledge from a large "teacher" model to a smaller "student" model by training the student to mimic the teacher's predictions.

#### 5.1.1 Soft Labels vs Hard Labels

| Type | Description | Example |
|------|-------------|---------|
| **Hard Labels** | Ground truth (0 or 1) | [1, 0, 0, 1, 0] |
| **Soft Labels** | Teacher's probability predictions | [0.92, 0.15, 0.08, 0.87, 0.23] |

**Why Soft Labels?** They contain "dark knowledge":
- Relationships between classes
- Uncertainty information
- Smoother gradients for training

#### 5.1.2 Temperature Scaling

Temperature (T) controls the "softness" of probability distributions:

$$P_i = \sigma\left(\frac{z_i}{T}\right) = \frac{1}{1 + e^{-z_i/T}}$$

Where:
- $z_i$ = logit for label $i$
- $T$ = temperature parameter
- Higher $T$ → softer (more uniform) distribution

**Example with logit z = 2.0:**

| Temperature | σ(z/T) | Interpretation |
|-------------|--------|----------------|
| T = 1 | 0.88 | Confident, less informative |
| T = 2 | 0.73 | Moderately soft |
| T = 4 | 0.62 | Soft, more informative |

#### 5.1.3 Multi-Label Adaptation

**Critical Difference**: Standard KD uses softmax (multiclass), but cyberbullying detection is multi-label.

| Aspect | Multiclass (Standard KD) | Multi-label (Our Adaptation) |
|--------|--------------------------|------------------------------|
| Activation | Softmax | Sigmoid |
| Outputs | Mutually exclusive | Independent |
| Loss | KL Divergence | Binary Cross-Entropy |

### 5.2 Loss Functions

#### 5.2.1 Hard Loss (Ground Truth)

Standard binary cross-entropy with optional class weights:

$$\mathcal{L}_{hard} = -\frac{1}{N \cdot L} \sum_{i=1}^{N} \sum_{j=1}^{L} \left[ y_{ij} \log(\hat{y}_{ij}) + (1-y_{ij}) \log(1-\hat{y}_{ij}) \right]$$

**Code** (`distillation.py`, Lines 517-520):
```python
hard_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
hard_loss = hard_loss_fn(student_logits, labels)
```

#### 5.2.2 Soft Loss (Teacher Mimicking)

BCE between temperature-scaled student and teacher predictions:

$$\mathcal{L}_{soft} = -\frac{1}{N \cdot L} \sum_{i=1}^{N} \sum_{j=1}^{L} \left[ p_{ij}^T \log(p_{ij}^S) + (1-p_{ij}^T) \log(1-p_{ij}^S) \right]$$

Where:
- $p^T = \sigma(z^T / T)$ = teacher soft probabilities
- $p^S = \sigma(z^S / T)$ = student soft probabilities

**Code** (`distillation.py`, Lines 586-602):
```python
# Temperature-scaled logits
s_scaled = student_logits / self.temperature
t_scaled = teacher_logits / self.temperature

# Soft probabilities (sigmoid for multi-label)
t_probs = torch.sigmoid(t_scaled).detach()

# Soft loss
soft_loss = F.binary_cross_entropy_with_logits(
    s_scaled,
    t_probs,
    reduction='mean'
)

# T² scaling correction (Hinton et al.)
soft_loss = soft_loss * (self.temperature ** 2)
```

#### 5.2.3 T² Scaling Correction

When using temperature scaling, gradients are reduced by factor T. To compensate:

$$\mathcal{L}_{soft}^{corrected} = T^2 \cdot \mathcal{L}_{soft}$$

**Derivation**:
$$\frac{\partial \mathcal{L}_{soft}}{\partial z} = \frac{1}{T} \cdot (\text{gradient without T})$$

Multiplying by $T^2$ ensures comparable gradient magnitudes.

#### 5.2.4 Combined Loss

$$\mathcal{L}_{total} = \alpha \cdot \mathcal{L}_{soft} + (1-\alpha) \cdot \mathcal{L}_{hard}$$

Where $\alpha$ controls the balance (default: 0.7 = 70% soft, 30% hard).

**Code** (`distillation.py`, Line 525):
```python
logit_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
```

### 5.3 KD Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `kd_alpha` | 0.7 | Soft/hard loss weight (0.7 = 70% soft) |
| `kd_temperature` | 4.0 | Temperature for softening |
| `kd_method` | 'logit' | Distillation method |

**KD Methods Available:**

| Method | Description | Speed |
|--------|-------------|-------|
| `logit` | Match final predictions only | Fastest |
| `hidden` | Also match hidden layer representations | Medium |
| `attention` | Match attention patterns | Medium |
| `multi_level` | Combine all methods | Slowest, best accuracy |

### 5.4 Critical Decision: Student-Matched Hyperparameters

**Rationale**: Since KD trains the same student architecture as Phase 2, we use the optimal hyperparameters discovered during Phase 2 finetuning.

| KD Model | Student Base | Hyperparameters | Source |
|----------|--------------|-----------------|--------|
| **KD1** | SahajBERT | batch=32, lr=2e-5, epochs=20 | FS1 best config |
| **KD2** | BanglaBERT-small | batch=16, lr=3e-5, epochs=30 | FS2 best config |
| **KD3** | SahajBERT | batch=32, lr=2e-5, epochs=20 | FS1 best config |
| **KD4** | BanglaBERT-small | batch=16, lr=3e-5, epochs=30 | FS2 best config |

---

## 6. Pruning Implementation

### 6.1 Pruning Overview

Pruning removes unnecessary weights from neural networks to reduce size and increase speed.

**Types of Pruning:**

| Type | What's Removed | Speedup | Compression |
|------|----------------|---------|-------------|
| **Unstructured** | Individual weights | Requires sparse libraries | High (50%+) |
| **Structured** | Entire neurons/heads | Works on any hardware | Medium (20-40%) |

### 6.2 Magnitude Pruning

**Algorithm**: Remove weights with smallest absolute values.

$$\text{mask}_i = \begin{cases} 1 & \text{if } |w_i| \geq \tau \\ 0 & \text{if } |w_i| < \tau \end{cases}$$

Where $\tau$ is the threshold at the target sparsity percentile.

**Code** (`pruning.py`, Lines 182-215):
```python
def apply_magnitude_pruning(self, sparsity=None):
    """
    Apply one-shot magnitude-based pruning.

    Example:
        Weights: [0.8, -0.05, 0.3, 0.01, -0.6]
        After 40% pruning: [0.8, 0, 0.3, 0, -0.6]
    """
    sparsity = sparsity or self.target_sparsity

    if self.global_pruning:
        # Same threshold across all layers (recommended)
        prune.global_unstructured(
            self.prunable_modules,
            pruning_method=prune.L1Unstructured,
            amount=sparsity
        )
    else:
        # Per-layer independent pruning
        for module, param_name in self.prunable_modules:
            prune.l1_unstructured(module, param_name, amount=sparsity)
```

### 6.3 WANDA Pruning (Weights AND Activations)

**Innovation**: Considers both weight magnitude AND activation magnitude.

$$\text{importance}_i = |w_i| \times \mathbb{E}[|a_i|]$$

Where:
- $w_i$ = weight value
- $a_i$ = input activation to that weight
- $\mathbb{E}[\cdot]$ = mean over calibration samples

**Why Better?**
- Small weight × high activation = potentially important
- Large weight × low activation = potentially unimportant

**Algorithm:**
1. Run calibration data through model
2. Collect activation statistics via forward hooks
3. Compute importance = weight × activation
4. Prune lowest importance weights

**Code** (`pruning.py`, Lines 527-592):
```python
def apply_wanda_pruning(self):
    """
    WANDA: Prune based on weight × activation importance.

    Must call collect_activations() first!
    """
    # Compute importance for each weight
    for name, module in self.model.named_modules():
        weight = module.weight.data.abs()  # [out, in]
        act_norm = self.activation_norms[name]  # [in]

        # importance = |weight| × mean(|activation|)
        importance = weight * act_norm.unsqueeze(0)
        all_importance_scores.append(importance.flatten())

    # Global threshold
    all_scores = torch.cat(all_importance_scores)
    threshold = torch.quantile(all_scores, self.target_sparsity)

    # Create mask
    mask = (importance >= threshold).float()
    prune.custom_from_mask(module, 'weight', mask)
```

### 6.4 Gradual Pruning

**Innovation**: Slowly increase sparsity during training, allowing model to adapt.

#### 6.4.1 Sparsity Schedules

**Linear Schedule:**
$$s(t) = s_{target} \cdot \frac{t - t_{start}}{t_{end} - t_{start}}$$

**Cubic Schedule (Recommended):**
$$s(t) = s_{target} \cdot (3p^2 - 2p^3)$$

Where $p = \frac{t - t_{start}}{t_{end} - t_{start}}$ (progress)

**Exponential Schedule:**
$$s(t) = s_{target} \cdot (1 - e^{-5p})$$

#### 6.4.2 Incremental Pruning Fix

**Problem**: Naive cumulative pruning doesn't achieve target sparsity.
- Apply 10% pruning 5 times: $(1-0.1)^5 = 0.59$ remaining = 41% sparsity (not 50%!)

**Solution**: Calculate incremental amount of **remaining** weights.

$$\text{incremental} = \frac{s_{target} - s_{current}}{1 - s_{current}}$$

**Code** (`pruning.py`, Lines 375-400):
```python
def step(self, current_step, total_steps):
    """Perform one pruning step if needed."""
    target = self._get_target_sparsity_at_step(current_step, total_steps)

    # KEY FIX: Calculate incremental pruning amount
    # If at 20% and want 30%, prune (30-20)/(100-20) = 12.5% of REMAINING
    incremental = (target - self.current_sparsity) / (1 - self.current_sparsity)
    incremental = min(incremental, 0.99)  # Safety cap

    # Apply incremental pruning
    prune.global_unstructured(
        self.prunable_modules,
        pruning_method=prune.L1Unstructured,
        amount=incremental
    )
```

### 6.5 Structured Pruning

**Innovation**: Remove entire attention heads, achieving **real parameter reduction**.

#### 6.5.1 Unstructured vs Structured

| Aspect | Unstructured | Structured |
|--------|--------------|------------|
| What's removed | Individual weights | Entire heads |
| Result | Zeros in weight matrix | Smaller matrix |
| Speedup | Requires sparse math | Works on any hardware |
| Compression | Higher (50%+) | Lower (20-40%) |
| Parameters | Same count (zeros) | Actually reduced |

#### 6.5.2 Head Importance Scoring

$$\text{importance}_{head} = \mathbb{E}\left[\left|\text{output}_{head}\right|\right]$$

Heads producing larger output magnitudes are more important.

**Code** (`pruning.py`, Lines 715-732):
```python
def make_attention_hook(layer_idx):
    def hook(module, input, output):
        attn_output = output[0]  # [batch, seq, hidden]

        # Reshape to per-head
        batch_size, seq_len, hidden = attn_output.shape
        head_dim = hidden // self.num_heads
        attn_output = attn_output.view(batch_size, seq_len,
                                       self.num_heads, head_dim)

        # Importance = mean magnitude per head
        head_magnitude = attn_output.abs().mean(dim=(0,1,3))
        attention_outputs[layer_idx].append(head_magnitude)
    return hook
```

#### 6.5.3 Head Selection and Pruning

```python
# Sort heads by importance (ascending)
all_scores.sort(key=lambda x: x[0])

# Prune least important, but keep ≥1 per layer
for score, layer_idx, head_idx in all_scores[:num_to_prune]:
    if heads_pruned_per_layer[layer_idx] < self.num_heads - 1:
        heads_to_prune[layer_idx].append(head_idx)

# Use HuggingFace's built-in pruning
layer.attention.prune_heads(set(head_indices))
```

### 6.6 Fine-tuning After Pruning

**Why Required**: Pruning removes knowledge, causing accuracy drop. Fine-tuning recovers it.

**Configuration:**
- Learning rate: 10× lower than initial training (`config.lr * 0.1`)
- Epochs: 3 (default)
- Model selection: F1 Weighted (primary metric)

**Code** (`pruning.py`, Lines 1003-1014):
```python
# Lower learning rate for fine-tuning
optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=config.lr * 0.1,  # 10x LOWER
    weight_decay=config.weight_decay
)
```

**Typical Recovery:**
```
Before pruning:  F1 = 0.75
After pruning:   F1 = 0.60 (drop 15%)
After fine-tune: F1 = 0.72 (recover 12%, net loss 3%)
```

---

## 7. Experiment Design

### 7.1 Experiment Scenarios

#### Scenario 1: Baselines (4 runs)
Evaluate pre-trained models without compression.

| Run | Model | eval_fold | Purpose |
|-----|-------|-----------|---------|
| 1.1 | T1 (XLM-RoBERTa) | 3 | Large teacher baseline |
| 1.2 | T2 (BanglaBERT) | 2 | Large teacher baseline |
| 1.3 | FS1 (SahajBERT-ft) | 2 | Small finetuned baseline |
| 1.4 | FS2 (BanglaBERT-small-ft) | 5 | Small finetuned baseline |

#### Scenario 2: Knowledge Distillation Only (4 runs)
Train raw students with teacher guidance.

| Run | Teacher | Student | eval_fold | Hyperparameters |
|-----|---------|---------|-----------|-----------------|
| 2.1 (KD1) | T1 | RS1 (SahajBERT) | 3 | batch=32, lr=2e-5, epochs=20 |
| 2.2 (KD2) | T1 | RS2 (BanglaBERT-small) | 3 | batch=16, lr=3e-5, epochs=30 |
| 2.3 (KD3) | T2 | RS1 (SahajBERT) | 2 | batch=32, lr=2e-5, epochs=20 |
| 2.4 (KD4) | T2 | RS2 (BanglaBERT-small) | 2 | batch=16, lr=3e-5, epochs=30 |

#### Scenario 3: Magnitude Pruning (6 runs)
Apply magnitude pruning with 50% sparsity.

| Run | Model | eval_fold | Description |
|-----|-------|-----------|-------------|
| 3.1 | FS1 | 2 | Prune finetuned SahajBERT |
| 3.2 | FS2 | 5 | Prune finetuned BanglaBERT-small |
| 3.3 | KD1 | 3 | Prune KD SahajBERT (from T1) |
| 3.4 | KD2 | 3 | Prune KD BanglaBERT-small (from T1) |
| 3.5 | KD3 | 2 | Prune KD SahajBERT (from T2) |
| 3.6 | KD4 | 2 | Prune KD BanglaBERT-small (from T2) |

#### Scenario 4: Pruning Methods Comparison (12 runs)
Compare WANDA, Gradual, and Structured pruning on KD models.

| Method | KD1 (fold=3) | KD2 (fold=3) | KD3 (fold=2) | KD4 (fold=2) |
|--------|--------------|--------------|--------------|--------------|
| WANDA | 4.1 | 4.2 | 4.3 | 4.4 |
| Gradual | 4.5 | 4.6 | 4.7 | 4.8 |
| Structured | 4.9 | 4.10 | 4.11 | 4.12 |

### 7.2 Critical Decision: eval_fold Assignment

**Problem**: Different models were trained with different validation folds in Phase 1 & 2. Using inconsistent folds would make comparison unfair.

**Solution**: Use each model's original eval_fold.

| Model | eval_fold | Reason |
|-------|-----------|--------|
| T1 | 3 | Original from Phase 1 |
| T2 | 2 | Original from Phase 1 |
| FS1 | 2 | Original from Phase 2 |
| FS2 | 5 | Original from Phase 2 |
| KD1, KD2 | 3 | Match teacher T1 |
| KD3, KD4 | 2 | Match teacher T2 |

**Rationale for KD eval_fold**: KD models should be compared against their teacher, so they use the teacher's fold.

### 7.3 Fair Evaluation Strategy

```python
# Recreate original fold splits
parser.add_argument('--use_original_folds', action='store_true')
parser.add_argument('--eval_fold', type=int, default=3, choices=[1,2,3,4,5])

# Using MultilabelStratifiedKFold with fixed seed
splitter = MultilabelStratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42  # Same as Phase 1 & 2
)
```

---

## 8. Evaluation Metrics

### 8.1 Primary Metric: F1 Weighted

**Critical Decision**: Changed from F1 Macro to F1 Weighted for consistency with Phase 1 & 2.

$$F1_{weighted} = \sum_{c=1}^{C} \frac{n_c}{N} \cdot F1_c$$

Where:
- $n_c$ = number of samples with label $c$
- $N$ = total samples
- $F1_c$ = F1 score for label $c$

**Code Change** (`pruning.py`, Lines 1107-1109):
```python
# Track best model using F1 weighted (primary metric)
if f1_weighted > best_f1:
    best_f1 = f1_weighted
    best_epoch = epoch + 1
```

### 8.2 Secondary Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| F1 Macro | $\frac{1}{C}\sum_{c=1}^{C} F1_c$ | Fair to all labels |
| F1 Micro | Global precision/recall | Overall performance |
| Accuracy (Exact) | All 5 labels correct | Strictest measure |
| Hamming Loss | Fraction of wrong labels | Multi-label specific |
| Per-label F1 | Individual label F1 | Identify weak spots |

### 8.3 Efficiency Metrics

| Metric | Description | Unit |
|--------|-------------|------|
| Model Size | Total parameter bytes | MB |
| Sparsity | Percentage of zero weights | % |
| Compression Ratio | Baseline/Compressed size | × |
| Latency | Inference time per sample | ms |
| Throughput | Samples processed per second | samples/sec |

---

## 9. Implementation Details

### 9.1 Codebase Structure

```
phase_3_final/
├── main.py                 # Main execution pipeline
├── distillation.py         # KD implementation
├── pruning.py              # Pruning methods
├── evaluation.py           # Metrics computation
├── data.py                 # Data loading/preprocessing
├── compression_config.py   # Configuration & arguments
├── kaggle_notebooks/       # Experiment notebooks
│   ├── phase_a_baselines.ipynb
│   ├── phase_b_kd_only.ipynb
│   └── phase_cd_kd_prune.ipynb
└── PHASE_3_RESEARCH_DOCUMENTATION.md
```

### 9.2 Key Functions

| File | Function | Purpose |
|------|----------|---------|
| `main.py` | `run_knowledge_distillation()` | Execute KD training |
| `main.py` | `run_pruning()` | Apply pruning method |
| `distillation.py` | `MultiLabelDistillationLoss` | KD loss computation |
| `pruning.py` | `fine_tune_after_pruning()` | Post-pruning recovery |
| `evaluation.py` | `CompressionEvaluator` | Compute all metrics |

### 9.3 Default Hyperparameters

| Category | Parameter | Default |
|----------|-----------|---------|
| **Training** | batch_size | 32 |
| | learning_rate | 2e-5 |
| | max_length | 128 |
| | dropout | 0.1 |
| | weight_decay | 0.01 |
| **KD** | alpha | 0.7 |
| | temperature | 4.0 |
| | epochs | 15 |
| **Pruning** | sparsity | 0.5 (50%) |
| | fine_tune_epochs | 3 |
| | fine_tune_lr | lr × 0.1 |

### 9.4 Hardware Requirements

- **GPU**: NVIDIA T4/P100 (Kaggle free tier)
- **Memory**: 16GB GPU RAM
- **Storage**: ~10GB for models and cache
- **Runtime**: 3-10 hours per notebook

---

## 10. Critical Research Decisions

### Summary Table

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| 1 | Primary metric | F1 Weighted | Consistency with Phase 1 & 2 |
| 2 | Model selection criterion | F1 Weighted | Same as above |
| 3 | eval_fold for T1 | 3 | Original from Phase 1 |
| 4 | eval_fold for T2 | 2 | Original from Phase 1 |
| 5 | eval_fold for FS1 | 2 | Original from Phase 2 |
| 6 | eval_fold for FS2 | 5 | Original from Phase 2 |
| 7 | eval_fold for KD1, KD2 | 3 | Match teacher T1 |
| 8 | eval_fold for KD3, KD4 | 2 | Match teacher T2 |
| 9 | KD hyperparameters | Student-matched | Use FS1/FS2 optimal configs |
| 10 | Fine-tune after prune | Always enabled | Essential for accuracy recovery |
| 11 | Fine-tune learning rate | 10× lower | Standard practice |
| 12 | Pruning target sparsity | 50% | Balance compression/accuracy |
| 13 | Structured sparsity | 30% | More conservative for head pruning |

### Detailed Rationale

#### Decision 1-2: F1 Weighted as Primary Metric

**Previous state**: Code used F1 Macro for model selection in `pruning.py`.

**Problem**: Phase 1 & 2 used F1 Weighted, creating inconsistency.

**Solution**: Modified `pruning.py` (Lines 1107-1109) to use F1 Weighted.

```python
# Before (f1_macro)
if f1_macro > best_f1:
    best_f1 = f1_macro

# After (f1_weighted)
if f1_weighted > best_f1:
    best_f1 = f1_weighted
```

#### Decision 7-8: KD Model eval_fold Matches Teacher

**Options considered**:
- Option A: Match teacher's fold (chosen)
- Option B: Use single consistent fold
- Option C: Match student architecture fold

**Rationale for Option A**: KD models should be compared against their teacher to evaluate knowledge transfer quality. Using teacher's fold ensures fair comparison.

#### Decision 9: Student-Matched Hyperparameters

**Problem**: KD trains the same student architecture as Phase 2. Should we use default hyperparameters or Phase 2 optimal configs?

**Solution**: Use Phase 2 optimal hyperparameters since:
1. Same model architecture
2. Similar training dynamics
3. Already validated as optimal for this task

---

## 11. Experiment Execution Plan

### 11.1 Notebook Organization

| Notebook | Scenarios | Runs | Estimated Time |
|----------|-----------|------|----------------|
| `phase_a_baselines.ipynb` | 1, 3.1-3.2 | 6 | 3-4 hours |
| `phase_b_kd_only.ipynb` | 2 | 4 | 8-10 hours |
| `phase_cd_kd_prune.ipynb` | 3.3-3.6, 4 | 16 | 6-8 hours |

### 11.2 Execution Dependencies

```
Phase A (Baselines)     Phase B (KD)
       │                    │
       │                    ▼
       │            Upload KD models
       │            to HuggingFace
       │                    │
       └───────┬────────────┘
               │
               ▼
        Phase C+D (Prune KD)
```

**Critical**: Phase B must complete and upload KD models before Phase C+D can run.

### 11.3 Repository and Data Configuration

**Git Repository**:
```
https://github.com/SaifSiddique009/kd_pruning_quantization_framework_for_nlp.git
```

**Data Path** (in Kaggle after git clone):
```
/kaggle/working/kd_pruning_quantization_framework_for_nlp/data/1_Multilablel_Cyberbully_Data.csv
```

---

## 12. Results (Placeholder)

### 12.1 Baseline Performance

| Model | F1 Weighted | F1 Macro | Parameters | Size (MB) |
|-------|-------------|----------|------------|-----------|
| T1 (XLM-RoBERTa) | TBD | TBD | ~110M | TBD |
| T2 (BanglaBERT) | TBD | TBD | ~110M | TBD |
| FS1 (SahajBERT-ft) | TBD | TBD | ~50M | TBD |
| FS2 (BanglaBERT-small-ft) | TBD | TBD | ~50M | TBD |

### 12.2 Knowledge Distillation Results

| KD Model | Teacher | Student | F1 Weighted | vs Teacher | vs FS |
|----------|---------|---------|-------------|------------|-------|
| KD1 | T1 | RS1 | TBD | TBD | TBD |
| KD2 | T1 | RS2 | TBD | TBD | TBD |
| KD3 | T2 | RS1 | TBD | TBD | TBD |
| KD4 | T2 | RS2 | TBD | TBD | TBD |

### 12.3 Pruning Methods Comparison

| Model | Baseline | Magnitude | WANDA | Gradual | Structured |
|-------|----------|-----------|-------|---------|------------|
| KD1 | TBD | TBD | TBD | TBD | TBD |
| KD2 | TBD | TBD | TBD | TBD | TBD |
| KD3 | TBD | TBD | TBD | TBD | TBD |
| KD4 | TBD | TBD | TBD | TBD | TBD |

### 12.4 Best Configuration

| Aspect | Best Choice | Performance |
|--------|-------------|-------------|
| Best Teacher | TBD | TBD |
| Best KD Combo | TBD | TBD |
| Best Pruning Method | TBD | TBD |
| Best Overall | TBD | TBD |

---

## 13. Appendices

### Appendix A: Complete Hyperparameter Reference

```python
# compression_config.py - All configurable parameters

# Dataset
--dataset_path          # Path to CSV file
--text_column           # Column name for text (default: 'comment')
--label_columns         # Label column names

# Models
--teacher_path          # Pre-trained teacher model path
--teacher_checkpoint    # Finetuned teacher checkpoint (HuggingFace)
--student_path          # Pre-trained student model path

# Training
--batch                 # Batch size (default: 32)
--lr                    # Learning rate (default: 2e-5)
--epochs                # Training epochs (default: 15)
--max_length            # Max sequence length (default: 128)
--dropout               # Dropout rate (default: 0.1)
--weight_decay          # L2 regularization (default: 0.01)
--warmup_ratio          # LR warmup ratio (default: 0.1)
--gradient_clip_norm    # Gradient clipping (default: 1.0)

# Knowledge Distillation
--kd_alpha              # Soft/hard loss balance (default: 0.7)
--kd_temperature        # Temperature scaling (default: 4.0)
--kd_method             # logit/hidden/attention/multi_level

# Pruning
--prune_method          # magnitude/wanda/gradual/structured
--prune_sparsity        # Target sparsity (default: 0.5)
--prune_schedule        # linear/cubic/exponential
--fine_tune_epochs      # Post-pruning epochs (default: 3)

# Evaluation
--use_original_folds    # Use original K-fold splits
--eval_fold             # Which fold for validation (1-5)
```

### Appendix B: Model HuggingFace Paths

**Teacher Models (Finetuned)**:
- T1: `Saif-Siddique/bangla-cyberbully-xlm-roberta-base`
- T2: `Saif-Siddique/bangla-cyberbully-sagor-bangla-bert-base`

**Finetuned Student Models**:
- FS1: `Saif-Siddique/bangla-cyberbully-neuropark-sahajBERT`
- FS2: `Saif-Siddique/bangla-cyberbully-csebuetnlp-banglabert_small`

**Raw Student Models** (for KD):
- RS1: `neuropark/sahajBERT`
- RS2: `csebuetnlp/banglabert_small`

**KD Models** (after Phase B upload):
- KD1: `Saif-Siddique/bangla-cyberbully-kd1-xlmroberta-to-sahajbert`
- KD2: `Saif-Siddique/bangla-cyberbully-kd2-xlmroberta-to-banglabert-small`
- KD3: `Saif-Siddique/bangla-cyberbully-kd3-banglabert-to-sahajbert`
- KD4: `Saif-Siddique/bangla-cyberbully-kd4-banglabert-to-banglabert-small`

### Appendix C: Key Code Snippets

#### C.1 Multi-Label Distillation Loss

```python
# distillation.py, Lines 563-604
class MultiLabelDistillationLoss(nn.Module):
    def forward(self, student_outputs, teacher_outputs, labels, pos_weight=None):
        # Temperature scaling
        s_scaled = student_logits / self.temperature
        t_scaled = teacher_logits / self.temperature

        # Soft probabilities (sigmoid for multi-label!)
        t_probs = torch.sigmoid(t_scaled).detach()

        # Soft loss with T² correction
        soft_loss = F.binary_cross_entropy_with_logits(s_scaled, t_probs)
        soft_loss = soft_loss * (self.temperature ** 2)

        # Hard loss
        hard_loss = F.binary_cross_entropy_with_logits(s_logits, labels)

        # Combined
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss
```

#### C.2 Gradual Pruning with Incremental Fix

```python
# pruning.py, Lines 375-400
def step(self, current_step, total_steps):
    target = self._get_target_sparsity_at_step(current_step, total_steps)

    # KEY FIX: Incremental pruning calculation
    incremental = (target - self.current_sparsity) / (1 - self.current_sparsity)

    prune.global_unstructured(
        self.prunable_modules,
        pruning_method=prune.L1Unstructured,
        amount=incremental
    )
```

#### C.3 Fine-tuning with F1 Weighted Selection

```python
# pruning.py, Lines 1085-1119
# Calculate metrics
f1_macro = f1_score(all_labels, all_preds, average='macro')
f1_weighted = f1_score(all_labels, all_preds, average='weighted')

# Track best model (F1 Weighted as primary)
if f1_weighted > best_f1:
    best_f1 = f1_weighted
    best_epoch = epoch + 1
    best_metrics = {
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        ...
    }
```

---

## References

1. Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. *arXiv preprint arXiv:1503.02531*.

2. Sun, M., et al. (2023). A simple and effective pruning approach for large language models. *arXiv preprint arXiv:2306.11695*. (WANDA)

3. Zhu, M., & Gupta, S. (2017). To prune, or not to prune: exploring the efficacy of pruning for model compression. *arXiv preprint arXiv:1710.01878*. (Gradual Pruning)

4. Michel, P., Levy, O., & Neubig, G. (2019). Are sixteen heads really better than one? *NeurIPS*. (Structured Pruning)

---

*Document generated for Phase 3 research on Knowledge Distillation and Pruning for Bangla Cyberbullying Detection.*
