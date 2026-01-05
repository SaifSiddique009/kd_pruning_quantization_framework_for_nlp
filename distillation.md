# 📘 Script 3: `research_distillation.py`

## Overview

This script implements **Knowledge Distillation (KD)** - the technique of training a smaller "student" model to mimic a larger "teacher" model. This is often the first and most impactful compression technique.

**Why KD is powerful:**
- Can reduce model size by 40-60% with only 2-5% accuracy loss
- Student learns "dark knowledge" - relationships between classes that labels don't capture
- Works as foundation for other compression techniques (pruning, quantization)

---

## Section 1: Imports and Setup

```python
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoModel, AutoConfig
import numpy as np
from typing import Dict, Optional, Tuple, List
from tqdm import tqdm
```

### Key Imports Explained:

| Import | Purpose |
|--------|---------|
| `torch.nn.functional as F` | Contains `softmax`, `sigmoid`, `kl_div` for loss computation |
| `autocast`, `GradScaler` | Automatic Mixed Precision (AMP) for faster training |
| `AutoModel`, `AutoConfig` | Load any HuggingFace transformer |

### Why AMP (Automatic Mixed Precision)?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ AUTOMATIC MIXED PRECISION (AMP)                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ Normal Training (FP32):                                                     │
│ ───────────────────────                                                     │
│   Forward pass:  FP32 (32-bit floats)                                      │
│   Backward pass: FP32                                                       │
│   Memory usage:  100%                                                       │
│   Speed:         1.0×                                                       │
│                                                                             │
│ Mixed Precision Training (AMP):                                            │
│ ────────────────────────────────                                            │
│   Forward pass:  FP16 (16-bit) where safe                                  │
│   Backward pass: FP32 for gradient accumulation                            │
│   Memory usage:  ~60-70%                                                   │
│   Speed:         1.3-2.0× faster!                                          │
│                                                                             │
│ How it works:                                                               │
│                                                                             │
│   with autocast():           # Automatically chooses FP16 or FP32          │
│       outputs = model(x)     # Forward in FP16 (fast)                      │
│       loss = criterion(...)  # Loss in FP32 (accurate)                     │
│                                                                             │
│   scaler.scale(loss).backward()  # Scale gradients to prevent underflow   │
│   scaler.step(optimizer)          # Unscale and apply                      │
│   scaler.update()                 # Adjust scale factor                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Section 2: Teacher Model Class

```python
class TeacherModel(nn.Module):
    """
    The large, accurate model that will teach the student.
    
    Architecture:
        Input → Transformer Encoder → [CLS] token → Classifier → Output
    """
    
    def __init__(
        self,
        model_name: str = 'csebuetnlp/banglabert',
        num_labels: int = 5,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Load pre-trained transformer
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Classification head
        hidden_size = self.config.hidden_size  # 768 for BERT-base
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels)
        )
        
        self.num_labels = num_labels
```

### Architecture Visualization:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ TEACHER MODEL ARCHITECTURE                                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Input: "তুমি বোকা" (You're stupid)                                        │
│          ↓                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ TOKENIZER                                                           │  │
│   │   → [CLS] তুমি বোকা [SEP] [PAD] [PAD] ...                          │  │
│   │   → input_ids: [101, 2345, 6789, 102, 0, 0, ...]                   │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│          ↓                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ TRANSFORMER ENCODER (BanglaBERT)                                    │  │
│   │                                                                     │  │
│   │   Embedding Layer                                                   │  │
│   │        ↓                                                            │  │
│   │   Transformer Block 1 (Self-Attention + FFN)                       │  │
│   │        ↓                                                            │  │
│   │   Transformer Block 2                                               │  │
│   │        ↓                                                            │  │
│   │   ... (12 layers total for BERT-base)                              │  │
│   │        ↓                                                            │  │
│   │   Transformer Block 12                                              │  │
│   │        ↓                                                            │  │
│   │   Output: (batch_size, seq_len, 768)                               │  │
│   │                                                                     │  │
│   │   Extract [CLS] token: (batch_size, 768)                           │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│          ↓                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ CLASSIFIER HEAD                                                     │  │
│   │                                                                     │  │
│   │   Linear(768 → 768) → ReLU → Dropout(0.1) → Linear(768 → 5)       │  │
│   │                                                                     │  │
│   │   Output logits: [1.2, -0.5, 0.1, 0.8, -1.0]                       │  │
│   │                                                                     │  │
│   │   After sigmoid: [0.77, 0.38, 0.52, 0.69, 0.27]                    │  │
│   │   Predictions:   [bully=1, sexual=0, religious=1, threat=1, spam=0]│  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Forward Method:

```python
def forward(
    self, 
    input_ids: torch.Tensor, 
    attention_mask: torch.Tensor,
    output_hidden_states: bool = False,
    output_attentions: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Forward pass through teacher model.
    
    Returns dict with:
        - logits: Final predictions (always)
        - hidden_states: Intermediate layers (if requested)
        - attentions: Attention weights (if requested)
    """
    # Get encoder outputs
    encoder_outputs = self.encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=output_hidden_states,
        output_attentions=output_attentions
    )
    
    # Extract [CLS] token representation (first token)
    cls_output = encoder_outputs.last_hidden_state[:, 0, :]
    
    # Get logits from classifier
    logits = self.classifier(cls_output)
    
    # Build output dictionary
    outputs = {'logits': logits}
    
    if output_hidden_states:
        outputs['hidden_states'] = encoder_outputs.hidden_states
    
    if output_attentions:
        outputs['attentions'] = encoder_outputs.attentions
    
    return outputs
```

### Why Return a Dictionary?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ WHY DICTIONARY OUTPUT?                                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ Different KD methods need different outputs:                               │
│                                                                             │
│   Method          │ Needs                                                  │
│   ────────────────┼─────────────────────────────────────────────────────   │
│   logit           │ logits only                                            │
│   hidden          │ logits + hidden_states                                 │
│   attention       │ logits + attentions                                    │
│   multi_level     │ logits + hidden_states + attentions                    │
│                                                                             │
│ Dictionary allows flexible access:                                          │
│                                                                             │
│   outputs = teacher(input_ids, attention_mask,                             │
│                     output_hidden_states=True,                             │
│                     output_attentions=True)                                │
│                                                                             │
│   logits = outputs['logits']                    # Always available         │
│   hidden = outputs.get('hidden_states', None)   # None if not requested   │
│   attn = outputs.get('attentions', None)        # None if not requested   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### What You Can Modify:

| Modification | Code Change | Effect |
|--------------|-------------|--------|
| Larger classifier | Add more layers | More capacity, slower |
| Different pooling | Use mean instead of [CLS] | Sometimes better for long texts |
| Freeze encoder | `self.encoder.requires_grad_(False)` | Faster training, less adaptable |

**Example - Mean Pooling Instead of [CLS]:**
```python
def forward(self, input_ids, attention_mask, ...):
    encoder_outputs = self.encoder(input_ids, attention_mask)
    
    # Instead of [CLS] token:
    # cls_output = encoder_outputs.last_hidden_state[:, 0, :]
    
    # Use mean pooling:
    hidden_states = encoder_outputs.last_hidden_state
    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
    sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
    sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
    mean_output = sum_embeddings / sum_mask
    
    logits = self.classifier(mean_output)
    return {'logits': logits}
```

---

## Section 3: Student Model Class

```python
class StudentModel(nn.Module):
    """
    The smaller, faster model that learns from the teacher.
    
    Key differences from teacher:
        - Smaller base model (DistilBERT: 66M vs BERT: 110M)
        - Optionally smaller classifier
    """
    
    def __init__(
        self,
        model_name: str = 'distilbert-base-multilingual-cased',
        num_labels: int = 5,
        dropout: float = 0.1,
        classifier_hidden_size: int = 256  # Smaller than teacher's 768!
    ):
        super().__init__()
        
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        
        hidden_size = self.config.hidden_size  # 768 for DistilBERT
        
        # Smaller classifier than teacher
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, classifier_hidden_size),  # 768 → 256
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_size, num_labels)    # 256 → 5
        )
```

### Teacher vs Student Comparison:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ TEACHER VS STUDENT ARCHITECTURE                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   TEACHER (BanglaBERT)              STUDENT (DistilBERT)                   │
│   ─────────────────────             ──────────────────────                  │
│                                                                             │
│   Parameters: 110M                  Parameters: 66M                         │
│   Size: 420 MB                      Size: 250 MB                            │
│   Layers: 12                        Layers: 6                               │
│   Hidden: 768                       Hidden: 768                             │
│   Attention heads: 12               Attention heads: 12                     │
│                                                                             │
│   Classifier:                       Classifier:                             │
│   768 → 768 → 5                     768 → 256 → 5                          │
│   (590K params)                     (200K params)                           │
│                                                                             │
│   Speed: 1.0×                       Speed: 1.6×                             │
│   Accuracy: 72% F1                  Accuracy: 68-70% F1                     │
│                                                                             │
│   ┌─────────────────┐               ┌─────────────────┐                    │
│   │ Layer 12        │               │                 │                    │
│   │ Layer 11        │               │                 │                    │
│   │ Layer 10        │               │                 │                    │
│   │ Layer 9         │               │                 │                    │
│   │ Layer 8         │               │                 │                    │
│   │ Layer 7         │               │ Layer 6         │                    │
│   │ Layer 6         │    ──────▶    │ Layer 5         │                    │
│   │ Layer 5         │   Distill     │ Layer 4         │                    │
│   │ Layer 4         │               │ Layer 3         │                    │
│   │ Layer 3         │               │ Layer 2         │                    │
│   │ Layer 2         │               │ Layer 1         │                    │
│   │ Layer 1         │               │                 │                    │
│   └─────────────────┘               └─────────────────┘                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### What You Can Modify:

| Modification | Effect | When to Use |
|--------------|--------|-------------|
| `model_name='bert-base-multilingual-cased'` | Larger student | When accuracy is critical |
| `model_name='distilroberta-base'` | Different architecture | Experiment with alternatives |
| `classifier_hidden_size=128` | Smaller classifier | Maximum compression |
| `classifier_hidden_size=512` | Larger classifier | Better accuracy |

**Example - Trying Different Student Models:**
```bash
# Smallest student (maximum compression)
python research_main.py --student_path "google/mobilebert-uncased" \
    --student_hidden_size 128

# Medium student (balanced)
python research_main.py --student_path "distilbert-base-multilingual-cased" \
    --student_hidden_size 256

# Large student (maximum accuracy)
python research_main.py --student_path "bert-base-multilingual-cased" \
    --student_hidden_size 512
```

---

## Section 4: Multi-Label Distillation Loss (THE MATHEMATICAL CORE)

This is the most important and most complex part of the script. Understanding this deeply will help you understand all of KD.

```python
class MultiLabelDistillationLoss(nn.Module):
    """
    Knowledge Distillation loss for multi-label classification.
    
    CRITICAL: This is DIFFERENT from single-label KD!
    
    Single-label KD uses:
        KL divergence on softmax outputs
        
    Multi-label KD uses:
        BCE on sigmoid outputs (per-label)
    
    Why? Because in multi-label, each label is independent!
    """
    
    def __init__(
        self,
        alpha: float = 0.7,
        temperature: float = 4.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.reduction = reduction
```

### Understanding Temperature in KD:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ TEMPERATURE SCALING EXPLAINED                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ Teacher's raw logits: [2.0, 0.5, -1.0, 1.5, -2.0]                          │
│                                                                             │
│ After sigmoid (T=1, normal inference):                                     │
│   [0.88, 0.62, 0.27, 0.82, 0.12]                                           │
│   └─ Very confident predictions                                            │
│                                                                             │
│ Divide by temperature, then sigmoid:                                       │
│                                                                             │
│   T = 1:   [2.0, 0.5, -1.0, 1.5, -2.0] / 1 = [2.0, 0.5, -1.0, 1.5, -2.0]  │
│            sigmoid → [0.88, 0.62, 0.27, 0.82, 0.12]                        │
│            Very sharp (hard to learn from)                                 │
│                                                                             │
│   T = 4:   [2.0, 0.5, -1.0, 1.5, -2.0] / 4 = [0.5, 0.125, -0.25, 0.375, -0.5] │
│            sigmoid → [0.62, 0.53, 0.44, 0.59, 0.38]                        │
│            Softer (reveals inter-label relationships)                      │
│                                                                             │
│   T = 10:  [2.0, 0.5, -1.0, 1.5, -2.0] / 10 = [0.2, 0.05, -0.1, 0.15, -0.2]│
│            sigmoid → [0.55, 0.51, 0.48, 0.54, 0.45]                        │
│            Too soft (almost uniform, loses information)                    │
│                                                                             │
│ Visual representation:                                                      │
│                                                                             │
│   T=1:  ████████░░  ██████░░░░  ██░░░░░░░░  ████████░░  █░░░░░░░░░         │
│         bully=0.88  sexual=0.62 relig=0.27  threat=0.82 spam=0.12          │
│                                                                             │
│   T=4:  ██████░░░░  █████░░░░░  ████░░░░░░  █████░░░░░  ███░░░░░░░         │
│         bully=0.62  sexual=0.53 relig=0.44  threat=0.59 spam=0.38          │
│         ↑ Reveals: "religious is more similar to sexual than to spam"     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### The Forward Method:

```python
def forward(
    self,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None
) -> Dict[str, torch.Tensor]:
    """
    Compute distillation loss.
    
    Args:
        student_logits: Student's raw outputs (before sigmoid)
        teacher_logits: Teacher's raw outputs (before sigmoid)
        labels: Ground truth labels (0 or 1)
        class_weights: Weights for imbalanced classes
    
    Returns:
        Dictionary with total_loss, soft_loss, hard_loss
    """
    
    # =========================================================================
    # SOFT LOSS: Learn from teacher
    # =========================================================================
    
    # Scale logits by temperature
    student_soft = student_logits / self.temperature
    teacher_soft = teacher_logits / self.temperature
    
    # Apply sigmoid (NOT softmax - this is multi-label!)
    student_probs = torch.sigmoid(student_soft)
    teacher_probs = torch.sigmoid(teacher_soft)
    
    # Binary cross-entropy between student and teacher
    soft_loss = F.binary_cross_entropy(
        student_probs,
        teacher_probs,
        reduction=self.reduction
    )
    
    # Scale by T² (compensates for smaller gradients at high T)
    soft_loss = soft_loss * (self.temperature ** 2)
    
    # =========================================================================
    # HARD LOSS: Learn from ground truth labels
    # =========================================================================
    
    if class_weights is not None:
        hard_loss = F.binary_cross_entropy_with_logits(
            student_logits,
            labels,
            pos_weight=class_weights,
            reduction=self.reduction
        )
    else:
        hard_loss = F.binary_cross_entropy_with_logits(
            student_logits,
            labels,
            reduction=self.reduction
        )
    
    # =========================================================================
    # COMBINE LOSSES
    # =========================================================================
    
    # α balances soft vs hard loss
    total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
    
    return {
        'total_loss': total_loss,
        'soft_loss': soft_loss,
        'hard_loss': hard_loss
    }
```

### Mathematical Formulation:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ DISTILLATION LOSS FORMULA                                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ Total Loss = α × Soft Loss + (1 - α) × Hard Loss                           │
│                                                                             │
│ ═══════════════════════════════════════════════════════════════════════════│
│                                                                             │
│ SOFT LOSS (learning from teacher):                                         │
│                                                                             │
│   For each label i:                                                        │
│                                                                             │
│   p_teacher = σ(logit_teacher / T)     where σ = sigmoid                   │
│   p_student = σ(logit_student / T)                                         │
│                                                                             │
│   soft_loss_i = -[p_teacher × log(p_student) +                             │
│                   (1-p_teacher) × log(1-p_student)]                        │
│                                                                             │
│   soft_loss = T² × mean(soft_loss_i)                                       │
│               ↑                                                             │
│               Compensates for smaller gradients at high T                  │
│                                                                             │
│ ═══════════════════════════════════════════════════════════════════════════│
│                                                                             │
│ HARD LOSS (learning from labels):                                          │
│                                                                             │
│   For each label i:                                                        │
│                                                                             │
│   hard_loss_i = -[y_i × log(σ(logit)) +                                    │
│                   (1-y_i) × log(1-σ(logit))]                               │
│                                                                             │
│   With class weights:                                                      │
│   hard_loss_i = -[weight_i × y_i × log(σ(logit)) +                         │
│                   (1-y_i) × log(1-σ(logit))]                               │
│                                                                             │
│ ═══════════════════════════════════════════════════════════════════════════│
│                                                                             │
│ WHY T² SCALING?                                                             │
│                                                                             │
│   When T is high, gradients through sigmoid become smaller.                │
│   Multiplying by T² compensates, keeping gradient magnitudes stable.       │
│                                                                             │
│   Gradient magnitude ∝ 1/T² (from sigmoid derivative at high T)            │
│   Compensation: multiply loss by T²                                        │
│   Result: stable training regardless of temperature                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Critical Difference: Multi-Label vs Single-Label KD

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ MULTI-LABEL VS SINGLE-LABEL KD                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ SINGLE-LABEL (ImageNet, MNIST):                                            │
│ ────────────────────────────────                                            │
│   Each sample has EXACTLY ONE correct label                                │
│   Labels are MUTUALLY EXCLUSIVE                                            │
│                                                                             │
│   Example: Image is either "cat" OR "dog", never both                      │
│                                                                             │
│   Use SOFTMAX: Probabilities sum to 1                                      │
│   [cat: 0.7, dog: 0.2, bird: 0.1]  →  sum = 1.0                           │
│                                                                             │
│   Loss: KL Divergence                                                      │
│   KL(student || teacher) = Σ p_t × log(p_t / p_s)                         │
│                                                                             │
│ ═══════════════════════════════════════════════════════════════════════════│
│                                                                             │
│ MULTI-LABEL (Cyberbullying):                                               │
│ ────────────────────────────                                                │
│   Each sample can have MULTIPLE correct labels                             │
│   Labels are INDEPENDENT                                                   │
│                                                                             │
│   Example: Comment is "bully" AND "threat", not spam                       │
│   [bully: 1, sexual: 0, religious: 0, threat: 1, spam: 0]                  │
│                                                                             │
│   Use SIGMOID: Each label independent (0-1)                                │
│   [bully: 0.8, sexual: 0.1, religious: 0.2, threat: 0.7, spam: 0.1]       │
│   Sum can be anything!                                                     │
│                                                                             │
│   Loss: Binary Cross-Entropy (per label)                                   │
│   BCE = -[y × log(p) + (1-y) × log(1-p)]                                  │
│                                                                             │
│ ═══════════════════════════════════════════════════════════════════════════│
│                                                                             │
│ ⚠️  COMMON MISTAKE: Using KL divergence for multi-label                   │
│                                                                             │
│   Many papers and implementations incorrectly use softmax + KL             │
│   for multi-label problems. This LOSES information because:               │
│                                                                             │
│   Softmax forces probabilities to sum to 1:                                │
│   If bully=0.8, threat can't also be 0.8!                                  │
│                                                                             │
│   Our implementation correctly uses sigmoid + BCE per label.               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### What You Can Modify:

| Modification | Code Change | Research Question |
|--------------|-------------|-------------------|
| Change alpha | `alpha=0.5` | How important is teacher vs labels? |
| Change temperature | `temperature=2.0` | How soft should predictions be? |
| Label-specific alpha | Per-label alpha values | Should threat use more teacher? |
| Remove T² scaling | Remove `* (self.temperature ** 2)` | What happens to training? |

**Example - Label-Specific Alpha:**
```python
# More weight to teacher for rare classes (threat)
# More weight to labels for common classes (bully)
label_alphas = {
    'bully': 0.5,      # 50% teacher, 50% labels
    'sexual': 0.6,
    'religious': 0.7,
    'threat': 0.9,     # 90% teacher (rare class, teacher is important!)
    'spam': 0.5
}

def forward(self, student_logits, teacher_logits, labels, ...):
    soft_losses = []
    hard_losses = []
    
    for i, (label, alpha) in enumerate(label_alphas.items()):
        soft_loss_i = F.binary_cross_entropy(
            torch.sigmoid(student_logits[:, i] / self.temperature),
            torch.sigmoid(teacher_logits[:, i] / self.temperature)
        )
        hard_loss_i = F.binary_cross_entropy_with_logits(
            student_logits[:, i], labels[:, i]
        )
        total_i = alpha * soft_loss_i + (1 - alpha) * hard_loss_i
        # ... aggregate
```

---

## Section 5: Hidden State and Attention Losses

### Hidden State Loss:

```python
def compute_hidden_state_loss(
    teacher_hidden: Tuple[torch.Tensor, ...],
    student_hidden: Tuple[torch.Tensor, ...],
    layer_mapping: Optional[Dict[int, int]] = None
) -> torch.Tensor:
    """
    Match intermediate representations between teacher and student.
    
    WHY: Hidden states contain rich information about HOW the model
         processes text, not just WHAT it predicts.
    
    CHALLENGE: Teacher has 12 layers, student has 6 layers.
               Which layers should match?
    """
    if layer_mapping is None:
        # Default: Map every 2nd teacher layer to student
        # Teacher [2, 4, 6, 8, 10, 12] → Student [1, 2, 3, 4, 5, 6]
        layer_mapping = {2: 1, 4: 2, 6: 3, 8: 4, 10: 5, 12: 6}
    
    total_loss = 0
    num_layers = 0
    
    for teacher_layer, student_layer in layer_mapping.items():
        # Get hidden states for this layer
        t_hidden = teacher_hidden[teacher_layer]  # (batch, seq, 768)
        s_hidden = student_hidden[student_layer]  # (batch, seq, 768)
        
        # MSE loss between hidden states
        layer_loss = F.mse_loss(s_hidden, t_hidden)
        total_loss += layer_loss
        num_layers += 1
    
    return total_loss / num_layers
```

### Visual Explanation:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ HIDDEN STATE MATCHING                                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   TEACHER (12 layers)                    STUDENT (6 layers)                │
│                                                                             │
│   Layer 12 ─────────────────────────────── Layer 6  (match!)               │
│   Layer 11                                                                  │
│   Layer 10 ─────────────────────────────── Layer 5  (match!)               │
│   Layer 9                                                                   │
│   Layer 8 ──────────────────────────────── Layer 4  (match!)               │
│   Layer 7                                                                   │
│   Layer 6 ──────────────────────────────── Layer 3  (match!)               │
│   Layer 5                                                                   │
│   Layer 4 ──────────────────────────────── Layer 2  (match!)               │
│   Layer 3                                                                   │
│   Layer 2 ──────────────────────────────── Layer 1  (match!)               │
│   Layer 1                                                                   │
│                                                                             │
│   Loss = MSE(teacher_layer, student_layer)                                 │
│                                                                             │
│   This teaches student to process text SIMILARLY to teacher,              │
│   not just produce similar outputs.                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Attention Loss:

```python
def compute_attention_loss(
    teacher_attentions: Tuple[torch.Tensor, ...],
    student_attentions: Tuple[torch.Tensor, ...],
    layer_mapping: Optional[Dict[int, int]] = None
) -> torch.Tensor:
    """
    Match attention patterns between teacher and student.
    
    WHY: Attention patterns show WHAT the model focuses on.
         "This word relates to that word"
    
    Shape of attention: (batch, num_heads, seq_len, seq_len)
    """
    if layer_mapping is None:
        layer_mapping = {2: 1, 4: 2, 6: 3, 8: 4, 10: 5, 12: 6}
    
    total_loss = 0
    num_layers = 0
    
    for teacher_layer, student_layer in layer_mapping.items():
        t_attn = teacher_attentions[teacher_layer]
        s_attn = student_attentions[student_layer]
        
        # Average across attention heads (teacher and student may have different heads)
        t_attn_avg = t_attn.mean(dim=1)  # (batch, seq, seq)
        s_attn_avg = s_attn.mean(dim=1)  # (batch, seq, seq)
        
        # MSE loss on attention patterns
        layer_loss = F.mse_loss(s_attn_avg, t_attn_avg)
        total_loss += layer_loss
        num_layers += 1
    
    return total_loss / num_layers
```

### Attention Pattern Visualization:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ ATTENTION PATTERN MATCHING                                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ Input: "তুমি বোকা মেয়ে" (You stupid girl)                                   │
│                                                                             │
│ Teacher's attention (what it focuses on):                                  │
│                                                                             │
│         তুমি    বোকা    মেয়ে                                               │
│   তুমি  [0.2    0.7     0.1 ]  ← "তুমি" attends strongly to "বোকা"         │
│   বোকা  [0.3    0.3     0.4 ]                                               │
│   মেয়ে [0.1    0.6     0.3 ]  ← "মেয়ে" also attends to "বোকা" (insult!)   │
│                                                                             │
│ We want student to learn: "বোকা is the key word to focus on"              │
│                                                                             │
│ Loss = MSE(teacher_attention, student_attention)                           │
│                                                                             │
│ This helps student learn WHAT MATTERS in the input,                        │
│ not just how to classify the whole thing.                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Section 6: Distillation Trainer

```python
class DistillationTrainer:
    """
    Orchestrates the training process for knowledge distillation.
    
    Handles:
        - Forward passes through both models
        - Loss computation (soft + hard)
        - Gradient updates
        - AMP (mixed precision)
    """
    
    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        config,
        device: str
    ):
        self.teacher = teacher
        self.student = student
        self.config = config
        self.device = device
        
        # Teacher is frozen (no gradient updates)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        # Loss function
        self.kd_loss = MultiLabelDistillationLoss(
            alpha=config.kd_alpha,
            temperature=config.kd_temperature
        )
        
        # AMP scaler for mixed precision
        self.scaler = GradScaler()
```

### Why Freeze Teacher?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ WHY TEACHER IS FROZEN                                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ The teacher is already trained and accurate.                               │
│ We want to TRANSFER its knowledge, not change it.                          │
│                                                                             │
│ If teacher were trainable:                                                 │
│                                                                             │
│   1. It would need gradients → 2× memory usage                            │
│   2. It might change during training → unstable target                    │
│   3. We'd be training two models → very slow                              │
│                                                                             │
│ With frozen teacher:                                                        │
│                                                                             │
│   1. No gradients needed → less memory                                     │
│   2. Stable predictions → consistent learning signal                       │
│   3. Only student is trained → faster                                      │
│                                                                             │
│ Code:                                                                       │
│   self.teacher.eval()                    # Set to evaluation mode          │
│   for param in self.teacher.parameters():                                  │
│       param.requires_grad = False        # Disable gradient computation   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Training Step:

```python
def train_step(
    self,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    class_weights: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Single training step.
    """
    self.student.train()
    
    input_ids = batch['input_ids'].to(self.device)
    attention_mask = batch['attention_mask'].to(self.device)
    labels = batch['labels'].to(self.device)
    
    # Determine what outputs we need based on KD method
    need_hidden = self.config.kd_method in ['hidden', 'multi_level']
    need_attention = self.config.kd_method in ['attention', 'multi_level']
    
    with autocast():  # Mixed precision
        # Teacher forward (no gradients)
        with torch.no_grad():
            teacher_outputs = self.teacher(
                input_ids, attention_mask,
                output_hidden_states=need_hidden,
                output_attentions=need_attention
            )
        
        # Student forward (with gradients)
        student_outputs = self.student(
            input_ids, attention_mask,
            output_hidden_states=need_hidden,
            output_attentions=need_attention
        )
        
        # Compute logit distillation loss
        losses = self.kd_loss(
            student_outputs['logits'],
            teacher_outputs['logits'],
            labels,
            class_weights
        )
        
        total_loss = losses['total_loss']
        
        # Add hidden state loss if using hidden/multi_level
        if need_hidden:
            hidden_loss = compute_hidden_state_loss(
                teacher_outputs['hidden_states'],
                student_outputs['hidden_states']
            )
            total_loss += self.config.hidden_loss_weight * hidden_loss
            losses['hidden_loss'] = hidden_loss.item()
        
        # Add attention loss if using attention/multi_level
        if need_attention:
            attn_loss = compute_attention_loss(
                teacher_outputs['attentions'],
                student_outputs['attentions']
            )
            total_loss += self.config.attention_loss_weight * attn_loss
            losses['attention_loss'] = attn_loss.item()
    
    # Backward pass with AMP
    optimizer.zero_grad()
    self.scaler.scale(total_loss).backward()
    self.scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.config.gradient_clip_norm)
    self.scaler.step(optimizer)
    self.scaler.update()
    
    return {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}
```

### Complete KD Method Comparison:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ KD METHOD COMPARISON                                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ Method       │ What's Matched          │ Loss Formula                      │
│ ─────────────┼─────────────────────────┼────────────────────────────────── │
│ logit        │ Final predictions only  │ L = α×Soft + (1-α)×Hard           │
│              │                         │                                   │
│ hidden       │ + Intermediate layers   │ L = α×Soft + (1-α)×Hard           │
│              │                         │   + β×MSE(hidden_t, hidden_s)     │
│              │                         │                                   │
│ attention    │ + Attention patterns    │ L = α×Soft + (1-α)×Hard           │
│              │                         │   + γ×MSE(attn_t, attn_s)         │
│              │                         │                                   │
│ multi_level  │ Everything!             │ L = α×Soft + (1-α)×Hard           │
│              │                         │   + β×MSE(hidden_t, hidden_s)     │
│              │                         │   + γ×MSE(attn_t, attn_s)         │
│                                                                             │
│ Default weights:                                                            │
│   α = 0.7 (kd_alpha)                                                       │
│   β = 0.3 (hidden_loss_weight)                                             │
│   γ = 0.2 (attention_loss_weight)                                          │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ WHICH TO CHOOSE?                                                            │
│                                                                             │
│   logit:       Fast, good baseline. Start here.                            │
│   hidden:      +5-10% accuracy, +50% training time                         │
│   attention:   +3-5% accuracy, +50% training time                          │
│   multi_level: Best accuracy, ~2× training time                            │
│                                                                             │
│   Recommendation: Use 'logit' for quick experiments,                       │
│                   'multi_level' for final results in paper                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Section 7: Teacher Verification

```python
def verify_teacher_performance(
    teacher: nn.Module,
    val_loader,
    device: str,
    min_f1: float = 0.4
) -> Tuple[bool, Dict]:
    """
    Verify that teacher model is properly fine-tuned before KD.
    
    WHY: KD from a bad teacher = bad student!
         "Garbage in, garbage out"
    
    A teacher with F1 < 0.4 is essentially random guessing.
    KD from such a teacher would be harmful.
    """
    teacher.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = teacher(input_ids, attention_mask)
            preds = (torch.sigmoid(outputs['logits']) > 0.5).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(batch['labels'].numpy())
    
    from sklearn.metrics import f1_score
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    is_valid = f1 >= min_f1
    
    if not is_valid:
        print(f"⚠️  WARNING: Teacher F1 = {f1:.4f} < {min_f1}")
        print(f"   The teacher is not well-trained!")
        print(f"   KD from this teacher may not be helpful.")
    else:
        print(f"✅ Teacher F1 = {f1:.4f} (acceptable)")
    
    return is_valid, {'f1': f1}
```

### Why Verification Matters:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ WHY VERIFY TEACHER BEFORE KD?                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ Scenario 1: Good Teacher (F1 = 0.72)                                       │
│                                                                             │
│   Teacher predictions: [0.85, 0.12, 0.08, 0.78, 0.05]                      │
│   Ground truth:        [1,    0,    0,    1,    0   ]                      │
│                                                                             │
│   Teacher is accurate → student learns good patterns                       │
│   Result: Student F1 ≈ 0.68-0.70                                           │
│                                                                             │
│ ═══════════════════════════════════════════════════════════════════════════│
│                                                                             │
│ Scenario 2: Bad Teacher (F1 = 0.30, not fine-tuned)                        │
│                                                                             │
│   Teacher predictions: [0.52, 0.48, 0.51, 0.49, 0.50]  ← random!           │
│   Ground truth:        [1,    0,    0,    1,    0   ]                      │
│                                                                             │
│   Teacher is random → student learns nothing useful                        │
│   Result: Student F1 ≈ 0.25-0.35 (worse than training alone!)              │
│                                                                             │
│ ═══════════════════════════════════════════════════════════════════════════│
│                                                                             │
│ The verification step catches this:                                        │
│                                                                             │
│   if teacher_f1 < 0.4:                                                     │
│       print("WARNING: Teacher is not good enough!")                        │
│       # Either: train teacher more, or skip KD                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Summary: What You Can Modify in This Script

| Category | What to Modify | Research Impact |
|----------|----------------|-----------------|
| **Architecture** | Student model choice | Compression vs accuracy |
| **Architecture** | Classifier size | Fine-tune capacity |
| **Loss** | `alpha` | Teacher vs label importance |
| **Loss** | `temperature` | Softness of predictions |
| **Loss** | Per-label alpha | Label-specific distillation |
| **Hidden** | Layer mapping | Which layers to match |
| **Hidden** | `hidden_loss_weight` | Importance of hidden matching |
| **Attention** | `attention_loss_weight` | Importance of attention |
| **Method** | `kd_method` | What knowledge to transfer |
| **Training** | AMP on/off | Speed vs precision |

---

## Experiments You Can Run

```bash
# Experiment 1: Compare KD methods
for method in logit hidden attention multi_level; do
    python research_main.py --pipeline kd_only --kd_method $method \
        --output_dir results/kd_method_$method
done

# Experiment 2: Alpha sweep
for alpha in 0.3 0.5 0.7 0.9; do
    python research_main.py --pipeline kd_only --kd_alpha $alpha \
        --output_dir results/alpha_$alpha
done

# Experiment 3: Temperature sweep
for temp in 2 4 6 10; do
    python research_main.py --pipeline kd_only --kd_temperature $temp \
        --output_dir results/temp_$temp
done

# Experiment 4: Different student models
python research_main.py --pipeline kd_only \
    --student_path "google/mobilebert-uncased" \
    --student_hidden_size 128 \
    --output_dir results/student_mobilebert
```

---

## Practice Exercise

Before moving to the next script:

1. **Trace the loss computation**: Write out by hand what happens when `alpha=0.7, temperature=4.0`
2. **Understand layer mapping**: Draw which teacher layers map to which student layers
3. **Try modifying alpha**: Run with `alpha=0.3` and `alpha=0.9` and compare results

---

**Ready for the next script? The next one is `research_pruning.py` which implements all pruning methods (magnitude, gradual, wanda, structured).**

Would you like me to continue?