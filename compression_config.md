Absolutely! Let me explain each script in-depth, building your intuition step by step. I'll start with the **configuration script** since it's the foundation that controls everything else.

---

# 📘 Script 1: `research_compression_config.py`

## Overview

This script is the **control center** of your entire compression framework. Think of it as the "settings panel" - every decision about how compression works flows from here.

**Why start here?** Because understanding configuration helps you understand what "knobs" you can turn in your experiments.

---

## Section 1: Imports and Constants

```python
import argparse
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict
import json

LABEL_COLUMNS = ['bully', 'sexual', 'religious', 'threat', 'spam']
```

### What This Does:
- `argparse`: Python's built-in library for parsing command-line arguments
- `dataclass`: A decorator that automatically creates `__init__`, `__repr__` methods
- `LABEL_COLUMNS`: Defines the 5 cyberbullying categories your model detects

### Why Here:
Configuration is loaded FIRST before any other code runs. We define constants here so they're available everywhere.

### What You Can Modify:

| Modification | Effect | When to Do |
|--------------|--------|------------|
| Add new label to `LABEL_COLUMNS` | Model will predict 6 classes instead of 5 | If your dataset has more categories |
| Remove a label | Model predicts fewer classes | If you want binary classification |
| Reorder labels | Changes column order in output | Cosmetic only |

**Example - Adding a new label:**
```python
# Original
LABEL_COLUMNS = ['bully', 'sexual', 'religious', 'threat', 'spam']

# Modified for 6-class detection
LABEL_COLUMNS = ['bully', 'sexual', 'religious', 'threat', 'spam', 'hate_speech']
```

⚠️ **Impact:** You must also update your dataset CSV to have this new column!

---

## Section 2: Pipeline Definitions

```python
PIPELINE_CONFIGS = {
    'baseline': {
        'enable_kd': False,
        'enable_pruning': False,
        'enable_quantization': False,
        'description': 'Evaluate teacher only (no compression)'
    },
    'kd_only': {
        'enable_kd': True,
        'enable_pruning': False,
        'enable_quantization': False,
        'description': 'KD only: Teacher → Student'
    },
    # ... more pipelines
}
```

### What This Does:

This dictionary defines **8 compression pipelines**. Each pipeline is a combination of techniques:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PIPELINE COMBINATIONS                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   baseline ─────────── No compression (just evaluate teacher)              │
│                                                                             │
│   Single techniques:                                                        │
│   ├── kd_only ─────── Create smaller student model                         │
│   ├── prune_only ──── Remove weights from teacher                          │
│   └── quant_only ──── Reduce precision of teacher                          │
│                                                                             │
│   Two techniques:                                                           │
│   ├── kd_prune ────── Student + remove weights                             │
│   ├── kd_quant ────── Student + reduce precision                           │
│   └── prune_quant ─── Remove weights + reduce precision                    │
│                                                                             │
│   All techniques:                                                           │
│   └── kd_prune_quant ─ Student + remove weights + reduce precision         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why This Design:

1. **Ablation Studies**: You can measure each technique's contribution
2. **Reproducibility**: Same pipeline name = same settings every time
3. **Clarity**: Easy to understand what each experiment does

### What You Can Modify:

| Modification | Effect | Research Use |
|--------------|--------|--------------|
| Add new pipeline | New experiment combination | Test specific hypotheses |
| Change default flags | Different baseline behavior | Customize for your needs |

**Example - Add a new pipeline that uses aggressive settings:**
```python
PIPELINE_CONFIGS['aggressive'] = {
    'enable_kd': True,
    'enable_pruning': True,
    'enable_quantization': True,
    'description': 'Aggressive compression with high sparsity',
    # You could add custom defaults here
    'default_sparsity': 0.7,  # 70% pruning
    'default_quant': 'int4'   # Maximum quantization
}
```

---

## Section 3: Argument Parser (The Heart of Configuration)

```python
def parse_compression_arguments(args_list: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(
        description="Compress Transformer models for Bangla Cyberbullying Detection"
    )
    
    # Required arguments
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--author_name', type=str, required=True)
```

### What This Does:

Creates a command-line interface. When you run:
```bash
python research_main.py --dataset_path data.csv --kd_alpha 0.5
```

This function parses those arguments into a Python object you can use.

### Understanding Argument Types:

```python
# 1. REQUIRED arguments (must provide)
parser.add_argument('--dataset_path', type=str, required=True)

# 2. OPTIONAL with default
parser.add_argument('--kd_alpha', type=float, default=0.7)

# 3. CHOICE arguments (limited options)
parser.add_argument('--kd_method', type=str, default='logit',
                   choices=['logit', 'hidden', 'attention', 'multi_level'])

# 4. FLAG arguments (True if present, False if absent)
parser.add_argument('--run_ablation', action='store_true')

# 5. INVERSE FLAG (True by default, False if --no_xxx provided)
parser.add_argument('--fine_tune_after_prune', default=True)
parser.add_argument('--no_fine_tune_after_prune', action='store_false',
                   dest='fine_tune_after_prune')
```

---

## Section 3.1: Teacher Configuration

```python
# Teacher Model Configuration
parser.add_argument('--teacher_path', type=str, default='csebuetnlp/banglabert')
parser.add_argument('--teacher_checkpoint', type=str, default=None)
parser.add_argument('--teacher_epochs', type=int, default=10)
```

### What Each Argument Does:

| Argument | Purpose | Default | Your Options |
|----------|---------|---------|--------------|
| `--teacher_path` | Base model architecture | BanglaBERT | Any HuggingFace model |
| `--teacher_checkpoint` | Pre-trained weights | None (train from scratch) | Your HuggingFace model |
| `--teacher_epochs` | Training epochs | 10 | 5-20 typical |

### Deep Dive - Why Two Arguments for Teacher?

```
--teacher_path vs --teacher_checkpoint

┌─────────────────────────────────────────────────────────────────────────────┐
│ Scenario 1: NO checkpoint (train from scratch)                             │
│                                                                             │
│   --teacher_path "csebuetnlp/banglabert"                                   │
│   --teacher_checkpoint (not provided)                                       │
│                                                                             │
│   What happens:                                                             │
│   1. Load raw BanglaBERT (not fine-tuned)                                  │
│   2. Train for 10 epochs on your dataset                                   │
│   3. Takes ~30 minutes                                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│ Scenario 2: WITH checkpoint (skip training!)                               │
│                                                                             │
│   --teacher_path "csebuetnlp/banglabert"                                   │
│   --teacher_checkpoint "your-username/your-finetuned-model"                │
│                                                                             │
│   What happens:                                                             │
│   1. Load YOUR pre-trained model directly                                  │
│   2. Skip training completely                                               │
│   3. Takes ~1 minute                                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### What You Can Modify:

**For Different Languages:**
```python
# Bengali (current)
parser.add_argument('--teacher_path', default='csebuetnlp/banglabert')

# Hindi
parser.add_argument('--teacher_path', default='ai4bharat/indic-bert')

# Multilingual
parser.add_argument('--teacher_path', default='bert-base-multilingual-cased')

# English
parser.add_argument('--teacher_path', default='bert-base-uncased')
```

**Impact of changing teacher:**
- Different languages supported
- Different model sizes
- Different accuracy baselines

---

## Section 3.2: Knowledge Distillation Parameters

```python
parser.add_argument('--kd_alpha', type=float, default=0.7)
parser.add_argument('--kd_temperature', type=float, default=4.0)
parser.add_argument('--kd_method', type=str, default='logit',
                   choices=['logit', 'hidden', 'attention', 'multi_level'])
```

### Visual Explanation of Each Parameter:

```
KD_ALPHA: Balance between teacher and ground truth
──────────────────────────────────────────────────

Loss = α × (learn from teacher) + (1-α) × (learn from labels)

α = 0.0: ████████████████████ 100% ground truth, 0% teacher
         Student learns ONLY from labels (no distillation!)
         
α = 0.5: ██████████░░░░░░░░░░ 50% ground truth, 50% teacher
         Equal balance
         
α = 0.7: ██████░░░░░░░░░░░░░░ 30% ground truth, 70% teacher  ← DEFAULT
         More weight to teacher (recommended for KD)
         
α = 1.0: ░░░░░░░░░░░░░░░░░░░░ 0% ground truth, 100% teacher
         Student learns ONLY from teacher (ignores labels!)


KD_TEMPERATURE: How "soft" are teacher's predictions
────────────────────────────────────────────────────

Original prediction: [bully: 0.95, threat: 0.03, spam: 0.02]
                      Very confident (sharp)
                      
T = 1.0: [0.95, 0.03, 0.02]  ← Sharp (normal inference)
         Student only learns "this is bully"
         
T = 4.0: [0.70, 0.18, 0.12]  ← Softer  ← DEFAULT
         Student learns "mostly bully, some threat/spam relationship"
         
T = 10:  [0.45, 0.30, 0.25]  ← Very soft
         Almost equal probabilities (too soft, loses information)


KD_METHOD: What knowledge to transfer
─────────────────────────────────────

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  TEACHER MODEL                              STUDENT MODEL                   │
│  ┌─────────────┐                           ┌─────────────┐                 │
│  │ Input       │                           │ Input       │                 │
│  ├─────────────┤                           ├─────────────┤                 │
│  │ Layer 1     │ ──── hidden ────────────▶ │ Layer 1     │                 │
│  │ Attention   │ ──── attention ─────────▶ │ Attention   │                 │
│  ├─────────────┤                           ├─────────────┤                 │
│  │ Layer 2     │ ──── hidden ────────────▶ │ Layer 2     │                 │
│  │ Attention   │ ──── attention ─────────▶ │ Attention   │                 │
│  ├─────────────┤                           ├─────────────┤                 │
│  │ ...         │                           │ ...         │                 │
│  ├─────────────┤                           ├─────────────┤                 │
│  │ Output      │ ──── logit ─────────────▶ │ Output      │                 │
│  └─────────────┘                           └─────────────┘                 │
│                                                                             │
│  logit:       Match only output predictions (simplest)                     │
│  hidden:      Match intermediate layer outputs                              │
│  attention:   Match attention patterns                                      │
│  multi_level: Match ALL of the above (best but slowest)                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Research Modifications:

| Parameter | Range to Try | Expected Effect |
|-----------|--------------|-----------------|
| `kd_alpha` | 0.3, 0.5, 0.7, 0.9 | Higher = more teacher influence |
| `kd_temperature` | 2, 4, 6, 10 | Higher = softer predictions |
| `kd_method` | all 4 options | multi_level usually best |

**Example Experiment - Alpha Sweep:**
```bash
for alpha in 0.3 0.5 0.7 0.9; do
    python research_main.py --pipeline kd_only --kd_alpha $alpha \
        --output_dir results/alpha_$alpha
done
```

---

## Section 3.3: Pruning Parameters

```python
parser.add_argument('--prune_method', type=str, default='magnitude',
                   choices=['magnitude', 'wanda', 'gradual', 'structured'])
parser.add_argument('--prune_sparsity', type=float, default=0.5)
parser.add_argument('--prune_schedule', type=str, default='cubic',
                   choices=['linear', 'cubic', 'exponential'])
```

### Visual Explanation:

```
PRUNE_SPARSITY: What fraction of weights to remove
──────────────────────────────────────────────────

Original weights: ████████████████████ 100% (110M parameters)

sparsity = 0.3:   ██████████████░░░░░░ 70% remain, 30% removed
                  Conservative, minimal accuracy loss
                  
sparsity = 0.5:   ██████████░░░░░░░░░░ 50% remain, 50% removed  ← DEFAULT
                  Balanced compression vs accuracy
                  
sparsity = 0.7:   ██████░░░░░░░░░░░░░░ 30% remain, 70% removed
                  Aggressive, may hurt accuracy significantly


PRUNE_METHOD: Algorithm for selecting which weights to remove
─────────────────────────────────────────────────────────────

magnitude:  Remove weights with smallest |value|
            Weights: [0.8, -0.02, 0.3, 0.01, -0.6]
            Remove:       ↑ small      ↑ small
            Result:  [0.8,   0,   0.3,   0,  -0.6]
            
wanda:      Remove based on |weight| × |activation|
            Even small weights can be important if activations are large!
            More sophisticated, state-of-the-art (2023)
            
gradual:    Slowly increase sparsity during training
            Epoch 1: 10% sparsity
            Epoch 5: 30% sparsity
            Epoch 10: 50% sparsity
            Model adapts as pruning progresses
            
structured: Remove entire neurons/attention heads
            Gives REAL speedup without special hardware


PRUNE_SCHEDULE (for gradual pruning):
─────────────────────────────────────

Sparsity
   │
50%├─────────────────────────────●●●●● Target
   │                          ●●
   │                       ●●
25%├────────────────────●●
   │              ●●●●  
   │         ●●●●      
   │    ●●●●           linear: constant rate ──────
   │ ●●●               cubic: slow-fast-slow ●●●●●●  ← DEFAULT
   │●                  exponential: slow-fast ······
 0%├──────────────────────────────────────────────────
   0  1  2  3  4  5  6  7  8  9  10  Epochs
```

### Research Modifications:

| Experiment | Parameters | Research Question |
|------------|------------|-------------------|
| Sparsity sweep | 0.3, 0.4, 0.5, 0.6, 0.7 | "What's the accuracy-compression tradeoff?" |
| Method comparison | magnitude, wanda, gradual | "Which pruning method works best?" |
| Layer-specific | attention, ffn, encoder | "Which layers are most compressible?" |

---

## Section 3.4: Quantization Parameters

```python
parser.add_argument('--quant_method', type=str, default='dynamic',
                   choices=['dynamic', 'static', 'qat', 'fp16', 'int4'])
parser.add_argument('--quant_dtype', type=str, default='int8',
                   choices=['int8', 'int4', 'fp16'])
```

### Visual Explanation:

```
QUANTIZATION METHODS:
────────────────────

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  Original FP32:  0.12345678901234567890123456789012                        │
│                  └──────────── 32 bits ────────────┘                        │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  FP16:           0.1234567                                                  │
│                  └── 16 bits ──┘                                            │
│                  2× compression, works on GPU, ~0% accuracy loss           │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  INT8:           0.12                                                       │
│                  └ 8 bits ┘                                                 │
│                  4× compression, CPU only, 1-2% accuracy loss              │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  INT4:           0.1                                                        │
│                  └4 bits┘                                                   │
│                  8× compression, GPU (bitsandbytes), 2-4% accuracy loss    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


WHEN TO USE EACH:
─────────────────

dynamic:  Easiest, weights pre-quantized, activations at runtime
          Use for: Quick experiments, deployment to CPU
          
static:   Better accuracy, needs calibration data
          Use for: Production deployment, maximum accuracy
          
fp16:     Works on GPU, minimal accuracy loss
          Use for: GPU inference, when 2× compression is enough
          
int4:     Maximum compression (8×)
          Use for: Large models that don't fit in memory
```

---

## Section 4: Helper Functions

```python
def _apply_pipeline_config(args):
    """Apply pipeline-specific settings."""
    pipeline_config = PIPELINE_CONFIGS.get(args.pipeline, {})
    args.enable_kd = pipeline_config.get('enable_kd', False)
    args.enable_pruning = pipeline_config.get('enable_pruning', False)
    args.enable_quantization = pipeline_config.get('enable_quantization', False)
```

### What This Does:

When you specify `--pipeline kd_prune`, this function automatically sets:
- `enable_kd = True`
- `enable_pruning = True`
- `enable_quantization = False`

### Why:

You don't want to manually specify `--enable_kd --enable_pruning` every time. The pipeline name handles it.

---

## Section 5: Configuration Printer

```python
def print_compression_config(config):
    """Print configuration summary."""
    print(f"\n📊 Pipeline: {config.pipeline.upper()}")
    # ... prints all settings
```

### What This Does:

Pretty-prints all your settings before running. Helps you verify:
- You're running the right pipeline
- Parameters are set correctly
- No mistakes before waiting 1+ hours

### Sample Output:

```
══════════════════════════════════════════════════════════════════════
🔧 COMPRESSION CONFIGURATION
══════════════════════════════════════════════════════════════════════

📊 Pipeline: KD_PRUNE_QUANT
   Full pipeline: KD → Prune STUDENT → Quantize

📋 COMPRESSION FLOW:
   Teacher → KD → Student → Prune Student → Fine-tune → Quantize → Final Model

🎓 Teacher: csebuetnlp/banglabert
   Using checkpoint: your-username/model (skip training!)

📚 Student: distilbert-base-multilingual-cased
   KD Method: logit
   Alpha: 0.7, Temperature: 4.0

✂️  Pruning:
   Method: magnitude
   Sparsity: 50%
   Fine-tune after: Yes

📉 Quantization:
   Method: fp16
══════════════════════════════════════════════════════════════════════
```

---

## Summary: What You Can Modify in This Script

| Category | Parameter | Research Use |
|----------|-----------|--------------|
| **Model** | `teacher_path` | Different base models |
| **Model** | `student_path` | Different student architectures |
| **KD** | `kd_alpha` | Hyperparameter search |
| **KD** | `kd_temperature` | Hyperparameter search |
| **KD** | `kd_method` | Compare distillation methods |
| **Pruning** | `prune_sparsity` | Compression-accuracy tradeoff |
| **Pruning** | `prune_method` | Compare pruning algorithms |
| **Quantization** | `quant_method` | Compare precision levels |
| **Training** | `lr`, `epochs`, `batch` | Standard hyperparameter tuning |
| **Labels** | `LABEL_COLUMNS` | Add/remove detection categories |

---

## Practice Exercise

Before moving to the next script, try this:

1. **Read the config file** and identify where each parameter is defined
2. **Trace the flow**: When you pass `--kd_alpha 0.5`, where does it end up?
3. **Modify something**: Add a new pipeline called `kd_heavy` that uses `kd_method=multi_level` by default

---

**Ready for the next script? The next one is `research_data.py` which handles data loading and the critical tokenization caching feature.**

Would you like me to continue with the data module?