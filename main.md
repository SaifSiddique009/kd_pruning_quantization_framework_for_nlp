# 📘 Script 7: `research_main.py`

## Overview

This is the **orchestration script** - the conductor that brings together all the instruments (data loading, KD, pruning, quantization, evaluation) into a cohesive symphony. When you run an experiment, this is the entry point that coordinates everything.

**Why this script is the heart of your framework:**
- Single entry point for all experiments
- Manages the flow between compression stages
- Handles configuration and command-line arguments
- Ensures proper sequencing (KD before pruning, pruning before quantization)
- Produces final reports and saves models

---

## The Big Picture: How Everything Fits Together

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ THE COMPLETE COMPRESSION PIPELINE                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Command Line                                                              │
│       │                                                                     │
│       ▼                                                                     │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ research_main.py                                                    │  │
│   │                                                                     │  │
│   │   1. Parse arguments (research_compression_config.py)              │  │
│   │   2. Load & cache data (research_data.py)                          │  │
│   │   3. Load/train teacher                                            │  │
│   │   4. Run compression stages:                                       │  │
│   │      ├─→ KD (research_distillation.py)                            │  │
│   │      ├─→ Pruning (research_pruning.py)                            │  │
│   │      └─→ Quantization (research_quantization.py)                  │  │
│   │   5. Evaluate each stage (research_evaluation.py)                  │  │
│   │   6. Save models and metrics                                       │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│       │                                                                     │
│       ▼                                                                     │
│   Outputs:                                                                  │
│   ├─→ Compressed models (HuggingFace format)                              │
│   ├─→ Metrics (CSV, JSON)                                                 │
│   ├─→ Comparison tables                                                   │
│   └─→ Safety reports                                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Section 1: Environment Setup (MUST BE FIRST!)

```python
#!/usr/bin/env python3
"""
================================================================================
MAIN COMPRESSION PIPELINE
================================================================================

Entry point for all compression experiments.

Usage:
    # Single pipeline
    python research_main.py --dataset_path data.csv --author_name "X" --pipeline kd_only
    
    # Full ablation study
    python research_main.py --dataset_path data.csv --author_name "X" --run_ablation
"""

# =============================================================================
# ENVIRONMENT SETUP (MUST BE FIRST!)
# =============================================================================
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Prevent tokenizer fork warning

import sys
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup
import numpy as np
from datetime import datetime
from tqdm import tqdm
import json
import warnings
import shutil
warnings.filterwarnings('ignore')
```

### Why Environment Setup Must Be First:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ WHY os.environ MUST COME BEFORE IMPORTS                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ WRONG ORDER (will cause warnings):                                         │
│                                                                             │
│   from transformers import AutoTokenizer  # ← Tokenizer initialized here!  │
│   os.environ["TOKENIZERS_PARALLELISM"] = "false"  # ← Too late!           │
│                                                                             │
│   What happens:                                                             │
│   1. transformers imports tokenizers library                               │
│   2. tokenizers checks TOKENIZERS_PARALLELISM env var                      │
│   3. Var not set → defaults to parallel mode                               │
│   4. Your later setting is ignored                                         │
│   5. Warning appears when DataLoader forks processes                       │
│                                                                             │
│ CORRECT ORDER (no warnings):                                                │
│                                                                             │
│   import os                                                                 │
│   os.environ["TOKENIZERS_PARALLELISM"] = "false"  # ← Set FIRST!          │
│   from transformers import AutoTokenizer  # ← Now respects the setting    │
│                                                                             │
│ LESSON: Environment variables must be set BEFORE importing libraries       │
│         that read them during initialization.                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Local Module Imports:

```python
# Local imports (our compression modules)
from research_compression_config import (
    parse_compression_arguments, print_compression_config,
    PIPELINE_CONFIGS, LABEL_COLUMNS, get_config_for_pipeline
)
from research_data import (
    load_and_preprocess_data, get_or_create_tokenized_dataset,
    prepare_kfold_splits, calculate_class_weights,
    create_data_loaders, IndexedDataset
)
from research_distillation import (
    TeacherModel, StudentModel, DistillationTrainer,
    MultiLabelDistillationLoss, verify_teacher_performance
)
from research_pruning import (
    PruningManager, GradualPruner, WandaPruner,
    get_pruner, fine_tune_after_pruning
)
from research_quantization import (
    QuantizationManager, quantize_model, benchmark_inference_speed
)
from research_evaluation import (
    CompressionEvaluator, CompressionStageMetrics,
    compare_stages, export_metrics_to_csv, export_metrics_to_json
)
```

---

## Section 2: Utility Functions

```python
def set_seed(seed: int):
    """
    Set random seeds for reproducibility.
    
    WHY REPRODUCIBILITY MATTERS:
    ────────────────────────────
    Without fixed seeds, running the same experiment twice gives different results.
    This makes it impossible to:
    - Debug issues ("it worked yesterday!")
    - Compare configurations fairly
    - Reproduce results for your paper
    
    We set seeds for ALL random sources:
    - Python's random module
    - NumPy's random
    - PyTorch CPU operations
    - PyTorch CUDA operations
    - cuDNN (CUDA's neural network library)
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # These settings ensure deterministic behavior but may slow down training
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    """
    Get the best available device and print info.
    
    Returns 'cuda' if GPU available, 'cpu' otherwise.
    Also prints useful diagnostic information.
    """
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"🖥️  Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"   CUDA Version: {torch.version.cuda}")
    else:
        device = 'cpu'
        print("🖥️  Using CPU (GPU not available)")
        print("   ⚠️  Training will be slow. Consider using Google Colab or Kaggle.")
    
    return device
```

### Understanding Reproducibility Settings:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ REPRODUCIBILITY VS PERFORMANCE TRADE-OFF                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ torch.backends.cudnn.deterministic = True                                  │
│ ─────────────────────────────────────────                                   │
│   What: Forces cuDNN to use deterministic algorithms                       │
│   Effect: Same input always gives same output                              │
│   Cost: Some operations are slower (can't use fastest algorithms)          │
│                                                                             │
│ torch.backends.cudnn.benchmark = False                                     │
│ ──────────────────────────────────────                                      │
│   What: Disables cuDNN auto-tuning                                         │
│   Effect: Uses same algorithm every run                                    │
│   Cost: May not use optimal algorithm for your GPU/input sizes             │
│                                                                             │
│ WHEN TO USE:                                                                │
│                                                                             │
│   During development/debugging:                                            │
│       deterministic=True, benchmark=False                                  │
│       → Reproducible results for fair comparisons                          │
│                                                                             │
│   During final training (when you need speed):                             │
│       deterministic=False, benchmark=True                                  │
│       → Fastest possible training                                          │
│       → Report average of multiple runs in paper                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Section 3: HuggingFace Model Saving

```python
def save_model_for_huggingface(model, save_path, tokenizer=None):
    """
    Save model in HuggingFace format for easy deployment.
    
    WHY HUGGINGFACE FORMAT?
    ───────────────────────
    1. Industry standard - everyone knows how to load it
    2. Easy deployment to HuggingFace Hub
    3. Compatible with transformers pipelines
    4. Includes tokenizer for complete solution
    
    WHAT'S SAVED:
    ─────────────
    - config.json: Model architecture configuration
    - pytorch_model.bin: Encoder weights
    - tokenizer files: Vocabulary and tokenization settings
    - classifier.pt: Our custom classifier head
    - how_to_load.py: Instructions for loading
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Save the encoder (transformer backbone)
    if hasattr(model, 'encoder'):
        model.encoder.save_pretrained(save_path)
        print(f"   ✓ Encoder saved to {save_path}")
    
    # Save classifier separately (not part of standard HuggingFace format)
    if hasattr(model, 'classifier'):
        classifier_path = os.path.join(save_path, 'classifier.pt')
        torch.save(model.classifier.state_dict(), classifier_path)
        
        # Save classifier architecture info
        classifier_config = {
            'type': 'sequential',
            'layers': str(model.classifier),
            'num_labels': getattr(model, 'num_labels', 5)
        }
        with open(os.path.join(save_path, 'classifier_config.json'), 'w') as f:
            json.dump(classifier_config, f, indent=2)
        print(f"   ✓ Classifier saved to {classifier_path}")
    
    # Save tokenizer
    if tokenizer is not None:
        tokenizer.save_pretrained(save_path)
        print(f"   ✓ Tokenizer saved")
    
    # Create a helper script for loading
    _create_loading_script(save_path)
    
    print(f"✅ Model saved in HuggingFace format: {save_path}")
```

### The Loading Script Generator:

```python
def _create_loading_script(save_path):
    """Generate a Python script showing how to load the saved model."""
    
    script = f'''#!/usr/bin/env python3
"""
How to load this compressed model for inference.
"""

from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn

def load_model(model_path="{save_path}"):
    """Load the compressed model."""
    
    # Load encoder
    encoder = AutoModel.from_pretrained(model_path)
    
    # Load classifier
    # First, check classifier config
    import json
    with open(f"{{model_path}}/classifier_config.json") as f:
        clf_config = json.load(f)
    
    num_labels = clf_config['num_labels']
    hidden_size = encoder.config.hidden_size
    
    # Recreate classifier architecture
    classifier = nn.Sequential(
        nn.Linear(hidden_size, 256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, num_labels)
    )
    
    # Load classifier weights
    classifier.load_state_dict(
        torch.load(f"{{model_path}}/classifier.pt", map_location='cpu')
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    return encoder, classifier, tokenizer


def predict(text, encoder, classifier, tokenizer, device='cuda'):
    """Run inference on a single text."""
    
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding='max_length'
    )
    
    # Move to device
    inputs = {{k: v.to(device) for k, v in inputs.items()}}
    encoder = encoder.to(device)
    classifier = classifier.to(device)
    
    # Inference
    encoder.eval()
    classifier.eval()
    
    with torch.no_grad():
        outputs = encoder(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        logits = classifier(cls_embedding)
        probabilities = torch.sigmoid(logits)
    
    # Format output
    labels = ['bully', 'sexual', 'religious', 'threat', 'spam']
    results = {{}}
    for i, label in enumerate(labels):
        results[label] = probabilities[0, i].item()
    
    return results


# Example usage
if __name__ == "__main__":
    encoder, classifier, tokenizer = load_model()
    
    # Test prediction
    text = "আপনার বাংলা টেক্সট এখানে"  # Your Bangla text here
    results = predict(text, encoder, classifier, tokenizer)
    
    print("Predictions:")
    for label, prob in results.items():
        print(f"  {{label}}: {{prob:.4f}}")
'''
    
    with open(os.path.join(save_path, 'how_to_load.py'), 'w') as f:
        f.write(script)
```

---

## Section 4: Metrics Saving Per Stage

```python
def save_stage_metrics(metrics: CompressionStageMetrics, output_dir: str, stage_name: str):
    """
    Save metrics immediately after each compression stage.
    
    WHY SAVE AFTER EACH STAGE?
    ──────────────────────────
    1. If experiment crashes later, you don't lose earlier results
    2. Can monitor progress while experiment is running
    3. Easier to debug which stage caused issues
    4. Results available for early analysis
    
    FILES CREATED:
    ──────────────
    - results_{stage_name}.csv: This stage's metrics
    - results_{stage_name}.json: Same, in JSON format
    - results_all.csv: Cumulative (appends each stage)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save individual stage metrics
    stage_csv = os.path.join(output_dir, f'results_{stage_name}.csv')
    stage_json = os.path.join(output_dir, f'results_{stage_name}.json')
    
    import pandas as pd
    df = pd.DataFrame([metrics.to_flat_dict()])
    df.to_csv(stage_csv, index=False)
    
    with open(stage_json, 'w') as f:
        json.dump(metrics.to_dict(), f, indent=2, default=str)
    
    # Append to cumulative results file
    all_csv = os.path.join(output_dir, 'results_all.csv')
    if os.path.exists(all_csv):
        # Append without header
        df.to_csv(all_csv, mode='a', header=False, index=False)
    else:
        # Create with header
        df.to_csv(all_csv, index=False)
    
    print(f"   📊 Metrics saved: {stage_csv}")
```

---

## Section 5: Teacher Loading/Training

This is Phase 1 of the pipeline - getting a working teacher model.

```python
def get_or_train_teacher(config, tokenized_data, train_idx, val_idx, device):
    """
    Get teacher model - either load from checkpoint or train from scratch.
    
    TWO MODES:
    ──────────
    1. WITH --teacher_checkpoint: Load pre-trained model (fast, ~1 min)
    2. WITHOUT checkpoint: Train from scratch (slow, ~30 min)
    
    For your experiments, you likely have a fine-tuned model on HuggingFace.
    Use --teacher_checkpoint to skip training!
    """
    print("\n" + "="*70)
    print("🎓 PHASE 1: TEACHER MODEL")
    print("="*70)
    
    # Always need the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.teacher_path)
    
    # Check if we can load from checkpoint
    if config.teacher_checkpoint:
        print(f"\n📥 Loading pre-trained teacher from: {config.teacher_checkpoint}")
        
        teacher = TeacherModel(
            model_name=config.teacher_checkpoint,  # Load from checkpoint
            num_labels=len(LABEL_COLUMNS),
            dropout=config.dropout
        ).to(device)
        
        print("   ✅ Teacher loaded successfully!")
        print(f"   Parameters: {sum(p.numel() for p in teacher.parameters())/1e6:.2f}M")
        
        return teacher, tokenizer
    
    # No checkpoint - train from scratch
    print(f"\n🔧 Training teacher from scratch...")
    print(f"   Base model: {config.teacher_path}")
    print(f"   Epochs: {config.teacher_epochs}")
    
    teacher = TeacherModel(
        model_name=config.teacher_path,
        num_labels=len(LABEL_COLUMNS),
        dropout=config.dropout
    ).to(device)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        tokenized_data, train_idx, val_idx,
        batch_size=config.batch, num_workers=2
    )
    
    # Calculate class weights for imbalanced data
    train_labels = tokenized_data['labels'][train_idx]
    class_weights = calculate_class_weights(train_labels.numpy()).to(device)
    
    # Setup optimizer and scheduler
    optimizer = AdamW(
        teacher.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )
    
    total_steps = len(train_loader) * config.teacher_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(config.warmup_ratio * total_steps),
        num_training_steps=total_steps
    )
    
    # Loss function with class weights
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    
    # Training loop
    best_f1 = 0
    
    for epoch in range(config.teacher_epochs):
        teacher.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Teacher Epoch {epoch+1}/{config.teacher_epochs}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            outputs = teacher(input_ids, attention_mask)
            loss = loss_fn(outputs['logits'], labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), config.gradient_clip_norm)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Validate
        val_f1 = _evaluate_f1(teacher, val_loader, device)
        print(f"   Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, F1={val_f1:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
    
    print(f"\n   ✅ Teacher training complete! Best F1: {best_f1:.4f}")
    return teacher, tokenizer


def _evaluate_f1(model, dataloader, device):
    """Quick F1 evaluation for training progress."""
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids, attention_mask)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(batch['labels'].numpy())
    
    from sklearn.metrics import f1_score
    return f1_score(all_labels, all_preds, average='macro')
```

### Understanding the Teacher Loading Flow:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ TEACHER MODEL LOADING DECISION TREE                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   --teacher_checkpoint provided?                                           │
│       │                                                                     │
│       ├── YES: "your-username/your-finetuned-model"                        │
│       │       │                                                             │
│       │       ▼                                                             │
│       │   ┌─────────────────────────────────────────────────────────────┐  │
│       │   │ Load directly from HuggingFace Hub                          │  │
│       │   │ - Downloads model weights (~400 MB)                         │  │
│       │   │ - Creates TeacherModel with these weights                   │  │
│       │   │ - Time: ~1-2 minutes                                        │  │
│       │   │ - F1: Whatever your model achieves                          │  │
│       │   └─────────────────────────────────────────────────────────────┘  │
│       │                                                                     │
│       └── NO: Train from scratch                                           │
│               │                                                             │
│               ▼                                                             │
│           ┌─────────────────────────────────────────────────────────────┐  │
│           │ Load base model (e.g., BanglaBERT)                          │  │
│           │ - Downloads pretrained weights                              │  │
│           │ - NOT fine-tuned on your task                               │  │
│           │                                                             │  │
│           │ Train for teacher_epochs (default: 10)                      │  │
│           │ - Time: ~30 minutes                                         │  │
│           │ - Learns to classify cyberbullying                          │  │
│           │                                                             │  │
│           │ Result: Fine-tuned teacher ready for KD                     │  │
│           └─────────────────────────────────────────────────────────────┘  │
│                                                                             │
│ RECOMMENDATION:                                                             │
│   If you already have a fine-tuned model, ALWAYS use --teacher_checkpoint  │
│   This saves 30 minutes per experiment!                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Section 6: Knowledge Distillation Phase

```python
def run_knowledge_distillation(config, teacher, tokenized_data, train_idx, val_idx, device):
    """
    Run knowledge distillation from teacher to student.
    
    THIS CREATES A NEW, SMALLER MODEL!
    
    The student model is:
    - Smaller architecture (e.g., DistilBERT vs BERT)
    - Trained to mimic teacher's predictions
    - The basis for further compression (pruning, quantization)
    """
    print("\n" + "="*70)
    print("🔄 PHASE 2: KNOWLEDGE DISTILLATION")
    print("="*70)
    print(f"   Teacher: {config.teacher_path}")
    print(f"   Student: {config.student_path}")
    print(f"   Method: {config.kd_method}")
    print(f"   Alpha: {config.kd_alpha}, Temperature: {config.kd_temperature}")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        tokenized_data, train_idx, val_idx,
        batch_size=config.batch, num_workers=2
    )
    
    # Verify teacher is good enough
    is_valid, teacher_metrics = verify_teacher_performance(
        teacher, val_loader, device, min_f1=0.4
    )
    if not is_valid:
        print("   ⚠️  Warning: Teacher F1 is low. KD may not help much.")
    
    # CREATE THE STUDENT MODEL
    student = StudentModel(
        model_name=config.student_path,
        num_labels=len(LABEL_COLUMNS),
        dropout=config.dropout,
        classifier_hidden_size=config.student_hidden_size
    ).to(device)
    
    # Print size comparison
    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())
    
    print(f"\n   📊 Model Size Comparison:")
    print(f"      Teacher: {teacher_params/1e6:.2f}M parameters")
    print(f"      Student: {student_params/1e6:.2f}M parameters")
    print(f"      Reduction: {(1 - student_params/teacher_params)*100:.1f}%")
    
    # Get class weights
    train_labels = tokenized_data['labels'][train_idx]
    class_weights = calculate_class_weights(train_labels.numpy()).to(device)
    
    # Create distillation trainer
    trainer = DistillationTrainer(teacher, student, config, device)
    
    # Optimizer and scheduler
    optimizer = AdamW(student.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    total_steps = len(train_loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(config.warmup_ratio * total_steps),
        num_training_steps=total_steps
    )
    
    # Training loop
    best_f1 = 0
    patience_counter = 0
    
    for epoch in range(config.epochs):
        epoch_losses = []
        
        pbar = tqdm(train_loader, desc=f"KD Epoch {epoch+1}/{config.epochs}")
        for batch in pbar:
            losses = trainer.train_step(batch, optimizer, class_weights)
            scheduler.step()
            
            epoch_losses.append(losses['total_loss'])
            pbar.set_postfix({
                'loss': f"{losses['total_loss']:.4f}",
                'soft': f"{losses['soft_loss']:.4f}",
                'hard': f"{losses['hard_loss']:.4f}"
            })
        
        # Evaluate
        val_f1 = _evaluate_f1(student, val_loader, device)
        print(f"   Epoch {epoch+1}: Loss={np.mean(epoch_losses):.4f}, F1={val_f1:.4f}")
        
        # Early stopping
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                print(f"   Early stopping at epoch {epoch+1}")
                break
    
    print(f"\n   ✅ Knowledge Distillation complete!")
    print(f"      Teacher F1: {teacher_metrics.get('f1', 'N/A')}")
    print(f"      Student F1: {best_f1:.4f}")
    print(f"      The STUDENT model will be used for subsequent compression.")
    
    return student
```

---

## Section 7: Pruning Phase

```python
def run_pruning(config, model, tokenized_data, train_idx, val_idx, device, model_name="model"):
    """
    Apply pruning to the model.
    
    IMPORTANT: This prunes whatever model is passed in!
    - After KD: prunes the STUDENT
    - Without KD: prunes the TEACHER
    
    The model_name parameter is just for logging.
    """
    print("\n" + "="*70)
    print("✂️  PHASE 3: PRUNING")
    print("="*70)
    print(f"   Target model: {model_name}")
    print(f"   Method: {config.prune_method}")
    print(f"   Target sparsity: {config.prune_sparsity*100:.0f}%")
    print(f"   Fine-tune after: {config.fine_tune_after_prune}")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        tokenized_data, train_idx, val_idx,
        batch_size=config.batch, num_workers=2
    )
    
    # Measure F1 before pruning
    f1_before = _evaluate_f1(model, val_loader, device)
    print(f"\n   F1 before pruning: {f1_before:.4f}")
    
    # Get the appropriate pruner
    if config.prune_method == 'magnitude':
        pruner = PruningManager(
            model=model,
            target_sparsity=config.prune_sparsity,
            prune_layers=config.prune_layers,
            global_pruning=True
        )
        pruner.apply_magnitude_pruning()
        
    elif config.prune_method == 'gradual':
        pruner = GradualPruner(
            model=model,
            target_sparsity=config.prune_sparsity,
            start_epoch=config.prune_start_epoch,
            end_epoch=config.prune_end_epoch,
            schedule=config.prune_schedule,
            prune_frequency=config.prune_frequency,
            prune_layers=config.prune_layers
        )
        _run_gradual_pruning_training(
            model, pruner, train_loader, config, device
        )
        
    elif config.prune_method == 'wanda':
        pruner = WandaPruner(
            model=model,
            target_sparsity=config.prune_sparsity,
            prune_layers=config.prune_layers
        )
        # Collect activation statistics
        pruner.collect_activations(train_loader, device, num_samples=config.calib_samples)
        pruner.apply_wanda_pruning()
        
    else:
        raise ValueError(f"Unknown pruning method: {config.prune_method}")
    
    # Make pruning permanent (remove masks, keep zeros)
    if hasattr(pruner, 'make_pruning_permanent'):
        pruner.make_pruning_permanent()
    
    # Measure F1 after pruning (before fine-tuning)
    f1_after_prune = _evaluate_f1(model, val_loader, device)
    print(f"\n   F1 after pruning: {f1_after_prune:.4f}")
    print(f"   F1 drop: {(f1_before - f1_after_prune)*100:.2f}%")
    
    # Fine-tune to recover accuracy (DEFAULT IS NOW TRUE!)
    if config.fine_tune_after_prune:
        print(f"\n   🔧 Fine-tuning for {config.fine_tune_epochs} epochs...")
        
        model = fine_tune_after_pruning(
            model, train_loader, val_loader, config, device
        )
        
        f1_after_finetune = _evaluate_f1(model, val_loader, device)
        print(f"\n   F1 after fine-tuning: {f1_after_finetune:.4f}")
        print(f"   Recovered: {(f1_after_finetune - f1_after_prune)*100:.2f}%")
    
    # Report final sparsity
    sparsity_info = pruner.get_sparsity()
    print(f"\n   ✅ Pruning complete!")
    print(f"      Final sparsity: {sparsity_info['overall']*100:.2f}%")
    
    return model
```

### Understanding the Pruning Flow:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ PRUNING PHASE DETAILED FLOW                                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ INPUT: Model (could be teacher OR student)                                 │
│                                                                             │
│ ┌─────────────────────────────────────────────────────────────────────────┐│
│ │ STEP 1: Measure baseline F1                                             ││
│ │         F1 = 0.70                                                       ││
│ └─────────────────────────────────────────────────────────────────────────┘│
│                           │                                                 │
│                           ▼                                                 │
│ ┌─────────────────────────────────────────────────────────────────────────┐│
│ │ STEP 2: Apply pruning (based on method)                                 ││
│ │                                                                         ││
│ │   magnitude: Remove smallest |weight| values                            ││
│ │   wanda: Collect activations, then prune low |w|×|a|                   ││
│ │   gradual: Train while slowly increasing sparsity                       ││
│ │                                                                         ││
│ │   Result: 50% of weights are now zero                                   ││
│ └─────────────────────────────────────────────────────────────────────────┘│
│                           │                                                 │
│                           ▼                                                 │
│ ┌─────────────────────────────────────────────────────────────────────────┐│
│ │ STEP 3: Measure F1 after pruning                                        ││
│ │         F1 = 0.58 (dropped 12%!)                                        ││
│ └─────────────────────────────────────────────────────────────────────────┘│
│                           │                                                 │
│                           ▼                                                 │
│ ┌─────────────────────────────────────────────────────────────────────────┐│
│ │ STEP 4: Fine-tune (if enabled, DEFAULT=TRUE)                            ││
│ │                                                                         ││
│ │   Train for 3 epochs with:                                              ││
│ │   - Lower learning rate (0.1× original)                                 ││
│ │   - Zeroed gradients for pruned weights                                 ││
│ │                                                                         ││
│ │   Remaining weights adjust to compensate                                ││
│ └─────────────────────────────────────────────────────────────────────────┘│
│                           │                                                 │
│                           ▼                                                 │
│ ┌─────────────────────────────────────────────────────────────────────────┐│
│ │ STEP 5: Measure F1 after fine-tuning                                    ││
│ │         F1 = 0.67 (recovered 9% of the 12% drop!)                       ││
│ └─────────────────────────────────────────────────────────────────────────┘│
│                                                                             │
│ OUTPUT: Pruned model with 50% sparsity, F1 drop of only 3%                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Section 8: Quantization Phase

```python
def run_quantization(config, model, tokenized_data, train_idx, val_idx, device):
    """
    Apply quantization to the model.
    
    IMPORTANT DEVICE CONSIDERATIONS:
    ─────────────────────────────────
    - dynamic/static (INT8): CPU only!
    - fp16: GPU only!
    - int4: GPU only (requires bitsandbytes)
    
    The function returns both the quantized model AND the device it runs on.
    """
    print("\n" + "="*70)
    print("📉 PHASE 4: QUANTIZATION")
    print("="*70)
    print(f"   Method: {config.quant_method}")
    
    # Create data loader for calibration (if needed)
    train_loader, val_loader = create_data_loaders(
        tokenized_data, train_idx, val_idx,
        batch_size=config.batch, num_workers=2
    )
    
    # Apply quantization based on method
    if config.quant_method == 'dynamic':
        print("   ⚠️  Dynamic INT8 quantization runs on CPU only")
        quant_device = 'cpu'
        
        model_cpu = model.cpu()
        quantized_model = torch.quantization.quantize_dynamic(
            model_cpu,
            {nn.Linear},
            dtype=torch.qint8
        )
        
    elif config.quant_method == 'static':
        print("   ⚠️  Static INT8 quantization runs on CPU only")
        quant_device = 'cpu'
        
        manager = QuantizationManager(model, method='static')
        manager.prepare_static_quantization()
        manager.calibrate(train_loader, device='cpu', 
                         num_batches=config.quant_calibration_batches)
        quantized_model = manager.convert_static_quantization()
        
    elif config.quant_method == 'fp16':
        print("   ✅ FP16 quantization works on GPU!")
        quant_device = device
        
        quantized_model = model.half().to(device)
        
    elif config.quant_method == 'int4':
        print("   Applying INT4 quantization (requires bitsandbytes)...")
        quant_device = device
        
        try:
            quantized_model = _apply_int4_quantization(model, device)
        except ImportError:
            print("   ❌ bitsandbytes not installed, falling back to dynamic INT8")
            quant_device = 'cpu'
            quantized_model = torch.quantization.quantize_dynamic(
                model.cpu(), {nn.Linear}, dtype=torch.qint8
            )
    
    else:
        raise ValueError(f"Unknown quantization method: {config.quant_method}")
    
    # Compare sizes
    original_size = _get_model_size(model)
    quantized_size = _get_model_size(quantized_model)
    compression = original_size / quantized_size if quantized_size > 0 else 0
    
    print(f"\n   ✅ Quantization complete!")
    print(f"      Original size: {original_size:.1f} MB")
    print(f"      Quantized size: {quantized_size:.1f} MB")
    print(f"      Compression: {compression:.2f}×")
    
    return quantized_model, quant_device


def _get_model_size(model):
    """Calculate model size in MB."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)
```

---

## Section 9: The Main Pipeline Orchestrator

This is where everything comes together!

```python
def run_compression_pipeline(config):
    """
    Run the complete compression pipeline.
    
    This is the MAIN FUNCTION that orchestrates:
    1. Data loading with caching
    2. Teacher model loading/training
    3. Knowledge distillation (if enabled)
    4. Pruning (if enabled)
    5. Quantization (if enabled)
    6. Evaluation at each stage
    7. Saving results
    
    The pipeline configuration determines which stages run.
    """
    print("\n" + "="*70)
    print("🚀 COMPRESSION PIPELINE")
    print("="*70)
    print(f"   Pipeline: {config.pipeline}")
    print(f"   KD: {'✅ Enabled' if config.enable_kd else '❌ Disabled'}")
    print(f"   Pruning: {'✅ Enabled' if config.enable_pruning else '❌ Disabled'}")
    print(f"   Quantization: {'✅ Enabled' if config.enable_quantization else '❌ Disabled'}")
    
    # Setup
    set_seed(config.seed)
    device = get_device()
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.cache_dir, exist_ok=True)
    
    # =========================================================================
    # DATA LOADING (with caching!)
    # =========================================================================
    print("\n📂 Loading data...")
    
    comments, labels = load_and_preprocess_data(config.dataset_path)
    
    tokenizer = AutoTokenizer.from_pretrained(config.teacher_path)
    
    # This uses caching - fast on subsequent runs!
    tokenized_data = get_or_create_tokenized_dataset(
        comments, labels, tokenizer, config.max_length, config.cache_dir
    )
    
    # Prepare K-fold splits
    splits = list(prepare_kfold_splits(
        comments, labels, config.num_folds,
        stratification_type='multiclass', seed=config.seed
    ))
    
    # Use first fold for experiments (can use all folds for final paper)
    train_idx, val_idx = splits[0]
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        tokenized_data, train_idx, val_idx,
        batch_size=config.batch, num_workers=2
    )
    
    # =========================================================================
    # INITIALIZE EVALUATOR AND METRICS STORAGE
    # =========================================================================
    evaluator = CompressionEvaluator()
    all_metrics = []
    
    # =========================================================================
    # PHASE 1: TEACHER
    # =========================================================================
    teacher, tokenizer = get_or_train_teacher(
        config, tokenized_data, train_idx, val_idx, device
    )
    
    # Evaluate and save baseline metrics
    baseline_metrics = evaluator.evaluate_model(
        teacher, val_loader, device, stage='baseline'
    )
    baseline_metrics.print_summary()
    all_metrics.append(baseline_metrics)
    save_stage_metrics(baseline_metrics, config.output_dir, 'baseline')
    
    # If baseline-only pipeline, we're done
    if config.pipeline == 'baseline':
        print("\n✅ Baseline evaluation complete!")
        save_model_for_huggingface(teacher, 
                                   os.path.join(config.output_dir, 'model_hf'),
                                   tokenizer)
        return all_metrics
    
    # Track current model (will change as we apply compression)
    current_model = teacher
    current_model_name = "teacher"
    
    # =========================================================================
    # PHASE 2: KNOWLEDGE DISTILLATION (if enabled)
    # =========================================================================
    if config.enable_kd:
        student = run_knowledge_distillation(
            config, teacher, tokenized_data, train_idx, val_idx, device
        )
        
        # Evaluate student
        kd_metrics = evaluator.evaluate_model(
            student, val_loader, device, stage='after_kd'
        )
        kd_metrics.print_summary()
        all_metrics.append(kd_metrics)
        save_stage_metrics(kd_metrics, config.output_dir, 'after_kd')
        
        # IMPORTANT: Now the student becomes our current model!
        current_model = student
        current_model_name = "student"
        
        if config.save_all_stages:
            save_model_for_huggingface(
                student,
                os.path.join(config.output_dir, 'model_after_kd_hf'),
                tokenizer
            )
    
    # =========================================================================
    # PHASE 3: PRUNING (if enabled)
    # =========================================================================
    if config.enable_pruning:
        print(f"\n   📌 Pruning will be applied to: {current_model_name}")
        
        pruned_model = run_pruning(
            config, current_model, tokenized_data, train_idx, val_idx, device,
            model_name=current_model_name
        )
        
        # Evaluate pruned model
        prune_metrics = evaluator.evaluate_model(
            pruned_model, val_loader, device, stage='after_pruning'
        )
        prune_metrics.print_summary()
        all_metrics.append(prune_metrics)
        save_stage_metrics(prune_metrics, config.output_dir, 'after_pruning')
        
        current_model = pruned_model
        current_model_name = f"pruned_{current_model_name}"
        
        if config.save_all_stages:
            save_model_for_huggingface(
                pruned_model,
                os.path.join(config.output_dir, 'model_after_pruning_hf'),
                tokenizer
            )
    
    # =========================================================================
    # PHASE 4: QUANTIZATION (if enabled)
    # =========================================================================
    if config.enable_quantization:
        quantized_model, quant_device = run_quantization(
            config, current_model, tokenized_data, train_idx, val_idx, device
        )
        
        # Need new dataloader for potentially different device
        quant_val_loader = create_data_loaders(
            tokenized_data, train_idx, val_idx,
            batch_size=config.batch, num_workers=0  # 0 workers for CPU
        )[1]
        
        # Evaluate quantized model
        quant_metrics = evaluator.evaluate_model(
            quantized_model, quant_val_loader, quant_device, stage='after_quantization'
        )
        quant_metrics.print_summary()
        all_metrics.append(quant_metrics)
        save_stage_metrics(quant_metrics, config.output_dir, 'after_quantization')
        
        current_model = quantized_model
    
    # =========================================================================
    # FINAL SUMMARY AND EXPORT
    # =========================================================================
    print("\n" + "="*70)
    print("📊 FINAL RESULTS SUMMARY")
    print("="*70)
    
    # Create comparison table
    comparison_df = compare_stages(all_metrics)
    print(comparison_df.to_string(index=False))
    
    # Save final results
    export_metrics_to_csv(all_metrics, os.path.join(config.output_dir, 'results_final.csv'))
    export_metrics_to_json(all_metrics, os.path.join(config.output_dir, 'results_final.json'))
    
    # Print final summary
    final_metrics = all_metrics[-1]
    print(f"\n🎯 COMPRESSION SUMMARY:")
    print(f"   Original F1: {baseline_metrics.f1_macro:.4f}")
    print(f"   Final F1: {final_metrics.f1_macro:.4f}")
    print(f"   F1 Retention: {(final_metrics.f1_macro / baseline_metrics.f1_macro)*100:.1f}%")
    print(f"")
    print(f"   Original Size: {baseline_metrics.model_size_mb:.1f} MB")
    print(f"   Final Size: {final_metrics.model_size_mb:.1f} MB")
    print(f"   Compression: {final_metrics.size_compression_ratio:.2f}×")
    
    return all_metrics
```

### Visual: Pipeline Flow for Different Configurations

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ PIPELINE FLOW FOR DIFFERENT CONFIGURATIONS                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ pipeline=baseline:                                                         │
│                                                                             │
│   Teacher ──▶ Evaluate ──▶ Done                                            │
│                                                                             │
│ ═══════════════════════════════════════════════════════════════════════════│
│                                                                             │
│ pipeline=kd_only:                                                          │
│                                                                             │
│   Teacher ──▶ KD ──▶ Student ──▶ Evaluate ──▶ Done                        │
│               │                                                             │
│               └── current_model = Student                                  │
│                                                                             │
│ ═══════════════════════════════════════════════════════════════════════════│
│                                                                             │
│ pipeline=prune_only:                                                       │
│                                                                             │
│   Teacher ──▶ Prune Teacher ──▶ Fine-tune ──▶ Evaluate ──▶ Done           │
│               │                                                             │
│               └── current_model = Pruned Teacher                           │
│                                                                             │
│ ═══════════════════════════════════════════════════════════════════════════│
│                                                                             │
│ pipeline=kd_prune_quant (FULL):                                            │
│                                                                             │
│   Teacher                                                                   │
│       │                                                                     │
│       ▼                                                                     │
│   KD ──▶ Student (current_model = Student)                                │
│              │                                                              │
│              ▼                                                              │
│          Prune STUDENT ──▶ Fine-tune (current_model = Pruned Student)     │
│                                │                                            │
│                                ▼                                            │
│                            Quantize (current_model = Quantized)            │
│                                │                                            │
│                                ▼                                            │
│                            Evaluate ──▶ Done                               │
│                                                                             │
│   Final model: Quantized Pruned Student                                    │
│   Compression: 8-10× smaller than original teacher!                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Section 10: Ablation Study Runner

```python
def run_ablation_study(config):
    """
    Run all 8 pipeline configurations for comprehensive ablation.
    
    WHAT IS AN ABLATION STUDY?
    ──────────────────────────
    Systematically testing each component to understand its contribution.
    
    By running all 8 pipelines, you can answer:
    - How much does KD alone help?
    - How much does pruning alone help?
    - Do KD + pruning combine well or conflict?
    - Which combination gives best trade-off?
    """
    print("\n" + "="*70)
    print("🔬 ABLATION STUDY")
    print("="*70)
    print(f"   Running {len(config.ablation_pipelines)} pipeline configurations")
    
    all_results = {}
    
    for pipeline in config.ablation_pipelines:
        print(f"\n{'='*70}")
        print(f"📊 ABLATION: {pipeline.upper()}")
        print(f"{'='*70}")
        
        # Create configuration for this pipeline
        pipeline_config = get_config_for_pipeline(
            pipeline,
            dataset_path=config.dataset_path,
            author_name=config.author_name,
            teacher_checkpoint=config.teacher_checkpoint,
            output_dir=os.path.join(config.output_dir, f'ablation_{pipeline}'),
            cache_dir=config.cache_dir,  # Share cache across all runs!
            kd_method=config.kd_method,
            prune_method=config.prune_method,
            prune_sparsity=config.prune_sparsity,
            quant_method=config.quant_method,
            fine_tune_after_prune=True
        )
        
        # Run this pipeline
        metrics = run_compression_pipeline(pipeline_config)
        all_results[pipeline] = metrics[-1]  # Store final metrics
    
    # =========================================================================
    # GENERATE ABLATION SUMMARY TABLE
    # =========================================================================
    print("\n" + "="*70)
    print("📊 ABLATION STUDY RESULTS")
    print("="*70)
    
    import pandas as pd
    rows = []
    
    for pipeline, metrics in all_results.items():
        rows.append({
            'Pipeline': pipeline,
            'F1 Macro': f"{metrics.f1_macro:.4f}",
            'F1 Weighted': f"{metrics.f1_weighted:.4f}",
            'Threat F1': f"{metrics.threat_f1:.4f}",
            'Size (MB)': f"{metrics.model_size_mb:.1f}",
            'Compression': f"{metrics.size_compression_ratio:.2f}×",
            'Sparsity': f"{metrics.sparsity_percent:.1f}%",
            'Latency P95': f"{metrics.latency_p95_ms:.1f} ms"
        })
    
    ablation_df = pd.DataFrame(rows)
    print(ablation_df.to_string(index=False))
    
    # Save summary
    ablation_df.to_csv(
        os.path.join(config.output_dir, 'ablation_summary.csv'),
        index=False
    )
    
    # =========================================================================
    # FIND BEST CONFIGURATIONS
    # =========================================================================
    print("\n🏆 BEST CONFIGURATIONS:")
    
    # Best accuracy
    best_accuracy = max(all_results.items(), key=lambda x: x[1].f1_macro)
    print(f"   Best F1: {best_accuracy[0]} ({best_accuracy[1].f1_macro:.4f})")
    
    # Best compression (among those with F1 > 90% of baseline)
    baseline_f1 = all_results.get('baseline', all_results[list(all_results.keys())[0]]).f1_macro
    acceptable = {k: v for k, v in all_results.items() if v.f1_macro >= 0.9 * baseline_f1}
    if acceptable:
        best_compression = max(acceptable.items(), key=lambda x: x[1].size_compression_ratio)
        print(f"   Best Compression (F1≥90%): {best_compression[0]} ({best_compression[1].size_compression_ratio:.2f}×)")
    
    # Best threat F1 (safety-critical)
    best_threat = max(all_results.items(), key=lambda x: x[1].threat_f1)
    print(f"   Best Threat F1: {best_threat[0]} ({best_threat[1].threat_f1:.4f})")
    
    return all_results
```

### Ablation Study Output Example:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ ABLATION STUDY RESULTS                                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ Pipeline        │ F1 Macro │ Threat F1 │ Size (MB) │ Compression │ Sparsity│
│ ────────────────┼──────────┼───────────┼───────────┼─────────────┼──────── │
│ baseline        │  0.7200  │   0.6500  │   420.0   │    1.00×    │   0.0%  │
│ kd_only         │  0.6980  │   0.6300  │   250.0   │    1.68×    │   0.0%  │
│ prune_only      │  0.7050  │   0.6400  │   320.0   │    1.31×    │  50.0%  │
│ quant_only      │  0.7150  │   0.6450  │   105.0   │    4.00×    │   0.0%  │
│ kd_prune        │  0.6750  │   0.6100  │   200.0   │    2.10×    │  50.0%  │
│ kd_quant        │  0.6900  │   0.6250  │    63.0   │    6.67×    │   0.0%  │
│ prune_quant     │  0.6950  │   0.6350  │    81.0   │    5.19×    │  50.0%  │
│ kd_prune_quant  │  0.6600  │   0.5900  │    50.0   │    8.40×    │  50.0%  │
│                                                                             │
│ 🏆 BEST CONFIGURATIONS:                                                     │
│    Best F1: baseline (0.7200)                                              │
│    Best Compression (F1≥90%): kd_quant (6.67×)                             │
│    Best Threat F1: baseline (0.6500)                                       │
│                                                                             │
│ INSIGHTS:                                                                   │
│ - quantization alone gives 4× compression with only 0.5% F1 drop           │
│ - kd_quant is the sweet spot: 6.67× compression, 2% F1 drop                │
│ - Full pipeline (kd_prune_quant) is too aggressive for threat detection    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Section 11: Entry Point

```python
def main():
    """
    Entry point for the compression framework.
    
    Parses command-line arguments and runs the appropriate pipeline.
    """
    # Parse arguments
    config = parse_compression_arguments()
    
    # Print configuration summary
    print_compression_config(config)
    
    # Run ablation study or single pipeline
    if config.run_ablation:
        return run_ablation_study(config)
    else:
        return run_compression_pipeline(config)


if __name__ == "__main__":
    main()
```

---

## Complete Usage Examples

### Basic Usage:

```bash
# 1. Evaluate baseline only (no compression)
python research_main.py \
    --dataset_path data.csv \
    --author_name "your_name" \
    --pipeline baseline \
    --teacher_checkpoint "your-username/your-finetuned-model"

# 2. Knowledge distillation only
python research_main.py \
    --dataset_path data.csv \
    --author_name "your_name" \
    --pipeline kd_only \
    --teacher_checkpoint "your-username/your-finetuned-model" \
    --kd_method logit \
    --kd_alpha 0.7 \
    --kd_temperature 4.0

# 3. Full compression pipeline
python research_main.py \
    --dataset_path data.csv \
    --author_name "your_name" \
    --pipeline kd_prune_quant \
    --teacher_checkpoint "your-username/your-finetuned-model" \
    --kd_method multi_level \
    --prune_method wanda \
    --prune_sparsity 0.5 \
    --quant_method fp16

# 4. Full ablation study
python research_main.py \
    --dataset_path data.csv \
    --author_name "your_name" \
    --run_ablation \
    --teacher_checkpoint "your-username/your-finetuned-model"
```

### Research Experiments:

```bash
# Experiment 1: KD method comparison
for method in logit hidden attention multi_level; do
    python research_main.py \
        --pipeline kd_only \
        --kd_method $method \
        --output_dir results/kd_method_$method \
        --dataset_path data.csv \
        --author_name "experiment" \
        --teacher_checkpoint "your-model"
done

# Experiment 2: Sparsity level study
for sparsity in 0.3 0.4 0.5 0.6 0.7; do
    python research_main.py \
        --pipeline prune_only \
        --prune_method magnitude \
        --prune_sparsity $sparsity \
        --output_dir results/sparsity_$sparsity \
        --dataset_path data.csv \
        --author_name "experiment" \
        --teacher_checkpoint "your-model"
done

# Experiment 3: Quantization method comparison
for method in dynamic fp16; do
    python research_main.py \
        --pipeline quant_only \
        --quant_method $method \
        --output_dir results/quant_$method \
        --dataset_path data.csv \
        --author_name "experiment" \
        --teacher_checkpoint "your-model"
done
```

---

## Output Directory Structure

After running a full pipeline, your output directory looks like this:

```
compressed_models/
├── results_baseline.csv           # Metrics after Phase 1
├── results_baseline.json
├── results_after_kd.csv           # Metrics after Phase 2
├── results_after_kd.json
├── results_after_pruning.csv      # Metrics after Phase 3
├── results_after_pruning.json
├── results_after_quantization.csv # Metrics after Phase 4
├── results_after_quantization.json
├── results_all.csv                # Cumulative metrics
├── results_final.csv              # Final summary
├── results_final.json
├── model_hf/                      # Final model in HuggingFace format
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   ├── classifier.pt
│   ├── classifier_config.json
│   └── how_to_load.py
├── model_after_kd_hf/             # (if save_all_stages=True)
├── model_after_pruning_hf/        # (if save_all_stages=True)
└── cache/                         # Tokenization cache (shared)
    └── banglabert_maxlen128_tokenized.pkl
```

---

## Summary: Complete Framework Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ COMPLETE COMPRESSION FRAMEWORK ARCHITECTURE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                        research_main.py                                    │
│                        (Orchestrator)                                      │
│                              │                                              │
│         ┌───────────────────┼───────────────────┐                          │
│         │                   │                   │                          │
│         ▼                   ▼                   ▼                          │
│ ┌───────────────┐   ┌───────────────┐   ┌───────────────┐                 │
│ │research_      │   │research_      │   │research_      │                 │
│ │compression_   │   │data.py        │   │evaluation.py  │                 │
│ │config.py      │   │               │   │               │                 │
│ │               │   │• Load CSV     │   │• F1, Precision│                 │
│ │• Arguments    │   │• Tokenize     │   │• Per-label    │                 │
│ │• Pipelines    │   │• Cache        │   │• Efficiency   │                 │
│ │• Defaults     │   │• K-fold       │   │• Safety       │                 │
│ └───────────────┘   └───────────────┘   └───────────────┘                 │
│                                                                             │
│         ┌───────────────────┼───────────────────┐                          │
│         │                   │                   │                          │
│         ▼                   ▼                   ▼                          │
│ ┌───────────────┐   ┌───────────────┐   ┌───────────────┐                 │
│ │research_      │   │research_      │   │research_      │                 │
│ │distillation.py│   │pruning.py     │   │quantization.py│                 │
│ │               │   │               │   │               │                 │
│ │• Teacher      │   │• Magnitude    │   │• Dynamic INT8 │                 │
│ │• Student      │   │• Gradual      │   │• Static INT8  │                 │
│ │• KD Loss      │   │• Wanda        │   │• FP16         │                 │
│ │• Multi-level  │   │• Structured   │   │• INT4         │                 │
│ └───────────────┘   └───────────────┘   └───────────────┘                 │
│                                                                             │
│                              │                                              │
│                              ▼                                              │
│                    ┌───────────────────┐                                   │
│                    │    OUTPUTS        │                                   │
│                    │                   │                                   │
│                    │ • Compressed model│                                   │
│                    │ • Metrics (CSV)   │                                   │
│                    │ • Comparison table│                                   │
│                    │ • Safety report   │                                   │
│                    └───────────────────┘                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## What You Can Modify for Research

| Area | What to Modify | Research Impact |
|------|----------------|-----------------|
| **Pipeline** | Add new pipeline configurations | Test new compression combinations |
| **Teacher** | Different base models | Language/domain adaptation |
| **Student** | Different student architectures | Architecture search |
| **KD** | Loss weights, methods | Optimal knowledge transfer |
| **Pruning** | Sparsity, methods, schedules | Compression-accuracy tradeoff |
| **Quantization** | Methods, calibration size | Precision vs efficiency |
| **Evaluation** | Custom metrics, priorities | Domain-specific evaluation |

---

## Final Thoughts

Congratulations! You now have a deep understanding of the entire compression framework. Here's a quick recap of what each script does:

1. **research_compression_config.py**: Configuration and arguments
2. **research_data.py**: Data loading and tokenization caching
3. **research_distillation.py**: Knowledge distillation (teacher→student)
4. **research_pruning.py**: Weight pruning (magnitude, gradual, wanda)
5. **research_quantization.py**: Precision reduction (INT8, FP16, INT4)
6. **research_evaluation.py**: Comprehensive metrics and safety checks
7. **research_main.py**: Orchestration of the complete pipeline

With this knowledge, you can confidently modify any part of the framework for your research needs. Good luck with your PhD research! 🎓