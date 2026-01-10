#!/usr/bin/env python3
"""
================================================================================
MAIN COMPRESSION PIPELINE (ENHANCED VERSION)
================================================================================

FIXES IN THIS VERSION:
1. ✅ HuggingFace format saving for deployment
2. ✅ Clear KD → Pruning → Quantization flow (always on student after KD)
3. ✅ Fine-tuning after pruning is now DEFAULT
4. ✅ INT4 quantization support (via bitsandbytes)
5. ✅ Metrics CSV saved after EACH stage
6. ✅ Better progress tracking

FLOW EXPLANATION:
─────────────────
Pipeline: kd_prune_quant

Step 1: Load/Train Teacher (large model, e.g., BanglaBERT)
        ↓
Step 2: Knowledge Distillation
        - Create Student (smaller model, e.g., DistilBERT)
        - Train Student to mimic Teacher
        - current_model = Student
        ↓
Step 3: Pruning (applied to Student, NOT teacher!)
        - Remove 50% of Student's weights
        - Fine-tune to recover accuracy
        - current_model = Pruned Student
        ↓
Step 4: Quantization (applied to Pruned Student)
        - Reduce precision (FP32 → INT8 or FP16)
        - current_model = Quantized Pruned Student
        ↓
Step 5: Final Evaluation & Save in HuggingFace format
"""

# =============================================================================
# ENVIRONMENT SETUP (MUST BE FIRST!)
# =============================================================================
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import gc
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

# Local imports (removed research_ prefix)
from compression_config import (
    parse_compression_arguments, print_compression_config,
    PIPELINE_CONFIGS, LABEL_COLUMNS, get_config_for_pipeline
)
from data import (
    load_and_preprocess_data, get_or_create_tokenized_dataset,
    prepare_kfold_splits, calculate_class_weights,
    create_data_loaders, IndexedDataset
)
from distillation import (
    TeacherModel, StudentModel, DistillationTrainer,
    MultiLabelDistillationLoss, verify_teacher_performance
)
from pruning import (
    PruningManager, GradualPruner, WandaPruner,
    get_pruner, fine_tune_after_pruning
)
from quantization import (
    QuantizationManager, quantize_model, benchmark_inference_speed,
    apply_int4_quantization
)
from evaluation import (
    CompressionEvaluator, CompressionStageMetrics,
    compare_stages, export_metrics_to_csv, export_metrics_to_json,
    calculate_aggregate_metrics, print_aggregated_metrics
)
from logging_utils import (
    setup_logging, get_logger, log_config, log_dataset_info,
    log_model_info, log_stage_start, log_stage_end, log_metrics,
    log_memory_usage, log_gpu_info, log_error, log_final_summary
)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"[Device] Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = 'cpu'
        print("[Device] Using CPU (GPU not available)")
    return device


def cleanup_memory(device):
    """Clean up memory between stages."""
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# =============================================================================
# HUGGINGFACE FORMAT SAVING (NEW!)
# =============================================================================

class HuggingFaceModelWrapper(nn.Module):
    """
    Wrapper to save models in HuggingFace format for easy deployment.
    
    This creates a model that can be:
    1. Pushed to HuggingFace Hub
    2. Loaded with AutoModel.from_pretrained()
    3. Used with transformers pipeline
    """
    
    def __init__(self, encoder, classifier, config, num_labels=5):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.config = config
        self.num_labels = num_labels
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        
        loss = None
        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels.float())
        
        return {'loss': loss, 'logits': logits}


def save_model_for_huggingface(model, save_path, tokenizer=None):
    """
    Save model in HuggingFace format for deployment.
    
    Creates a folder with:
    - config.json (model configuration)
    - pytorch_model.bin (weights)
    - tokenizer files (if tokenizer provided)
    - classifier_config.json (classifier head info)
    
    After saving, you can:
    1. Push to Hub: `huggingface-cli upload ./model your-username/model-name`
    2. Load: `model = AutoModel.from_pretrained('./model')`
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Save encoder
    if hasattr(model, 'encoder'):
        model.encoder.save_pretrained(save_path)
    
    # Save classifier separately
    if hasattr(model, 'classifier'):
        classifier_path = os.path.join(save_path, 'classifier.pt')
        torch.save(model.classifier.state_dict(), classifier_path)
        
        # Save classifier config
        classifier_config = {
            'type': 'sequential',
            'layers': str(model.classifier),
            'num_labels': model.num_labels if hasattr(model, 'num_labels') else 5
        }
        with open(os.path.join(save_path, 'classifier_config.json'), 'w') as f:
            json.dump(classifier_config, f, indent=2)
    
    # Save tokenizer
    if tokenizer is not None:
        tokenizer.save_pretrained(save_path)
    
    # Create a loading script
    loading_script = '''
# How to load this model:

from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn

# Load encoder
encoder = AutoModel.from_pretrained("{save_path}")

# Load classifier
classifier = nn.Sequential(
    nn.Linear(encoder.config.hidden_size, 256),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(256, 5)  # 5 labels
)
classifier.load_state_dict(torch.load("{save_path}/classifier.pt"))

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("{save_path}")

# Inference
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = encoder(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        logits = classifier(cls_embedding)
        probs = torch.sigmoid(logits)
    return probs

# Example
probs = predict("আপনার বাংলা টেক্সট এখানে")
print(probs)
'''.format(save_path=save_path)
    
    with open(os.path.join(save_path, 'how_to_load.py'), 'w') as f:
        f.write(loading_script)
    
    print(f"[OK] Model saved in HuggingFace format: {save_path}")
    print(f"   Files created:")
    for f in os.listdir(save_path):
        print(f"      - {f}")


def push_to_huggingface_hub(model, tokenizer, repo_name, token=None):
    """
    Push model directly to HuggingFace Hub.
    
    Args:
        model: Model to push
        tokenizer: Tokenizer to push
        repo_name: "your-username/model-name"
        token: HuggingFace API token (or set HF_TOKEN env var)
    """
    from huggingface_hub import HfApi, create_repo
    
    # Create temporary directory
    temp_dir = './temp_hf_upload'
    save_model_for_huggingface(model, temp_dir, tokenizer)
    
    # Create repo if doesn't exist
    api = HfApi()
    try:
        create_repo(repo_name, token=token, exist_ok=True)
    except Exception as e:
        print(f"Note: {e}")
    
    # Upload
    api.upload_folder(
        folder_path=temp_dir,
        repo_id=repo_name,
        token=token
    )
    
    # Cleanup
    shutil.rmtree(temp_dir)
    
    print(f"[OK] Model pushed to: https://huggingface.co/{repo_name}")


# =============================================================================
# INT4 QUANTIZATION HELPER (uses quantization module)
# =============================================================================

def apply_int4_quantization_local(model, device='cuda'):
    """
    Apply INT4 quantization using bitsandbytes library.

    WHAT: Reduces weights to 4-bit precision (vs 32-bit original)
    WHY: 8x compression with minimal accuracy loss
    HOW: Uses bitsandbytes library for 4-bit quantization

    REQUIREMENTS:
        pip install bitsandbytes
        GPU with CUDA support

    NOTE: INT4 is primarily used for inference, not training.
    """
    try:
        import bitsandbytes as bnb
        from transformers import BitsAndBytesConfig
    except ImportError:
        print("[ERROR] bitsandbytes not installed!")
        print("   Install: pip install bitsandbytes")
        print("   Falling back to INT8...")
        return apply_int8_quantization_local(model)

    print("\n   Applying INT4 quantization (bitsandbytes)...")

    # Create quantization config
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,  # Nested quantization for more compression
        bnb_4bit_quant_type="nf4"  # Normalized float 4-bit
    )

    # For existing model, we need to convert Linear layers
    def replace_linear_with_4bit(model):
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
                # Copy weights
                new_layer.weight = bnb.nn.Params4bit(
                    module.weight.data,
                    requires_grad=False,
                    quant_type="nf4"
                )
                if module.bias is not None:
                    new_layer.bias = nn.Parameter(module.bias.data)
                setattr(model, name, new_layer)
            else:
                replace_linear_with_4bit(module)
        return model

    model = replace_linear_with_4bit(model)
    model = model.to(device)

    print("   [OK] INT4 quantization applied")
    return model


def apply_int8_quantization_local(model):
    """Apply INT8 dynamic quantization (fallback for INT4)."""
    print("\n   Applying INT8 dynamic quantization...")

    model_cpu = model.cpu()
    quantized = torch.quantization.quantize_dynamic(
        model_cpu,
        {nn.Linear},
        dtype=torch.qint8
    )

    print("   [OK] INT8 quantization applied")
    return quantized


# =============================================================================
# METRICS SAVING (ENHANCED)
# =============================================================================

def save_stage_metrics(metrics: CompressionStageMetrics, output_dir: str, stage_name: str):
    """
    Save metrics immediately after each stage (not just at the end).
    
    Creates:
    - results_{stage_name}.csv
    - results_{stage_name}.json
    - results_all.csv (appends)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save individual stage
    stage_csv = os.path.join(output_dir, f'results_{stage_name}.csv')
    stage_json = os.path.join(output_dir, f'results_{stage_name}.json')
    
    import pandas as pd
    df = pd.DataFrame([metrics.to_flat_dict()])
    df.to_csv(stage_csv, index=False)
    
    with open(stage_json, 'w') as f:
        json.dump(metrics.to_dict(), f, indent=2, default=str)
    
    # Append to cumulative results
    all_csv = os.path.join(output_dir, 'results_all.csv')
    if os.path.exists(all_csv):
        df.to_csv(all_csv, mode='a', header=False, index=False)
    else:
        df.to_csv(all_csv, index=False)
    
    print(f"   [Metrics] Saved: {stage_csv}")


# =============================================================================
# TEACHER LOADING/TRAINING
# =============================================================================

def get_or_train_teacher(config, tokenized_data, train_idx, val_idx, device, logger=None):
    """
    Get teacher model - either load from checkpoint or train from scratch.

    Args:
        config: Configuration object
        tokenized_data: Tokenized dataset
        train_idx: Training indices
        val_idx: Validation indices
        device: Device to use
        logger: Optional logger instance

    Returns:
        Tuple of (teacher model, tokenizer, training_metrics dict)
    """
    print("\n" + "="*70)
    print("[PHASE 1] TEACHER MODEL")
    print("="*70)

    num_labels = len(config.label_columns)
    training_metrics = {}

    if config.teacher_checkpoint:
        print(f"\n[Loading] Pre-trained teacher from: {config.teacher_checkpoint}")

        # IMPORTANT: Use tokenizer from checkpoint, not teacher_path!
        tokenizer = AutoTokenizer.from_pretrained(config.teacher_checkpoint)
        print(f"   [Tokenizer] Loaded from: {config.teacher_checkpoint}")

        # Try to load using custom format (encoder/ + classifier_head.pt)
        # This matches the format used by TransformerMultiLabelClassifier
        try:
            from huggingface_hub import hf_hub_download, snapshot_download
            import tempfile

            print("   [Format] Detected custom format (encoder/ + classifier_head.pt)")

            # Download the entire repo to get the structure
            local_dir = snapshot_download(
                repo_id=config.teacher_checkpoint,
                allow_patterns=["encoder/*", "classifier_*.json", "classifier_head.pt", "config.json"]
            )
            print(f"   [Download] Files cached to: {local_dir}")

            # Check for encoder subfolder
            encoder_path = os.path.join(local_dir, 'encoder')
            if os.path.exists(encoder_path):
                print(f"   [Encoder] Loading from: {encoder_path}")
                encoder_model_path = encoder_path
            else:
                # Fallback: encoder might be at root level
                encoder_model_path = local_dir
                print(f"   [Encoder] Loading from root: {local_dir}")

            # Create TeacherModel with encoder
            teacher = TeacherModel(
                model_name=encoder_model_path,
                num_labels=num_labels,
                dropout=config.dropout
            )
            print(f"   [Encoder] Loaded successfully!")

            # Load classifier weights from classifier_head.pt
            classifier_path = os.path.join(local_dir, 'classifier_head.pt')
            if not os.path.exists(classifier_path):
                # Try alternative name
                classifier_path = os.path.join(local_dir, 'classifier.pt')

            if os.path.exists(classifier_path):
                classifier_state = torch.load(classifier_path, map_location='cpu')
                teacher.classifier.load_state_dict(classifier_state)
                print(f"   [Classifier] Loaded from: {os.path.basename(classifier_path)}")
            else:
                print(f"   [Classifier] No saved weights found, using fresh classifier")

            # Load classifier config if available
            config_path = os.path.join(local_dir, 'classifier_config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    classifier_config = json.load(f)
                print(f"   [Config] Classifier config: {classifier_config}")

            teacher = teacher.to(device)

        except Exception as e:
            print(f"   [Warning] Custom format loading failed: {e}")
            print("   [Fallback] Trying AutoModelForSequenceClassification...")

            try:
                from transformers import AutoModelForSequenceClassification

                hf_model = AutoModelForSequenceClassification.from_pretrained(
                    config.teacher_checkpoint,
                    num_labels=num_labels,
                    ignore_mismatched_sizes=True
                )

                teacher = TeacherModel(
                    model_name=config.teacher_checkpoint,
                    num_labels=num_labels,
                    dropout=config.dropout
                )

                # Copy encoder weights
                if hasattr(hf_model, 'roberta'):
                    teacher.encoder.load_state_dict(hf_model.roberta.state_dict())
                    print("   [Encoder] Loaded from roberta attribute")
                elif hasattr(hf_model, 'bert'):
                    teacher.encoder.load_state_dict(hf_model.bert.state_dict())
                    print("   [Encoder] Loaded from bert attribute")

                del hf_model
                torch.cuda.empty_cache()
                teacher = teacher.to(device)

            except Exception as e2:
                print(f"   [Error] All loading methods failed: {e2}")
                print("   [Fallback] Creating fresh TeacherModel...")
                teacher = TeacherModel(
                    model_name=config.teacher_checkpoint,
                    num_labels=num_labels,
                    dropout=config.dropout
                ).to(device)

        print("   [OK] Teacher loaded successfully!")
        if logger:
            logger.info(f"Loaded pre-trained teacher from {config.teacher_checkpoint}")
        return teacher, tokenizer, training_metrics

    # If no checkpoint, use teacher_path for tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.teacher_path)

    # Train from scratch
    print(f"\n[Training] Teacher from scratch ({config.teacher_epochs} epochs)")

    teacher = TeacherModel(
        model_name=config.teacher_path,
        num_labels=num_labels,
        dropout=config.dropout
    ).to(device)

    train_loader, val_loader = create_data_loaders(
        tokenized_data, train_idx, val_idx,
        batch_size=config.batch, num_workers=2
    )

    train_labels = tokenized_data['labels'][train_idx]
    class_weights = calculate_class_weights(train_labels.numpy(), config.label_columns)

    optimizer = AdamW(teacher.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    total_steps = len(train_loader) * config.teacher_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(config.warmup_ratio * total_steps),
        num_training_steps=total_steps
    )

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights.to(device))
    best_f1 = 0
    best_epoch = 0
    epoch_history = []

    for epoch in range(config.teacher_epochs):
        teacher.train()
        total_loss = 0
        num_batches = 0

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
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = total_loss / num_batches if num_batches > 0 else 0

        # Evaluate
        teacher.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = teacher(input_ids, attention_mask)
                preds = (torch.sigmoid(outputs['logits']) > 0.5).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch['labels'].numpy())

        from sklearn.metrics import f1_score, precision_score, recall_score
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)

        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'f1_macro': f1,
            'precision_macro': precision,
            'recall_macro': recall
        }
        epoch_history.append(epoch_metrics)

        print(f"   Epoch {epoch+1}: Loss={avg_train_loss:.4f}, F1={f1:.4f}")

        if logger:
            logger.info(f"Teacher Epoch {epoch+1}: Loss={avg_train_loss:.4f}, F1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch + 1

    training_metrics = {
        'best_f1': best_f1,
        'best_epoch': best_epoch,
        'final_f1': epoch_history[-1]['f1_macro'] if epoch_history else 0,
        'final_loss': epoch_history[-1]['train_loss'] if epoch_history else 0,
        'epoch_history': epoch_history,
        'total_epochs': config.teacher_epochs
    }

    print(f"\n   [OK] Teacher training complete! Best F1: {best_f1:.4f} (Epoch {best_epoch})")
    return teacher, tokenizer, training_metrics


# =============================================================================
# KNOWLEDGE DISTILLATION
# =============================================================================

def run_knowledge_distillation(config, teacher, tokenized_data, train_idx, val_idx, device, logger=None):
    """
    Run knowledge distillation from teacher to student.

    CREATES A NEW STUDENT MODEL and trains it to mimic the teacher.
    The student is SMALLER than the teacher.

    Args:
        config: Configuration object
        teacher: Teacher model
        tokenized_data: Tokenized dataset (may contain student tokens)
        train_idx: Training indices
        val_idx: Validation indices
        device: Device to use
        logger: Optional logger instance

    Returns:
        Tuple of (student model, training_metrics dict)
    """
    print("\n" + "="*70)
    print("[PHASE 2] KNOWLEDGE DISTILLATION")
    print("="*70)
    print(f"   Teacher: {config.teacher_path} (large)")
    print(f"   Student: {config.student_path} (smaller)")
    print(f"   Method: {config.kd_method}")
    print(f"   Alpha: {config.kd_alpha} (0=hard labels only, 1=soft labels only)")
    print(f"   Temperature: {config.kd_temperature}")

    # Check if we have dual tokenization (student tokens in data)
    has_student_tokens = 'student_input_ids' in tokenized_data

    train_loader, val_loader = create_data_loaders(
        tokenized_data, train_idx, val_idx,
        batch_size=config.batch, num_workers=2
    )

    # Verify teacher
    is_valid, teacher_f1 = verify_teacher_performance(teacher, val_loader, device, min_f1=0.4)
    if not is_valid:
        print("   [WARN] Teacher F1 is low. Consider using a better teacher.")

    num_labels = len(config.label_columns)

    # CREATE NEW STUDENT MODEL (smaller than teacher!)
    student = StudentModel(
        model_name=config.student_path,
        num_labels=num_labels,
        dropout=config.dropout,
        classifier_hidden_size=config.student_hidden_size
    ).to(device)

    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())

    print(f"\n   [Model Size Comparison]:")
    print(f"      Teacher: {teacher_params/1e6:.2f}M parameters")
    print(f"      Student: {student_params/1e6:.2f}M parameters")
    print(f"      Reduction: {(1 - student_params/teacher_params)*100:.1f}%")

    if has_student_tokens:
        print(f"   [Dual Tokenization] Using separate tokenizers for teacher/student")

    # Get class weights
    train_labels = tokenized_data['labels'][train_idx]
    class_weights = calculate_class_weights(train_labels.numpy(), config.label_columns)

    # Create trainer
    trainer = DistillationTrainer(teacher, student, config, device)

    optimizer = AdamW(student.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    total_steps = len(train_loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(config.warmup_ratio * total_steps),
        num_training_steps=total_steps
    )

    best_f1 = 0
    best_epoch = 0
    patience_counter = 0
    epoch_history = []

    for epoch in range(config.epochs):
        epoch_losses = []
        soft_losses = []
        hard_losses = []

        pbar = tqdm(train_loader, desc=f"KD Epoch {epoch+1}/{config.epochs}")
        for batch in pbar:
            losses = trainer.train_step(batch, optimizer, class_weights)
            scheduler.step()
            epoch_losses.append(losses['total_loss'])
            soft_losses.append(losses.get('soft_loss', 0))
            hard_losses.append(losses.get('hard_loss', 0))
            pbar.set_postfix({
                'loss': f"{losses['total_loss']:.4f}",
                'soft': f"{losses.get('soft_loss', 0):.4f}"
            })

        avg_loss = np.mean(epoch_losses)
        avg_soft_loss = np.mean(soft_losses)
        avg_hard_loss = np.mean(hard_losses)

        # Evaluate - use student tokens if available
        eval_results = trainer.evaluate(val_loader, class_weights, use_student_input_ids=has_student_tokens)
        from sklearn.metrics import f1_score, precision_score, recall_score
        preds = (eval_results['predictions'] > 0.5).astype(int)
        f1 = f1_score(eval_results['labels'], preds, average='macro', zero_division=0)
        precision = precision_score(eval_results['labels'], preds, average='macro', zero_division=0)
        recall = recall_score(eval_results['labels'], preds, average='macro', zero_division=0)

        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': avg_loss,
            'soft_loss': avg_soft_loss,
            'hard_loss': avg_hard_loss,
            'f1_macro': f1,
            'precision_macro': precision,
            'recall_macro': recall
        }
        epoch_history.append(epoch_metrics)

        print(f"   Epoch {epoch+1}: Loss={avg_loss:.4f}, F1={f1:.4f}")

        if logger:
            logger.info(f"KD Epoch {epoch+1}: Loss={avg_loss:.4f}, F1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                print(f"   [Early stopping] at epoch {epoch+1}")
                break

    training_metrics = {
        'best_f1': best_f1,
        'best_epoch': best_epoch,
        'final_f1': epoch_history[-1]['f1_macro'] if epoch_history else 0,
        'final_loss': epoch_history[-1]['train_loss'] if epoch_history else 0,
        'epoch_history': epoch_history,
        'total_epochs': len(epoch_history),
        'teacher_f1': teacher_f1,
        'teacher_params': teacher_params,
        'student_params': student_params,
        'param_reduction': (1 - student_params/teacher_params) * 100
    }

    print(f"\n   [OK] KD complete!")
    print(f"      Student F1: {best_f1:.4f} (Epoch {best_epoch})")
    print(f"      The STUDENT model will be used for subsequent compression steps.")

    return student, training_metrics


# =============================================================================
# PRUNING (APPLIES TO CURRENT MODEL - STUDENT IF KD WAS DONE)
# =============================================================================

def run_pruning(config, model, tokenized_data, train_idx, val_idx, device, model_name="model",
                use_student_input_ids=False, logger=None):
    """
    Apply pruning to the model.

    IMPORTANT: This prunes whatever model is passed in:
    - If called after KD: prunes the STUDENT
    - If called without KD (prune_only): prunes the TEACHER

    Always fine-tunes after pruning by default (config.fine_tune_after_prune=True).

    Args:
        config: Configuration object
        model: Model to prune
        tokenized_data: Tokenized dataset
        train_idx: Training indices
        val_idx: Validation indices
        device: Device to use
        model_name: Name of the model being pruned
        use_student_input_ids: If True, use student tokens for evaluation
        logger: Optional logger instance

    Returns:
        Tuple of (pruned model, pruning_metrics dict)
    """
    print("\n" + "="*70)
    print("[PHASE 3] PRUNING")
    print("="*70)
    print(f"   Target model: {model_name}")
    print(f"   Method: {config.prune_method}")
    print(f"   Target sparsity: {config.prune_sparsity*100:.0f}%")
    print(f"   Fine-tune after: {'Yes' if config.fine_tune_after_prune else 'No'}")

    train_loader, val_loader = create_data_loaders(
        tokenized_data, train_idx, val_idx,
        batch_size=config.batch, num_workers=2
    )

    # Helper function to get input tensors based on tokenization mode
    def get_input_tensors(batch):
        if use_student_input_ids and 'student_input_ids' in batch:
            return batch['student_input_ids'].to(device), batch['student_attention_mask'].to(device)
        return batch['input_ids'].to(device), batch['attention_mask'].to(device)

    # Get initial metrics
    from sklearn.metrics import f1_score
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask = get_input_tensors(batch)
            outputs = model(input_ids, attention_mask)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch['labels'].numpy())

    f1_before = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    print(f"\n   F1 before pruning: {f1_before:.4f}")

    # Apply pruning based on method
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

        optimizer = AdamW(model.parameters(), lr=config.lr * 0.1)
        loss_fn = nn.BCEWithLogitsLoss()

        total_steps = len(train_loader) * config.prune_end_epoch
        current_step = 0

        for epoch in range(config.prune_end_epoch):
            model.train()
            for batch in tqdm(train_loader, desc=f"Gradual Prune Epoch {epoch+1}"):
                input_ids, attention_mask = get_input_tensors(batch)
                labels = batch['labels'].to(device)

                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                loss = loss_fn(logits, labels)
                loss.backward()
                optimizer.step()

                pruner.step(current_step, total_steps)
                current_step += 1

            sparsity = pruner.get_sparsity()['overall']
            print(f"   Epoch {epoch+1}: Sparsity = {sparsity*100:.1f}%")

    elif config.prune_method == 'wanda':
        pruner = WandaPruner(
            model=model,
            target_sparsity=config.prune_sparsity,
            prune_layers=config.prune_layers
        )
        pruner.collect_activations(train_loader, device, num_samples=config.calib_samples)
        pruner.apply_wanda_pruning()

    else:
        raise ValueError(f"Unknown pruning method: {config.prune_method}")

    # Make pruning permanent
    if hasattr(pruner, 'make_pruning_permanent'):
        pruner.make_pruning_permanent()

    # Get F1 after pruning (before fine-tuning)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask = get_input_tensors(batch)
            outputs = model(input_ids, attention_mask)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch['labels'].numpy())

    f1_after_prune = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    print(f"\n   F1 after pruning (before fine-tune): {f1_after_prune:.4f}")
    print(f"   F1 drop: {(f1_before - f1_after_prune)*100:.2f}%")

    fine_tune_metrics = {}

    # ALWAYS fine-tune after pruning (default behavior now)
    if config.fine_tune_after_prune:
        print(f"\n   [Fine-tuning] for {config.fine_tune_epochs} epochs to recover accuracy...")
        fine_tune_metrics = fine_tune_after_pruning(
            model, train_loader, val_loader, config, device,
            use_student_input_ids=use_student_input_ids
        )

        # Get F1 after fine-tuning
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask = get_input_tensors(batch)
                outputs = model(input_ids, attention_mask)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch['labels'].numpy())

        f1_after_finetune = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        print(f"\n   F1 after fine-tuning: {f1_after_finetune:.4f}")
        print(f"   Recovery: {(f1_after_finetune - f1_after_prune)*100:.2f}%")

    final_sparsity = pruner.get_sparsity()

    pruning_metrics = {
        'f1_before': f1_before,
        'f1_after_prune': f1_after_prune,
        'f1_drop': (f1_before - f1_after_prune) * 100,
        'final_sparsity': final_sparsity['overall'],
        'nonzero_params': final_sparsity['nonzero_params'],
        'total_params': final_sparsity['total_params'],
        'fine_tune_metrics': fine_tune_metrics
    }

    if fine_tune_metrics:
        pruning_metrics['f1_after_finetune'] = fine_tune_metrics.get('best_f1', f1_after_prune)
        pruning_metrics['recovery'] = (fine_tune_metrics.get('best_f1', f1_after_prune) - f1_after_prune) * 100

    print(f"\n   [OK] Pruning complete!")
    print(f"      Final sparsity: {final_sparsity['overall']*100:.2f}%")

    if logger:
        logger.info(f"Pruning complete: sparsity={final_sparsity['overall']*100:.2f}%")

    return model, pruning_metrics


# =============================================================================
# QUANTIZATION (APPLIES TO CURRENT MODEL)
# =============================================================================

def run_quantization(config, model, tokenized_data, train_idx, val_idx, device, logger=None):
    """
    Apply quantization to the model.

    Supports: dynamic, static, fp16, int4

    Args:
        config: Configuration object
        model: Model to quantize
        tokenized_data: Tokenized dataset
        train_idx: Training indices
        val_idx: Validation indices
        device: Device to use
        logger: Optional logger instance

    Returns:
        Tuple of (quantized model, quant_device, quantization_metrics dict)
    """
    print("\n" + "="*70)
    print("[PHASE 4] QUANTIZATION")
    print("="*70)
    print(f"   Method: {config.quant_method}")
    print(f"   Data type: {config.quant_dtype}")

    train_loader, val_loader = create_data_loaders(
        tokenized_data, train_idx, val_idx,
        batch_size=config.batch, num_workers=2
    )

    # Determine quantization method
    if config.quant_method == 'dynamic':
        print("   [NOTE] Dynamic INT8 runs on CPU only")
        quant_device = 'cpu'
        model_cpu = model.cpu()
        quantized_model = torch.quantization.quantize_dynamic(
            model_cpu, {nn.Linear}, dtype=torch.qint8
        )

    elif config.quant_method == 'static':
        print("   [NOTE] Static INT8 runs on CPU only")
        quant_device = 'cpu'
        manager = QuantizationManager(model, method='static', dtype=config.quant_dtype)
        manager.prepare_static_quantization()
        manager.calibrate(train_loader, device='cpu', num_batches=config.quant_calibration_batches)
        quantized_model = manager.convert_static_quantization()

    elif config.quant_method == 'fp16':
        print("   [OK] FP16 works on GPU!")
        quant_device = device
        quantized_model = model.half().to(device)

    elif config.quant_method == 'int4':
        print("   Applying INT4 quantization (bitsandbytes)...")
        quant_device = device
        quantized_model = apply_int4_quantization(model, device)

    else:
        raise ValueError(f"Unknown quantization method: {config.quant_method}")

    # Compare sizes
    manager = QuantizationManager(model, method=config.quant_method)
    manager.quantized_model = quantized_model
    size_info = manager.compare_sizes()

    quantization_metrics = {
        'method': config.quant_method,
        'dtype': config.quant_dtype,
        'original_size_mb': size_info['original_size_mb'],
        'quantized_size_mb': size_info['quantized_size_mb'],
        'compression_ratio': size_info['compression_ratio'],
        'size_reduction_pct': size_info['size_reduction_pct'],
        'device': quant_device
    }

    print(f"\n   [OK] Quantization complete!")
    print(f"      Compression: {size_info['compression_ratio']:.2f}x")

    if logger:
        logger.info(f"Quantization complete: {config.quant_method}, compression={size_info['compression_ratio']:.2f}x")

    return quantized_model, quant_device, quantization_metrics


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_compression_pipeline(config):
    """
    Run the complete compression pipeline.

    FLOW:
    1. Load/Train Teacher
    2. KD: Create Student, train to mimic Teacher
    3. Pruning: Applied to Student (or Teacher if no KD)
    4. Quantization: Applied to Pruned model
    5. Save in HuggingFace format

    Supports:
    - data_fraction for quick testing
    - run_full_kfold for robust evaluation
    - Dual tokenization for different teacher/student models
    - Comprehensive logging
    """
    print("\n" + "="*70)
    print("[COMPRESSION PIPELINE]")
    print("="*70)
    print(f"   Pipeline: {config.pipeline}")
    print(f"   KD: {'Yes' if config.enable_kd else 'No'}")
    print(f"   Pruning: {'Yes' if config.enable_pruning else 'No'}")
    print(f"   Quantization: {'Yes' if config.enable_quantization else 'No'}")

    # Setup
    set_seed(config.seed)
    device = get_device()
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.cache_dir, exist_ok=True)

    # Setup logging
    log_file = setup_logging(config.output_dir)
    logger = get_logger('main')
    logger.info("="*70)
    logger.info("COMPRESSION PIPELINE STARTED")
    logger.info("="*70)
    log_config(config)
    log_gpu_info()

    # Load data with caching
    logger.info("Loading data...")
    comments, labels, label_distribution = load_and_preprocess_data(
        config.dataset_path, label_columns=config.label_columns
    )
    log_dataset_info(len(comments), label_distribution, config.label_columns)

    # Apply data_fraction for quick testing
    if config.data_fraction < 1.0:
        n_samples = int(len(comments) * config.data_fraction)
        np.random.seed(config.seed)
        indices = np.random.choice(len(comments), n_samples, replace=False)
        comments = [comments[i] for i in indices]
        labels = labels[indices]
        logger.info(f"Using {config.data_fraction*100:.0f}% of data: {n_samples} samples")
        print(f"   [Data Fraction] Using {n_samples} samples ({config.data_fraction*100:.0f}%)")

    # Load tokenizers
    # IMPORTANT: If teacher_checkpoint is provided, use its tokenizer for consistency
    teacher_tokenizer_path = config.teacher_checkpoint if config.teacher_checkpoint else config.teacher_path
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_tokenizer_path)
    logger.info(f"Teacher tokenizer loaded from: {teacher_tokenizer_path}")
    print(f"   [Tokenizer] Teacher: {teacher_tokenizer_path}")

    # Load student tokenizer if different from teacher (for dual tokenization)
    student_tokenizer = None
    if config.enable_kd and config.student_path != teacher_tokenizer_path:
        student_tokenizer = AutoTokenizer.from_pretrained(config.student_path)
        logger.info(f"Loaded student tokenizer from {config.student_path} (dual tokenization enabled)")
        print(f"   [Dual Tokenization] Student tokenizer: {config.student_path}")

    # Tokenize data (with optional dual tokenization)
    tokenized_data = get_or_create_tokenized_dataset(
        comments, labels, teacher_tokenizer, config.max_length, config.cache_dir,
        student_tokenizer=student_tokenizer
    )

    has_student_tokens = 'student_input_ids' in tokenized_data

    # K-fold splits
    num_folds = config.num_folds if config.run_full_kfold else 1
    splits = list(prepare_kfold_splits(
        comments, labels, num_folds,
        stratification_type='multilabel', seed=config.seed
    ))

    if config.run_full_kfold:
        logger.info(f"Running full {num_folds}-fold cross-validation")
        print(f"   [K-Fold] Running all {num_folds} folds")
    else:
        logger.info("Running single fold (fold 1)")

    # Initialize evaluator
    evaluator = CompressionEvaluator()

    # Storage for all fold results
    all_fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        fold_num = fold_idx + 1

        if num_folds > 1:
            print(f"\n{'='*70}")
            print(f"[FOLD {fold_num}/{num_folds}]")
            print(f"{'='*70}")
            logger.info(f"Starting Fold {fold_num}/{num_folds}")

        all_metrics = []
        training_metrics = {}

        train_loader, val_loader = create_data_loaders(
            tokenized_data, train_idx, val_idx,
            batch_size=config.batch, num_workers=2
        )

        # ======================================================================
        # PHASE 1: Teacher
        # ======================================================================
        log_stage_start('Teacher')
        teacher, tokenizer, teacher_training_metrics = get_or_train_teacher(
            config, tokenized_data, train_idx, val_idx, device, logger
        )
        training_metrics['teacher'] = teacher_training_metrics
        log_model_info(teacher, 'Teacher')

        baseline_metrics = evaluator.evaluate_model(
            teacher, val_loader, device, stage='baseline',
            latency_batch_size=config.latency_batch_size,
            config=config
        )
        baseline_metrics.print_summary()
        all_metrics.append(baseline_metrics)
        save_stage_metrics(baseline_metrics, config.output_dir, f'baseline_fold{fold_num}')
        log_stage_end('Teacher', baseline_metrics.f1_macro)

        # Cleanup memory
        cleanup_memory(device)

        if config.pipeline == 'baseline':
            if fold_idx == 0:
                save_model_for_huggingface(teacher, os.path.join(config.output_dir, 'model_hf'), tokenizer)
            all_fold_results.append({'fold': fold_num, 'metrics': all_metrics, 'training': training_metrics})
            continue

        # Track current model
        current_model = teacher
        current_model_name = "teacher"
        use_student_tokens_for_eval = False

        # ======================================================================
        # PHASE 2: Knowledge Distillation
        # ======================================================================
        if config.enable_kd:
            log_stage_start('Knowledge Distillation')
            student, kd_training_metrics = run_knowledge_distillation(
                config, teacher, tokenized_data, train_idx, val_idx, device, logger
            )
            training_metrics['kd'] = kd_training_metrics

            # Cleanup teacher to free memory
            del teacher
            cleanup_memory(device)

            # Evaluate student - use student tokens if available
            kd_metrics = evaluator.evaluate_model(
                student, val_loader, device, stage='after_kd',
                use_student_input_ids=has_student_tokens,
                latency_batch_size=config.latency_batch_size,
                config=config
            )
            kd_metrics.print_summary()
            all_metrics.append(kd_metrics)
            save_stage_metrics(kd_metrics, config.output_dir, f'after_kd_fold{fold_num}')
            log_stage_end('Knowledge Distillation', kd_metrics.f1_macro)

            # NOW THE STUDENT BECOMES THE CURRENT MODEL
            current_model = student
            current_model_name = "student"
            use_student_tokens_for_eval = has_student_tokens

            if config.save_all_stages and fold_idx == 0:
                save_model_for_huggingface(
                    student, os.path.join(config.output_dir, 'model_after_kd_hf'), tokenizer
                )

            cleanup_memory(device)

        # ======================================================================
        # PHASE 3: Pruning
        # ======================================================================
        if config.enable_pruning:
            log_stage_start('Pruning')
            print(f"\n   [NOTE] Pruning will be applied to: {current_model_name}")

            pruned_model, pruning_metrics_dict = run_pruning(
                config, current_model, tokenized_data, train_idx, val_idx, device,
                model_name=current_model_name,
                use_student_input_ids=use_student_tokens_for_eval,
                logger=logger
            )
            training_metrics['pruning'] = pruning_metrics_dict

            prune_metrics = evaluator.evaluate_model(
                pruned_model, val_loader, device, stage='after_pruning',
                use_student_input_ids=use_student_tokens_for_eval,
                latency_batch_size=config.latency_batch_size,
                config=config
            )
            prune_metrics.print_summary()
            all_metrics.append(prune_metrics)
            save_stage_metrics(prune_metrics, config.output_dir, f'after_pruning_fold{fold_num}')
            log_stage_end('Pruning', prune_metrics.f1_macro)

            current_model = pruned_model
            current_model_name = f"pruned_{current_model_name}"

            if config.save_all_stages and fold_idx == 0:
                save_model_for_huggingface(
                    pruned_model, os.path.join(config.output_dir, 'model_after_pruning_hf'), tokenizer
                )

            cleanup_memory(device)

        # ======================================================================
        # PHASE 4: Quantization
        # ======================================================================
        if config.enable_quantization:
            log_stage_start('Quantization')
            quantized_model, quant_device, quant_metrics_dict = run_quantization(
                config, current_model, tokenized_data, train_idx, val_idx, device, logger
            )
            training_metrics['quantization'] = quant_metrics_dict

            # Create loader with appropriate settings for quantized model
            quant_loader = create_data_loaders(
                tokenized_data, train_idx, val_idx,
                batch_size=config.latency_batch_size, num_workers=0
            )[1]

            quant_metrics = evaluator.evaluate_model(
                quantized_model, quant_loader, quant_device, stage='after_quantization',
                use_student_input_ids=use_student_tokens_for_eval,
                latency_batch_size=config.latency_batch_size,
                config=config
            )
            quant_metrics.print_summary()
            all_metrics.append(quant_metrics)
            save_stage_metrics(quant_metrics, config.output_dir, f'after_quantization_fold{fold_num}')
            log_stage_end('Quantization', quant_metrics.f1_macro)

            current_model = quantized_model
            cleanup_memory(device)

        # Store fold results
        all_fold_results.append({
            'fold': fold_num,
            'metrics': all_metrics,
            'training': training_metrics
        })

        # Save final model (only for first fold or single fold)
        if fold_idx == 0:
            save_model_for_huggingface(
                current_model, os.path.join(config.output_dir, 'model_final_hf'), tokenizer
            )

    # ==========================================================================
    # Aggregate Results (K-Fold)
    # ==========================================================================
    print("\n" + "="*70)
    print("[FINAL RESULTS]")
    print("="*70)

    if config.run_full_kfold and num_folds > 1:
        # Aggregate metrics across folds
        aggregated = calculate_aggregate_metrics(all_fold_results)
        print_aggregated_metrics(aggregated)

        # Save aggregated results
        with open(os.path.join(config.output_dir, 'results_aggregated.json'), 'w') as f:
            json.dump(aggregated, f, indent=2, default=str)

        logger.info("Aggregated results saved")
    else:
        # Single fold - use all_metrics directly
        all_metrics = all_fold_results[0]['metrics']
        comparison_df = compare_stages(all_metrics)
        print(comparison_df.to_string(index=False))

        export_metrics_to_csv(all_metrics, os.path.join(config.output_dir, 'results_final.csv'), config=config)
        export_metrics_to_json(all_metrics, os.path.join(config.output_dir, 'results_final.json'), config=config)

    # Final summary
    final_metrics = all_fold_results[0]['metrics'][-1]
    baseline_metrics = all_fold_results[0]['metrics'][0]

    print(f"\n[SUMMARY]:")
    print(f"   Compression: {final_metrics.size_compression_ratio:.2f}x")
    print(f"   F1: {baseline_metrics.f1_macro:.4f} -> {final_metrics.f1_macro:.4f}")
    print(f"   Size: {baseline_metrics.model_size_mb:.1f} MB -> {final_metrics.model_size_mb:.1f} MB")
    print(f"   Log file: {log_file}")

    log_final_summary({
        'compression_ratio': final_metrics.size_compression_ratio,
        'f1_baseline': baseline_metrics.f1_macro,
        'f1_final': final_metrics.f1_macro,
        'size_baseline_mb': baseline_metrics.model_size_mb,
        'size_final_mb': final_metrics.model_size_mb
    })

    logger.info("Pipeline completed successfully")

    return all_fold_results


# =============================================================================
# ABLATION STUDY
# =============================================================================

def run_ablation_study(config):
    """Run all pipeline configurations for ablation study."""
    print("\n" + "="*70)
    print("[ABLATION STUDY]")
    print("="*70)

    pipelines = config.ablation_pipelines
    all_results = {}

    for pipeline in pipelines:
        print(f"\n{'='*70}")
        print(f"[ABLATION] {pipeline.upper()}")
        print(f"{'='*70}")

        pipeline_config = get_config_for_pipeline(
            pipeline,
            dataset_path=config.dataset_path,
            author_name=config.author_name,
            teacher_checkpoint=config.teacher_checkpoint,
            output_dir=os.path.join(config.output_dir, f'ablation_{pipeline}'),
            cache_dir=config.cache_dir,
            kd_method=config.kd_method,
            prune_method=config.prune_method,
            quant_method=config.quant_method,
            fine_tune_after_prune=True  # Always fine-tune!
        )

        fold_results = run_compression_pipeline(pipeline_config)
        # Get final metrics from first fold
        all_results[pipeline] = fold_results[0]['metrics'][-1]

    # Summary table
    print("\n" + "="*70)
    print("[ABLATION RESULTS]")
    print("="*70)

    import pandas as pd
    rows = []
    for pipeline, metrics in all_results.items():
        rows.append({
            'Pipeline': pipeline,
            'F1 Macro': f"{metrics.f1_macro:.4f}",
            'F1 Threat': f"{metrics.per_label_f1.get('threat', 0):.4f}",
            'Size (MB)': f"{metrics.model_size_mb:.1f}",
            'Compression': f"{metrics.size_compression_ratio:.2f}x",
            'Sparsity': f"{metrics.sparsity_percent:.1f}%"
        })

    ablation_df = pd.DataFrame(rows)
    print(ablation_df.to_string(index=False))
    ablation_df.to_csv(os.path.join(config.output_dir, 'ablation_summary.csv'), index=False)

    return all_results


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    config = parse_compression_arguments()
    print_compression_config(config)
    
    if config.run_ablation:
        return run_ablation_study(config)
    else:
        return run_compression_pipeline(config)


if __name__ == "__main__":
    main()
