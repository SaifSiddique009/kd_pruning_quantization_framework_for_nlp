"""
================================================================================
COMPRESSION CONFIGURATION (ENHANCED VERSION)
================================================================================

FIXES IN THIS VERSION:
1. ✅ fine_tune_after_prune is now DEFAULT=TRUE
2. ✅ Added INT4 quantization option
3. ✅ Better documentation of what happens at each stage
"""

import argparse
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict
import json


LABEL_COLUMNS = ['bully', 'sexual', 'religious', 'threat', 'spam']

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
    'prune_only': {
        'enable_kd': False,
        'enable_pruning': True,
        'enable_quantization': False,
        'description': 'Prune teacher directly (no student)'
    },
    'quant_only': {
        'enable_kd': False,
        'enable_pruning': False,
        'enable_quantization': True,
        'description': 'Quantize teacher directly'
    },
    'kd_prune': {
        'enable_kd': True,
        'enable_pruning': True,
        'enable_quantization': False,
        'description': 'KD → Prune STUDENT'
    },
    'kd_quant': {
        'enable_kd': True,
        'enable_pruning': False,
        'enable_quantization': True,
        'description': 'KD → Quantize STUDENT'
    },
    'prune_quant': {
        'enable_kd': False,
        'enable_pruning': True,
        'enable_quantization': True,
        'description': 'Prune teacher → Quantize'
    },
    'kd_prune_quant': {
        'enable_kd': True,
        'enable_pruning': True,
        'enable_quantization': True,
        'description': 'KD → Prune STUDENT → Quantize (full pipeline)'
    }
}


def parse_compression_arguments(args_list: Optional[List[str]] = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compress Transformer models for Bangla Cyberbullying Detection"
    )
    
    # Required
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--author_name', type=str, required=True)

    # Label Configuration
    parser.add_argument('--label_columns', type=str, nargs='+', default=LABEL_COLUMNS,
                       help='List of label columns in the dataset (default: bully, sexual, religious, threat, spam)')
    parser.add_argument('--label_priority', type=str, default=None,
                       help='JSON string of label priorities for weighted metrics (e.g. \'{"threat": 3, "sexual": 2}\')')

    # Pipeline
    parser.add_argument('--pipeline', type=str, default='kd_only',
                       choices=list(PIPELINE_CONFIGS.keys()))
    
    # Teacher
    parser.add_argument('--teacher_path', type=str, default='csebuetnlp/banglabert')
    parser.add_argument('--teacher_checkpoint', type=str, default=None,
                       help='HuggingFace model path to SKIP teacher training')
    parser.add_argument('--teacher_epochs', type=int, default=10)
    
    # Student
    parser.add_argument('--student_path', type=str, default='distilbert-base-multilingual-cased')
    parser.add_argument('--student_hidden_size', type=int, default=256)
    
    # Training
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--num_folds', type=int, default=5)
    parser.add_argument('--run_full_kfold', action='store_true',
                       help='Run all K folds and report mean/std metrics (slower but more robust)')
    parser.add_argument('--data_fraction', type=float, default=1.0,
                       help='Fraction of data to use (0.0-1.0). Use smaller values for quick testing.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--gradient_clip_norm', type=float, default=1.0)
    parser.add_argument('--early_stopping_patience', type=int, default=5)
    
    # KD Parameters
    parser.add_argument('--kd_alpha', type=float, default=0.7,
                       help='0=hard labels only, 1=soft labels only')
    parser.add_argument('--kd_temperature', type=float, default=4.0)
    parser.add_argument('--kd_method', type=str, default='logit',
                       choices=['logit', 'hidden', 'attention', 'multi_level'])
    parser.add_argument('--hidden_loss_weight', type=float, default=0.3)
    parser.add_argument('--attention_loss_weight', type=float, default=0.2)
    
    # Pruning Parameters
    parser.add_argument('--prune_method', type=str, default='magnitude',
                       choices=['magnitude', 'wanda', 'gradual', 'structured'])
    parser.add_argument('--prune_sparsity', type=float, default=0.5)
    parser.add_argument('--prune_schedule', type=str, default='cubic',
                       choices=['linear', 'cubic', 'exponential'])
    parser.add_argument('--prune_start_epoch', type=int, default=0)
    parser.add_argument('--prune_end_epoch', type=int, default=10)
    parser.add_argument('--prune_frequency', type=int, default=100)
    parser.add_argument('--prune_layers', type=str, default='all',
                       choices=['all', 'attention', 'ffn', 'encoder'])
    parser.add_argument('--calib_samples', type=int, default=512)
    
    # FIXED: Fine-tune after pruning is now DEFAULT=TRUE!
    parser.add_argument('--fine_tune_after_prune', action='store_true', default=True,
                       help='Fine-tune after pruning (DEFAULT: True)')
    parser.add_argument('--no_fine_tune_after_prune', action='store_false', 
                       dest='fine_tune_after_prune',
                       help='Skip fine-tuning after pruning')
    parser.add_argument('--fine_tune_epochs', type=int, default=3)
    
    # Quantization Parameters (ADDED INT4!)
    parser.add_argument('--quant_method', type=str, default='dynamic',
                       choices=['dynamic', 'static', 'qat', 'fp16', 'int4'])
    parser.add_argument('--quant_dtype', type=str, default='int8',
                       choices=['int8', 'int4', 'fp16'])
    parser.add_argument('--quant_calibration_batches', type=int, default=100)
    parser.add_argument('--latency_batch_size', type=int, default=1,
                       help='Batch size for latency measurement (default: 1 for realistic inference)')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./compressed_models')
    parser.add_argument('--cache_dir', type=str, default='./cache')
    parser.add_argument('--save_all_stages', action='store_true', default=True)
    
    # Ablation
    parser.add_argument('--run_ablation', action='store_true')
    parser.add_argument('--ablation_pipelines', type=str, nargs='+',
                       default=list(PIPELINE_CONFIGS.keys()))
    
    args = parser.parse_args(args_list)

    # Parse label_priority if provided
    if args.label_priority:
        try:
            args.label_priority = json.loads(args.label_priority)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse label_priority JSON: {args.label_priority}")
            args.label_priority = {col: 1 for col in args.label_columns}
    else:
        # Default: equal priority (weight=1) for all labels
        args.label_priority = {col: 1 for col in args.label_columns}

    # Validate data_fraction
    if args.data_fraction <= 0 or args.data_fraction > 1:
        print(f"Warning: data_fraction must be in (0, 1]. Setting to 1.0")
        args.data_fraction = 1.0

    _apply_pipeline_config(args)
    return args


def _apply_pipeline_config(args):
    """Apply pipeline-specific settings."""
    pipeline_config = PIPELINE_CONFIGS.get(args.pipeline, {})
    args.enable_kd = pipeline_config.get('enable_kd', False)
    args.enable_pruning = pipeline_config.get('enable_pruning', False)
    args.enable_quantization = pipeline_config.get('enable_quantization', False)


def get_config_for_pipeline(pipeline: str, **overrides):
    """Create config for a specific pipeline programmatically."""
    args_list = [
        '--pipeline', pipeline,
        '--dataset_path', overrides.pop('dataset_path', 'data.csv'),
        '--author_name', overrides.pop('author_name', 'experiment'),
    ]
    
    for key, value in overrides.items():
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                args_list.append(f'--{key}')
        else:
            args_list.extend([f'--{key}', str(value)])
    
    return parse_compression_arguments(args_list)


def print_compression_config(config):
    """Print configuration summary."""
    print("\n" + "="*70)
    print("COMPRESSION CONFIGURATION")
    print("="*70)

    pipeline_desc = PIPELINE_CONFIGS.get(config.pipeline, {}).get('description', '')
    print(f"\n[Pipeline]: {config.pipeline.upper()}")
    print(f"   {pipeline_desc}")

    # Data configuration
    print(f"\n[Data]:")
    print(f"   Labels: {config.label_columns}")
    if hasattr(config, 'data_fraction') and config.data_fraction < 1.0:
        print(f"   Data Fraction: {config.data_fraction*100:.0f}%")
    if hasattr(config, 'label_priority'):
        non_default = {k: v for k, v in config.label_priority.items() if v != 1}
        if non_default:
            print(f"   Priority Labels: {non_default}")
    if hasattr(config, 'run_full_kfold') and config.run_full_kfold:
        print(f"   K-Fold Mode: Full (all {config.num_folds} folds)")
    
    # Show what will happen
    print(f"\n[Compression Flow]:")
    if config.pipeline == 'baseline':
        print("   Teacher -> Evaluate -> Done")
    else:
        flow = ["Teacher"]
        if config.enable_kd:
            flow.append("KD -> Student")
        if config.enable_pruning:
            target = "Student" if config.enable_kd else "Teacher"
            flow.append(f"Prune {target}")
            if config.fine_tune_after_prune:
                flow.append("Fine-tune")
        if config.enable_quantization:
            flow.append("Quantize")
        flow.append("Final Model")
        print("   " + " -> ".join(flow))

    print(f"\n[Teacher]: {config.teacher_path}")
    if config.teacher_checkpoint:
        print(f"   Using checkpoint: {config.teacher_checkpoint} (skip training!)")

    if config.enable_kd:
        print(f"\n[Student]: {config.student_path}")
        print(f"   KD Method: {config.kd_method}")
        print(f"   Alpha: {config.kd_alpha}, Temperature: {config.kd_temperature}")

    if config.enable_pruning:
        print(f"\n[Pruning]:")
        print(f"   Method: {config.prune_method}")
        print(f"   Sparsity: {config.prune_sparsity*100:.0f}%")
        print(f"   Fine-tune after: {'Yes' if config.fine_tune_after_prune else 'No'}")

    if config.enable_quantization:
        print(f"\n[Quantization]:")
        print(f"   Method: {config.quant_method}")
        print(f"   Latency batch size: {config.latency_batch_size}")
        if config.quant_method in ['dynamic', 'static']:
            print(f"   Note: {config.quant_method} runs on CPU only")
        elif config.quant_method == 'int4':
            print(f"   Note: Requires bitsandbytes library")

    print("="*70 + "\n")
