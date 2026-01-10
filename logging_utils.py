"""
================================================================================
LOGGING UTILITIES MODULE
================================================================================

Centralized logging system for the compression pipeline.
Provides structured logging to both console and file for debugging and monitoring.

Features:
- Dual output: Console (colored) + File (timestamped)
- Structured log format with timestamps
- Memory usage tracking
- GPU status monitoring
- Training progress logging
- Stage transition logging

Usage:
    from logging_utils import setup_logging, get_logger, log_config, log_metrics

    # Setup at start of main.py
    log_file = setup_logging(output_dir='./output', log_level='INFO')
    logger = get_logger(__name__)

    # Log configuration
    log_config(config)

    # Log metrics after evaluation
    log_metrics(metrics_dict, stage='after_kd')
"""

import logging
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional
import json

# Try to import torch for GPU monitoring
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Try to import psutil for memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


# =============================================================================
# CUSTOM FORMATTER WITH COLORS (Console only)
# =============================================================================

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }

    def format(self, record):
        # Add color to log level
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"

        return super().format(record)


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(
    output_dir: str = './output',
    log_level: str = 'INFO',
    log_filename: Optional[str] = None
) -> str:
    """
    Setup dual logging to file and console.

    Args:
        output_dir: Directory to save log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_filename: Custom log filename (default: training_YYYYMMDD_HHMMSS.log)

    Returns:
        Path to the log file
    """
    # Create logs directory
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # Generate timestamped log filename if not provided
    if log_filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = f'training_{timestamp}.log'

    log_file = os.path.join(log_dir, log_filename)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    root_logger.handlers = []

    # File handler (detailed, no colors)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)  # Capture all levels to file
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Console handler (colored, less verbose)
    console_formatter = ColoredFormatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # Log initial message
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")

    return log_file


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# =============================================================================
# CONFIGURATION LOGGING
# =============================================================================

def log_config(config, logger: Optional[logging.Logger] = None):
    """
    Log all configuration parameters.

    Args:
        config: Configuration namespace/object with hyperparameters
        logger: Logger instance (creates one if not provided)
    """
    if logger is None:
        logger = get_logger('config')

    logger.info("=" * 70)
    logger.info("CONFIGURATION")
    logger.info("=" * 70)

    # Convert to dict if needed
    if hasattr(config, '__dict__'):
        config_dict = vars(config)
    elif hasattr(config, '_asdict'):
        config_dict = config._asdict()
    else:
        config_dict = dict(config)

    # Group parameters by category
    categories = {
        'Pipeline': ['pipeline', 'enable_kd', 'enable_pruning', 'enable_quantization'],
        'Dataset': ['dataset_path', 'label_columns', 'label_priority', 'data_fraction', 'max_length'],
        'Teacher': ['teacher_path', 'teacher_checkpoint', 'teacher_epochs'],
        'Student': ['student_path', 'student_hidden_size'],
        'Training': ['batch', 'lr', 'epochs', 'num_folds', 'run_full_kfold', 'seed', 'dropout',
                     'weight_decay', 'warmup_ratio', 'gradient_clip_norm', 'early_stopping_patience'],
        'KD': ['kd_alpha', 'kd_temperature', 'kd_method', 'hidden_loss_weight', 'attention_loss_weight'],
        'Pruning': ['prune_method', 'prune_sparsity', 'prune_schedule', 'fine_tune_after_prune', 'fine_tune_epochs'],
        'Quantization': ['quant_method', 'quant_dtype', 'latency_batch_size'],
        'Output': ['output_dir', 'cache_dir', 'save_all_stages']
    }

    logged_keys = set()

    for category, keys in categories.items():
        category_params = {k: config_dict.get(k) for k in keys if k in config_dict}
        if category_params:
            logger.info(f"\n[{category}]")
            for key, value in category_params.items():
                logger.info(f"  {key}: {value}")
                logged_keys.add(key)

    # Log any remaining parameters
    remaining = {k: v for k, v in config_dict.items() if k not in logged_keys}
    if remaining:
        logger.info(f"\n[Other]")
        for key, value in remaining.items():
            logger.info(f"  {key}: {value}")

    logger.info("=" * 70)


# =============================================================================
# DATASET LOGGING
# =============================================================================

def log_dataset_info(
    num_samples: int,
    label_distribution: Dict[str, Dict[str, int]],
    label_columns: list,
    logger: Optional[logging.Logger] = None
):
    """
    Log dataset statistics.

    Args:
        num_samples: Total number of samples
        label_distribution: Dict with label counts {label: {0: count, 1: count}}
        label_columns: List of label column names
        logger: Logger instance
    """
    if logger is None:
        logger = get_logger('data')

    logger.info("=" * 70)
    logger.info("DATASET STATISTICS")
    logger.info("=" * 70)
    logger.info(f"Total samples: {num_samples}")
    logger.info(f"Labels: {label_columns}")

    logger.info("\nLabel Distribution:")
    for label in label_columns:
        if label in label_distribution:
            dist = label_distribution[label]
            pos = dist.get(1, 0)
            neg = dist.get(0, 0)
            ratio = pos / (pos + neg) * 100 if (pos + neg) > 0 else 0
            logger.info(f"  {label}: {pos} positive ({ratio:.1f}%), {neg} negative")

    logger.info("=" * 70)


# =============================================================================
# MODEL LOGGING
# =============================================================================

def log_model_info(
    model,
    model_name: str,
    logger: Optional[logging.Logger] = None
):
    """
    Log model architecture information.

    Args:
        model: PyTorch model
        model_name: Name/description of the model
        logger: Logger instance
    """
    if logger is None:
        logger = get_logger('model')

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"\n[{model_name}]")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Non-trainable parameters: {total_params - trainable_params:,}")

    # Estimate model size
    param_size = total_params * 4 / (1024 * 1024)  # Assuming float32
    logger.info(f"  Estimated size (FP32): {param_size:.2f} MB")


# =============================================================================
# TRAINING PROGRESS LOGGING
# =============================================================================

def log_epoch_start(epoch: int, total_epochs: int, logger: Optional[logging.Logger] = None):
    """Log start of training epoch."""
    if logger is None:
        logger = get_logger('training')
    logger.info(f"\n{'='*50}")
    logger.info(f"Epoch {epoch + 1}/{total_epochs}")
    logger.info(f"{'='*50}")


def log_epoch_end(
    epoch: int,
    train_loss: float,
    val_loss: float,
    val_metrics: Dict[str, float],
    logger: Optional[logging.Logger] = None
):
    """
    Log end of training epoch with metrics.

    Args:
        epoch: Current epoch number
        train_loss: Training loss
        val_loss: Validation loss
        val_metrics: Dictionary of validation metrics
        logger: Logger instance
    """
    if logger is None:
        logger = get_logger('training')

    logger.info(f"Epoch {epoch + 1} Summary:")
    logger.info(f"  Train Loss: {train_loss:.4f}")
    logger.info(f"  Val Loss: {val_loss:.4f}")

    # Log key metrics
    key_metrics = ['f1_macro', 'f1_weighted', 'accuracy', 'hamming_loss']
    for metric in key_metrics:
        if metric in val_metrics:
            logger.info(f"  Val {metric}: {val_metrics[metric]:.4f}")


def log_training_step(
    step: int,
    total_steps: int,
    loss: float,
    lr: float,
    logger: Optional[logging.Logger] = None,
    log_every: int = 100
):
    """Log training step (periodically)."""
    if step % log_every == 0:
        if logger is None:
            logger = get_logger('training')
        logger.debug(f"Step {step}/{total_steps} | Loss: {loss:.4f} | LR: {lr:.2e}")


# =============================================================================
# STAGE TRANSITION LOGGING
# =============================================================================

def log_stage_start(stage: str, logger: Optional[logging.Logger] = None):
    """Log start of a pipeline stage."""
    if logger is None:
        logger = get_logger('pipeline')

    logger.info("")
    logger.info("=" * 70)
    logger.info(f"STARTING STAGE: {stage.upper()}")
    logger.info("=" * 70)
    log_memory_usage(logger)


def log_stage_end(
    stage: str,
    metrics: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None
):
    """Log end of a pipeline stage with optional metrics summary."""
    if logger is None:
        logger = get_logger('pipeline')

    logger.info("")
    logger.info(f"COMPLETED STAGE: {stage.upper()}")

    if metrics is not None:
        # Handle case where metrics is just a single float (f1_macro value)
        if isinstance(metrics, (int, float)):
            logger.info(f"  F1 Macro: {float(metrics):.4f}")
        elif isinstance(metrics, dict):
            logger.info("Stage Metrics Summary:")
            key_metrics = ['f1_macro', 'model_size_mb', 'sparsity', 'compression_ratio']
            for metric in key_metrics:
                if metric in metrics:
                    value = metrics[metric]
                    if isinstance(value, float):
                        logger.info(f"  {metric}: {value:.4f}")
                    else:
                        logger.info(f"  {metric}: {value}")

    logger.info("=" * 70)
    log_memory_usage(logger)


# =============================================================================
# METRICS LOGGING
# =============================================================================

def log_metrics(
    metrics: Dict[str, Any],
    stage: str = '',
    logger: Optional[logging.Logger] = None
):
    """
    Log evaluation metrics.

    Args:
        metrics: Dictionary of metrics
        stage: Stage name (e.g., 'teacher_baseline', 'after_kd')
        logger: Logger instance
    """
    if logger is None:
        logger = get_logger('evaluation')

    logger.info(f"\n[Metrics - {stage}]")

    # Group metrics by type
    classification_metrics = ['f1_macro', 'f1_weighted', 'f1_micro', 'accuracy',
                             'precision_macro', 'recall_macro', 'roc_auc', 'hamming_loss']
    efficiency_metrics = ['latency_mean_ms', 'latency_p50_ms', 'latency_p95_ms',
                         'throughput_samples_per_sec', 'peak_memory_mb']
    compression_metrics = ['model_size_mb', 'num_parameters', 'sparsity', 'compression_ratio']

    def log_metric_group(name, metric_keys):
        group_metrics = {k: metrics.get(k) for k in metric_keys if k in metrics and metrics.get(k) is not None}
        if group_metrics:
            logger.info(f"\n  {name}:")
            for key, value in group_metrics.items():
                if isinstance(value, float):
                    logger.info(f"    {key}: {value:.4f}")
                else:
                    logger.info(f"    {key}: {value}")

    log_metric_group("Classification", classification_metrics)
    log_metric_group("Efficiency", efficiency_metrics)
    log_metric_group("Compression", compression_metrics)

    # Log per-label metrics if present
    per_label = {k: v for k, v in metrics.items() if k.startswith('f1_') and k not in classification_metrics}
    if per_label:
        logger.info(f"\n  Per-Label F1:")
        for key, value in per_label.items():
            if isinstance(value, float):
                logger.info(f"    {key}: {value:.4f}")


def log_comparison_table(
    all_metrics: Dict[str, Dict[str, Any]],
    logger: Optional[logging.Logger] = None
):
    """
    Log a comparison table of metrics across stages.

    Args:
        all_metrics: Dict of {stage_name: metrics_dict}
        logger: Logger instance
    """
    if logger is None:
        logger = get_logger('comparison')

    logger.info("\n" + "=" * 70)
    logger.info("COMPRESSION RESULTS COMPARISON")
    logger.info("=" * 70)

    # Key metrics to compare
    compare_metrics = ['f1_macro', 'model_size_mb', 'sparsity', 'latency_mean_ms', 'compression_ratio']

    # Header
    stages = list(all_metrics.keys())
    header = f"{'Metric':<25}" + "".join(f"{s:<18}" for s in stages)
    logger.info(header)
    logger.info("-" * len(header))

    # Rows
    for metric in compare_metrics:
        row = f"{metric:<25}"
        for stage in stages:
            value = all_metrics[stage].get(metric, 'N/A')
            if isinstance(value, float):
                row += f"{value:<18.4f}"
            else:
                row += f"{str(value):<18}"
        logger.info(row)

    logger.info("=" * 70)


# =============================================================================
# MEMORY & GPU LOGGING
# =============================================================================

def log_memory_usage(logger: Optional[logging.Logger] = None):
    """Log current memory usage (CPU and GPU if available)."""
    if logger is None:
        logger = get_logger('system')

    memory_info = []

    # CPU memory
    if PSUTIL_AVAILABLE:
        process = psutil.Process()
        cpu_mem = process.memory_info().rss / (1024 * 1024)
        memory_info.append(f"CPU: {cpu_mem:.1f} MB")

    # GPU memory
    if TORCH_AVAILABLE and torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / (1024 * 1024)
        gpu_cached = torch.cuda.memory_reserved() / (1024 * 1024)
        memory_info.append(f"GPU: {gpu_mem:.1f} MB (cached: {gpu_cached:.1f} MB)")

    if memory_info:
        logger.debug(f"Memory Usage - {' | '.join(memory_info)}")


def log_gpu_info(logger: Optional[logging.Logger] = None):
    """Log GPU information if available."""
    if logger is None:
        logger = get_logger('system')

    if TORCH_AVAILABLE and torch.cuda.is_available():
        logger.info("\n[GPU Information]")
        logger.info(f"  Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"  CUDA Version: {torch.version.cuda}")
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"  Total Memory: {total_memory:.1f} GB")
    else:
        logger.info("\n[GPU Information]")
        logger.info("  No GPU available, using CPU")


# =============================================================================
# ERROR LOGGING
# =============================================================================

def log_error(
    error: Exception,
    context: str = '',
    logger: Optional[logging.Logger] = None
):
    """
    Log an error with context and stack trace.

    Args:
        error: Exception that occurred
        context: Context description
        logger: Logger instance
    """
    if logger is None:
        logger = get_logger('error')

    logger.error(f"ERROR in {context}: {type(error).__name__}: {str(error)}")
    logger.exception("Stack trace:")


def log_warning(
    message: str,
    logger: Optional[logging.Logger] = None
):
    """Log a warning message."""
    if logger is None:
        logger = get_logger('warning')
    logger.warning(message)


# =============================================================================
# FINAL SUMMARY
# =============================================================================

def log_final_summary(
    config,
    all_metrics: Dict[str, Dict[str, Any]],
    total_time_seconds: float,
    log_file: str,
    logger: Optional[logging.Logger] = None
):
    """
    Log final summary at end of pipeline.

    Args:
        config: Configuration used
        all_metrics: All metrics from all stages
        total_time_seconds: Total execution time
        log_file: Path to log file
        logger: Logger instance
    """
    if logger is None:
        logger = get_logger('summary')

    logger.info("\n" + "=" * 70)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 70)

    logger.info(f"\nPipeline: {config.pipeline}")
    logger.info(f"Total Time: {total_time_seconds / 60:.2f} minutes")
    logger.info(f"Log File: {log_file}")

    # Best metrics
    if all_metrics:
        final_stage = list(all_metrics.keys())[-1]
        final_metrics = all_metrics[final_stage]

        logger.info(f"\nFinal Model ({final_stage}):")
        if 'f1_macro' in final_metrics:
            logger.info(f"  F1 Macro: {final_metrics['f1_macro']:.4f}")
        if 'model_size_mb' in final_metrics:
            logger.info(f"  Model Size: {final_metrics['model_size_mb']:.2f} MB")
        if 'compression_ratio' in final_metrics:
            logger.info(f"  Compression Ratio: {final_metrics['compression_ratio']:.1f}x")
        if 'latency_mean_ms' in final_metrics:
            logger.info(f"  Latency: {final_metrics['latency_mean_ms']:.2f} ms")

    logger.info("\n" + "=" * 70)
    logger.info("Pipeline completed successfully!")
    logger.info("=" * 70)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    # Test logging setup
    log_file = setup_logging(output_dir='./test_output', log_level='DEBUG')
    logger = get_logger(__name__)

    logger.info("Testing logging module...")
    logger.debug("Debug message")
    logger.warning("Warning message")
    logger.error("Error message")

    log_gpu_info(logger)
    log_memory_usage(logger)

    # Test metrics logging
    test_metrics = {
        'f1_macro': 0.85,
        'model_size_mb': 256.5,
        'latency_mean_ms': 15.2
    }
    log_metrics(test_metrics, stage='test', logger=logger)

    print(f"\nLog file created at: {log_file}")
