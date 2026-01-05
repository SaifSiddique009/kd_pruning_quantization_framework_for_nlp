"""
================================================================================
EVALUATION MODULE
================================================================================

This module provides comprehensive evaluation for compressed models.
It measures both classification performance and efficiency metrics.

WHAT THIS MODULE DOES:
1. Classification metrics (F1, accuracy, precision, recall, etc.)
2. Per-label breakdown (critical for cyberbullying detection)
3. Efficiency metrics (latency, throughput, memory)
4. Compression metrics (size, sparsity, compression ratio)
5. Trade-off analysis (accuracy vs compression)

WHY COMPREHENSIVE EVALUATION?
─────────────────────────────
For your research paper, you need to show:
- How much accuracy is lost?
- How much compression is gained?
- Is the trade-off worth it?
- Which labels suffer most from compression?

The last point is CRITICAL for cyberbullying:
- Losing accuracy on 'spam' is acceptable
- Losing accuracy on 'threat' could be dangerous!
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    hamming_loss, roc_auc_score, classification_report,
    multilabel_confusion_matrix
)
from dataclasses import dataclass, asdict
import time
import json
import pandas as pd
from tqdm import tqdm


# =============================================================================
# LABEL CONFIGURATION
# =============================================================================

LABEL_COLUMNS = ['bully', 'sexual', 'religious', 'threat', 'spam']

LABEL_PRIORITY = {
    'threat': 5,     # HIGHEST - safety critical
    'sexual': 4,     # HIGH - harmful content
    'religious': 3,  # MEDIUM - hate speech
    'bully': 2,      # MEDIUM - general harassment
    'spam': 1        # LOW - just noise
}


# =============================================================================
# METRICS DATACLASS
# =============================================================================

@dataclass
class CompressionStageMetrics:
    """Container for all metrics at a compression stage."""
    stage: str
    model_size_mb: float
    num_parameters: int
    trainable_parameters: int
    sparsity_percent: float
    
    accuracy_exact: float
    accuracy_per_label: float
    f1_macro: float
    f1_weighted: float
    f1_micro: float
    precision_macro: float
    recall_macro: float
    hamming_loss: float
    
    per_label_f1: Dict[str, float]
    per_label_precision: Dict[str, float]
    per_label_recall: Dict[str, float]
    per_label_accuracy: Dict[str, float]
    
    roc_auc_macro: Optional[float] = None
    roc_auc_per_label: Optional[Dict[str, float]] = None
    
    inference_latency_mean_ms: float = 0.0
    inference_latency_p50_ms: float = 0.0
    inference_latency_p95_ms: float = 0.0
    inference_latency_p99_ms: float = 0.0
    throughput_samples_per_sec: float = 0.0
    peak_memory_mb: float = 0.0
    
    size_compression_ratio: float = 1.0
    speedup_ratio: float = 1.0
    priority_weighted_f1: float = 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def to_flat_dict(self) -> Dict:
        result = {
            'stage': self.stage,
            'size_mb': self.model_size_mb,
            'params_M': self.num_parameters / 1e6,
            'sparsity_%': self.sparsity_percent,
            'acc_exact': self.accuracy_exact,
            'f1_macro': self.f1_macro,
            'f1_weighted': self.f1_weighted,
            'precision': self.precision_macro,
            'recall': self.recall_macro,
            'hamming': self.hamming_loss,
            'latency_ms': self.inference_latency_mean_ms,
            'throughput': self.throughput_samples_per_sec,
            'compression': self.size_compression_ratio,
            'speedup': self.speedup_ratio,
            'priority_f1': self.priority_weighted_f1,
        }
        for label, f1 in self.per_label_f1.items():
            result[f'f1_{label}'] = f1
        return result
    
    def print_summary(self):
        print(f"\n{'='*60}")
        print(f"📊 METRICS: {self.stage.upper()}")
        print(f"{'='*60}")
        print(f"\n📦 Model Size:")
        print(f"   Size: {self.model_size_mb:.2f} MB")
        print(f"   Parameters: {self.num_parameters/1e6:.2f}M")
        print(f"   Sparsity: {self.sparsity_percent:.1f}%")
        print(f"   Compression: {self.size_compression_ratio:.2f}×")
        print(f"\n🎯 Classification Performance:")
        print(f"   F1 Macro: {self.f1_macro:.4f}")
        print(f"   F1 Weighted: {self.f1_weighted:.4f}")
        print(f"   Accuracy (exact): {self.accuracy_exact:.4f}")
        print(f"\n📋 Per-Label F1 Scores:")
        for label in LABEL_COLUMNS:
            f1 = self.per_label_f1.get(label, 0)
            priority = LABEL_PRIORITY.get(label, 1)
            bar = '█' * int(f1 * 20)
            print(f"   {label:10s}: {f1:.4f} {bar} (priority: {priority})")
        print(f"\n   Priority-Weighted F1: {self.priority_weighted_f1:.4f}")
        if self.inference_latency_mean_ms > 0:
            print(f"\n⚡ Efficiency:")
            print(f"   Latency (mean): {self.inference_latency_mean_ms:.2f} ms")
            print(f"   Throughput: {self.throughput_samples_per_sec:.1f} samples/sec")
            print(f"   Speedup: {self.speedup_ratio:.2f}×")
        print(f"{'='*60}\n")


# =============================================================================
# EVALUATOR CLASS
# =============================================================================

class CompressionEvaluator:
    """Comprehensive evaluator for compressed models."""
    
    def __init__(
        self,
        label_columns: List[str] = LABEL_COLUMNS,
        label_priority: Dict[str, int] = LABEL_PRIORITY,
        threshold: float = 0.5
    ):
        self.label_columns = label_columns
        self.label_priority = label_priority
        self.threshold = threshold
        self.baseline_metrics = None
    
    def evaluate_model(
        self,
        model: nn.Module,
        dataloader,
        device: str,
        stage: str = 'unknown',
        measure_latency: bool = True,
        measure_memory: bool = True,
        latency_iterations: int = 100
    ) -> CompressionStageMetrics:
        """Evaluate a model comprehensively."""
        print(f"\n📊 Evaluating: {stage}")
        
        predictions, labels, probabilities = self._get_predictions(model, dataloader, device)
        class_metrics = self._compute_classification_metrics(predictions, labels, probabilities)
        size_metrics = self._compute_size_metrics(model)
        
        efficiency_metrics = {}
        if measure_latency:
            efficiency_metrics = self._measure_latency(model, dataloader, device, latency_iterations)
        if measure_memory and device == 'cuda':
            efficiency_metrics.update(self._measure_memory(model, dataloader, device))
        
        compression_ratio = 1.0
        speedup_ratio = 1.0
        if self.baseline_metrics is not None:
            compression_ratio = self.baseline_metrics.model_size_mb / max(size_metrics['size_mb'], 0.01)
            if efficiency_metrics.get('latency_mean_ms', 0) > 0:
                speedup_ratio = self.baseline_metrics.inference_latency_mean_ms / efficiency_metrics['latency_mean_ms']
        
        metrics = CompressionStageMetrics(
            stage=stage,
            model_size_mb=size_metrics['size_mb'],
            num_parameters=size_metrics['num_params'],
            trainable_parameters=size_metrics['trainable_params'],
            sparsity_percent=size_metrics['sparsity_percent'],
            accuracy_exact=class_metrics['accuracy_exact'],
            accuracy_per_label=class_metrics['accuracy_per_label'],
            f1_macro=class_metrics['f1_macro'],
            f1_weighted=class_metrics['f1_weighted'],
            f1_micro=class_metrics['f1_micro'],
            precision_macro=class_metrics['precision_macro'],
            recall_macro=class_metrics['recall_macro'],
            hamming_loss=class_metrics['hamming_loss'],
            per_label_f1=class_metrics['per_label_f1'],
            per_label_precision=class_metrics['per_label_precision'],
            per_label_recall=class_metrics['per_label_recall'],
            per_label_accuracy=class_metrics['per_label_accuracy'],
            roc_auc_macro=class_metrics.get('roc_auc_macro'),
            roc_auc_per_label=class_metrics.get('roc_auc_per_label'),
            inference_latency_mean_ms=efficiency_metrics.get('latency_mean_ms', 0),
            inference_latency_p50_ms=efficiency_metrics.get('latency_p50_ms', 0),
            inference_latency_p95_ms=efficiency_metrics.get('latency_p95_ms', 0),
            inference_latency_p99_ms=efficiency_metrics.get('latency_p99_ms', 0),
            throughput_samples_per_sec=efficiency_metrics.get('throughput', 0),
            peak_memory_mb=efficiency_metrics.get('peak_memory_mb', 0),
            size_compression_ratio=compression_ratio,
            speedup_ratio=speedup_ratio,
            priority_weighted_f1=class_metrics['priority_weighted_f1']
        )
        
        if stage == 'baseline':
            self.baseline_metrics = metrics
        
        return metrics
    
    def _get_predictions(self, model, dataloader, device):
        model.eval()
        model.to(device)
        all_preds, all_labels, all_probs = [], [], []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="   Predicting"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels']
                
                outputs = model(input_ids, attention_mask)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs[0] if isinstance(outputs, tuple) else outputs
                
                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs > self.threshold).astype(int)
                
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
        
        return np.array(all_preds), np.array(all_labels), np.array(all_probs)
    
    def _compute_classification_metrics(self, predictions, labels, probabilities):
        accuracy_exact = accuracy_score(labels, predictions)
        accuracy_per_label = np.mean(np.mean(labels == predictions, axis=1))
        
        f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
        f1_weighted = f1_score(labels, predictions, average='weighted', zero_division=0)
        f1_micro = f1_score(labels, predictions, average='micro', zero_division=0)
        precision_macro = precision_score(labels, predictions, average='macro', zero_division=0)
        recall_macro = recall_score(labels, predictions, average='macro', zero_division=0)
        h_loss = hamming_loss(labels, predictions)
        
        per_label_f1, per_label_precision, per_label_recall, per_label_accuracy = {}, {}, {}, {}
        for i, label in enumerate(self.label_columns):
            per_label_f1[label] = f1_score(labels[:, i], predictions[:, i], zero_division=0)
            per_label_precision[label] = precision_score(labels[:, i], predictions[:, i], zero_division=0)
            per_label_recall[label] = recall_score(labels[:, i], predictions[:, i], zero_division=0)
            per_label_accuracy[label] = accuracy_score(labels[:, i], predictions[:, i])
        
        total_weight = sum(self.label_priority.values())
        priority_f1 = sum(per_label_f1[l] * self.label_priority[l] for l in self.label_columns) / total_weight
        
        roc_auc_macro, roc_auc_per_label = None, None
        try:
            roc_auc_macro = roc_auc_score(labels, probabilities, average='macro')
            roc_auc_per_label = {self.label_columns[i]: roc_auc_score(labels[:, i], probabilities[:, i]) 
                                 for i in range(len(self.label_columns)) if len(np.unique(labels[:, i])) > 1}
        except:
            pass
        
        return {
            'accuracy_exact': accuracy_exact, 'accuracy_per_label': accuracy_per_label,
            'f1_macro': f1_macro, 'f1_weighted': f1_weighted, 'f1_micro': f1_micro,
            'precision_macro': precision_macro, 'recall_macro': recall_macro, 'hamming_loss': h_loss,
            'per_label_f1': per_label_f1, 'per_label_precision': per_label_precision,
            'per_label_recall': per_label_recall, 'per_label_accuracy': per_label_accuracy,
            'priority_weighted_f1': priority_f1, 'roc_auc_macro': roc_auc_macro, 'roc_auc_per_label': roc_auc_per_label
        }
    
    def _compute_size_metrics(self, model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
        size_mb = (param_size + buffer_size) / (1024 ** 2)
        zero_params = sum((p == 0).sum().item() for p in model.parameters())
        sparsity = (zero_params / total_params * 100) if total_params > 0 else 0
        return {'size_mb': size_mb, 'num_params': total_params, 'trainable_params': trainable_params, 'sparsity_percent': sparsity}
    
    def _measure_latency(self, model, dataloader, device, num_iterations=100):
        model.eval()
        model.to(device)
        batch = next(iter(dataloader))
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        batch_size = input_ids.shape[0]
        
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_ids, attention_mask)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        latencies = []
        with torch.no_grad():
            for _ in range(num_iterations):
                if device == 'cuda':
                    torch.cuda.synchronize()
                start = time.perf_counter()
                _ = model(input_ids, attention_mask)
                if device == 'cuda':
                    torch.cuda.synchronize()
                latencies.append((time.perf_counter() - start) * 1000)
        
        latencies = np.array(latencies)
        return {
            'latency_mean_ms': np.mean(latencies), 'latency_p50_ms': np.percentile(latencies, 50),
            'latency_p95_ms': np.percentile(latencies, 95), 'latency_p99_ms': np.percentile(latencies, 99),
            'throughput': (batch_size / np.mean(latencies)) * 1000
        }
    
    def _measure_memory(self, model, dataloader, device):
        if device != 'cuda':
            return {'peak_memory_mb': 0}
        model.to(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        batch = next(iter(dataloader))
        with torch.no_grad():
            _ = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
        return {'peak_memory_mb': torch.cuda.max_memory_allocated() / (1024 ** 2)}


def compare_stages(metrics_list: List[CompressionStageMetrics]) -> pd.DataFrame:
    return pd.DataFrame([m.to_flat_dict() for m in metrics_list])


def export_metrics_to_csv(metrics_list: List[CompressionStageMetrics], output_path: str):
    compare_stages(metrics_list).to_csv(output_path, index=False)
    print(f"✅ Metrics exported to: {output_path}")


def export_metrics_to_json(metrics_list: List[CompressionStageMetrics], output_path: str):
    with open(output_path, 'w') as f:
        json.dump([m.to_dict() for m in metrics_list], f, indent=2)
    print(f"✅ Metrics exported to: {output_path}")
