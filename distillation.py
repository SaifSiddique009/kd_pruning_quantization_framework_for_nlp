"""
================================================================================
KNOWLEDGE DISTILLATION MODULE
================================================================================

This module implements Knowledge Distillation (KD) for multi-label classification.
It transfers knowledge from a large "teacher" model to a smaller "student" model.

WHAT IS KNOWLEDGE DISTILLATION?
─────────────────────────────────
Imagine you're a professor (teacher) explaining a concept to a student.
You don't just say "the answer is X" - you explain your reasoning,
show intermediate steps, and convey your confidence in each part.

Similarly, KD doesn't just train the student on hard labels (0 or 1).
It also trains on "soft labels" - the teacher's probability predictions.

Example:
    Hard label: [bully=1, sexual=0, threat=0, spam=0, religious=0]
    Soft label: [bully=0.85, sexual=0.12, threat=0.03, spam=0.02, religious=0.01]
    
    The soft label tells the student:
    - "This is mostly bully, but has some sexual undertones"
    - "There's a tiny bit of threat-like content"
    
    This extra information helps the student learn better!

WHY IS THIS MODULE SPECIAL FOR MULTI-LABEL?
───────────────────────────────────────────────
Most KD papers focus on single-label (multiclass) classification.
They use KL divergence on softmax outputs.

BUT: For multi-label, each label is INDEPENDENT binary classification!
You can't use softmax (labels aren't mutually exclusive).

This module correctly implements multi-label KD using:
- Sigmoid (not softmax) for each label
- Softened binary cross-entropy loss
- Optional hidden state and attention matching

DISTILLATION METHODS:
─────────────────────
1. 'logit' (simplest, recommended to start):
   - Match student's output predictions to teacher's
   - Fast, works well
   
2. 'hidden' (more complex):
   - Also match intermediate layer representations
   - Better knowledge transfer
   - Slower training
   
3. 'attention' (experimental):
   - Match attention patterns
   - Helps student learn "what to focus on"
   
4. 'multi_level' (best accuracy, slowest):
   - All of the above combined
   - Maximum knowledge transfer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, AutoTokenizer
from typing import Optional, Dict, List, Tuple
import numpy as np
import os
import json


# =============================================================================
# TEACHER MODEL
# =============================================================================

class TeacherModel(nn.Module):
    """
    Teacher model wrapper for knowledge distillation.
    
    WHAT: A large, accurate model that teaches a smaller student
    WHY: Teachers have high accuracy but are too big for deployment
    HOW: Wraps a pre-trained transformer + custom classifier head
    
    CRITICAL: Teacher MUST be fine-tuned on the task BEFORE distillation!
    If you use a raw pre-trained model, it won't have task knowledge to teach.
    
    Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                     TEACHER MODEL                           │
    ├─────────────────────────────────────────────────────────────┤
    │  Input: Token IDs [batch, seq_len]                         │
    │                    ↓                                        │
    │  ┌─────────────────────────────────────┐                   │
    │  │   Pre-trained Encoder               │                   │
    │  │   (BanglaBERT, mBERT, etc.)         │                   │
    │  │   Output: [batch, seq_len, 768]     │                   │
    │  └─────────────────────────────────────┘                   │
    │                    ↓                                        │
    │            Take [CLS] token                                 │
    │            [batch, 768]                                     │
    │                    ↓                                        │
    │  ┌─────────────────────────────────────┐                   │
    │  │   Classifier Head                   │                   │
    │  │   Linear(768→256) + ReLU + Dropout  │                   │
    │  │   Linear(256→5)                     │                   │
    │  └─────────────────────────────────────┘                   │
    │                    ↓                                        │
    │  Output: Logits [batch, 5]                                 │
    └─────────────────────────────────────────────────────────────┘
    """
    
    def __init__(self, model_name: str, num_labels: int = 5, dropout: float = 0.1):
        """
        Initialize teacher model.
        
        Args:
            model_name: HuggingFace model path (e.g., "csebuetnlp/banglabert")
            num_labels: Number of output labels (5 for cyberbullying)
            dropout: Dropout rate for regularization
        """
        super().__init__()
        
        # Load pre-trained encoder
        # This is the "brain" of the model - all the language understanding
        self.encoder = AutoModel.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        
        # Get hidden size from config (usually 768 for BERT-base)
        hidden_size = self.config.hidden_size
        
        # Classifier head
        # Transforms [CLS] embedding → label predictions
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),  # Reduce dimension
            nn.ReLU(),                     # Non-linearity
            nn.Dropout(dropout),           # Regularization
            nn.Linear(256, num_labels)     # Output logits
        )
        
        # Store for later
        self.num_labels = num_labels
        self.model_name = model_name
        self.hidden_size = hidden_size
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs [batch_size, seq_length]
            attention_mask: Mask for padding tokens [batch_size, seq_length]
            output_hidden_states: Whether to return all hidden states
            output_attentions: Whether to return attention weights
        
        Returns:
            Dict with:
            - 'logits': Raw predictions [batch_size, num_labels]
            - 'cls_embedding': [CLS] token embedding [batch_size, hidden_size]
            - 'hidden_states': (optional) All layer outputs
            - 'attentions': (optional) All attention weights
        """
        # Pass through encoder
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions
        )
        
        # Get [CLS] token embedding (first token of last layer)
        # This represents the "summary" of the input
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Pass through classifier
        logits = self.classifier(cls_output)
        
        # Build result dict
        result = {
            'logits': logits,
            'cls_embedding': cls_output
        }
        
        # Add optional outputs (for hidden/attention distillation)
        if output_hidden_states:
            result['hidden_states'] = outputs.hidden_states
        if output_attentions:
            result['attentions'] = outputs.attentions
        
        return result
    
    def save_pretrained(self, save_path: str):
        """
        Save the teacher model for later use.
        
        Saves:
        - Encoder weights (HuggingFace format)
        - Classifier weights (PyTorch format)
        - Config (JSON)
        """
        os.makedirs(save_path, exist_ok=True)
        
        # Save encoder
        self.encoder.save_pretrained(os.path.join(save_path, 'encoder'))
        
        # Save classifier
        torch.save(
            self.classifier.state_dict(),
            os.path.join(save_path, 'classifier.pt')
        )
        
        # Save config
        config = {
            'model_name': self.model_name,
            'num_labels': self.num_labels,
            'hidden_size': self.hidden_size
        }
        with open(os.path.join(save_path, 'teacher_config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Teacher model saved to: {save_path}")
    
    @classmethod
    def from_pretrained(cls, load_path: str, device: str = 'cpu') -> 'TeacherModel':
        """
        Load a pre-trained teacher model.
        
        This is how you load YOUR fine-tuned model from HuggingFace!
        
        Args:
            load_path: Path to saved model (local or HuggingFace)
            device: Device to load on
        
        Returns:
            Loaded TeacherModel
        """
        # Check if it's a local path with our format
        config_path = os.path.join(load_path, 'teacher_config.json')
        
        if os.path.exists(config_path):
            # Local format (saved by save_pretrained)
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            model = cls(
                model_name=os.path.join(load_path, 'encoder'),
                num_labels=config['num_labels']
            )
            
            # Load classifier weights
            classifier_state = torch.load(
                os.path.join(load_path, 'classifier.pt'),
                map_location=device
            )
            model.classifier.load_state_dict(classifier_state)
        else:
            # Assume it's a HuggingFace model path
            # Need to create fresh model and load weights differently
            model = cls(model_name=load_path, num_labels=5)
        
        return model.to(device)


# =============================================================================
# STUDENT MODEL
# =============================================================================

class StudentModel(nn.Module):
    """
    Student model for knowledge distillation.
    
    WHAT: A smaller model that learns from the teacher
    WHY: Students are faster/smaller for deployment
    HOW: Same architecture as teacher but with smaller encoder
    
    Typical student choices:
    - DistilBERT: 66M params (vs BERT's 110M)
    - TinyBERT: 14M params
    - MobileBERT: Optimized for mobile
    
    The student learns in two ways:
    1. From ground truth labels (hard loss)
    2. From teacher's predictions (soft loss)
    """
    
    def __init__(
        self, 
        model_name: str, 
        num_labels: int = 5, 
        dropout: float = 0.1,
        classifier_hidden_size: int = 256
    ):
        """
        Initialize student model.
        
        Args:
            model_name: HuggingFace model path for student
            num_labels: Number of output labels
            dropout: Dropout rate
            classifier_hidden_size: Hidden size for classifier head
        """
        super().__init__()
        
        self.encoder = AutoModel.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        hidden_size = self.config.hidden_size
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, classifier_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_size, num_labels)
        )
        
        self.num_labels = num_labels
        self.model_name = model_name
        self.hidden_size = hidden_size
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass (same as teacher)."""
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions
        )
        
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        
        result = {
            'logits': logits,
            'cls_embedding': cls_output
        }
        
        if output_hidden_states:
            result['hidden_states'] = outputs.hidden_states
        if output_attentions:
            result['attentions'] = outputs.attentions
        
        return result
    
    def save_pretrained(self, save_path: str):
        """Save student model."""
        os.makedirs(save_path, exist_ok=True)
        self.encoder.save_pretrained(os.path.join(save_path, 'encoder'))
        torch.save(
            self.classifier.state_dict(),
            os.path.join(save_path, 'classifier.pt')
        )

        config = {
            'model_name': self.model_name,
            'num_labels': self.num_labels,
            'hidden_size': self.hidden_size
        }
        with open(os.path.join(save_path, 'student_config.json'), 'w') as f:
            json.dump(config, f, indent=2)

    @classmethod
    def from_pretrained(
        cls,
        load_path: str,
        dropout: float = 0.1,
        classifier_hidden_size: int = 256,
        **kwargs
    ):
        """
        Load a saved student model.

        Args:
            load_path: Path where student model was saved
            dropout: Dropout rate for classifier
            classifier_hidden_size: Hidden size for classifier
            **kwargs: Additional arguments (for quantization config, etc.)

        Returns:
            StudentModel instance
        """
        # Load config
        config_path = os.path.join(load_path, 'student_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            model_name = config.get('model_name', 'distilbert-base-multilingual-cased')
            num_labels = config.get('num_labels', 5)
        else:
            # Fallback to defaults
            model_name = 'distilbert-base-multilingual-cased'
            num_labels = 5

        # Create model
        model = cls(
            model_name=model_name,
            num_labels=num_labels,
            dropout=dropout,
            classifier_hidden_size=classifier_hidden_size
        )

        # Load encoder if saved separately
        encoder_path = os.path.join(load_path, 'encoder')
        if os.path.exists(encoder_path):
            model.encoder = AutoModel.from_pretrained(encoder_path, **kwargs)

        # Load classifier
        classifier_path = os.path.join(load_path, 'classifier.pt')
        if os.path.exists(classifier_path):
            model.classifier.load_state_dict(torch.load(classifier_path, map_location='cpu'))

        return model


# =============================================================================
# DISTILLATION LOSS
# =============================================================================

class MultiLabelDistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss for Multi-Label Classification.
    
    WHAT: Custom loss function that combines learning from teacher and ground truth
    
    WHY: Standard KD uses KL divergence on softmax, but that's WRONG for multi-label!
         Multi-label = multiple independent binary classifications
         We need to use sigmoid, not softmax
    
    HOW: 
        Total Loss = α × Soft_Loss + (1-α) × Hard_Loss
        
        Where:
        - Soft_Loss: Match teacher's soft predictions
        - Hard_Loss: Match ground truth labels
        - α: Balance factor (typically 0.7)
    
    TEMPERATURE SCALING:
    ─────────────────────
    To make soft labels more informative, we use temperature scaling:
    
    Normal sigmoid: σ(z) = 1 / (1 + e^(-z))
    With temp T:    σ(z/T) = 1 / (1 + e^(-z/T))
    
    Higher T = Softer probabilities = More information
    
    Example (logit z = 2.0):
        T=1: σ(2/1) = 0.88 (confident)
        T=4: σ(2/4) = 0.62 (less confident, more informative)
    """
    
    def __init__(
        self,
        alpha: float = 0.7,
        temperature: float = 4.0,
        method: str = 'logit',
        hidden_weight: float = 0.3,
        attention_weight: float = 0.2
    ):
        """
        Initialize distillation loss.
        
        Args:
            alpha: Weight for soft loss (0.7 = 70% teacher, 30% ground truth)
            temperature: Softening factor for teacher predictions
            method: 'logit', 'hidden', 'attention', or 'multi_level'
            hidden_weight: Weight for hidden state loss (if using hidden/multi_level)
            attention_weight: Weight for attention loss (if using attention/multi_level)
        """
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.method = method
        self.hidden_weight = hidden_weight
        self.attention_weight = attention_weight
    
    def forward(
        self,
        student_outputs: Dict[str, torch.Tensor],
        teacher_outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        pos_weight: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate distillation loss.
        
        Args:
            student_outputs: Dict with 'logits' and optionally hidden_states/attentions
            teacher_outputs: Dict with 'logits' and optionally hidden_states/attentions
            labels: Ground truth labels [batch_size, num_labels]
            pos_weight: Class weights for imbalanced data
        
        Returns:
            Dict with 'total_loss' and component losses
        """
        s_logits = student_outputs['logits']
        t_logits = teacher_outputs['logits']
        
        # =====================================================================
        # SOFT LOSS: Learn from teacher's predictions
        # =====================================================================
        # This is the "knowledge transfer" part
        soft_loss = self._compute_soft_loss(s_logits, t_logits)
        
        # =====================================================================
        # HARD LOSS: Learn from ground truth
        # =====================================================================
        # This is standard supervised learning
        if pos_weight is not None:
            hard_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            hard_loss_fn = nn.BCEWithLogitsLoss()
        hard_loss = hard_loss_fn(s_logits, labels)
        
        # =====================================================================
        # COMBINED LOSS
        # =====================================================================
        logit_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        total_loss = logit_loss
        
        # Build result dict
        loss_dict = {
            'total_loss': total_loss,
            'soft_loss': soft_loss.detach(),
            'hard_loss': hard_loss.detach(),
            'logit_loss': logit_loss.detach()
        }
        
        # =====================================================================
        # OPTIONAL: Hidden state matching
        # =====================================================================
        if self.method in ['hidden', 'multi_level']:
            if 'hidden_states' in student_outputs and 'hidden_states' in teacher_outputs:
                hidden_loss = self._compute_hidden_loss(
                    student_outputs['hidden_states'],
                    teacher_outputs['hidden_states']
                )
                total_loss = total_loss + self.hidden_weight * hidden_loss
                loss_dict['hidden_loss'] = hidden_loss.detach()
        
        # =====================================================================
        # OPTIONAL: Attention matching
        # =====================================================================
        if self.method in ['attention', 'multi_level']:
            if 'attentions' in student_outputs and 'attentions' in teacher_outputs:
                attention_loss = self._compute_attention_loss(
                    student_outputs['attentions'],
                    teacher_outputs['attentions']
                )
                total_loss = total_loss + self.attention_weight * attention_loss
                loss_dict['attention_loss'] = attention_loss.detach()
        
        loss_dict['total_loss'] = total_loss
        return loss_dict
    
    def _compute_soft_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute soft distillation loss.

        THIS IS THE KEY FUNCTION FOR MULTI-LABEL KD!

        Instead of KL divergence on softmax (wrong for multi-label),
        we use softened binary cross-entropy.

        Steps:
        1. Apply temperature scaling to both logits
        2. Convert teacher to soft targets with sigmoid
        3. Compute BCE with logits (AMP-safe) between student logits and soft targets
        4. Scale by T^2 (see Hinton paper)

        NOTE: Using binary_cross_entropy_with_logits is more numerically stable
        and AMP-safe than applying sigmoid then binary_cross_entropy.
        """
        # Temperature-scaled logits
        s_scaled = student_logits / self.temperature
        t_scaled = teacher_logits / self.temperature

        # Soft probabilities from teacher (sigmoid for multi-label!)
        t_probs = torch.sigmoid(t_scaled).detach()  # Detach teacher (no gradients)

        # Binary cross-entropy with logits (AMP-safe version)
        # This applies sigmoid internally and is numerically stable
        soft_loss = F.binary_cross_entropy_with_logits(
            s_scaled,
            t_probs,
            reduction='mean'
        )

        # Scale by T^2 (compensates for gradient scaling from temperature)
        # See: Hinton et al., "Distilling the Knowledge in a Neural Network"
        soft_loss = soft_loss * (self.temperature ** 2)

        return soft_loss
    
    def _compute_hidden_loss(
        self,
        student_hidden: Tuple[torch.Tensor, ...],
        teacher_hidden: Tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        """
        Compute hidden state matching loss.
        
        Makes student's intermediate representations similar to teacher's.
        This helps transfer "how the teacher thinks", not just final answers.
        """
        # Get number of layers
        num_student = len(student_hidden)
        num_teacher = len(teacher_hidden)
        
        # Create layer mapping (student layer → teacher layer)
        layer_mapping = self._get_layer_mapping(num_student, num_teacher)
        
        total_loss = 0.0
        for s_idx, t_idx in layer_mapping:
            s_layer = student_hidden[s_idx]
            t_layer = teacher_hidden[t_idx]
            
            # Handle dimension mismatch
            if s_layer.shape[-1] != t_layer.shape[-1]:
                # Project student to teacher dimension using adaptive pooling
                s_layer = F.adaptive_avg_pool1d(
                    s_layer.transpose(1, 2), 
                    t_layer.shape[-1]
                ).transpose(1, 2)
            
            # Cosine similarity loss
            # 1.0 = identical, 0.0 = orthogonal
            cos_sim = F.cosine_similarity(s_layer, t_layer, dim=-1)
            cos_loss = 1 - cos_sim.mean()
            total_loss += cos_loss
        
        return total_loss / len(layer_mapping)
    
    def _compute_attention_loss(
        self,
        student_attentions: Tuple[torch.Tensor, ...],
        teacher_attentions: Tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        """
        Compute attention pattern matching loss.
        
        Makes student attend to same things as teacher.
        """
        num_student = len(student_attentions)
        num_teacher = len(teacher_attentions)
        
        layer_mapping = self._get_layer_mapping(num_student, num_teacher)
        
        total_loss = 0.0
        for s_idx, t_idx in layer_mapping:
            s_attn = student_attentions[s_idx]  # [batch, heads, seq, seq]
            t_attn = teacher_attentions[t_idx]
            
            # Average over heads if different
            if s_attn.shape[1] != t_attn.shape[1]:
                s_attn = s_attn.mean(dim=1, keepdim=True)
                t_attn = t_attn.mean(dim=1, keepdim=True)
            
            # KL divergence on attention distributions
            # Attention is already a probability distribution over tokens
            s_attn_log = torch.log(s_attn + 1e-10)
            attn_loss = F.kl_div(s_attn_log, t_attn, reduction='batchmean')
            total_loss += attn_loss
        
        return total_loss / len(layer_mapping)
    
    def _get_layer_mapping(
        self, 
        num_student: int, 
        num_teacher: int
    ) -> List[Tuple[int, int]]:
        """
        Create mapping between student and teacher layers.
        
        Since student has fewer layers, we sample teacher layers
        to create pairs for matching.
        """
        if num_student <= 2 or num_teacher <= 2:
            return [(0, 0)]
        
        # Match 4 intermediate layers (or fewer if not enough layers)
        num_matches = min(4, num_student - 1, num_teacher - 1)
        
        student_indices = np.linspace(1, num_student - 1, num_matches, dtype=int)
        teacher_indices = np.linspace(1, num_teacher - 1, num_matches, dtype=int)
        
        return list(zip(student_indices, teacher_indices))


# =============================================================================
# DISTILLATION TRAINER
# =============================================================================

class DistillationTrainer:
    """
    Trainer for knowledge distillation.
    
    WHAT: Handles the complete training loop for KD
    WHY: Encapsulates all KD logic in one place
    HOW: Freezes teacher, trains student with distillation loss
    
    Usage:
        trainer = DistillationTrainer(teacher, student, config, device)
        for epoch in range(epochs):
            for batch in train_loader:
                losses = trainer.train_step(batch, optimizer)
            metrics = trainer.evaluate(val_loader)
    """
    
    def __init__(
        self,
        teacher: TeacherModel,
        student: StudentModel,
        config,
        device: str = 'cuda'
    ):
        """
        Initialize distillation trainer.
        
        Args:
            teacher: Pre-trained teacher model
            student: Student model to train
            config: Configuration with KD parameters
            device: Device to train on
        """
        self.teacher = teacher.to(device)
        self.student = student.to(device)
        self.config = config
        self.device = device
        
        # CRITICAL: Freeze teacher!
        # Teacher is already trained; we don't want to change it
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        # Determine what outputs we need
        output_hidden = config.kd_method in ['hidden', 'multi_level']
        output_attention = config.kd_method in ['attention', 'multi_level']
        
        # Create loss function
        self.loss_fn = MultiLabelDistillationLoss(
            alpha=config.kd_alpha,
            temperature=config.kd_temperature,
            method=config.kd_method,
            hidden_weight=config.hidden_loss_weight if output_hidden else 0,
            attention_weight=config.attention_loss_weight if output_attention else 0
        )
        
        self.output_hidden = output_hidden
        self.output_attention = output_attention
        
        # For AMP (Automatic Mixed Precision)
        self.use_amp = torch.cuda.is_available()
        if self.use_amp:
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        class_weights: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Perform one training step.

        Supports DUAL TOKENIZATION for Knowledge Distillation:
        If batch contains 'student_input_ids', uses separate tokenizations
        for teacher (input_ids) and student (student_input_ids).

        Args:
            batch: Dict with 'input_ids', 'attention_mask', 'labels'
                   and optionally 'student_input_ids', 'student_attention_mask'
            optimizer: Optimizer for student
            class_weights: Optional class weights for imbalanced data

        Returns:
            Dict with loss values
        """
        self.student.train()

        # Check if we have separate student tokenization
        has_student_tokens = 'student_input_ids' in batch

        # Teacher tokens (always use input_ids for teacher)
        t_input_ids = batch['input_ids'].to(self.device)
        t_attention_mask = batch['attention_mask'].to(self.device)

        # Student tokens (use student_input_ids if available, else same as teacher)
        if has_student_tokens:
            s_input_ids = batch['student_input_ids'].to(self.device)
            s_attention_mask = batch['student_attention_mask'].to(self.device)
        else:
            s_input_ids = t_input_ids
            s_attention_mask = t_attention_mask

        labels = batch['labels'].to(self.device)

        # Get teacher outputs (no gradients!)
        with torch.no_grad():
            teacher_outputs = self.teacher(
                t_input_ids, t_attention_mask,
                output_hidden_states=self.output_hidden,
                output_attentions=self.output_attention
            )

        # Forward pass with optional mixed precision
        if self.use_amp:
            from torch.cuda.amp import autocast
            with autocast():
                student_outputs = self.student(
                    s_input_ids, s_attention_mask,
                    output_hidden_states=self.output_hidden,
                    output_attentions=self.output_attention
                )

                pos_weight = class_weights.to(self.device) if class_weights is not None else None
                loss_dict = self.loss_fn(student_outputs, teacher_outputs, labels, pos_weight)

            # Backward pass with scaling
            optimizer.zero_grad()
            self.scaler.scale(loss_dict['total_loss']).backward()
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.config.gradient_clip_norm)
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            # Standard forward/backward
            student_outputs = self.student(
                s_input_ids, s_attention_mask,
                output_hidden_states=self.output_hidden,
                output_attentions=self.output_attention
            )

            pos_weight = class_weights.to(self.device) if class_weights is not None else None
            loss_dict = self.loss_fn(student_outputs, teacher_outputs, labels, pos_weight)

            optimizer.zero_grad()
            loss_dict['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.config.gradient_clip_norm)
            optimizer.step()

        # Return losses as floats
        return {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
    
    @torch.no_grad()
    def evaluate(
        self,
        dataloader,
        class_weights: Optional[torch.Tensor] = None,
        use_student_input_ids: bool = True
    ) -> Dict:
        """
        Evaluate student model.

        Supports DUAL TOKENIZATION:
        If use_student_input_ids=True and batch contains 'student_input_ids',
        uses student tokenization for evaluation. Otherwise uses 'input_ids'.

        Args:
            dataloader: Validation dataloader
            class_weights: Optional class weights
            use_student_input_ids: Whether to use student tokenization if available

        Returns:
            Dict with 'predictions', 'labels', and 'loss'
        """
        self.student.eval()

        all_preds = []
        all_labels = []
        total_loss = 0.0

        for batch in dataloader:
            # Use student tokens if available and requested
            if use_student_input_ids and 'student_input_ids' in batch:
                input_ids = batch['student_input_ids'].to(self.device)
                attention_mask = batch['student_attention_mask'].to(self.device)
            else:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

            labels = batch['labels'].to(self.device)

            outputs = self.student(input_ids, attention_mask)

            # Simple BCE for evaluation
            loss = F.binary_cross_entropy_with_logits(outputs['logits'], labels)
            total_loss += loss.item()

            preds = torch.sigmoid(outputs['logits'])
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        return {
            'predictions': np.array(all_preds),
            'labels': np.array(all_labels),
            'loss': total_loss / len(dataloader)
        }


# =============================================================================
# TEACHER VERIFICATION
# =============================================================================

def verify_teacher_performance(
    teacher: TeacherModel,
    dataloader,
    device: str,
    min_f1: float = 0.5
) -> Tuple[bool, Dict]:
    """
    Verify that teacher is properly fine-tuned before distillation.
    
    WHAT: Checks teacher performance on validation data
    WHY: If teacher hasn't learned the task, it has nothing to teach!
    HOW: Run inference, compute F1 score, compare to threshold
    
    This is CRITICAL! Many KD implementations fail because they use
    a pre-trained model without fine-tuning as the teacher.
    
    Args:
        teacher: Teacher model to verify
        dataloader: Validation dataloader
        device: Device to run on
        min_f1: Minimum acceptable F1 score
    
    Returns:
        Tuple of (is_valid, metrics_dict)
    """
    from sklearn.metrics import f1_score, accuracy_score
    
    print("\nVerifying teacher performance...")

    teacher.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']

            outputs = teacher(input_ids, attention_mask)
            preds = torch.sigmoid(outputs['logits']) > 0.5

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute metrics
    f1 = f1_score(all_labels, all_preds, average='macro')
    acc = accuracy_score(all_labels, all_preds)

    metrics = {
        'f1_macro': f1,
        'accuracy': acc
    }

    # Check if valid
    is_valid = f1 >= min_f1

    if is_valid:
        print(f"   Teacher verification PASSED!")
        print(f"      F1 Macro: {f1:.4f}")
        print(f"      Accuracy: {acc:.4f}")
    else:
        print(f"   Teacher verification FAILED!")
        print(f"      F1 Macro: {f1:.4f} (minimum required: {min_f1})")
        print(f"      The teacher model doesn't appear to be fine-tuned!")
        print(f"      KD will not work well without a trained teacher.")

    return is_valid, metrics


def load_teacher_from_huggingface(
    model_path: str,
    num_labels: int = 5,
    device: str = 'cuda'
) -> TeacherModel:
    """
    Load a fine-tuned teacher model from HuggingFace.
    
    This is for loading YOUR fine-tuned model that you previously
    uploaded to HuggingFace Hub.
    
    Args:
        model_path: HuggingFace model path (e.g., "your-username/your-model")
        num_labels: Number of labels
        device: Device to load on
    
    Returns:
        Loaded TeacherModel
    """
    print(f"\n📥 Loading teacher from HuggingFace: {model_path}")
    
    # Create model
    teacher = TeacherModel(
        model_name=model_path,
        num_labels=num_labels
    )
    
    return teacher.to(device)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing distillation module...")
    
    # Test loss function
    loss_fn = MultiLabelDistillationLoss(alpha=0.7, temperature=4.0)
    
    # Dummy data
    student_outputs = {'logits': torch.randn(4, 5)}
    teacher_outputs = {'logits': torch.randn(4, 5)}
    labels = torch.randint(0, 2, (4, 5)).float()
    
    loss_dict = loss_fn(student_outputs, teacher_outputs, labels)
    print(f"Total loss: {loss_dict['total_loss'].item():.4f}")
    print(f"Soft loss: {loss_dict['soft_loss'].item():.4f}")
    print(f"Hard loss: {loss_dict['hard_loss'].item():.4f}")
    
    print("\n✅ Distillation module tests passed!")
