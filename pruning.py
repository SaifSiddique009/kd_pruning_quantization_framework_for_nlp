"""
================================================================================
PRUNING MODULE
================================================================================

This module implements various pruning strategies to remove unnecessary weights
from neural networks, making them smaller and faster.

WHAT IS PRUNING?
────────────────
Pruning removes weights (connections) from the neural network.
Think of it like trimming a tree - you remove branches that don't contribute much.

Example:
    Before: 110 million weights
    After 50% pruning: 55 million weights (half removed!)
    
    The removed weights are set to 0 and can be stored efficiently.

WHY PRUNE?
──────────
1. Smaller model size (fewer parameters to store)
2. Faster inference (fewer multiplications)
3. Works well with quantization for maximum compression

TYPES OF PRUNING:
─────────────────
1. UNSTRUCTURED PRUNING:
   - Removes individual weights anywhere in the network
   - Highest compression, but needs sparse math libraries for speedup
   - Example: [0.5, 0.1, 0.8, 0.02] → [0.5, 0, 0.8, 0]

2. STRUCTURED PRUNING:
   - Removes entire neurons/filters/attention heads
   - Lower compression, but gives real speedup on any hardware
   - Example: Remove entire row of weight matrix

PRUNING METHODS IN THIS MODULE:
───────────────────────────────
1. Magnitude Pruning: Remove smallest weights (simple but effective)
2. Gradual Pruning: Slowly increase sparsity during training
3. Wanda Pruning: Consider activations too (state-of-the-art)
4. Structured Pruning: Remove entire attention heads
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import List, Tuple, Dict, Optional
import numpy as np
from tqdm import tqdm
from collections import defaultdict


# =============================================================================
# PRUNING MANAGER (Base Class)
# =============================================================================

class PruningManager:
    """
    Base class for managing pruning operations.
    
    WHAT: Handles finding prunable layers and tracking sparsity
    WHY: Common functionality shared by all pruning methods
    HOW: Scans model for Linear layers, applies pruning, tracks stats
    
    Usage:
        manager = PruningManager(model, target_sparsity=0.5)
        manager.apply_magnitude_pruning()
        print(manager.get_sparsity())
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_sparsity: float = 0.5,
        prune_layers: str = 'all',
        global_pruning: bool = True
    ):
        """
        Initialize pruning manager.
        
        Args:
            model: Model to prune
            target_sparsity: Fraction of weights to remove (0.5 = 50%)
            prune_layers: Which layers to prune
                - 'all': All Linear layers
                - 'attention': Only attention layers
                - 'ffn': Only feed-forward layers
                - 'encoder': Encoder only (not classifier)
            global_pruning: If True, use global threshold across all layers
                            If False, prune each layer independently
        """
        self.model = model
        self.target_sparsity = target_sparsity
        self.prune_layers = prune_layers
        self.global_pruning = global_pruning
        self.current_sparsity = 0.0
        
        # Find prunable modules
        self.prunable_modules = self._get_prunable_modules()
        
        print(f"\n[Pruning] PruningManager initialized:")
        print(f"   Target sparsity: {target_sparsity * 100:.1f}%")
        print(f"   Prunable modules: {len(self.prunable_modules)}")
        print(f"   Layer filter: {prune_layers}")
        print(f"   Global pruning: {global_pruning}")
    
    def _get_prunable_modules(self) -> List[Tuple[nn.Module, str]]:
        """
        Find all modules that can be pruned.
        
        Returns list of (module, parameter_name) tuples.
        We only prune Linear layers' weights (not biases).
        """
        modules = []
        
        for name, module in self.model.named_modules():
            # Only prune Linear layers
            if not isinstance(module, nn.Linear):
                continue
            
            # Filter by layer type
            if self.prune_layers == 'attention':
                # Only attention-related layers
                if not any(x in name.lower() for x in ['attention', 'query', 'key', 'value', 'q_proj', 'k_proj', 'v_proj']):
                    continue
            elif self.prune_layers == 'ffn':
                # Only feed-forward layers
                if not any(x in name.lower() for x in ['intermediate', 'output', 'dense', 'fc', 'mlp']):
                    continue
            elif self.prune_layers == 'encoder':
                # Skip classifier head
                if 'classifier' in name.lower():
                    continue
            
            modules.append((module, 'weight'))
        
        return modules
    
    def get_sparsity(self) -> Dict[str, float]:
        """
        Calculate current sparsity of the model.
        
        Returns:
            Dict with:
            - 'overall': Total sparsity across all weights
            - 'per_layer': Dict mapping layer name to its sparsity
            - 'total_params': Total number of parameters
            - 'zero_params': Number of zero parameters
        """
        total_params = 0
        zero_params = 0
        layer_sparsity = {}
        
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight'):
                weight = module.weight
                
                # Handle pruned modules (have weight_mask)
                if hasattr(module, 'weight_mask'):
                    weight = module.weight * module.weight_mask
                
                total = weight.numel()
                zeros = (weight == 0).sum().item()
                total_params += total
                zero_params += zeros
                
                if total > 0:
                    layer_sparsity[name] = zeros / total
        
        overall = zero_params / total_params if total_params > 0 else 0
        
        return {
            'overall': overall,
            'per_layer': layer_sparsity,
            'total_params': total_params,
            'zero_params': zero_params,
            'nonzero_params': total_params - zero_params
        }
    
    def apply_magnitude_pruning(self, sparsity: Optional[float] = None):
        """
        Apply one-shot magnitude-based pruning.
        
        WHAT: Removes weights with smallest absolute values
        WHY: Small weights contribute little to output
        HOW: Sort all weights by |value|, remove smallest fraction
        
        This is the simplest and most common pruning method.
        
        Example:
            Weights: [0.8, -0.05, 0.3, 0.01, -0.6]
            After 40% pruning: [0.8, 0, 0.3, 0, -0.6]
            (Removed 0.05 and 0.01, the smallest by magnitude)
        """
        sparsity = sparsity or self.target_sparsity
        
        print(f"\n   Applying magnitude pruning (target: {sparsity*100:.1f}%)")
        
        if self.global_pruning:
            # Global pruning: same threshold across all layers
            # This is usually better as it removes least important weights globally
            prune.global_unstructured(
                self.prunable_modules,
                pruning_method=prune.L1Unstructured,
                amount=sparsity
            )
        else:
            # Per-layer pruning: each layer pruned independently
            for module, param_name in self.prunable_modules:
                prune.l1_unstructured(module, param_name, amount=sparsity)
        
        self.current_sparsity = self.get_sparsity()['overall']
        print(f"   [OK] Applied magnitude pruning: {self.current_sparsity*100:.2f}% sparsity")
    
    def make_pruning_permanent(self):
        """
        Make pruning permanent by applying masks to weights.
        
        WHAT: Removes the pruning reparameterization, makes zeros permanent
        WHY: After training, we want actual zeros, not masked values
        HOW: weight = weight * mask, then remove mask
        
        Call this after training is complete!
        """
        for module, param_name in self.prunable_modules:
            if prune.is_pruned(module):
                prune.remove(module, param_name)
        
        print("   [OK] Pruning made permanent (masks applied to weights)")
    
    def get_pruning_masks(self) -> Dict[str, torch.Tensor]:
        """
        Extract and return pruning masks BEFORE making pruning permanent.
        
        WHAT: Saves the binary masks that indicate which weights are pruned
        WHY: Needed to enforce sparsity during fine-tuning
        HOW: Iterates through prunable modules and extracts weight_mask tensors
        
        IMPORTANT: Call this BEFORE make_pruning_permanent()!
                   After make_pruning_permanent(), masks are removed.
        
        Returns:
            Dict mapping parameter names to their mask tensors.
            Mask values: 1 = keep weight, 0 = pruned weight
        """
        masks = {}
        
        for name, module in self.model.named_modules():
            # Check if module has a pruning mask
            if hasattr(module, 'weight_mask'):
                # Store mask with full parameter path
                mask_name = f"{name}.weight"
                masks[mask_name] = module.weight_mask.clone().detach()
                
            if hasattr(module, 'bias_mask'):
                mask_name = f"{name}.bias"
                masks[mask_name] = module.bias_mask.clone().detach()
        
        print(f"   [OK] Extracted {len(masks)} pruning masks")
        return masks


# =============================================================================
# GRADUAL PRUNING
# =============================================================================

class GradualPruner(PruningManager):
    """
    Gradual magnitude pruning with sparsity scheduling.
    
    WHAT: Slowly increases sparsity over training epochs
    WHY: Model can adapt to pruning, less accuracy loss
    HOW: At each step, prune a little more until target reached
    
    The key insight: If you prune 50% at once, the model may not recover.
    But if you prune 5% every epoch for 10 epochs, it can adapt.
    
    Sparsity Schedules:
    ───────────────────
    1. Linear: Constant rate (sparsity grows linearly)
    2. Cubic: Slow start, fast middle, slow end (recommended)
    3. Exponential: Slow start, fast end
    
    IMPORTANT: Fixes cumulative pruning bug!
    ──────────────────────────────────────────
    Wrong way: Apply 10% pruning 5 times = 41% total (not 50%!)
    Right way: Calculate incremental amount needed each time
    
    This module does it the RIGHT way.
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_sparsity: float = 0.5,
        start_epoch: int = 0,
        end_epoch: int = 10,
        schedule: str = 'cubic',
        prune_frequency: int = 100,
        prune_layers: str = 'all',
        global_pruning: bool = True
    ):
        """
        Initialize gradual pruner.
        
        Args:
            model: Model to prune
            target_sparsity: Final sparsity to reach
            start_epoch: Epoch to start pruning
            end_epoch: Epoch to reach target sparsity
            schedule: 'linear', 'cubic', or 'exponential'
            prune_frequency: Steps between pruning updates
            prune_layers: Which layers to prune
            global_pruning: Use global vs per-layer threshold
        """
        super().__init__(model, target_sparsity, prune_layers, global_pruning)
        
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.schedule = schedule
        self.prune_frequency = prune_frequency
        self.total_steps = 0
        
        # Initialize masks (no pruning yet)
        self._initialize_masks()
        
        print(f"   Schedule: {schedule}")
        print(f"   Start epoch: {start_epoch}, End epoch: {end_epoch}")
        print(f"   Prune every {prune_frequency} steps")
    
    def _initialize_masks(self):
        """Initialize pruning masks with no pruning (identity)."""
        for module, param_name in self.prunable_modules:
            # Identity pruning = no pruning, just sets up the mask
            prune.identity(module, param_name)
    
    def _get_target_sparsity_at_step(self, current_step: int, total_steps: int) -> float:
        """
        Calculate target sparsity based on schedule and current step.
        
        This is where the schedule magic happens!
        """
        # Calculate pruning phase bounds
        start_step = int(self.start_epoch / self.end_epoch * total_steps)
        end_step = total_steps
        
        # Before pruning phase
        if current_step < start_step:
            return 0.0
        
        # After pruning phase
        if current_step >= end_step:
            return self.target_sparsity
        
        # During pruning phase: calculate progress
        progress = (current_step - start_step) / (end_step - start_step)
        
        # Apply schedule
        if self.schedule == 'linear':
            # Constant rate: sparsity grows linearly
            return self.target_sparsity * progress
        
        elif self.schedule == 'cubic':
            # Cubic schedule: slow start, fast middle, slow end
            # This is the default in most papers
            # Formula: 3t² - 2t³ (smooth S-curve)
            return self.target_sparsity * (3 * progress**2 - 2 * progress**3)
        
        elif self.schedule == 'exponential':
            # Exponential: slow start, fast end
            # Good when you want to prune aggressively at the end
            return self.target_sparsity * (1 - np.exp(-5 * progress))
        
        else:
            return self.target_sparsity * progress
    
    def step(self, current_step: int, total_steps: int):
        """
        Perform one pruning step if needed.
        
        Call this every training step!
        
        Args:
            current_step: Current training step
            total_steps: Total training steps
        """
        self.total_steps = total_steps
        
        # Only prune at specified frequency
        if current_step % self.prune_frequency != 0:
            return
        
        # Get target sparsity at this step
        target = self._get_target_sparsity_at_step(current_step, total_steps)
        
        # Already at or above target
        if target <= self.current_sparsity:
            return
        
        # Safety check
        if self.current_sparsity >= 0.99:
            return
        
        # =================================================================
        # KEY FIX: Calculate INCREMENTAL pruning amount
        # =================================================================
        # If we're at 20% sparsity and want 30%, we need to prune
        # (30% - 20%) / (100% - 20%) = 12.5% of REMAINING weights
        # Not 10% of total weights!
        
        incremental = (target - self.current_sparsity) / (1 - self.current_sparsity)
        incremental = min(incremental, 0.99)  # Safety cap
        
        if incremental < 0.001:
            return  # Too small to matter
        
        # Apply incremental pruning
        if self.global_pruning:
            prune.global_unstructured(
                self.prunable_modules,
                pruning_method=prune.L1Unstructured,
                amount=incremental
            )
        else:
            for module, param_name in self.prunable_modules:
                prune.l1_unstructured(module, param_name, amount=incremental)
        
        # Update current sparsity
        self.current_sparsity = self.get_sparsity()['overall']


# =============================================================================
# WANDA PRUNING (State-of-the-Art)
# =============================================================================

class WandaPruner(PruningManager):
    """
    Wanda (Weights AND Activations) Pruning.
    
    WHAT: Prunes based on both weight magnitude AND activation magnitude
    WHY: A small weight connected to high activations might be important!
    HOW: Importance = |weight| × |activation|, prune lowest importance
    
    This is state-of-the-art pruning from 2023 (Sun et al., "A Simple and
    Effective Pruning Approach for Large Language Models").
    
    Key Insight:
    ────────────
    Standard magnitude pruning: Remove weights with small |w|
    
    Problem: What if a small weight connects to features that are always
             activated strongly? That weight might actually be important!
    
    Wanda solution: Importance = |w| × mean(|activation|)
    
    Example:
        Weight A: |w| = 0.1 (small), avg |activation| = 10.0 (high)
            → Importance = 0.1 × 10.0 = 1.0 (important!)
        
        Weight B: |w| = 0.5 (medium), avg |activation| = 0.1 (low)
            → Importance = 0.5 × 0.1 = 0.05 (not important!)
    
    Requires: Calibration data to collect activation statistics
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_sparsity: float = 0.5,
        prune_layers: str = 'all',
        global_pruning: bool = True
    ):
        super().__init__(model, target_sparsity, prune_layers, global_pruning)
        
        # Storage for activation statistics
        self.activation_norms = {}
        self.hooks = []
    
    def collect_activations(
        self,
        dataloader,
        device: str,
        num_samples: int = 512
    ):
        """
        Collect activation statistics from calibration data.
        
        WHAT: Run forward passes and record activation magnitudes
        WHY: Need to know which activations are typically large
        HOW: Register hooks on Linear layers to capture inputs
        
        Args:
            dataloader: Calibration dataloader
            device: Device to run on
            num_samples: Number of samples for calibration
        """
        print(f"\n   Collecting activations on {num_samples} samples...")
        
        # Storage for activations
        activation_sums = defaultdict(lambda: None)
        activation_counts = defaultdict(int)
        
        # Register hooks to capture activations
        def make_hook(name):
            def hook(module, input, output):
                if len(input) > 0 and input[0] is not None:
                    # Get L2 norm of input activations
                    act = input[0].detach()
                    act_norm = act.abs().mean(dim=(0, 1))  # [hidden_size] - average over batch AND sequence
                    
                    if activation_sums[name] is None:
                        activation_sums[name] = act_norm
                    else:
                        activation_sums[name] += act_norm
                    activation_counts[name] += 1
            return hook
        
        # Register hooks on prunable modules
        self.hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                is_prunable = any(module is m for m, _ in self.prunable_modules)
                if is_prunable:
                    hook = module.register_forward_hook(make_hook(name))
                    self.hooks.append(hook)
        
        # Run calibration
        self.model.eval()
        samples_seen = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Calibrating"):
                if samples_seen >= num_samples:
                    break
                
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch.get('token_type_ids')
                if token_type_ids is None:
                    token_type_ids = torch.zeros_like(input_ids)
                else:
                    token_type_ids = token_type_ids.to(device)
                
                # Forward pass (hooks will capture activations)
                self.model(input_ids, attention_mask, token_type_ids)
                
                samples_seen += input_ids.shape[0]
        
        # Remove hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
        # Average activations
        for name in activation_sums:
            if activation_counts[name] > 0:
                self.activation_norms[name] = activation_sums[name] / activation_counts[name]
        
        print(f"   [OK] Collected activations for {len(self.activation_norms)} layers")
    
    def apply_wanda_pruning(self):
        """
        Apply Wanda pruning using collected activation statistics.
        
        Must call collect_activations() first!
        """
        if not self.activation_norms:
            raise ValueError("Must call collect_activations() first!")
        
        print(f"\n   Applying Wanda pruning (target: {self.target_sparsity*100:.1f}%)")
        
        # Collect importance scores for all weights
        all_importance_scores = []
        layer_info = []
        
        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            
            # Check if this module is prunable
            is_prunable = any(module is m for m, _ in self.prunable_modules)
            if not is_prunable:
                continue
            
            if name not in self.activation_norms:
                continue
            
            # Get weight magnitudes
            weight = module.weight.data.abs()  # [out, in]
            
            # Get activation norms
            act_norm = self.activation_norms[name]  # [in] or [out, in]
            
            # Handle dimension mismatch
            if act_norm.dim() == 1:
                # Broadcast: importance = |weight| * |activation|
                importance = weight * act_norm.unsqueeze(0)
            else:
                importance = weight * act_norm
            
            all_importance_scores.append(importance.flatten())
            layer_info.append((name, module, importance.shape))
        
        if not all_importance_scores:
            print("   [WARN] No importance scores computed!")
            return
        
        # Calculate global threshold
        all_scores = torch.cat(all_importance_scores)
        threshold = torch.quantile(all_scores, self.target_sparsity)
        
        # Apply pruning masks
        score_idx = 0
        for name, module, shape in layer_info:
            importance = all_importance_scores[score_idx].view(shape)
            
            # Create mask: keep weights with importance >= threshold
            mask = (importance >= threshold).float()
            
            # Apply custom mask
            prune.custom_from_mask(module, 'weight', mask)
            
            score_idx += 1
        
        self.current_sparsity = self.get_sparsity()['overall']
        print(f"   [OK] Applied Wanda pruning: {self.current_sparsity*100:.2f}% sparsity")


# =============================================================================
# STRUCTURED PRUNING
# =============================================================================

class StructuredPruner:
    """
    Structured pruning for transformer models.

    WHAT: Removes entire structures (attention heads, neurons) instead of individual weights
    WHY: Gives REAL speedup without special sparse libraries - works on ANY hardware!
    HOW: Identify unimportant heads/neurons, remove them entirely

    Unstructured vs Structured:
    ──────────────────────────
    Unstructured: [0.5, 0, 0.3, 0, 0.8] → Still 5 elements, need sparse math
    Structured:   [0.5, 0.3, 0.8] → Only 3 elements, regular math works!

    Trade-off:
    - Structured gives less compression than unstructured (typically 20-40%)
    - But structured gives actual wall-clock speedup on standard hardware (Kaggle, etc.)
    - Model parameters are ACTUALLY reduced (not just zeroed)

    Methods implemented:
    1. Attention head pruning - removes entire attention heads
    2. Importance scoring via:
       - Gradient-based (Michel et al., 2019)
       - Activation-based (mean attention output magnitude)
    """

    def __init__(
        self,
        model: nn.Module,
        target_sparsity: float = 0.3,
        importance_method: str = 'activation'
    ):
        """
        Initialize structured pruner.

        Args:
            model: Model to prune (must have encoder with attention layers)
            target_sparsity: Fraction of heads to remove (0.3 = 30% of heads)
            importance_method: 'gradient' or 'activation'
        """
        self.model = model
        self.target_sparsity = target_sparsity
        self.importance_method = importance_method
        self.head_importance = {}
        self.num_layers = 0
        self.num_heads = 0

        # Find encoder structure
        self._find_encoder_structure()

        print(f"\n[Structured Pruning] Initialized:")
        print(f"   Target sparsity: {target_sparsity * 100:.1f}% of heads")
        print(f"   Importance method: {importance_method}")
        print(f"   Detected layers: {self.num_layers}, heads per layer: {self.num_heads}")

    def _find_encoder_structure(self):
        """Find the encoder structure and number of attention heads."""
        # Try different encoder access patterns
        encoder = None

        # Pattern 1: model.encoder (standard BERT)
        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layer'):
            encoder = self.model.encoder
        # Pattern 2: model.model.encoder (wrapped models)
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'encoder'):
            encoder = self.model.model.encoder
        # Pattern 3: model.bert.encoder (BertModel)
        elif hasattr(self.model, 'bert') and hasattr(self.model.bert, 'encoder'):
            encoder = self.model.bert.encoder
        # Pattern 4: model.roberta.encoder (RoBERTa)
        elif hasattr(self.model, 'roberta') and hasattr(self.model.roberta, 'encoder'):
            encoder = self.model.roberta.encoder

        if encoder is not None and hasattr(encoder, 'layer'):
            self.num_layers = len(encoder.layer)
            # Get number of heads from first layer
            first_layer = encoder.layer[0]
            if hasattr(first_layer, 'attention'):
                if hasattr(first_layer.attention, 'self'):
                    self.num_heads = first_layer.attention.self.num_attention_heads
                elif hasattr(first_layer.attention, 'num_attention_heads'):
                    self.num_heads = first_layer.attention.num_attention_heads

        self.encoder = encoder

    def compute_head_importance(
        self,
        dataloader,
        device: str,
        num_samples: int = 256
    ) -> Dict[int, torch.Tensor]:
        """
        Compute importance scores for attention heads.

        Uses activation-based importance: heads that produce larger output
        magnitudes are considered more important.

        For gradient-based, we would compute gradients w.r.t. head outputs.

        Args:
            dataloader: Data for computing importance
            device: Device to run on
            num_samples: Number of samples for estimation

        Returns:
            Dict mapping layer_idx to tensor of head importance scores
        """
        if self.encoder is None:
            print("   [WARN] Could not find encoder structure!")
            return {}

        print(f"\n   Computing head importance ({self.importance_method} method)...")

        # Storage for attention outputs
        attention_outputs = {i: [] for i in range(self.num_layers)}
        hooks = []

        def make_attention_hook(layer_idx):
            """Hook to capture attention outputs."""
            def hook(module, input, output):
                # output is typically (attention_output, attention_weights) or just attention_output
                if isinstance(output, tuple):
                    attn_output = output[0]  # [batch, seq, hidden]
                else:
                    attn_output = output

                # Reshape to per-head: [batch, seq, num_heads, head_dim]
                batch_size, seq_len, hidden = attn_output.shape
                if self.num_heads > 0:
                    head_dim = hidden // self.num_heads
                    attn_output = attn_output.view(batch_size, seq_len, self.num_heads, head_dim)
                    # Compute mean magnitude per head: [num_heads]
                    head_magnitude = attn_output.abs().mean(dim=(0, 1, 3)).detach().cpu()
                    attention_outputs[layer_idx].append(head_magnitude)
            return hook

        # Register hooks on attention output layers
        for layer_idx, layer in enumerate(self.encoder.layer):
            if hasattr(layer, 'attention'):
                if hasattr(layer.attention, 'output'):
                    # BERT-style: attention.output.dense
                    hook = layer.attention.output.register_forward_hook(make_attention_hook(layer_idx))
                    hooks.append(hook)
                elif hasattr(layer.attention, 'self'):
                    # Alternative: hook on self-attention
                    hook = layer.attention.self.register_forward_hook(make_attention_hook(layer_idx))
                    hooks.append(hook)

        # Run forward passes
        self.model.eval()
        samples_seen = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Computing head importance"):
                if samples_seen >= num_samples:
                    break

                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch.get('token_type_ids')
                if token_type_ids is None:
                    token_type_ids = torch.zeros_like(input_ids)
                else:
                    token_type_ids = token_type_ids.to(device)

                try:
                    self.model(input_ids, attention_mask, token_type_ids)
                except:
                    # Some models return different outputs
                    pass

                samples_seen += input_ids.shape[0]

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Average importance scores
        for layer_idx in range(self.num_layers):
            if attention_outputs[layer_idx]:
                stacked = torch.stack(attention_outputs[layer_idx])
                self.head_importance[layer_idx] = stacked.mean(dim=0)
            else:
                # Fallback: uniform importance
                self.head_importance[layer_idx] = torch.ones(self.num_heads)

        print(f"   [OK] Computed importance for {len(self.head_importance)} layers")
        return self.head_importance

    def get_heads_to_prune(self) -> Dict[int, List[int]]:
        """
        Determine which heads to prune based on importance scores.

        Returns:
            Dict mapping layer_idx to list of head indices to prune
        """
        if not self.head_importance:
            print("   [WARN] No importance scores. Call compute_head_importance first!")
            return {}

        # Collect all head scores with their (layer, head) indices
        all_scores = []
        for layer_idx, scores in self.head_importance.items():
            for head_idx, score in enumerate(scores):
                all_scores.append((score.item(), layer_idx, head_idx))

        # Sort by importance (ascending - least important first)
        all_scores.sort(key=lambda x: x[0])

        # Calculate number of heads to prune
        total_heads = len(all_scores)
        num_to_prune = int(total_heads * self.target_sparsity)

        # Don't prune all heads in any layer - keep at least 1
        heads_to_prune = defaultdict(list)
        heads_pruned_per_layer = defaultdict(int)

        for score, layer_idx, head_idx in all_scores[:num_to_prune]:
            # Keep at least 1 head per layer
            if heads_pruned_per_layer[layer_idx] < self.num_heads - 1:
                heads_to_prune[layer_idx].append(head_idx)
                heads_pruned_per_layer[layer_idx] += 1

        total_pruned = sum(len(heads) for heads in heads_to_prune.values())
        print(f"   Selected {total_pruned}/{total_heads} heads for pruning ({total_pruned/total_heads*100:.1f}%)")

        return dict(heads_to_prune)

    def prune_attention_heads(self, heads_to_prune: Optional[Dict[int, List[int]]] = None):
        """
        Prune specified attention heads from the model.

        This ACTUALLY removes the heads, reducing model parameters!

        Args:
            heads_to_prune: Dict mapping layer index to list of head indices to prune.
                           If None, uses get_heads_to_prune() to determine automatically.
        """
        if heads_to_prune is None:
            heads_to_prune = self.get_heads_to_prune()

        if not heads_to_prune:
            print("   [WARN] No heads to prune!")
            return

        print(f"\n   Pruning attention heads...")

        # Count parameters before
        params_before = sum(p.numel() for p in self.model.parameters())

        # Apply pruning using HuggingFace's built-in method
        pruned_count = 0
        for layer_idx, head_indices in heads_to_prune.items():
            if layer_idx < len(self.encoder.layer):
                layer = self.encoder.layer[layer_idx]
                if hasattr(layer, 'attention'):
                    # BERT/RoBERTa style
                    if hasattr(layer.attention, 'prune_heads'):
                        layer.attention.prune_heads(set(head_indices))
                        pruned_count += len(head_indices)
                    elif hasattr(layer.attention, 'self') and hasattr(layer.attention.self, 'prune_heads'):
                        layer.attention.self.prune_heads(set(head_indices))
                        pruned_count += len(head_indices)

        # Count parameters after
        params_after = sum(p.numel() for p in self.model.parameters())
        reduction = (params_before - params_after) / params_before * 100

        print(f"   [OK] Pruned {pruned_count} attention heads")
        print(f"   Parameters: {params_before:,} → {params_after:,} ({reduction:.1f}% reduction)")

        return {
            'heads_pruned': pruned_count,
            'params_before': params_before,
            'params_after': params_after,
            'reduction_pct': reduction
        }

    def apply_structured_pruning(self, dataloader, device: str, num_samples: int = 256):
        """
        Complete structured pruning pipeline.

        1. Compute head importance scores
        2. Identify heads to prune
        3. Remove heads from model

        Args:
            dataloader: Data for importance estimation
            device: Device to run on
            num_samples: Samples for importance computation

        Returns:
            Dict with pruning statistics
        """
        print(f"\n[Structured Pruning] Starting pipeline...")

        # Step 1: Compute importance
        self.compute_head_importance(dataloader, device, num_samples)

        # Step 2 & 3: Get heads and prune
        heads_to_prune = self.get_heads_to_prune()
        stats = self.prune_attention_heads(heads_to_prune)

        return stats


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def get_pruner(method: str, model: nn.Module, config):
    """
    Factory function to create appropriate pruner.

    WHAT: Creates the right pruner based on method name
    WHY: Clean API - just specify method name
    HOW: Switch statement on method, create with config

    Args:
        method: 'magnitude', 'gradual', 'wanda', or 'structured'
        model: Model to prune
        config: Configuration with pruning parameters

    Returns:
        Appropriate pruner (PruningManager subclass or StructuredPruner)
    """
    if method == 'magnitude':
        return PruningManager(
            model=model,
            target_sparsity=config.prune_sparsity,
            prune_layers=config.prune_layers,
            global_pruning=True
        )

    elif method == 'gradual':
        return GradualPruner(
            model=model,
            target_sparsity=config.prune_sparsity,
            start_epoch=config.prune_start_epoch,
            end_epoch=config.prune_end_epoch,
            schedule=config.prune_schedule,
            prune_frequency=config.prune_frequency,
            prune_layers=config.prune_layers,
            global_pruning=True
        )

    elif method == 'wanda':
        return WandaPruner(
            model=model,
            target_sparsity=config.prune_sparsity,
            prune_layers=config.prune_layers,
            global_pruning=True
        )

    elif method == 'structured':
        return StructuredPruner(
            model=model,
            target_sparsity=config.prune_sparsity,
            importance_method='activation'
        )

    else:
        raise ValueError(f"Unknown pruning method: {method}. Available: magnitude, gradual, wanda, structured")


# =============================================================================
# POST-PRUNING FINE-TUNING
# =============================================================================

def fine_tune_after_pruning(
    model: nn.Module,
    train_loader,
    val_loader,
    config,
    device: str,
    use_student_input_ids: bool = False,
    pruning_masks: Optional[Dict[str, torch.Tensor]] = None,
    class_weights: Optional[torch.Tensor] = None
) -> Dict:
    """
    Fine-tune model after pruning to recover accuracy.

    WHAT: Continue training the pruned model for a few epochs
    WHY: Pruning hurts accuracy; fine-tuning helps recover
    HOW: Standard training loop with lower learning rate
    
    IMPORTANT: If pruning_masks is provided, gradients for pruned weights
               will be zeroed out to maintain sparsity during fine-tuning.

    Args:
        model: Pruned model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        config: Training configuration
        device: Device to train on
        use_student_input_ids: If True, use student_input_ids from batch
                               (for models that were trained with student tokenizer)
        pruning_masks: Optional dict mapping parameter names to mask tensors.
                      If provided, enforces sparsity during fine-tuning by
                      zeroing gradients for pruned weights.
        class_weights: Optional tensor of class weights for handling imbalanced
                      multi-label classification. Prevents minority class collapse.

    Returns:
        Dict with detailed fine-tuning metrics including:
        - best_f1: Best validation F1 achieved
        - best_epoch: Epoch with best F1
        - final_f1: F1 at final epoch
        - final_loss: Loss at final epoch
        - epoch_history: List of dicts with per-epoch metrics
    """
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup
    from sklearn.metrics import f1_score, precision_score, recall_score

    print(f"\n[Fine-tune] Fine-tuning pruned model for {config.fine_tune_epochs} epochs...")

    model.train()

    # Use lower learning rate for fine-tuning
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.lr * 0.1,  # 10x lower than initial training
        weight_decay=config.weight_decay
    )

    total_steps = len(train_loader) * config.fine_tune_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # Use weighted loss to prevent minority class collapse after pruning
    if class_weights is not None:
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights.to(device))
        print(f"   [Fine-tune] Using weighted loss (class_weights provided)")
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    # Tracking metrics
    best_f1 = 0
    best_epoch = 0
    best_metrics = {}
    epoch_history = []

    for epoch in range(config.fine_tune_epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        for batch in tqdm(train_loader, desc=f'Fine-tune Epoch {epoch+1}'):
            # Handle dual tokenization - use student tokens if specified
            if use_student_input_ids and 'student_input_ids' in batch:
                input_ids = batch['student_input_ids'].to(device)
                attention_mask = batch['student_attention_mask'].to(device)
                token_type_ids = batch.get('student_token_type_ids')
            else:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch.get('token_type_ids')
            
            # Create default token_type_ids (zeros) if not in batch
            if token_type_ids is None:
                token_type_ids = torch.zeros_like(input_ids)
            else:
                token_type_ids = token_type_ids.to(device)

            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, token_type_ids)
            loss = loss_fn(outputs['logits'], labels)
            loss.backward()
            
            # Enforce sparsity by zeroing gradients for pruned weights
            if pruning_masks is not None:
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if name in pruning_masks and param.grad is not None:
                            # Multiply gradient by mask: pruned positions get 0 gradient
                            param.grad.mul_(pruning_masks[name].to(param.device))

            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1

        avg_train_loss = total_loss / num_batches if num_batches > 0 else 0

        # Evaluate
        model.eval()
        all_preds = []
        all_labels = []
        val_loss = 0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                # Handle dual tokenization for validation
                if use_student_input_ids and 'student_input_ids' in batch:
                    input_ids = batch['student_input_ids'].to(device)
                    attention_mask = batch['student_attention_mask'].to(device)
                    token_type_ids = batch.get('student_token_type_ids')
                else:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    token_type_ids = batch.get('token_type_ids')
                
                # Create default token_type_ids (zeros) if not in batch
                if token_type_ids is None:
                    token_type_ids = torch.zeros_like(input_ids)
                else:
                    token_type_ids = token_type_ids.to(device)

                labels_batch = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask, token_type_ids)

                # Calculate validation loss
                batch_loss = loss_fn(outputs['logits'], labels_batch)
                val_loss += batch_loss.item()
                val_batches += 1

                preds = (torch.sigmoid(outputs['logits']) > 0.5).cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(batch['labels'].numpy())

        # Calculate metrics
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)

        # Record epoch metrics
        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro
        }
        epoch_history.append(epoch_metrics)

        print(f"   Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, "
              f"F1_weighted={f1_weighted:.4f}, F1_macro={f1_macro:.4f}")

        # Track best model (using f1_weighted as primary metric)
        if f1_weighted > best_f1:
            best_f1 = f1_weighted
            best_epoch = epoch + 1
            best_metrics = {
                'f1_weighted': f1_weighted,
                'f1_macro': f1_macro,
                'precision_macro': precision_macro,
                'recall_macro': recall_macro,
                'val_loss': avg_val_loss
            }

    print(f"   [OK] Fine-tuning complete. Best F1_weighted: {best_f1:.4f} (Epoch {best_epoch})")

    # Return detailed metrics
    return {
        'best_f1': best_f1,
        'best_epoch': best_epoch,
        'best_metrics': best_metrics,
        'final_f1': epoch_history[-1]['f1_weighted'] if epoch_history else 0,
        'final_loss': epoch_history[-1]['val_loss'] if epoch_history else 0,
        'epoch_history': epoch_history,
        'total_epochs': config.fine_tune_epochs
    }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing pruning module...")
    
    # Create a simple model for testing
    model = nn.Sequential(
        nn.Linear(768, 256),
        nn.ReLU(),
        nn.Linear(256, 5)
    )
    
    # Test magnitude pruning
    manager = PruningManager(model, target_sparsity=0.5)
    
    print(f"Before pruning:")
    sparsity = manager.get_sparsity()
    print(f"  Overall sparsity: {sparsity['overall']*100:.2f}%")
    
    manager.apply_magnitude_pruning()
    
    print(f"\nAfter pruning:")
    sparsity = manager.get_sparsity()
    print(f"  Overall sparsity: {sparsity['overall']*100:.2f}%")
    
    # Make permanent
    manager.make_pruning_permanent()
    
    print("\n[OK] Pruning module tests passed!")
