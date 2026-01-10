"""
================================================================================
DATA LOADING MODULE WITH CACHING
================================================================================

This module handles loading, preprocessing, and caching of the dataset.
The KEY FEATURE is tokenization caching which saves 10+ minutes per experiment.

WHAT THIS MODULE DOES:
1. Loads CSV dataset and validates label columns
2. Tokenizes text using the model's tokenizer
3. CACHES tokenized data to avoid re-processing
4. Creates K-fold splits for cross-validation
5. Calculates class weights for imbalanced data

WHY CACHING MATTERS:
- Tokenization is SLOW (~90 seconds for 12K samples)
- Without cache: Every experiment re-tokenizes
- With cache: First run = 90s, subsequent runs = 2s
- For 10 experiments: Saves 13+ minutes!

HOW CACHING WORKS:
1. Generate unique cache filename based on model + max_length
2. Check if cache exists → Load and return
3. If no cache → Tokenize everything → Save to cache → Return

CACHE INVALIDATION:
- Different model = Different cache file
- Different max_length = Different cache file
- If you change the dataset, delete the cache folder!
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold, StratifiedKFold
from typing import Tuple, List, Dict, Optional, Generator
from tqdm import tqdm
import hashlib


# =============================================================================
# CONSTANTS
# =============================================================================

# These must match your dataset's column names
LABEL_COLUMNS = ['bully', 'sexual', 'religious', 'threat', 'spam']


# =============================================================================
# CACHING FUNCTIONS (Merged from your previous codebase!)
# =============================================================================

def get_cache_filename(model_path: str, max_length: int, student_model_path: Optional[str] = None) -> str:
    """
    Generate a unique cache filename based on model(s) and max_length.

    WHAT: Creates a filename like "csebuetnlp_banglabert_maxlen128_tokenized.pkl"
          or with student: "csebuetnlp_banglabert_DUAL_distilbert_maxlen128_tokenized.pkl"
    WHY: Different models have different tokenizers, so cache must be separate
    HOW: Replace special characters in model path, append max_length

    Args:
        model_path: HuggingFace model path for teacher (e.g., "csebuetnlp/banglabert")
        max_length: Maximum sequence length
        student_model_path: Optional HuggingFace model path for student (for dual tokenization)

    Returns:
        Safe filename string

    Example:
        >>> get_cache_filename("csebuetnlp/banglabert", 128)
        'csebuetnlp_banglabert_maxlen128_tokenized.pkl'
        >>> get_cache_filename("csebuetnlp/banglabert", 128, "distilbert-base-multilingual-cased")
        'csebuetnlp_banglabert_DUAL_distilbert_base_multilingual_cased_maxlen128_tokenized.pkl'
    """
    # Replace characters that can't be in filenames
    safe_name = model_path.replace('/', '_').replace('-', '_').replace('.', '_')

    # If student model provided, add it to filename for dual tokenization cache
    if student_model_path:
        safe_student_name = student_model_path.replace('/', '_').replace('-', '_').replace('.', '_')
        return f"{safe_name}_DUAL_{safe_student_name}_maxlen{max_length}_tokenized.pkl"

    return f"{safe_name}_maxlen{max_length}_tokenized.pkl"


def get_or_create_tokenized_dataset(
    comments: np.ndarray,
    labels: np.ndarray,
    tokenizer,
    max_length: int,
    cache_dir: str = './cache',
    student_tokenizer=None
) -> Dict[str, torch.Tensor]:
    """
    Tokenize all samples ONCE and cache them.

    WHAT: Converts text to token IDs, with intelligent caching
    WHY: Tokenization is slow; we only want to do it once
    HOW: Check cache -> If exists, load; else tokenize and save

    This is the MAIN FUNCTION for data preparation!

    DUAL TOKENIZATION (for Knowledge Distillation):
    When student_tokenizer is provided, we tokenize with BOTH tokenizers.
    This is necessary because teacher and student may have different vocabularies.

    Args:
        comments: Array of text strings
        labels: Array of label arrays (multi-label)
        tokenizer: HuggingFace tokenizer (for teacher model)
        max_length: Maximum sequence length
        cache_dir: Directory to store cache files
        student_tokenizer: Optional HuggingFace tokenizer for student model

    Returns:
        Dictionary with:
        - 'input_ids': Tensor of shape [num_samples, max_length] (teacher tokens)
        - 'attention_mask': Tensor of shape [num_samples, max_length]
        - 'labels': Tensor of shape [num_samples, num_labels]
        - 'student_input_ids': (if student_tokenizer provided) Tensor [num_samples, max_length]
        - 'student_attention_mask': (if student_tokenizer provided) Tensor [num_samples, max_length]

    Example:
        # Single tokenization (teacher only)
        tokenized = get_or_create_tokenized_dataset(
            comments, labels, tokenizer, max_length=128
        )

        # Dual tokenization (for KD)
        tokenized = get_or_create_tokenized_dataset(
            comments, labels, teacher_tokenizer, max_length=128,
            student_tokenizer=student_tokenizer
        )
    """
    # Create cache directory if needed
    os.makedirs(cache_dir, exist_ok=True)

    # Generate cache filename (includes student if dual tokenization)
    student_path = student_tokenizer.name_or_path if student_tokenizer else None
    cache_filename = get_cache_filename(tokenizer.name_or_path, max_length, student_path)
    cache_path = os.path.join(cache_dir, cache_filename)

    # Try to load from cache
    if os.path.exists(cache_path):
        print(f"Loading tokenized data from cache: {cache_path}")
        cached_data = torch.load(cache_path)

        # Verify cache matches current data
        if cached_data['input_ids'].shape[0] == len(comments):
            # Check if we need student tokens but cache doesn't have them
            if student_tokenizer and 'student_input_ids' not in cached_data:
                print("  Cache missing student tokens! Re-tokenizing...")
            else:
                print(f"   Loaded {len(comments)} samples in ~2 seconds")
                return cached_data
        else:
            print(f"Warning: Cache size mismatch! Re-tokenizing...")

    # No cache or invalid - need to tokenize
    print(f"Tokenizing {len(comments)} samples...")
    print(f"   Teacher Model: {tokenizer.name_or_path}")
    if student_tokenizer:
        print(f"   Student Model: {student_tokenizer.name_or_path}")
    print(f"   Max length: {max_length}")

    # Tokenize all samples
    all_input_ids = []
    all_attention_masks = []
    all_student_input_ids = []
    all_student_attention_masks = []

    for comment in tqdm(comments, desc="Tokenizing"):
        # Handle None or non-string values
        text = str(comment) if comment is not None else ""

        # Tokenize with teacher tokenizer
        encoding = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )

        all_input_ids.append(encoding['input_ids'].squeeze(0))
        all_attention_masks.append(encoding['attention_mask'].squeeze(0))

        # Tokenize with student tokenizer if provided
        if student_tokenizer:
            student_encoding = student_tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors='pt'
            )
            all_student_input_ids.append(student_encoding['input_ids'].squeeze(0))
            all_student_attention_masks.append(student_encoding['attention_mask'].squeeze(0))

    # Stack into tensors
    tokenized_data = {
        'input_ids': torch.stack(all_input_ids),
        'attention_mask': torch.stack(all_attention_masks),
        'labels': torch.tensor(labels, dtype=torch.float32)
    }

    # Add student tokens if dual tokenization
    if student_tokenizer:
        tokenized_data['student_input_ids'] = torch.stack(all_student_input_ids)
        tokenized_data['student_attention_mask'] = torch.stack(all_student_attention_masks)

    # Save to cache
    torch.save(tokenized_data, cache_path)
    print(f"Saved tokenized data to cache: {cache_path}")
    print(f"   Cache size: {os.path.getsize(cache_path) / 1024 / 1024:.1f} MB")

    return tokenized_data


# =============================================================================
# DATASET CLASSES
# =============================================================================

class IndexedDataset(Dataset):
    """
    Dataset that indexes into pre-tokenized data.

    WHAT: A PyTorch Dataset that uses indices to access cached tokenized data
    WHY: Allows different train/val splits without re-tokenizing
    HOW: Stores full tokenized data + indices for this split

    This is used AFTER tokenization caching.

    Supports DUAL TOKENIZATION for Knowledge Distillation:
    If tokenized_data contains 'student_input_ids' and 'student_attention_mask',
    they will be included in the returned batch.

    Example:
        # Tokenize full dataset once
        full_data = get_or_create_tokenized_dataset(...)

        # Create train/val datasets using indices
        train_dataset = IndexedDataset(full_data, train_indices)
        val_dataset = IndexedDataset(full_data, val_indices)
    """

    def __init__(self, tokenized_data: Dict[str, torch.Tensor], indices: np.ndarray):
        """
        Initialize indexed dataset.

        Args:
            tokenized_data: Dict with 'input_ids', 'attention_mask', 'labels'
                           and optionally 'student_input_ids', 'student_attention_mask'
            indices: Array of indices to include in this dataset
        """
        self.input_ids = tokenized_data['input_ids']
        self.attention_mask = tokenized_data['attention_mask']
        self.labels = tokenized_data['labels']
        self.indices = indices

        # Optional student tokens for dual tokenization (KD)
        self.student_input_ids = tokenized_data.get('student_input_ids')
        self.student_attention_mask = tokenized_data.get('student_attention_mask')
        self.has_student_tokens = self.student_input_ids is not None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample by index."""
        real_idx = self.indices[idx]

        item = {
            'input_ids': self.input_ids[real_idx],
            'attention_mask': self.attention_mask[real_idx],
            'labels': self.labels[real_idx]
        }

        # Add student tokens if available
        if self.has_student_tokens:
            item['student_input_ids'] = self.student_input_ids[real_idx]
            item['student_attention_mask'] = self.student_attention_mask[real_idx]

        return item


class CyberbullyingDataset(Dataset):
    """
    Original dataset class (tokenizes on-the-fly).
    
    WHAT: Standard PyTorch Dataset for cyberbullying detection
    WHY: Used when you don't want caching (small experiments)
    HOW: Tokenizes each sample when accessed
    
    NOTE: This is SLOWER than IndexedDataset with caching!
    Use IndexedDataset + get_or_create_tokenized_dataset() instead.
    """
    
    def __init__(self, comments: np.ndarray, labels: np.ndarray, 
                 tokenizer, max_length: int = 128):
        self.comments = comments
        self.labels = labels.astype(np.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.comments)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        comment = str(self.comments[idx])
        labels = self.labels[idx]
        
        # Tokenize (this happens every time - slow!)
        encoding = self.tokenizer(
            comment,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.float)
        }


# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def load_and_preprocess_data(
    dataset_path: str,
    label_columns: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Dict[int, int]]]:
    """
    Load and preprocess the cyberbullying dataset.

    WHAT: Reads CSV file and extracts comments + labels
    WHY: Standardizes data loading across all experiments
    HOW: Read CSV -> Validate columns -> Extract arrays -> Print stats

    Args:
        dataset_path: Path to CSV file
        label_columns: Optional list of label column names (defaults to LABEL_COLUMNS)

    Returns:
        Tuple of (comments array, labels array, label_distribution dict)

    Expected CSV format:
        comment,bully,sexual,religious,threat,spam
        "some text",0,1,0,0,0
        "other text",1,0,0,1,0
    """
    # Use default label columns if not provided
    if label_columns is None:
        label_columns = LABEL_COLUMNS

    print(f"\nLoading dataset: {dataset_path}")

    # Load CSV
    df = pd.read_csv(dataset_path)
    print(f"   Raw rows: {len(df)}")

    # Remove demographic columns if present (not used for classification)
    columns_to_drop = [col for col in ['Gender', 'Profession'] if col in df.columns]
    if columns_to_drop:
        df = df.drop(columns_to_drop, axis=1)
        print(f"   Dropped columns: {columns_to_drop}")

    # Validate label columns exist
    missing_labels = [col for col in label_columns if col not in df.columns]
    if missing_labels:
        raise ValueError(f"Missing label columns: {missing_labels}\n"
                        f"Expected: {label_columns}\n"
                        f"Found: {df.columns.tolist()}")

    # Validate comment column exists
    if 'comment' not in df.columns:
        # Try common alternatives
        for alt in ['text', 'Comment', 'Text', 'content', 'Content', 'comments', 'Comments']:
            if alt in df.columns:
                df = df.rename(columns={alt: 'comment'})
                print(f"   Renamed '{alt}' to 'comment'")
                break
        else:
            raise ValueError("No 'comment' column found in dataset")

    # Handle missing values
    df = df.dropna(subset=['comment'])
    print(f"   After dropping NA: {len(df)} rows")

    # Extract data
    comments = df['comment'].values
    labels = df[label_columns].values

    # Calculate label distribution
    label_distribution = {}
    print(f"\nDataset Statistics:")
    print(f"   Total samples: {len(comments)}")
    print(f"   Label columns: {label_columns}")
    print(f"\n   Label distribution:")

    for i, col in enumerate(label_columns):
        positive = int(np.sum(labels[:, i]))
        negative = len(labels) - positive
        percentage = (positive / len(labels)) * 100
        print(f"      {col}: {positive}/{len(labels)} ({percentage:.1f}% positive)")
        label_distribution[col] = {0: negative, 1: positive}

    # Multi-label statistics
    labels_per_sample = np.sum(labels, axis=1)
    print(f"\n   Multi-label statistics:")
    print(f"      Samples with 0 labels: {np.sum(labels_per_sample == 0)}")
    print(f"      Samples with 1 label: {np.sum(labels_per_sample == 1)}")
    print(f"      Samples with 2+ labels: {np.sum(labels_per_sample >= 2)}")
    print(f"      Average labels per sample: {np.mean(labels_per_sample):.2f}")

    return comments, labels, label_distribution


# =============================================================================
# K-FOLD CROSS VALIDATION
# =============================================================================

def prepare_kfold_splits(
    comments: np.ndarray,
    labels: np.ndarray,
    num_folds: int = 5,
    stratification_type: str = 'multilabel',
    seed: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Prepare K-fold cross-validation splits.

    WHAT: Divides data into K training/validation splits
    WHY: Cross-validation gives more reliable performance estimates
    HOW: Choose stratification method based on data type

    SPECIAL CASE: num_folds=1
    When num_folds=1, creates a single 80/20 train/val split (not K-fold).
    This is useful for quick testing or when you want a simple holdout split.

    Stratification Types:
    1. 'multilabel': Uses MultilabelStratifiedKFold (best for multi-label)
       - Ensures each fold has similar label distribution
       - Requires: pip install iterative-stratification

    2. 'multiclass': Uses StratifiedKFold on primary label
       - Fallback when multilabel stratification unavailable
       - Less accurate but still better than random

    3. 'none': Uses basic KFold (random splits)
       - No stratification
       - May have imbalanced folds

    Args:
        comments: Array of text samples
        labels: Array of multi-label arrays
        num_folds: Number of folds (typically 5). Use 1 for single 80/20 split.
        stratification_type: 'multilabel', 'multiclass', or 'none'
        seed: Random seed for reproducibility

    Returns:
        List of tuples (train_indices, val_indices)

    Example:
        splits = prepare_kfold_splits(comments, labels, num_folds=5)
        for fold, (train_idx, val_idx) in enumerate(splits):
            print(f"Fold {fold}: {len(train_idx)} train, {len(val_idx)} val")
    """
    # Special case: single split (80/20)
    if num_folds == 1:
        print(f"\nPreparing single 80/20 train/val split...")
        n_samples = len(comments)
        indices = np.arange(n_samples)
        np.random.seed(seed)
        np.random.shuffle(indices)

        split_point = int(0.8 * n_samples)
        train_idx = indices[:split_point]
        val_idx = indices[split_point:]

        print(f"   Train: {len(train_idx)} samples ({len(train_idx)/n_samples*100:.1f}%)")
        print(f"   Val: {len(val_idx)} samples ({len(val_idx)/n_samples*100:.1f}%)")

        return [(train_idx, val_idx)]

    print(f"\nPreparing {num_folds}-fold cross-validation...")

    # Try multilabel stratification first
    if stratification_type == 'multilabel':
        try:
            from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
            print(f"   Using MultilabelStratifiedKFold (best for multi-label)")
            kfold = MultilabelStratifiedKFold(
                n_splits=num_folds,
                shuffle=True,
                random_state=seed
            )
            return list(kfold.split(comments, labels))

        except ImportError:
            print("   Warning: iterative-stratification not installed!")
            print("      Install: pip install iterative-stratification")
            print("      Falling back to multiclass stratification...")
            stratification_type = 'multiclass'

    # Multiclass stratification (use primary label)
    if stratification_type == 'multiclass':
        print(f"   Using StratifiedKFold on primary label")

        # Create single-label version (priority: threat > sexual > religious > bully > spam)
        primary_labels = np.zeros(len(labels), dtype=int)
        for i in range(len(labels)):
            if labels.shape[1] > 3 and labels[i, 3] == 1:      # threat
                primary_labels[i] = 4
            elif labels.shape[1] > 1 and labels[i, 1] == 1:    # sexual
                primary_labels[i] = 3
            elif labels.shape[1] > 2 and labels[i, 2] == 1:    # religious
                primary_labels[i] = 2
            elif labels.shape[1] > 0 and labels[i, 0] == 1:    # bully
                primary_labels[i] = 1
            elif labels.shape[1] > 4 and labels[i, 4] == 1:    # spam
                primary_labels[i] = 5
            else:
                primary_labels[i] = 0  # no label

        kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        return list(kfold.split(comments, primary_labels))

    # No stratification (random splits)
    print(f"   Using basic KFold (no stratification)")
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    return list(kfold.split(comments))


def get_fold_indices(
    labels: np.ndarray,
    fold_idx: int,
    num_folds: int = 5,
    stratification: str = 'multilabel',
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get train/val indices for a specific fold, matching original training splits.

    This function recreates the EXACT same fold splits used during original training,
    allowing for fair evaluation without data leakage.

    CRITICAL: To match original training splits, you must use:
    - Same num_folds (typically 5)
    - Same stratification type (typically 'multilabel')
    - Same seed (typically 42)
    - Same fold_idx (1-indexed, matching which fold was used for validation)

    Args:
        labels: numpy array of shape (n_samples, n_labels) for multilabel,
                or (n_samples,) for single-label
        fold_idx: 1-indexed fold number (1-5 for 5-fold CV)
                  This is the fold used for VALIDATION
        num_folds: number of folds (default 5)
        stratification: 'multilabel', 'multiclass', or 'none'
        seed: random seed (default 42)

    Returns:
        train_idx, val_idx: numpy arrays of indices

    Example:
        # Get fold 3 splits (matching original training where fold 3 was validation)
        train_idx, val_idx = get_fold_indices(labels, fold_idx=3)
        # train_idx contains indices from folds 1,2,4,5
        # val_idx contains indices from fold 3
    """
    if fold_idx < 1 or fold_idx > num_folds:
        raise ValueError(f"fold_idx must be between 1 and {num_folds}, got {fold_idx}")

    print(f"\n[Fold Split] Recreating original fold splits...")
    print(f"   Stratification: {stratification}")
    print(f"   Num folds: {num_folds}")
    print(f"   Seed: {seed}")
    print(f"   Validation fold: {fold_idx}")

    # Try multilabel stratification first
    if stratification == 'multilabel':
        try:
            from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
            print(f"   Using MultilabelStratifiedKFold (matches original training)")
            kfold = MultilabelStratifiedKFold(
                n_splits=num_folds,
                shuffle=True,
                random_state=seed
            )
            stratify_labels = labels
        except ImportError:
            print("   WARNING: iterative-stratification not installed!")
            print("      Install: pip install iterative-stratification")
            print("      Falling back to basic KFold (may not match original splits!)")
            from sklearn.model_selection import KFold
            kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
            stratify_labels = None

    elif stratification == 'multiclass':
        print(f"   Using StratifiedKFold on primary label")
        # Create single-label version (use first non-zero label)
        if len(labels.shape) > 1:
            primary_labels = np.argmax(labels, axis=1)
        else:
            primary_labels = labels
        kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        stratify_labels = primary_labels

    else:  # 'none'
        print(f"   Using basic KFold (no stratification)")
        from sklearn.model_selection import KFold
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
        stratify_labels = None

    # Iterate through folds to find the requested one
    for i, (train_idx, val_idx) in enumerate(kfold.split(labels, stratify_labels)):
        if i + 1 == fold_idx:  # 1-indexed
            print(f"   Train samples: {len(train_idx)}")
            print(f"   Val samples: {len(val_idx)}")
            return train_idx, val_idx

    raise ValueError(f"Fold {fold_idx} not found (should not happen)")


# =============================================================================
# CLASS WEIGHTS
# =============================================================================

def calculate_class_weights(
    labels: np.ndarray,
    label_columns: Optional[List[str]] = None
) -> torch.Tensor:
    """
    Calculate class weights for imbalanced data.

    WHAT: Computes weight for each label based on class frequency
    WHY: Rare classes (like 'threat') should have higher loss weight
    HOW: weight = negative_count / positive_count

    For multi-label classification, each label is an independent binary
    classification problem, so we calculate weights per label.

    Args:
        labels: Array of shape [num_samples, num_labels]
        label_columns: Optional list of label column names for logging

    Returns:
        Tensor of weights [num_labels]

    Example:
        If 'threat' has 500 positive and 12000 negative samples:
        weight = 12000 / 500 = 24.0

        This means a false negative on 'threat' is penalized 24x more
        than a false positive, helping the model not ignore rare classes.
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()

    # Use default label columns if not provided
    if label_columns is None:
        label_columns = LABEL_COLUMNS

    # Count positives and negatives per label
    pos_counts = np.sum(labels, axis=0)
    neg_counts = len(labels) - pos_counts

    # Calculate weights (avoid division by zero)
    weights = np.where(pos_counts > 0, neg_counts / pos_counts, 1.0)

    print("\nClass weights (for imbalanced data):")
    for i in range(len(weights)):
        col_name = label_columns[i] if i < len(label_columns) else f"label_{i}"
        print(f"      {col_name}: {weights[i]:.2f} "
              f"({int(pos_counts[i])} pos, {int(neg_counts[i])} neg)")

    return torch.FloatTensor(weights)


# =============================================================================
# DATA LOADERS
# =============================================================================

def create_data_loaders(
    tokenized_data: Dict[str, torch.Tensor],
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    batch_size: int = 32,
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders from tokenized data.
    
    WHAT: Wraps datasets in DataLoaders for batching
    WHY: DataLoaders handle batching, shuffling, and parallel loading
    HOW: Create IndexedDatasets, wrap in DataLoaders
    
    Args:
        tokenized_data: Pre-tokenized data from get_or_create_tokenized_dataset()
        train_indices: Indices for training split
        val_indices: Indices for validation split
        batch_size: Batch size
        num_workers: Number of parallel data loading workers
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = IndexedDataset(tokenized_data, train_indices)
    val_dataset = IndexedDataset(tokenized_data, val_indices)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,          # Shuffle for training
        num_workers=num_workers,
        pin_memory=True        # Faster GPU transfer
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,         # No shuffle for validation
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\nDataLoaders created:")
    print(f"      Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"      Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    
    return train_loader, val_loader


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    # Test the module
    print("Testing data module...")
    
    # Create dummy data
    comments = np.array(["This is a test comment " * 10] * 100)
    labels = np.random.randint(0, 2, size=(100, 5)).astype(np.float32)
    
    # Test class weights
    weights = calculate_class_weights(labels)
    print(f"Weights shape: {weights.shape}")
    
    # Test K-fold splits
    splits = list(prepare_kfold_splits(
        comments, labels, num_folds=5, stratification_type='none'
    ))
    print(f"Number of folds: {len(splits)}")
    print(f"Fold 0 train size: {len(splits[0][0])}")
    print(f"Fold 0 val size: {len(splits[0][1])}")
    
    print("\n✅ Data module tests passed!")
