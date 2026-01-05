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

def get_cache_filename(model_path: str, max_length: int) -> str:
    """
    Generate a unique cache filename based on model and max_length.
    
    WHAT: Creates a filename like "csebuetnlp_banglabert_maxlen128_tokenized.pkl"
    WHY: Different models have different tokenizers, so cache must be separate
    HOW: Replace special characters in model path, append max_length
    
    Args:
        model_path: HuggingFace model path (e.g., "csebuetnlp/banglabert")
        max_length: Maximum sequence length
    
    Returns:
        Safe filename string
    
    Example:
        >>> get_cache_filename("csebuetnlp/banglabert", 128)
        'csebuetnlp_banglabert_maxlen128_tokenized.pkl'
    """
    # Replace characters that can't be in filenames
    safe_name = model_path.replace('/', '_').replace('-', '_').replace('.', '_')
    return f"{safe_name}_maxlen{max_length}_tokenized.pkl"


def get_or_create_tokenized_dataset(
    comments: np.ndarray,
    labels: np.ndarray,
    tokenizer,
    max_length: int,
    cache_dir: str = './cache'
) -> Dict[str, torch.Tensor]:
    """
    Tokenize all samples ONCE and cache them.
    
    WHAT: Converts text to token IDs, with intelligent caching
    WHY: Tokenization is slow; we only want to do it once
    HOW: Check cache → If exists, load; else tokenize and save
    
    This is the MAIN FUNCTION for data preparation!
    
    Args:
        comments: Array of text strings
        labels: Array of label arrays (multi-label)
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        cache_dir: Directory to store cache files
    
    Returns:
        Dictionary with:
        - 'input_ids': Tensor of shape [num_samples, max_length]
        - 'attention_mask': Tensor of shape [num_samples, max_length]
        - 'labels': Tensor of shape [num_samples, num_labels]
    
    Example:
        tokenized = get_or_create_tokenized_dataset(
            comments, labels, tokenizer, max_length=128
        )
        # First run: ~90 seconds (tokenizing)
        # Second run: ~2 seconds (loading cache)
    """
    # Create cache directory if needed
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate cache filename
    cache_filename = get_cache_filename(tokenizer.name_or_path, max_length)
    cache_path = os.path.join(cache_dir, cache_filename)
    
    # Try to load from cache
    if os.path.exists(cache_path):
        print(f"✅ Loading tokenized data from cache: {cache_path}")
        cached_data = torch.load(cache_path)
        
        # Verify cache matches current data
        if cached_data['input_ids'].shape[0] == len(comments):
            print(f"   Loaded {len(comments)} samples in ~2 seconds")
            return cached_data
        else:
            print(f"⚠️  Cache size mismatch! Re-tokenizing...")
    
    # No cache or invalid - need to tokenize
    print(f"🔄 Tokenizing {len(comments)} samples...")
    print(f"   Model: {tokenizer.name_or_path}")
    print(f"   Max length: {max_length}")
    
    # Tokenize all samples
    all_input_ids = []
    all_attention_masks = []
    
    for comment in tqdm(comments, desc="Tokenizing"):
        # Handle None or non-string values
        text = str(comment) if comment is not None else ""
        
        # Tokenize
        encoding = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        all_input_ids.append(encoding['input_ids'].squeeze(0))
        all_attention_masks.append(encoding['attention_mask'].squeeze(0))
    
    # Stack into tensors
    tokenized_data = {
        'input_ids': torch.stack(all_input_ids),
        'attention_mask': torch.stack(all_attention_masks),
        'labels': torch.tensor(labels, dtype=torch.float32)
    }
    
    # Save to cache
    torch.save(tokenized_data, cache_path)
    print(f"✅ Saved tokenized data to cache: {cache_path}")
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
            indices: Array of indices to include in this dataset
        """
        self.input_ids = tokenized_data['input_ids']
        self.attention_mask = tokenized_data['attention_mask']
        self.labels = tokenized_data['labels']
        self.indices = indices
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample by index."""
        real_idx = self.indices[idx]
        return {
            'input_ids': self.input_ids[real_idx],
            'attention_mask': self.attention_mask[real_idx],
            'labels': self.labels[real_idx]
        }


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

def load_and_preprocess_data(dataset_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess the cyberbullying dataset.
    
    WHAT: Reads CSV file and extracts comments + labels
    WHY: Standardizes data loading across all experiments
    HOW: Read CSV → Validate columns → Extract arrays → Print stats
    
    Args:
        dataset_path: Path to CSV file
    
    Returns:
        Tuple of (comments array, labels array)
    
    Expected CSV format:
        comment,bully,sexual,religious,threat,spam
        "some text",0,1,0,0,0
        "other text",1,0,0,1,0
    """
    print(f"\n📁 Loading dataset: {dataset_path}")
    
    # Load CSV
    df = pd.read_csv(dataset_path)
    print(f"   Raw rows: {len(df)}")
    
    # Remove demographic columns if present (not used for classification)
    columns_to_drop = [col for col in ['Gender', 'Profession'] if col in df.columns]
    if columns_to_drop:
        df = df.drop(columns_to_drop, axis=1)
        print(f"   Dropped columns: {columns_to_drop}")
    
    # Validate label columns exist
    missing_labels = [col for col in LABEL_COLUMNS if col not in df.columns]
    if missing_labels:
        raise ValueError(f"Missing label columns: {missing_labels}\n"
                        f"Expected: {LABEL_COLUMNS}\n"
                        f"Found: {df.columns.tolist()}")
    
    # Validate comment column exists
    if 'comment' not in df.columns:
        # Try common alternatives
        for alt in ['text', 'Comment', 'Text', 'content', 'Content']:
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
    labels = df[LABEL_COLUMNS].values
    
    # Print statistics
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total samples: {len(comments)}")
    print(f"   Label columns: {LABEL_COLUMNS}")
    print(f"\n   Label distribution:")
    
    for i, col in enumerate(LABEL_COLUMNS):
        positive = np.sum(labels[:, i])
        percentage = (positive / len(labels)) * 100
        print(f"      {col}: {int(positive)}/{len(labels)} ({percentage:.1f}% positive)")
    
    # Multi-label statistics
    labels_per_sample = np.sum(labels, axis=1)
    print(f"\n   Multi-label statistics:")
    print(f"      Samples with 0 labels: {np.sum(labels_per_sample == 0)}")
    print(f"      Samples with 1 label: {np.sum(labels_per_sample == 1)}")
    print(f"      Samples with 2+ labels: {np.sum(labels_per_sample >= 2)}")
    print(f"      Average labels per sample: {np.mean(labels_per_sample):.2f}")
    
    return comments, labels


# =============================================================================
# K-FOLD CROSS VALIDATION
# =============================================================================

def prepare_kfold_splits(
    comments: np.ndarray,
    labels: np.ndarray,
    num_folds: int = 5,
    stratification_type: str = 'multilabel',
    seed: int = 42
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Prepare K-fold cross-validation splits.
    
    WHAT: Divides data into K training/validation splits
    WHY: Cross-validation gives more reliable performance estimates
    HOW: Choose stratification method based on data type
    
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
        num_folds: Number of folds (typically 5)
        stratification_type: 'multilabel', 'multiclass', or 'none'
        seed: Random seed for reproducibility
    
    Yields:
        Tuples of (train_indices, val_indices)
    
    Example:
        splits = prepare_kfold_splits(comments, labels, num_folds=5)
        for fold, (train_idx, val_idx) in enumerate(splits):
            print(f"Fold {fold}: {len(train_idx)} train, {len(val_idx)} val")
    """
    print(f"\n🔀 Preparing {num_folds}-fold cross-validation...")
    
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
            return kfold.split(comments, labels)
        
        except ImportError:
            print("   ⚠️  iterative-stratification not installed!")
            print("      Install: pip install iterative-stratification")
            print("      Falling back to multiclass stratification...")
            stratification_type = 'multiclass'
    
    # Multiclass stratification (use primary label)
    if stratification_type == 'multiclass':
        print(f"   Using StratifiedKFold on primary label")
        
        # Create single-label version (priority: threat > sexual > religious > bully > spam)
        primary_labels = np.zeros(len(labels), dtype=int)
        for i in range(len(labels)):
            if labels[i, 3] == 1:      # threat
                primary_labels[i] = 4
            elif labels[i, 1] == 1:    # sexual
                primary_labels[i] = 3
            elif labels[i, 2] == 1:    # religious
                primary_labels[i] = 2
            elif labels[i, 0] == 1:    # bully
                primary_labels[i] = 1
            elif labels[i, 4] == 1:    # spam
                primary_labels[i] = 5
            else:
                primary_labels[i] = 0  # no label
        
        kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        return kfold.split(comments, primary_labels)
    
    # No stratification (random splits)
    print(f"   Using basic KFold (no stratification)")
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    return kfold.split(comments)


# =============================================================================
# CLASS WEIGHTS
# =============================================================================

def calculate_class_weights(labels: np.ndarray) -> torch.Tensor:
    """
    Calculate class weights for imbalanced data.
    
    WHAT: Computes weight for each label based on class frequency
    WHY: Rare classes (like 'threat') should have higher loss weight
    HOW: weight = negative_count / positive_count
    
    For multi-label classification, each label is an independent binary
    classification problem, so we calculate weights per label.
    
    Args:
        labels: Array of shape [num_samples, num_labels]
    
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
    
    # Count positives and negatives per label
    pos_counts = np.sum(labels, axis=0)
    neg_counts = len(labels) - pos_counts
    
    # Calculate weights (avoid division by zero)
    weights = np.where(pos_counts > 0, neg_counts / pos_counts, 1.0)
    
    print("\n⚖️  Class weights (for imbalanced data):")
    for i, col in enumerate(LABEL_COLUMNS):
        print(f"      {col}: {weights[i]:.2f} "
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
    
    print(f"\n📦 DataLoaders created:")
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
