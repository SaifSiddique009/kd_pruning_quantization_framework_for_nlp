"""
================================================================================
KAGGLE NOTEBOOK TEMPLATE FOR COMPRESSION EXPERIMENTS
================================================================================

Copy this entire file to a Kaggle notebook and run cell by cell.

ESTIMATED TIME: ~5-6 hours for full ablation study
KAGGLE LIMIT: 12 hours GPU → You have plenty of time!

PREREQUISITES:
1. Upload your fine-tuned teacher model to HuggingFace
2. Upload your dataset to Kaggle
"""

# =============================================================================
# CELL 1: SETUP AND INSTALL DEPENDENCIES
# =============================================================================

# Install required packages
!pip install transformers datasets scikit-learn pandas tqdm --quiet
!pip install iterative-stratification --quiet  # For multi-label stratification

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("✅ Dependencies installed!")

# =============================================================================
# CELL 2: CREATE COMPRESSION FRAMEWORK FILES
# =============================================================================

# Create directory structure
!mkdir -p /kaggle/working/compression_framework
!mkdir -p /kaggle/working/cache
!mkdir -p /kaggle/working/results

# Option 1: Upload files manually (recommended)
# Upload all research_*.py files to /kaggle/working/compression_framework/

# Option 2: Write files inline (if you can't upload)
# [Files would be written here - too long for this template]

# =============================================================================
# CELL 3: VERIFY GPU
# =============================================================================

import torch

if torch.cuda.is_available():
    print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠️ No GPU available! Training will be slow.")

# =============================================================================
# CELL 4: CONFIGURATION
# =============================================================================

# ==================== MODIFY THESE ====================
DATASET_PATH = "/kaggle/input/YOUR-DATASET/data.csv"  # Change this!
TEACHER_CHECKPOINT = "YOUR-USERNAME/YOUR-MODEL"  # Change this!
AUTHOR_NAME = "Your Name"  # Change this!
# ======================================================

# Verify paths
import os
if not os.path.exists(DATASET_PATH):
    print(f"❌ Dataset not found: {DATASET_PATH}")
    print("   Please update DATASET_PATH to your dataset location")
else:
    print(f"✅ Dataset found: {DATASET_PATH}")

# =============================================================================
# CELL 5: RUN SINGLE EXPERIMENT (KD ONLY - QUICK TEST)
# =============================================================================

# Quick test to make sure everything works
!cd /kaggle/working/compression_framework && python research_main.py \
    --dataset_path {DATASET_PATH} \
    --author_name "{AUTHOR_NAME}" \
    --pipeline kd_only \
    --teacher_checkpoint {TEACHER_CHECKPOINT} \
    --epochs 3 \
    --output_dir /kaggle/working/results/test_run \
    --cache_dir /kaggle/working/cache

print("\n✅ Test run complete! Check results in /kaggle/working/results/test_run/")

# =============================================================================
# CELL 6: RUN FULL ABLATION STUDY
# =============================================================================

# This runs all 8 pipeline configurations
# Estimated time: ~5-6 hours

!cd /kaggle/working/compression_framework && python research_main.py \
    --dataset_path {DATASET_PATH} \
    --author_name "{AUTHOR_NAME}" \
    --run_ablation \
    --teacher_checkpoint {TEACHER_CHECKPOINT} \
    --output_dir /kaggle/working/results/ablation \
    --cache_dir /kaggle/working/cache

# =============================================================================
# CELL 7: ANALYZE RESULTS
# =============================================================================

import pandas as pd

# Load ablation results
results_path = "/kaggle/working/results/ablation/ablation_results.csv"
if os.path.exists(results_path):
    df = pd.read_csv(results_path)
    print("\n📊 ABLATION STUDY RESULTS")
    print("="*60)
    print(df.to_string(index=False))
    
    # Find best configuration
    best_idx = df['F1 Macro'].idxmax()
    best = df.loc[best_idx]
    print(f"\n🏆 Best Configuration: {best['Pipeline']}")
    print(f"   F1 Macro: {best['F1 Macro']:.4f}")
    print(f"   Compression: {best['Compression']:.2f}×")
else:
    print("❌ Results file not found. Did the ablation study complete?")

# =============================================================================
# CELL 8: VISUALIZE RESULTS
# =============================================================================

import matplotlib.pyplot as plt
import seaborn as sns

if os.path.exists(results_path):
    df = pd.read_csv(results_path)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: F1 vs Compression
    ax1 = axes[0]
    ax1.scatter(df['Compression'], df['F1 Macro'], s=100, c='blue', alpha=0.7)
    for i, row in df.iterrows():
        ax1.annotate(row['Pipeline'], (row['Compression'], row['F1 Macro']),
                    fontsize=8, ha='center', va='bottom')
    ax1.set_xlabel('Compression Ratio (×)')
    ax1.set_ylabel('F1 Macro')
    ax1.set_title('Accuracy vs Compression Trade-off')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Bar chart of F1 scores
    ax2 = axes[1]
    colors = ['green' if x == df['F1 Macro'].max() else 'steelblue' 
              for x in df['F1 Macro']]
    bars = ax2.barh(df['Pipeline'], df['F1 Macro'], color=colors)
    ax2.set_xlabel('F1 Macro')
    ax2.set_title('F1 Scores by Configuration')
    ax2.set_xlim(0.5, 0.8)
    
    plt.tight_layout()
    plt.savefig('/kaggle/working/results/ablation_plot.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ Plot saved to /kaggle/working/results/ablation_plot.png")

# =============================================================================
# CELL 9: RUN ADDITIONAL EXPERIMENTS (OPTIONAL)
# =============================================================================

# KD Method Comparison
for kd_method in ['logit', 'hidden', 'attention', 'multi_level']:
    print(f"\n{'='*60}")
    print(f"Running KD Method: {kd_method}")
    print('='*60)
    
    !cd /kaggle/working/compression_framework && python research_main.py \
        --dataset_path {DATASET_PATH} \
        --author_name "{AUTHOR_NAME}" \
        --pipeline kd_only \
        --teacher_checkpoint {TEACHER_CHECKPOINT} \
        --kd_method {kd_method} \
        --output_dir /kaggle/working/results/kd_{kd_method} \
        --cache_dir /kaggle/working/cache

# =============================================================================
# CELL 10: EXPORT RESULTS
# =============================================================================

# Zip all results for download
!cd /kaggle/working && zip -r results_all.zip results/

print("\n✅ All results zipped!")
print("   Download: /kaggle/working/results_all.zip")

# List output files
print("\n📁 Output Files:")
!find /kaggle/working/results -name "*.csv" -o -name "*.json" | head -20

# =============================================================================
# CELL 11: GENERATE PAPER TABLES (LATEX)
# =============================================================================

if os.path.exists(results_path):
    df = pd.read_csv(results_path)
    
    # Generate LaTeX table
    latex_table = df.to_latex(index=False, float_format="%.4f")
    
    print("\n📝 LATEX TABLE FOR PAPER:")
    print("="*60)
    print(latex_table)
    
    # Save to file
    with open('/kaggle/working/results/table_for_paper.tex', 'w') as f:
        f.write(latex_table)
    print("\n✅ LaTeX table saved to /kaggle/working/results/table_for_paper.tex")

print("\n" + "="*60)
print("🎉 ALL EXPERIMENTS COMPLETE!")
print("="*60)
