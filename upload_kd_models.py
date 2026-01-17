#!/usr/bin/env python3
"""
Upload KD Models to HuggingFace Hub

This script uploads all 4 distilled models from Scenario 2 to HuggingFace Hub
so they can be reused in Scenarios 3 and 4 without redoing KD.

Usage:
    python upload_kd_models.py --author_name YOUR_USERNAME

    # Or upload individual models:
    python upload_kd_models.py --author_name YOUR_USERNAME --kd_id KD1
"""

import argparse
import os
import sys
from pathlib import Path


# KD Model Registry
KD_MODELS = {
    "KD1": {
        "name": "kd1-xlmroberta-to-sahajbert",
        "description": "Knowledge Distillation: XLM-RoBERTa -> SahajBERT",
        "local_path": "./results/scenario2/KD1_T1_RS1/model_final_hf",
        "teacher": "XLM-RoBERTa (T1)",
        "student": "SahajBERT (RS1)"
    },
    "KD2": {
        "name": "kd2-xlmroberta-to-banglabert-small",
        "description": "Knowledge Distillation: XLM-RoBERTa -> BanglaBERT-small",
        "local_path": "./results/scenario2/KD2_T1_RS2/model_final_hf",
        "teacher": "XLM-RoBERTa (T1)",
        "student": "BanglaBERT-small (RS2)"
    },
    "KD3": {
        "name": "kd3-banglabert-to-sahajbert",
        "description": "Knowledge Distillation: BanglaBERT -> SahajBERT",
        "local_path": "./results/scenario2/KD3_T2_RS1/model_final_hf",
        "teacher": "BanglaBERT (T2)",
        "student": "SahajBERT (RS1)"
    },
    "KD4": {
        "name": "kd4-banglabert-to-banglabert-small",
        "description": "Knowledge Distillation: BanglaBERT -> BanglaBERT-small",
        "local_path": "./results/scenario2/KD4_T2_RS2/model_final_hf",
        "teacher": "BanglaBERT (T2)",
        "student": "BanglaBERT-small (RS2)"
    }
}


def create_model_card(kd_id: str, kd_info: dict, repo_name: str) -> str:
    """Generate a README.md model card for the KD model."""
    return f"""---
language:
  - bn
license: mit
tags:
  - knowledge-distillation
  - bangla
  - cyberbullying-detection
  - text-classification
---

# {kd_info['description']}

This is a **distilled student model** for Bangla Cyberbullying Detection.

## Model Details

| Property | Value |
|----------|-------|
| **KD ID** | {kd_id} |
| **Teacher** | {kd_info['teacher']} |
| **Student** | {kd_info['student']} |
| **Task** | Multi-label Cyberbullying Classification |
| **Language** | Bangla (bn) |

## Training Details

- **Method**: Knowledge Distillation with soft labels
- **KD Alpha**: 0.7 (70% soft labels, 30% hard labels)
- **Temperature**: 4.0
- **Epochs**: 15
- **Evaluation**: Fold 3 with original 5-fold split

## Usage

```python
from transformers import AutoModel, AutoTokenizer
import torch

# Load model
encoder = AutoModel.from_pretrained("{repo_name}")
tokenizer = AutoTokenizer.from_pretrained("{repo_name}")

# Load classifier head
import json
with open("classifier_config.json") as f:
    config = json.load(f)

classifier = torch.nn.Linear(config['hidden_size'], config['num_labels'])
classifier.load_state_dict(torch.load("classifier.pt"))
```

## Part of Phase 3 Experiments

This model is part of a comprehensive study comparing:
- Knowledge Distillation only
- Pruning only
- KD + Pruning combinations
- Different pruning methods (magnitude, wanda, gradual, structured)

## Citation

If you use this model, please cite our work on Bangla Cyberbullying Detection.
"""


def upload_kd_model(kd_id: str, author_name: str, private: bool = False) -> bool:
    """Upload a single KD model to HuggingFace Hub."""
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("Error: huggingface_hub not installed")
        print("Install with: pip install huggingface_hub")
        return False

    if kd_id not in KD_MODELS:
        print(f"Error: Unknown KD ID '{kd_id}'. Valid: {list(KD_MODELS.keys())}")
        return False

    kd_info = KD_MODELS[kd_id]
    model_path = Path(kd_info['local_path'])

    if not model_path.exists():
        print(f"Error: Model path does not exist: {model_path}")
        print(f"  Make sure you've run Scenario 2 first!")
        return False

    repo_name = f"{author_name}/bangla-cyberbully-{kd_info['name']}"

    api = HfApi()

    # Check login status
    try:
        user_info = api.whoami()
        print(f"Logged in as: {user_info['name']}")
    except Exception:
        print("Not logged in to HuggingFace Hub")
        print("Run: huggingface-cli login")
        return False

    # Create repository
    try:
        create_repo(repo_name, private=private, exist_ok=True)
        visibility = "private" if private else "public"
        print(f"Repository ready ({visibility}): https://huggingface.co/{repo_name}")
    except Exception as e:
        print(f"Repository note: {e}")

    # Create and save model card
    model_card = create_model_card(kd_id, kd_info, repo_name)
    readme_path = model_path / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(model_card)

    # Upload
    print(f"Uploading {kd_id} from {model_path}...")

    try:
        api.upload_folder(
            folder_path=str(model_path),
            repo_id=repo_name,
            commit_message=f"Upload {kd_id}: {kd_info['description']}"
        )

        print(f"SUCCESS: {kd_id} uploaded to {repo_name}")
        return True

    except Exception as e:
        print(f"Upload failed: {e}")
        return False


def upload_all_kd_models(author_name: str, private: bool = False) -> dict:
    """Upload all 4 KD models."""
    results = {}

    print("=" * 60)
    print("UPLOADING ALL KD MODELS TO HUGGINGFACE")
    print("=" * 60)

    for kd_id in KD_MODELS:
        print(f"\n--- {kd_id} ---")
        success = upload_kd_model(kd_id, author_name, private)
        results[kd_id] = success

    print("\n" + "=" * 60)
    print("UPLOAD SUMMARY")
    print("=" * 60)

    for kd_id, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        repo = f"{author_name}/bangla-cyberbully-{KD_MODELS[kd_id]['name']}"
        print(f"  {kd_id}: {status} -> {repo}")

    return results


def get_kd_model_path(kd_id: str, author_name: str) -> str:
    """Get the HuggingFace path for a KD model."""
    if kd_id not in KD_MODELS:
        raise ValueError(f"Unknown KD ID: {kd_id}")
    return f"{author_name}/bangla-cyberbully-{KD_MODELS[kd_id]['name']}"


def print_kd_paths(author_name: str):
    """Print all KD model paths for use in Scenarios 3 and 4."""
    print("\n" + "=" * 60)
    print("KD MODEL PATHS FOR SCENARIOS 3 & 4")
    print("=" * 60)
    print(f"\nUse these paths with --teacher_checkpoint in prune_only pipeline:\n")

    for kd_id in KD_MODELS:
        path = get_kd_model_path(kd_id, author_name)
        print(f"{kd_id}: {path}")

    print("\nExample command:")
    print(f"""
python main.py \\
    --pipeline prune_only \\
    --teacher_checkpoint "{author_name}/bangla-cyberbully-kd1-xlmroberta-to-sahajbert" \\
    --prune_method magnitude \\
    --prune_sparsity 0.5 \\
    --fine_tune_after_prune \\
    --output_dir ./results/scenario3/KD1_magnitude
""")


def main():
    parser = argparse.ArgumentParser(
        description="Upload KD models to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Upload all 4 KD models
    python upload_kd_models.py --author_name Saif-Siddique

    # Upload a specific KD model
    python upload_kd_models.py --author_name Saif-Siddique --kd_id KD1

    # Just print the paths (no upload)
    python upload_kd_models.py --author_name Saif-Siddique --print_paths

Before running:
    1. Complete Scenario 2 (all 4 KD experiments)
    2. Login to HuggingFace: huggingface-cli login
        """
    )

    parser.add_argument(
        '--author_name',
        type=str,
        required=True,
        help='Your HuggingFace username (e.g., Saif-Siddique)'
    )

    parser.add_argument(
        '--kd_id',
        type=str,
        choices=['KD1', 'KD2', 'KD3', 'KD4'],
        help='Upload specific KD model (default: all)'
    )

    parser.add_argument(
        '--private',
        action='store_true',
        help='Create private repositories'
    )

    parser.add_argument(
        '--print_paths',
        action='store_true',
        help='Just print the model paths, do not upload'
    )

    args = parser.parse_args()

    if args.print_paths:
        print_kd_paths(args.author_name)
        return

    if args.kd_id:
        success = upload_kd_model(args.kd_id, args.author_name, args.private)
        exit(0 if success else 1)
    else:
        results = upload_all_kd_models(args.author_name, args.private)
        all_success = all(results.values())

        # Print paths for next steps
        print_kd_paths(args.author_name)

        exit(0 if all_success else 1)


if __name__ == "__main__":
    main()
