#!/usr/bin/env python3
"""
================================================================================
INFERENCE SCRIPT FOR CYBERBULLYING DETECTION
================================================================================

This script provides easy-to-use inference functionality for trained models.

USAGE EXAMPLES:
---------------

1. Single text prediction:
   python inference.py --model_path ./compressed_models/model_hf --text "your text here"

2. Batch prediction from CSV:
   python inference.py --model_path ./compressed_models/model_hf \
       --input_csv test.csv --output_csv predictions.csv

3. Interactive mode:
   python inference.py --model_path ./compressed_models/model_hf --interactive

LABEL COLUMNS:
- bully: General cyberbullying
- sexual: Sexual harassment
- religious: Religious hate
- threat: Threats/violence
- spam: Spam content
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict, Optional, Union
from tqdm import tqdm


# Default label columns
LABEL_COLUMNS = ['bully', 'sexual', 'religious', 'threat', 'spam']


class CyberbullyingClassifier:
    """
    Inference class for cyberbullying detection models.

    Loads models saved in HuggingFace format and provides prediction methods.
    """

    def __init__(
        self,
        model_path: str,
        label_columns: Optional[List[str]] = None,
        device: Optional[str] = None,
        max_length: int = 128
    ):
        """
        Initialize the classifier.

        Args:
            model_path: Path to the saved model directory (HuggingFace format)
            label_columns: List of label names (default: bully, sexual, religious, threat, spam)
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            max_length: Maximum sequence length for tokenization
        """
        self.model_path = model_path
        self.label_columns = label_columns or LABEL_COLUMNS
        self.max_length = max_length

        # Auto-detect device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"[Inference] Loading model from: {model_path}")
        print(f"[Inference] Device: {self.device}")
        print(f"[Inference] Labels: {self.label_columns}")

        # Load model components
        self._load_model()

    def _load_model(self):
        """Load the encoder, classifier, and tokenizer from saved path."""
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        # Load encoder
        self.encoder = AutoModel.from_pretrained(self.model_path)

        # Load classifier config
        classifier_config_path = os.path.join(self.model_path, 'classifier_config.json')
        if os.path.exists(classifier_config_path):
            with open(classifier_config_path, 'r') as f:
                classifier_config = json.load(f)
            num_labels = classifier_config.get('num_labels', len(self.label_columns))
        else:
            num_labels = len(self.label_columns)

        # Build classifier (matching the structure from training)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_labels)
        )

        # Load classifier weights
        classifier_path = os.path.join(self.model_path, 'classifier.pt')
        if os.path.exists(classifier_path):
            self.classifier.load_state_dict(torch.load(classifier_path, map_location=self.device))
            print(f"[Inference] Loaded classifier weights from {classifier_path}")
        else:
            print(f"[WARN] Classifier weights not found at {classifier_path}")
            print("       Using randomly initialized classifier")

        # Move to device
        self.encoder.to(self.device)
        self.classifier.to(self.device)

        # Set to evaluation mode
        self.encoder.eval()
        self.classifier.eval()

        print(f"[OK] Model loaded successfully")

    def predict(
        self,
        texts: Union[str, List[str]],
        threshold: float = 0.5,
        return_probabilities: bool = True
    ) -> Dict:
        """
        Predict labels for input text(s).

        Args:
            texts: Single text string or list of text strings
            threshold: Probability threshold for positive prediction (default: 0.5)
            return_probabilities: If True, include raw probabilities in output

        Returns:
            Dict with predictions:
            - 'labels': List of predicted label names for each text
            - 'binary': Binary predictions (0/1) for each label
            - 'probabilities': Raw probabilities (if return_probabilities=True)
        """
        # Handle single text input
        if isinstance(texts, str):
            texts = [texts]

        # Tokenize
        encodings = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)

        # Forward pass
        with torch.no_grad():
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
            logits = self.classifier(cls_embedding)
            probabilities = torch.sigmoid(logits).cpu().numpy()

        # Convert to binary predictions
        binary_preds = (probabilities >= threshold).astype(int)

        # Get label names for each prediction
        label_names = []
        for i in range(len(texts)):
            pred_labels = [
                self.label_columns[j]
                for j in range(len(self.label_columns))
                if binary_preds[i, j] == 1
            ]
            label_names.append(pred_labels if pred_labels else ['none'])

        result = {
            'texts': texts,
            'labels': label_names,
            'binary': binary_preds.tolist()
        }

        if return_probabilities:
            result['probabilities'] = probabilities.tolist()

        return result

    def predict_single(self, text: str, threshold: float = 0.5) -> Dict:
        """
        Predict labels for a single text.

        Args:
            text: Input text string
            threshold: Probability threshold

        Returns:
            Dict with prediction for single text:
            - 'text': Input text
            - 'labels': List of predicted label names
            - 'probabilities': Dict mapping label name to probability
        """
        result = self.predict(text, threshold=threshold, return_probabilities=True)

        # Format for single text
        probs_dict = {
            label: float(result['probabilities'][0][i])
            for i, label in enumerate(self.label_columns)
        }

        return {
            'text': text,
            'labels': result['labels'][0],
            'probabilities': probs_dict
        }

    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        threshold: float = 0.5,
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Predict labels for a batch of texts.

        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            threshold: Probability threshold
            show_progress: Show progress bar

        Returns:
            DataFrame with predictions
        """
        all_results = {
            'text': [],
            'predicted_labels': []
        }

        # Add columns for each label's probability and binary prediction
        for label in self.label_columns:
            all_results[f'{label}_prob'] = []
            all_results[f'{label}_pred'] = []

        # Process in batches
        n_batches = (len(texts) + batch_size - 1) // batch_size
        iterator = range(0, len(texts), batch_size)

        if show_progress:
            iterator = tqdm(iterator, total=n_batches, desc="Predicting")

        for i in iterator:
            batch_texts = texts[i:i + batch_size]
            result = self.predict(batch_texts, threshold=threshold, return_probabilities=True)

            for j, text in enumerate(batch_texts):
                all_results['text'].append(text)
                all_results['predicted_labels'].append(', '.join(result['labels'][j]))

                for k, label in enumerate(self.label_columns):
                    all_results[f'{label}_prob'].append(result['probabilities'][j][k])
                    all_results[f'{label}_pred'].append(result['binary'][j][k])

        return pd.DataFrame(all_results)


def predict_from_csv(
    model_path: str,
    input_csv: str,
    output_csv: str,
    text_column: str = 'comment',
    batch_size: int = 32,
    threshold: float = 0.5,
    label_columns: Optional[List[str]] = None
):
    """
    Run predictions on a CSV file.

    Args:
        model_path: Path to saved model
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file
        text_column: Name of the column containing text
        batch_size: Batch size for processing
        threshold: Probability threshold
        label_columns: List of label names
    """
    # Load input data
    print(f"[Inference] Reading input from: {input_csv}")
    df = pd.read_csv(input_csv)

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in input CSV. Available: {df.columns.tolist()}")

    texts = df[text_column].tolist()
    print(f"[Inference] Found {len(texts)} texts to process")

    # Initialize classifier
    classifier = CyberbullyingClassifier(
        model_path=model_path,
        label_columns=label_columns
    )

    # Run predictions
    results_df = classifier.predict_batch(texts, batch_size=batch_size, threshold=threshold)

    # Save results
    results_df.to_csv(output_csv, index=False)
    print(f"[OK] Predictions saved to: {output_csv}")

    return results_df


def interactive_mode(model_path: str, label_columns: Optional[List[str]] = None):
    """
    Run interactive prediction mode.

    Args:
        model_path: Path to saved model
        label_columns: List of label names
    """
    classifier = CyberbullyingClassifier(
        model_path=model_path,
        label_columns=label_columns
    )

    print("\n" + "="*60)
    print("INTERACTIVE MODE")
    print("="*60)
    print("Enter text to classify. Type 'quit' to exit.\n")

    while True:
        try:
            text = input("Enter text: ").strip()

            if text.lower() in ['quit', 'exit', 'q']:
                print("Exiting...")
                break

            if not text:
                print("Please enter some text.\n")
                continue

            result = classifier.predict_single(text)

            print(f"\n[Predicted Labels]: {', '.join(result['labels'])}")
            print("[Probabilities]:")
            for label, prob in result['probabilities'].items():
                bar = '#' * int(prob * 20)
                print(f"  {label:12s}: {prob:.3f} |{bar}")
            print()

        except KeyboardInterrupt:
            print("\nExiting...")
            break


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Inference script for Bangla Cyberbullying Detection"
    )

    parser.add_argument(
        '--model_path', type=str, required=True,
        help='Path to saved model directory (HuggingFace format)'
    )

    # Input mode (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        '--text', type=str,
        help='Single text to classify'
    )
    input_group.add_argument(
        '--input_csv', type=str,
        help='Path to input CSV file for batch prediction'
    )
    input_group.add_argument(
        '--interactive', action='store_true',
        help='Run in interactive mode'
    )

    # Output
    parser.add_argument(
        '--output_csv', type=str, default='predictions.csv',
        help='Path to output CSV file (for batch prediction)'
    )
    parser.add_argument(
        '--output_json', type=str,
        help='Path to output JSON file (for single text)'
    )

    # Options
    parser.add_argument(
        '--text_column', type=str, default='comment',
        help='Name of text column in input CSV (default: comment)'
    )
    parser.add_argument(
        '--batch_size', type=int, default=32,
        help='Batch size for processing (default: 32)'
    )
    parser.add_argument(
        '--threshold', type=float, default=0.5,
        help='Probability threshold for positive prediction (default: 0.5)'
    )
    parser.add_argument(
        '--label_columns', type=str, nargs='+', default=LABEL_COLUMNS,
        help='List of label columns'
    )
    parser.add_argument(
        '--device', type=str, choices=['cuda', 'cpu'],
        help='Device to use (auto-detect if not specified)'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()

    if args.interactive:
        # Interactive mode
        interactive_mode(args.model_path, args.label_columns)

    elif args.input_csv:
        # Batch prediction from CSV
        predict_from_csv(
            model_path=args.model_path,
            input_csv=args.input_csv,
            output_csv=args.output_csv,
            text_column=args.text_column,
            batch_size=args.batch_size,
            threshold=args.threshold,
            label_columns=args.label_columns
        )

    elif args.text:
        # Single text prediction
        classifier = CyberbullyingClassifier(
            model_path=args.model_path,
            label_columns=args.label_columns,
            device=args.device
        )

        result = classifier.predict_single(args.text, threshold=args.threshold)

        print(f"\n[Input Text]: {result['text']}")
        print(f"[Predicted Labels]: {', '.join(result['labels'])}")
        print("[Probabilities]:")
        for label, prob in result['probabilities'].items():
            bar = '#' * int(prob * 20)
            print(f"  {label:12s}: {prob:.3f} |{bar}")

        # Save to JSON if requested
        if args.output_json:
            with open(args.output_json, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\n[OK] Result saved to: {args.output_json}")

    else:
        print("Please specify --text, --input_csv, or --interactive")
        print("Run with --help for usage information")


if __name__ == "__main__":
    main()
