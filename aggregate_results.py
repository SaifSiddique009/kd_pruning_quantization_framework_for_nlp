#!/usr/bin/env python3
"""
================================================================================
RESULTS AGGREGATION SCRIPT
================================================================================

Aggregates results from multiple experiment runs into a single summary CSV.
Useful for comparing metrics across different scenarios and configurations.

Usage:
    python aggregate_results.py --results_dir ./results --output summary.csv
    python aggregate_results.py --results_dir ./results --format markdown

"""

import os
import json
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


def find_result_files(results_dir: str, pattern: str = "results_final.json") -> List[Path]:
    """Find all result files matching pattern in directory tree."""
    results_path = Path(results_dir)
    return list(results_path.rglob(pattern))


def extract_experiment_info(file_path: Path) -> Dict:
    """Extract experiment metadata from file path."""
    parts = file_path.parts

    # Try to extract scenario and experiment name from path
    # Expected structure: results/scenario1_baselines/T1_xlm_roberta/results_final.json
    scenario = "unknown"
    experiment = "unknown"

    for i, part in enumerate(parts):
        if part.startswith("scenario"):
            scenario = part
            if i + 1 < len(parts) - 1:  # -1 to skip the filename
                experiment = parts[i + 1]
            break

    return {
        "scenario": scenario,
        "experiment": experiment,
        "file_path": str(file_path)
    }


def load_result_file(file_path: Path) -> Optional[Dict]:
    """Load a single result file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Warning: Could not load {file_path}: {e}")
        return None


def extract_metrics(data: Dict, experiment_info: Dict) -> Dict:
    """Extract relevant metrics from result data.

    Handles nested JSON structure where metrics are in data["metrics"] array.
    Uses the LAST stage (final results after all compression stages).
    """
    # Handle nested structure: data["metrics"] is a list of stages
    if "metrics" in data and isinstance(data["metrics"], list):
        # Use the last stage (final results after all compression)
        stage_data = data["metrics"][-1] if data["metrics"] else {}
    else:
        # Fallback to flat structure for backwards compatibility
        stage_data = data

    # Extract per-label F1 scores from nested dict
    per_label_f1 = stage_data.get("per_label_f1", {})

    metrics = {
        **experiment_info,
        # Classification metrics
        "f1_macro": stage_data.get("f1_macro"),
        "f1_weighted": stage_data.get("f1_weighted"),
        "f1_micro": stage_data.get("f1_micro"),
        "accuracy": stage_data.get("accuracy_exact"),
        "hamming_loss": stage_data.get("hamming_loss"),
        # Per-label F1 scores (from nested dict)
        "f1_bully": per_label_f1.get("bully"),
        "f1_sexual": per_label_f1.get("sexual"),
        "f1_religious": per_label_f1.get("religious"),
        "f1_threat": per_label_f1.get("threat"),
        "f1_spam": per_label_f1.get("spam"),
        # Efficiency metrics
        "model_size_mb": stage_data.get("model_size_mb"),
        "latency_mean_ms": stage_data.get("inference_latency_mean_ms"),
        "latency_p95_ms": stage_data.get("inference_latency_p95_ms"),
        "throughput_samples_per_sec": stage_data.get("throughput_samples_per_sec"),
        # Compression metrics
        "sparsity_pct": stage_data.get("sparsity_percent"),
        "compression_ratio": stage_data.get("size_compression_ratio"),
        "parameters": stage_data.get("num_parameters"),
        # Stage info
        "stage": stage_data.get("stage"),
    }

    # Clean up None values for numeric fields
    for key in ["f1_macro", "f1_weighted", "accuracy", "model_size_mb", "latency_mean_ms"]:
        if metrics[key] is None:
            metrics[key] = float('nan')

    return metrics


def aggregate_results(results_dir: str) -> pd.DataFrame:
    """Aggregate all results into a single DataFrame."""
    result_files = find_result_files(results_dir)

    if not result_files:
        print(f"No result files found in {results_dir}")
        return pd.DataFrame()

    print(f"Found {len(result_files)} result files")

    all_metrics = []
    for file_path in result_files:
        data = load_result_file(file_path)
        if data is None:
            continue

        experiment_info = extract_experiment_info(file_path)
        metrics = extract_metrics(data, experiment_info)
        all_metrics.append(metrics)

    df = pd.DataFrame(all_metrics)

    # Sort by scenario and experiment
    if not df.empty:
        df = df.sort_values(["scenario", "experiment"]).reset_index(drop=True)

    return df


def generate_comparison_tables(df: pd.DataFrame) -> str:
    """Generate markdown comparison tables from aggregated data."""
    output = []

    # Table 1: Overall Performance Summary
    output.append("## Overall Performance Summary\n")
    summary_cols = ["experiment", "f1_macro", "f1_weighted", "accuracy", "model_size_mb", "sparsity_pct"]
    available_cols = [c for c in summary_cols if c in df.columns]

    if available_cols:
        summary_df = df[available_cols].copy()
        summary_df = summary_df.round(4)
        output.append(summary_df.to_markdown(index=False))
        output.append("\n")

    # Table 2: Per-Label F1 Scores
    output.append("\n## Per-Label F1 Scores\n")
    label_cols = ["experiment", "f1_bully", "f1_sexual", "f1_religious", "f1_threat", "f1_spam"]
    available_label_cols = [c for c in label_cols if c in df.columns]

    if available_label_cols:
        label_df = df[available_label_cols].copy()
        label_df = label_df.round(4)
        output.append(label_df.to_markdown(index=False))
        output.append("\n")

    # Table 3: Efficiency Metrics
    output.append("\n## Efficiency Metrics\n")
    eff_cols = ["experiment", "latency_mean_ms", "latency_p95_ms", "throughput_samples_per_sec", "parameters"]
    available_eff_cols = [c for c in eff_cols if c in df.columns]

    if available_eff_cols:
        eff_df = df[available_eff_cols].copy()
        eff_df = eff_df.round(2)
        output.append(eff_df.to_markdown(index=False))
        output.append("\n")

    return "\n".join(output)


def generate_latex_table(df: pd.DataFrame, columns: List[str], caption: str) -> str:
    """Generate LaTeX table for paper."""
    available_cols = [c for c in columns if c in df.columns]
    if not available_cols:
        return ""

    subset = df[available_cols].copy()

    # Round numeric columns
    for col in subset.select_dtypes(include=['float64', 'float32']).columns:
        subset[col] = subset[col].round(4)

    latex = subset.to_latex(index=False, caption=caption, label=f"tab:{caption.lower().replace(' ', '_')}")
    return latex


def main():
    parser = argparse.ArgumentParser(description="Aggregate experiment results")
    parser.add_argument("--results_dir", type=str, default="./results",
                       help="Directory containing experiment results")
    parser.add_argument("--output", type=str, default="results_summary.csv",
                       help="Output file path")
    parser.add_argument("--format", type=str, choices=["csv", "markdown", "latex", "all"],
                       default="csv", help="Output format")

    args = parser.parse_args()

    print(f"Aggregating results from: {args.results_dir}")
    df = aggregate_results(args.results_dir)

    if df.empty:
        print("No results to aggregate!")
        return

    print(f"Aggregated {len(df)} experiment results")

    # Save based on format
    if args.format in ["csv", "all"]:
        csv_path = args.output if args.output.endswith(".csv") else f"{args.output}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV: {csv_path}")

    if args.format in ["markdown", "all"]:
        md_path = args.output.replace(".csv", ".md") if args.output.endswith(".csv") else f"{args.output}.md"
        md_content = f"# Experiment Results Summary\n\nGenerated: {datetime.now().isoformat()}\n\n"
        md_content += generate_comparison_tables(df)
        with open(md_path, 'w') as f:
            f.write(md_content)
        print(f"Saved Markdown: {md_path}")

    if args.format in ["latex", "all"]:
        tex_path = args.output.replace(".csv", ".tex") if args.output.endswith(".csv") else f"{args.output}.tex"
        latex_content = "% Auto-generated LaTeX tables\n\n"
        latex_content += generate_latex_table(
            df,
            ["experiment", "f1_macro", "f1_weighted", "accuracy"],
            "Classification Performance"
        )
        with open(tex_path, 'w') as f:
            f.write(latex_content)
        print(f"Saved LaTeX: {tex_path}")

    # Print summary to console
    print("\n" + "="*60)
    print("QUICK SUMMARY")
    print("="*60)

    if "f1_macro" in df.columns and df["f1_macro"].notna().any():
        best_idx = df["f1_macro"].idxmax()
        print(f"\nBest F1 Macro: {df.loc[best_idx, 'f1_macro']:.4f}")
        print(f"  Experiment: {df.loc[best_idx, 'experiment']}")
        print(f"  Scenario: {df.loc[best_idx, 'scenario']}")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()
