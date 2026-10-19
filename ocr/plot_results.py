"""
OCR Evaluation Results Plotter
===============================
Reads ocr_evaluation_results.csv and produces a grouped bar chart comparing
CER and WER (raw + normalized) across all evaluated models.

Usage:
    uv run plot_results.py
    uv run plot_results.py --input results.csv --output results.png
    uv run plot_results.py --show
    uv run plot_results.py --all-runs
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# =============================================================================
# Constants
# =============================================================================

METRIC_COLS = ["CER_raw", "CER_normalized", "WER_raw", "WER_normalized"]

METRIC_LABELS = {
    "CER_raw": "CER (raw)",
    "CER_normalized": "CER (normalized)",
    "WER_raw": "WER (raw)",
    "WER_normalized": "WER (normalized)",
}

# Two hues (blue for CER, orange for WER) with light / dark variants
METRIC_COLORS = {
    "CER_raw": "#4C72B0",
    "CER_normalized": "#8DA8D4",
    "WER_raw": "#DD8452",
    "WER_normalized": "#E8B796",
}


# =============================================================================
# Data loading
# =============================================================================


def load_results(csv_path: str, latest_only: bool = True) -> pd.DataFrame:
    """Load evaluation results from CSV.

    Args:
        csv_path: Path to the CSV file produced by eval_ocr.py.
        latest_only: If True, keep only the most recent run per model.

    Returns:
        DataFrame with one row per model (or per model-run).
    """
    df = pd.read_csv(csv_path)

    # Ensure metric columns are numeric
    for col in METRIC_COLS + ["time_taken_seconds"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if latest_only and "evaluation_date" in df.columns:
        df["evaluation_date"] = pd.to_datetime(df["evaluation_date"])
        idx = df.groupby("model")["evaluation_date"].idxmax()
        df = df.loc[idx].reset_index(drop=True)

    # Use model_key as a short label if available, else derive from model name
    if "model_key" in df.columns:
        df["label"] = df["model_key"]
    else:
        df["label"] = df["model"].apply(lambda m: m.split("/")[-1])

    return df


# =============================================================================
# Plotting
# =============================================================================


def plot_evaluation(df: pd.DataFrame, output_path: str, show: bool = False):
    """Create grouped bar chart and save / display it."""
    sns.set_theme(style="whitegrid", font_scale=1.05)

    models = df["label"].tolist()
    n_models = len(models)
    n_metrics = len(METRIC_COLS)

    # Bar geometry
    bar_width = 0.18
    group_width = n_metrics * bar_width
    x = np.arange(n_models)

    fig, ax = plt.subplots(figsize=(max(10, n_models * 2.2), 6))

    for i, metric in enumerate(METRIC_COLS):
        offsets = x - group_width / 2 + (i + 0.5) * bar_width
        values = df[metric].values * 100  # convert to percentage
        bars = ax.bar(
            offsets,
            values,
            width=bar_width,
            label=METRIC_LABELS[metric],
            color=METRIC_COLORS[metric],
            edgecolor="white",
            linewidth=0.5,
        )
        # Value labels on top of bars
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{val:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="medium",
                )

    # Build x-axis labels with duration annotation
    if "time_taken_seconds" in df.columns:
        xlabels = []
        for _, row in df.iterrows():
            secs = row["time_taken_seconds"]
            if pd.notna(secs):
                mins, s = divmod(int(secs), 60)
                duration = f"{mins}m{s:02d}s" if mins else f"{int(secs)}s"
                xlabels.append(f"{row['label']}\n({duration})")
            else:
                xlabels.append(row["label"])
    else:
        xlabels = models

    # Axes
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Error Rate (%)", fontsize=11)
    ax.set_ylim(bottom=0)

    # Title block
    num_samples = df["num_samples"].iloc[0] if "num_samples" in df.columns else "?"
    eval_date = (
        df["evaluation_date"].max().strftime("%Y-%m-%d")
        if "evaluation_date" in df.columns
        else "N/A"
    )
    ax.set_title(
        "OCR Model Evaluation \u2014 German Handwriting (CER / WER)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.text(
        0.5,
        1.02,
        f"Dataset: fhswf/german_handwriting  |  Samples: {num_samples}  |  Date: {eval_date}",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        color="gray",
    )

    # Legend
    ax.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.9,
        fontsize=9,
    )

    # Grid — only horizontal
    ax.yaxis.grid(True, linewidth=0.5, alpha=0.7)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Plot saved to {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# =============================================================================
# CLI
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot OCR evaluation results from CSV."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="ocr_evaluation_results.csv",
        help="Path to the evaluation CSV (default: ocr_evaluation_results.csv).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ocr_evaluation_results.png",
        help="Path to save the plot image (default: ocr_evaluation_results.png).",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Plot all evaluation runs instead of only the latest per model.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively (in addition to saving).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    csv_path = Path(args.input)
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        print("Run eval_ocr.py first to generate evaluation results.")
        raise SystemExit(1)

    df = load_results(str(csv_path), latest_only=not args.all_runs)

    if df.empty:
        print("No results found in CSV.")
        raise SystemExit(1)

    print(f"Loaded {len(df)} model result(s) from {csv_path}")
    print(f"Models: {', '.join(df['label'].tolist())}")

    plot_evaluation(df, output_path=args.output, show=args.show)


if __name__ == "__main__":
    main()
