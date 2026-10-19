#!/usr/bin/env python3
"""Generate a comparison bar chart of best ASR evaluation scores per model.

Usage:
    python -m asr_eval.plot
    python -m asr_eval.plot --top 5
    python -m asr_eval.plot --csv results/asr_evaluation_results.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_CSV = Path("results/asr_evaluation_results.csv")


def load_best_per_model(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in ("WER", "CER", "BLEU"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["WER", "model"])

    best = (
        df.loc[df.groupby("model")["WER"].idxmin()]
        .sort_values("WER")
        .reset_index(drop=True)
    )
    return best


def plot_metrics(df: pd.DataFrame, out_path: Path) -> None:
    models = df["model"].tolist()
    labels = [m if len(m) <= 35 else m[:32] + "…" for m in models]

    x = range(len(models))
    width = 0.25

    wer_pct = (df["WER"] * 100).round(1)
    cer_pct = (df["CER"] * 100).round(1)
    bleu_vals = df["BLEU"].round(1)

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 2.5), 6))

    bars_wer = ax.bar([i - width for i in x], df["WER"], width, label="WER", color="C0")
    bars_cer = ax.bar(x, df["CER"], width, label="CER", color="C1")

    ax2 = ax.twinx()
    bars_bleu = ax2.bar(
        [i + width for i in x], df["BLEU"], width, label="BLEU", color="C2", alpha=0.8
    )

    ax.bar_label(bars_wer, labels=wer_pct.astype(str), fontsize=8)
    ax.bar_label(bars_cer, labels=cer_pct.astype(str), fontsize=8)
    ax2.bar_label(bars_bleu, labels=bleu_vals.astype(str), fontsize=8)

    ax.set_xlabel("Model (best run by WER)")
    ax.set_ylabel("WER / CER (lower is better)")
    ax2.set_ylabel("BLEU (higher is better)", color="C2")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    fig.suptitle("ASR Evaluation: WER, CER, BLEU (best run per model)")
    plt.tight_layout()

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ASR evaluation results.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to results CSV.")
    parser.add_argument("--top", type=int, default=None, help="Only plot top N models by WER.")
    parser.add_argument("-o", "--output", type=Path, default=Path("results/asr_metrics_plot.png"))
    args = parser.parse_args()

    df = load_best_per_model(args.csv)
    if args.top:
        df = df.head(args.top)

    print("Models included (best WER per model):")
    print(df[["model", "WER", "CER", "BLEU"]].to_string(index=False))
    plot_metrics(df, out_path=args.output)


if __name__ == "__main__":
    main()
