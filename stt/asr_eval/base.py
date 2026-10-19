"""Base class for ASR evaluation pipelines."""

from __future__ import annotations

import abc
import csv
import glob
import os
import re
import textwrap
from datetime import datetime
from pathlib import Path

import jiwer
from sacrebleu.metrics import BLEU


class ASRPipeline(abc.ABC):
    """Subclass this to add a new ASR backend.

    You only need to implement ``transcribe`` — everything else
    (data loading, metrics, CSV output) is handled here.
    """

    def __init__(self, model: str, api_url: str, data_dir: str = "./data", **_ignored):
        self.model = model
        self.api_url = api_url
        self.data_dir = data_dir
        self._bleu = BLEU()

    @abc.abstractmethod
    async def transcribe(self, audio_path: str) -> str:
        """Return the transcription text for a single audio file."""
        ...

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def load_test_cases(self) -> list[dict]:
        test_cases = []
        subdirs = sorted(
            d
            for d in glob.glob(os.path.join(self.data_dir, "*"))
            if os.path.isdir(d)
        )
        for subdir in subdirs:
            audio_path = os.path.join(subdir, "RecordedAudio.wav")
            transcript_path = os.path.join(subdir, "ground_truth_transcript.txt")
            if os.path.exists(audio_path) and os.path.exists(transcript_path):
                with open(transcript_path, "r", encoding="utf-8") as f:
                    ground_truth = f.read().strip()
                test_cases.append(
                    {"audio_path": audio_path, "ground_truth": ground_truth}
                )
        return test_cases

    def calculate_metrics(
        self, ground_truths: list[str], predictions: list[str]
    ) -> dict[str, float]:
        return {
            "WER": jiwer.wer(ground_truths, predictions),
            "CER": jiwer.cer(ground_truths, predictions),
            "BLEU": self._bleu.corpus_score(predictions, [ground_truths]).score,
        }

    def per_case_rows(
        self,
        audio_paths: list[str],
        ground_truths: list[str],
        predictions: list[str],
    ) -> list[dict]:
        """One row per utterance with its own WER/CER — for cross-model diffing."""
        rows = []
        for ap, gt, pred in zip(audio_paths, ground_truths, predictions):
            rows.append(
                {
                    "audio_path": ap,
                    "ground_truth": gt,
                    "prediction": pred,
                    "WER": jiwer.wer([gt], [pred]),
                    "CER": jiwer.cer([gt], [pred]),
                }
            )
        return rows

    # ------------------------------------------------------------------
    # Evaluation orchestration
    # ------------------------------------------------------------------

    async def evaluate(self) -> dict:
        """Run transcription on all test cases and return a results dict.

        The returned dict carries the aggregate metrics plus the raw
        ``audio_paths``/``ground_truths``/``predictions`` lists so callers can
        persist per-case predictions (see ``write_predictions_csv``).
        """
        import asyncio

        test_cases = self.load_test_cases()
        audio_paths = [tc["audio_path"] for tc in test_cases]
        ground_truths = [tc["ground_truth"] for tc in test_cases]
        tasks = [self.transcribe(ap) for ap in audio_paths]
        predictions = list(await asyncio.gather(*tasks))

        empty = [ap for ap, pred in zip(audio_paths, predictions) if not (pred or "").strip()]
        if empty:
            print(f"WARNING: {len(empty)}/{len(predictions)} empty prediction(s):")
            for ap in empty:
                print(f"  - {ap}")

        self.print_comparison_table(ground_truths, predictions)
        metrics = self.calculate_metrics(ground_truths, predictions)

        return {
            "evaluation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "num_test_cases": len(test_cases),
            "evaluation_api": self.api_url,
            "model": self.model,
            **metrics,
            "audio_paths": audio_paths,
            "ground_truths": ground_truths,
            "predictions": predictions,
        }

    # ------------------------------------------------------------------
    # CSV persistence
    # ------------------------------------------------------------------

    RESULTS_CSV = Path("results/asr_evaluation_results.csv")
    CSV_FIELDS = [
        "evaluation_date",
        "num_test_cases",
        "evaluation_api",
        "model",
        "WER",
        "CER",
        "BLEU",
    ]

    PREDICTIONS_DIR = Path("results/predictions")
    PREDICTION_FIELDS = ["audio_path", "ground_truth", "prediction", "WER", "CER"]

    def write_results_to_csv(self, results: dict) -> None:
        self.RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.RESULTS_CSV.exists()
        with open(self.RESULTS_CSV, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_FIELDS)
            if not file_exists:
                writer.writeheader()
            writer.writerow({k: results.get(k) for k in self.CSV_FIELDS})

    def run_label(self) -> str:
        """Filesystem-safe label for this run: sanitized model + timestamp."""
        safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", self.model).strip("_")
        return f"{safe_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def write_predictions_csv(self, results: dict, label: str | None = None) -> Path:
        """Persist per-case predictions for later inspection / cross-model diff."""
        rows = self.per_case_rows(
            results["audio_paths"], results["ground_truths"], results["predictions"]
        )
        self.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = self.PREDICTIONS_DIR / f"{label or self.run_label()}.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.PREDICTION_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        return out_path

    # ------------------------------------------------------------------
    # Pretty-print comparison table
    # ------------------------------------------------------------------

    @staticmethod
    def print_comparison_table(
        ground_truths: list[str],
        predictions: list[str],
        *,
        col_width: int = 50,
        max_cases: int | None = None,
        output_txt_path: str | Path | None = None,
    ) -> None:
        n = len(ground_truths)
        indices = range(min(n, max_cases) if max_cases is not None else n)

        idx_width = max(2, len(str(n)))
        sep = " │ "
        header = (
            f" #{sep}{'Ground truth'.ljust(col_width)[:col_width]}"
            f"{sep}{'Predicted'.ljust(col_width)[:col_width]}"
        )
        rule = "─" * len(header)
        lines: list[str] = [rule, header, rule]

        for i in indices:
            gt_lines = textwrap.wrap((ground_truths[i] or "").strip(), width=col_width) or [""]
            pred_lines = textwrap.wrap((predictions[i] or "").strip(), width=col_width) or [""]
            row_height = max(len(gt_lines), len(pred_lines))
            gt_lines += [""] * (row_height - len(gt_lines))
            pred_lines += [""] * (row_height - len(pred_lines))

            for j in range(row_height):
                idx_str = str(i + 1).rjust(idx_width) if j == 0 else " " * idx_width
                gt_cell = gt_lines[j][:col_width].ljust(col_width)
                pred_cell = pred_lines[j][:col_width].ljust(col_width)
                lines.append(f"{idx_str}{sep}{gt_cell}{sep}{pred_cell}")
            lines.append(rule)

        if max_cases is not None and n > max_cases:
            lines.append(f" ... and {n - max_cases} more case(s)")

        text = "\n".join(lines)
        print(text)
        if output_txt_path is not None:
            Path(output_txt_path).write_text(text, encoding="utf-8")
