"""
OCR Model Evaluation Suite
===========================
Evaluates OCR VLMs served via vLLM against the fhswf/german_handwriting dataset.
Reports CER and WER for both raw and normalized outputs.

Usage:
    # Evaluate a single model (must be running via serve_models.sh)
    uv run eval_ocr.py --model glm-ocr

    # Evaluate all models
    uv run eval_ocr.py --model all

    # Quick test with 50 samples
    uv run eval_ocr.py --model glm-ocr --limit 50

    # Custom output file
    uv run eval_ocr.py --model all --output my_results.csv
"""

import argparse
import asyncio
import base64
import csv
import os
import re
import string
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from typing import Optional

import httpx
import jiwer
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

# =============================================================================
# Model configurations
# =============================================================================


@dataclass
class OCRModelConfig:
    """Configuration for a single OCR model served via vLLM."""

    key: str
    hf_name: str
    port: int
    prompt: Optional[str] = None
    max_tokens: int = 512
    temperature: float = 0.1
    image_first: bool = False
    prompt_prefix: str = ""
    # Extra sampling parameters sent as-is in the request body (vLLM extensions)
    extra_body: dict = field(default_factory=dict)
    # Regex for markup the model emits around the text (boxes, classes); removed before scoring
    strip_pattern: Optional[str] = None

    @property
    def api_url(self) -> str:
        return f"http://localhost:{self.port}/v1"


# Each model uses the prompt format documented on its HuggingFace model card.
MODEL_CONFIGS: dict[str, OCRModelConfig] = {
    "olmocr": OCRModelConfig(
        key="olmocr",
        hf_name="allenai/olmOCR-2-7B-1025-FP8",
        port=8001,
        prompt="Read the text in the image.",
        max_tokens=512,
    ),
    "glm-ocr": OCRModelConfig(
        key="glm-ocr",
        hf_name="zai-org/GLM-OCR",
        port=8002,
        prompt="Text Recognition:",
        max_tokens=512,
    ),
    "paddleocr": OCRModelConfig(
        key="paddleocr",
        hf_name="PaddlePaddle/PaddleOCR-VL-1.5",
        port=8003,
        prompt="OCR:",
        max_tokens=512,
    ),
    "deepseek-ocr": OCRModelConfig(
        key="deepseek-ocr",
        hf_name="deepseek-ai/DeepSeek-OCR-2",
        port=8004,
        prompt="Free OCR.",
        max_tokens=512,
    ),
    "lighton-ocr": OCRModelConfig(
        key="lighton-ocr",
        hf_name="lightonai/LightOnOCR-2-1B",
        port=8005,
        prompt=None,  # LightOnOCR expects image-only input
        max_tokens=512,
    ),
    "dots-ocr": OCRModelConfig(
        key="dots-ocr",
        hf_name="rednote-hilab/dots.ocr",
        port=8006,
        prompt="Extract the text content from this image.",
        max_tokens=16384,
        image_first=True,
        prompt_prefix="<|img|><|imgpad|><|endofimg|>",
    ),
    "lighton-ocr-3": OCRModelConfig(
        key="lighton-ocr-3",
        hf_name="lightonai/LightOnOCR-3-4B",
        port=8007,
        prompt=None,  # empty prompt = transcription only ("grounding" adds boxes)
        max_tokens=4096,
        temperature=0.2,
        extra_body={"top_p": 0.9},
    ),
    "nemotron-parse": OCRModelConfig(
        key="nemotron-parse",
        hf_name="nvidia/Nemotron-3.5-Parse",
        port=8008,
        prompt="</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>",
        max_tokens=8744,
        temperature=0.0,
        extra_body={"top_k": 1, "repetition_penalty": 1.1, "skip_special_tokens": False},
        # Output is <x_..><y_..>text<x_..><y_..><class_..> per region
        strip_pattern=r"<x_[\d.]+><y_[\d.]+>|<class_[^>]+>|</?s>",
    ),
    "paddleocr-1.6": OCRModelConfig(
        key="paddleocr-1.6",
        hf_name="PaddlePaddle/PaddleOCR-VL-1.6",
        port=8009,
        prompt="OCR:",
        max_tokens=512,
    ),
}


# =============================================================================
# Text normalization
# =============================================================================

# Punctuation table for stripping
_PUNCT_TABLE = str.maketrans(
    "", "", string.punctuation + "\u2013\u2014\u2018\u2019\u201c\u201d"
)


def normalize_text(text: str) -> str:
    """Normalize text for fair OCR metric comparison.

    1. Unicode NFC normalization
    2. Lowercase
    3. Strip leading/trailing whitespace
    4. Collapse multiple whitespace to single space
    5. Remove punctuation
    """
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = text.translate(_PUNCT_TABLE)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


# =============================================================================
# Evaluator
# =============================================================================


class OCREvaluator:
    """Evaluate OCR models against the german_handwriting dataset."""

    DATASET_NAME = "fhswf/german_handwriting"
    SANITY_COUNT = 10

    def __init__(
        self,
        concurrency: int = 4,
        timeout: float = 120.0,
        sanity_dir: str = "results/sanity_checks",
    ):
        self.concurrency = concurrency
        self.timeout = timeout
        self.sanity_dir = sanity_dir
        self._dataset = None
        self._sanity_indices: list[int] = []

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    def load_data(self, limit: Optional[int] = None):
        """Load the HuggingFace dataset (train split)."""
        ds = load_dataset(self.DATASET_NAME, split="train")
        if limit is not None and limit > 0:
            ds = ds.select(range(min(limit, len(ds))))
        self._dataset = ds

        # Pick fixed sanity-check indices (evenly spaced across the dataset)
        n = len(ds)
        count = min(self.SANITY_COUNT, n)
        self._sanity_indices = [int(i * n / count) for i in range(count)]

        # Save sanity-check images once
        self._save_sanity_images(ds)

        return ds

    def _save_sanity_images(self, ds):
        """Save the sanity-check input images to disk (once per run)."""
        img_dir = os.path.join(self.sanity_dir, "images")
        os.makedirs(img_dir, exist_ok=True)
        for idx in self._sanity_indices:
            path = os.path.join(img_dir, f"sample_{idx:05d}.png")
            if not os.path.exists(path):
                ds[idx]["image"].save(path)
        print(f"  Sanity-check images saved to {img_dir}/")

    # ------------------------------------------------------------------
    # Image encoding
    # ------------------------------------------------------------------

    @staticmethod
    def image_to_base64(image: Image.Image) -> str:
        """Convert a PIL Image to a base64-encoded PNG string."""
        buf = BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _build_messages(self, config: OCRModelConfig, image_b64: str) -> list[dict]:
        """Build the OpenAI-compatible chat messages for one image."""
        image_part = {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
        }

        text_part = None
        if config.prompt is not None:
            text_part = {
                "type": "text",
                "text": f"{config.prompt_prefix}{config.prompt}",
            }

        # Some models (e.g. dots.ocr) expect image before text
        if config.image_first:
            content = [image_part] + ([text_part] if text_part else [])
        else:
            content = ([text_part] if text_part else []) + [image_part]

        return [{"role": "user", "content": content}]

    async def _ocr_single(
        self,
        client: httpx.AsyncClient,
        config: OCRModelConfig,
        image_b64: str,
        semaphore: asyncio.Semaphore,
    ) -> str:
        """Send one image to the model and return the predicted text."""
        async with semaphore:
            messages = self._build_messages(config, image_b64)
            payload = {
                "model": config.hf_name,
                "messages": messages,
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                **config.extra_body,
            }
            try:
                resp = await client.post(
                    f"{config.api_url}/chat/completions",
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"] or ""
                if config.strip_pattern:
                    content = re.sub(config.strip_pattern, " ", content)
                    content = re.sub(r"\s+", " ", content)
                return content.strip()
            except Exception as exc:
                # Return empty string on failure so metrics still compute
                print(f"  [WARN] Inference failed: {exc}")
                return ""

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_metrics(
        ground_truths: list[str], predictions: list[str]
    ) -> dict[str, float]:
        """Calculate raw and normalized CER / WER."""

        # Raw (only whitespace-stripped, guard against None)
        gt_raw = [(gt or "").strip() for gt in ground_truths]
        pred_raw = [(p or "").strip() for p in predictions]

        wer_raw = jiwer.wer(gt_raw, pred_raw)
        cer_raw = jiwer.cer(gt_raw, pred_raw)

        # Normalized
        gt_norm = [normalize_text(gt) for gt in ground_truths]
        pred_norm = [normalize_text(p) for p in predictions]

        wer_norm = jiwer.wer(gt_norm, pred_norm)
        cer_norm = jiwer.cer(gt_norm, pred_norm)

        return {
            "WER_raw": round(wer_raw, 6),
            "CER_raw": round(cer_raw, 6),
            "WER_normalized": round(wer_norm, 6),
            "CER_normalized": round(cer_norm, 6),
        }

    # ------------------------------------------------------------------
    # Full evaluation
    # ------------------------------------------------------------------

    async def evaluate(self, config: OCRModelConfig) -> dict:
        """Run full evaluation for a single model."""
        if self._dataset is None:
            raise RuntimeError("Call load_data() before evaluate()")

        dataset = self._dataset
        semaphore = asyncio.Semaphore(self.concurrency)

        print(f"\nEvaluating {config.hf_name} ({len(dataset)} samples) ...")
        print(f"  API: {config.api_url}")

        # Encode all images to base64 up front
        images_b64: list[str] = []
        ground_truths: list[str] = []
        for sample in dataset:
            images_b64.append(self.image_to_base64(sample["image"]))
            ground_truths.append(sample["text"] or "")

        # Run inference with concurrency limit and ordered results
        t0 = time.monotonic()
        async with httpx.AsyncClient() as client:
            tasks = [
                self._ocr_single(client, config, img_b64, semaphore)
                for img_b64 in images_b64
            ]
            predictions = list(await tqdm_gather(tasks, desc=f"  {config.key}"))
        time_taken = round(time.monotonic() - t0, 2)

        metrics = self.calculate_metrics(ground_truths, predictions)

        result = {
            "evaluation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": config.hf_name,
            "model_key": config.key,
            "api_url": config.api_url,
            "num_samples": len(dataset),
            **metrics,
            "time_taken_seconds": time_taken,
        }

        # Print summary
        print(f"  Results for {config.key}:")
        print(f"    time_taken_seconds: {time_taken}")
        for k, v in metrics.items():
            print(f"    {k}: {v:.4f}  ({v * 100:.2f}%)")

        # Write sanity-check file for this model
        self._write_sanity_check(config.key, ground_truths, predictions)

        return result

    def _write_sanity_check(
        self,
        model_key: str,
        ground_truths: list[str],
        predictions: list[str],
    ):
        """Write a text file comparing GT vs prediction for the fixed sanity samples."""
        os.makedirs(self.sanity_dir, exist_ok=True)
        path = os.path.join(self.sanity_dir, f"{model_key}.txt")

        with open(path, "w", encoding="utf-8") as f:
            f.write(f"Sanity check: {model_key}\n")
            f.write(f"{'=' * 60}\n\n")
            for i, idx in enumerate(self._sanity_indices):
                gt = ground_truths[idx]
                pred = predictions[idx]
                f.write(f"--- Sample {i + 1}  (dataset index {idx}) ---\n")
                f.write(f"Image: images/sample_{idx:05d}.png\n\n")
                f.write(f"  Ground truth : {gt}\n")
                f.write(f"  Prediction   : {pred}\n")
                f.write(
                    f"  Match        : {'YES' if gt.strip() == pred.strip() else 'NO'}\n\n"
                )

        print(f"  Sanity check written to {path}")

    # ------------------------------------------------------------------
    # CSV output
    # ------------------------------------------------------------------

    @staticmethod
    def write_results_to_csv(
        results: dict, output_file: str = "ocr_evaluation_results.csv"
    ):
        """Append one row of results to the CSV file."""
        file_exists = os.path.exists(output_file)
        with open(output_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(results)
        print(f"  Results written to {output_file}")


# =============================================================================
# Async helpers
# =============================================================================


async def tqdm_gather(tasks: list, desc: str = "") -> list:
    """Run tasks concurrently with asyncio.gather but show a tqdm progress bar."""
    results = [None] * len(tasks)
    pbar = tqdm(total=len(tasks), desc=desc)

    async def _wrap(idx, coro):
        result = await coro
        results[idx] = result
        pbar.update(1)
        return result

    wrapped = [_wrap(i, t) for i, t in enumerate(tasks)]
    await asyncio.gather(*wrapped)
    pbar.close()
    return results


# =============================================================================
# CLI
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate OCR models on the fhswf/german_handwriting dataset."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_CONFIGS.keys()) + ["all"],
        help="Model key to evaluate, or 'all' for every model.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of dataset samples (for quick testing).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ocr_evaluation_results.csv",
        help="Path to the output CSV file (default: ocr_evaluation_results.csv).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Max concurrent API requests (default: 4).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Per-request timeout in seconds (default: 120).",
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    # Resolve model list
    if args.model == "all":
        models = list(MODEL_CONFIGS.values())
    else:
        models = [MODEL_CONFIGS[args.model]]

    evaluator = OCREvaluator(
        concurrency=args.concurrency,
        timeout=args.timeout,
    )

    # Load dataset once
    print(f"Loading dataset: {OCREvaluator.DATASET_NAME} ...")
    ds = evaluator.load_data(limit=args.limit)
    print(f"  Loaded {len(ds)} samples.")

    # Evaluate each model
    for config in models:
        try:
            results = await evaluator.evaluate(config)
            evaluator.write_results_to_csv(results, output_file=args.output)
        except Exception as exc:
            print(f"\n  [ERROR] Failed to evaluate {config.key}: {exc}")
            continue

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
