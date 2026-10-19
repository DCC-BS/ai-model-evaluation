#!/usr/bin/env python3
"""Single CLI entry point for evaluating any ASR model.

Usage examples:

  # Voxtral (OpenAI-compatible API)
  python -m asr_eval.evaluate openai_api \
      --model "RedHatAI/Voxtral-Small-24B-2507-FP8-dynamic"

  # Qwen ASR with post-processing
  python -m asr_eval.evaluate openai_api \
      --model "Qwen/Qwen3-ASR-1.7B" \
      --arg post_process=qwen_asr

  # IBM Granite
  python -m asr_eval.evaluate openai_api \
      --model "ibm-granite/granite-4.0-1b-speech"

  # VibeVoice
  python -m asr_eval.evaluate vibevoice

To benchmark many models back-to-back, use ``asr-bench`` (see asr_eval.suite).

"""

from __future__ import annotations

import argparse
import asyncio
import datetime

from asr_eval.pipelines import PIPELINES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate an ASR model and append results to CSV."
    )
    parser.add_argument(
        "pipeline",
        choices=list(PIPELINES.keys()),
        help="Pipeline type to use.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name/ID (passed to the API). Required for openai_api pipeline.",
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000/v1",
        help="Base URL for the ASR API (default: http://localhost:8000/v1).",
    )
    parser.add_argument(
        "--data-dir",
        default="./data",
        help="Path to the test data directory (default: ./data).",
    )
    parser.add_argument(
        "--arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        dest="extra_args",
        help=(
            "Extra pipeline constructor kwarg (repeatable), e.g. "
            "--arg language=de --arg post_process=qwen_asr."
        ),
    )
    return parser


def parse_kwargs(pairs: list[str]) -> dict:
    """Turn ``["language=de", "post_process=qwen_asr"]`` into a kwargs dict."""
    out: dict = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--arg must be KEY=VALUE, got: {pair!r}")
        key, value = pair.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def build_pipeline(pipeline: str, model: str | None, api_url: str, data_dir: str, extra: dict):
    """Construct a pipeline generically — no per-pipeline branching here.

    ``model`` is only passed when provided so each pipeline can fall back to
    its own default; ``extra`` is forwarded verbatim to the constructor.
    """
    pipeline_cls = PIPELINES[pipeline]
    kwargs: dict = {"api_url": api_url, "data_dir": data_dir, **extra}
    if model is not None:
        kwargs["model"] = model
    try:
        return pipeline_cls(**kwargs)
    except TypeError as exc:
        raise SystemExit(f"Cannot build '{pipeline}' pipeline: {exc}")


async def run(args: argparse.Namespace) -> None:
    pipeline = build_pipeline(
        args.pipeline,
        args.model,
        args.api_url,
        args.data_dir,
        parse_kwargs(args.extra_args),
    )

    print(f"Evaluating: {pipeline.model} via {args.pipeline} pipeline")
    print(f"API: {pipeline.api_url}")
    print(f"Data: {pipeline.data_dir}")
    print()

    start = datetime.datetime.now()
    results = await pipeline.evaluate()
    duration = datetime.datetime.now() - start

    pipeline.write_results_to_csv(results)
    pred_path = pipeline.write_predictions_csv(results)

    print()
    print(f"WER:  {results['WER']:.4f}")
    print(f"CER:  {results['CER']:.4f}")
    print(f"BLEU: {results['BLEU']:.2f}")
    print(f"Duration: {duration}")
    print(f"Results appended to {pipeline.RESULTS_CSV}")
    print(f"Predictions written to {pred_path}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
