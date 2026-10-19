#!/usr/bin/env python3
"""Run a whole benchmark suite from a YAML file, one model after another.

Models share a single GPU/port, so runs execute strictly serially: for each
run the configured server is started, polled until ready, evaluated, then torn
down before the next run begins (VRAM is freed between runs).

Usage:
    asr-bench benchmark.yaml
    asr-bench benchmark.yaml --data-dir ./data --keep-going

Config shape (see benchmark.example.yaml):

    defaults:                       # merged into every run (run wins on conflict)
      api_url: http://localhost:8000/v1
      data_dir: ./data
    runs:
      - name: voxtral               # optional label; defaults to model
        pipeline: openai_api
        model: RedHatAI/Voxtral-Small-24B-2507-FP8-dynamic
        args: {language: de}        # forwarded verbatim to the pipeline ctor
        server:                     # optional; omit if the server is already up
          script: servers/start_voxtral.sh
          container: voxtral-vllm   # docker --name, used for teardown
          ready_timeout: 900        # seconds to wait for readiness (default 900)
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import subprocess
import time
from pathlib import Path

import httpx
import yaml

from asr_eval.evaluate import build_pipeline


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict) or "runs" not in cfg:
        raise SystemExit(f"{path}: expected a mapping with a 'runs' list.")
    return cfg


def merge_run(defaults: dict, run: dict) -> dict:
    merged = {**defaults, **run}
    # args is the one nested dict we deep-merge so defaults can set shared kwargs.
    merged["args"] = {**defaults.get("args", {}), **run.get("args", {})}
    return merged


# ----------------------------------------------------------------------
# Server lifecycle
# ----------------------------------------------------------------------


def start_server(server: dict) -> None:
    script = server["script"]
    print(f"  → starting server: {script}")
    subprocess.run(["bash", script], check=True)


def stop_server(server: dict) -> None:
    container = server.get("container")
    if not container:
        return
    print(f"  → stopping server: {container}")
    subprocess.run(
        ["docker", "rm", "-f", container],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Give the GPU a moment to release VRAM before the next run.
    time.sleep(5)


def wait_for_server(api_url: str, timeout: float) -> None:
    """Poll ``{api_url}/models`` until it answers 200 or we time out."""
    url = api_url.rstrip("/") + "/models"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if httpx.get(url, timeout=5.0).status_code == 200:
                print("  → server ready")
                return
        except httpx.HTTPError:
            pass
        time.sleep(5)
    raise TimeoutError(f"server not ready after {timeout:.0f}s ({url})")


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------


async def run_one(run: dict) -> dict:
    pipeline = build_pipeline(
        run["pipeline"],
        run.get("model"),
        run.get("api_url", "http://localhost:8000/v1"),
        run.get("data_dir", "./data"),
        run.get("args", {}),
    )
    server = run.get("server")
    label = run.get("name") or pipeline.run_label()

    try:
        if server:
            start_server(server)
            wait_for_server(pipeline.api_url, server.get("ready_timeout", 900))

        print(f"  → evaluating {pipeline.model}")
        start = datetime.datetime.now()
        results = await pipeline.evaluate()
        duration = datetime.datetime.now() - start

        pipeline.write_results_to_csv(results)
        pred_path = pipeline.write_predictions_csv(results, label=label)
        print(
            f"  ✓ WER={results['WER']:.4f} CER={results['CER']:.4f} "
            f"BLEU={results['BLEU']:.2f} ({duration}) → {pred_path}"
        )
        return {"name": label, "ok": True, **{k: results[k] for k in ("WER", "CER", "BLEU")}}
    finally:
        if server:
            stop_server(server)


async def run_suite(config_path: Path, overrides: dict, keep_going: bool) -> int:
    cfg = load_config(config_path)
    defaults = {**cfg.get("defaults", {}), **{k: v for k, v in overrides.items() if v is not None}}
    runs = cfg["runs"]

    print(f"Benchmark suite: {len(runs)} run(s) from {config_path}\n")
    summary: list[dict] = []
    failures = 0

    for i, raw in enumerate(runs, 1):
        run = merge_run(defaults, raw)
        name = run.get("name") or run.get("model") or run["pipeline"]
        print(f"[{i}/{len(runs)}] {name}")
        try:
            summary.append(await run_one(run))
        except Exception as exc:  # noqa: BLE001 — keep the suite resilient
            failures += 1
            print(f"  ✗ FAILED: {exc}")
            summary.append({"name": name, "ok": False, "error": str(exc)})
            if not keep_going:
                print("\nAborting (use --keep-going to continue past failures).")
                break
        print()

    print("Summary")
    print("-------")
    for row in summary:
        if row.get("ok"):
            print(f"  {row['name']:<30} WER={row['WER']:.4f} CER={row['CER']:.4f} BLEU={row['BLEU']:.2f}")
        else:
            print(f"  {row['name']:<30} FAILED: {row.get('error', '')}")

    return 1 if failures else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an ASR benchmark suite from YAML.")
    parser.add_argument("config", type=Path, help="Path to the benchmark YAML file.")
    parser.add_argument("--data-dir", default=None, help="Override data_dir for all runs.")
    parser.add_argument("--api-url", default=None, help="Override api_url for all runs.")
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue with remaining runs after a failure.",
    )
    args = parser.parse_args()

    overrides = {"data_dir": args.data_dir, "api_url": args.api_url}
    raise SystemExit(asyncio.run(run_suite(args.config, overrides, args.keep_going)))


if __name__ == "__main__":
    main()
