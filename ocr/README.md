# OCR evaluation suite

This suite measures how well vision-language models (VLMs) read handwriting. It sends each image to a model served by vLLM through an OpenAI-compatible API, compares the output with the ground-truth text, and appends the scores to a CSV file.

## What we test

We want to know how a model reads German text that is not clean print. The public dataset below is the part of that question we can share. It does not cover our own document types, such as forms, tables or letters. For layout and document conversion, see the benchmarks linked under Related.

## Metrics

| Metric | Library | What it measures |
|---|---|---|
| CER | `jiwer` | Character error rate over the test set |
| WER | `jiwer` | Word error rate over the test set |

Each metric comes in two variants:

- **Raw.** Model output against ground truth. Only whitespace is stripped.
- **Normalised.** Both sides lower-cased, punctuation removed, whitespace collapsed, Unicode NFC-normalised.

Raw scores count case and punctuation differences as errors. Normalised scores show how much of the error is only formatting.

## Data

The suite uses [`fhswf/german_handwriting`](https://huggingface.co/datasets/fhswf/german_handwriting), a public dataset of about 10,900 images of German handwriting with ground-truth text. It is downloaded at runtime and not stored in this repository.

Limit: handwriting lines in one dataset say little about scans of administrative documents. Treat the scores as one input among several.

## Install

You need a CUDA GPU, Docker and [uv](https://docs.astral.sh/uv/).

```bash
cd ocr
uv sync
```

## How to run

`run_benchmark.sh` starts a model container, runs the evaluation, plots the results and stops the container. It stops the container on exit, also when the evaluation fails.

```bash
./run_benchmark.sh <key> --limit 50    # quick check with 50 samples
./run_benchmark.sh <key>               # full dataset
./run_benchmark.sh all                 # every model, one after another
```

Keys are defined in `serve_models.sh` and `eval_ocr.py`: `olmocr`, `glm-ocr`, `paddleocr`, `deepseek-ocr`, `lighton-ocr`, `dots-ocr`, `lighton-ocr-3`, `nemotron-parse`, `paddleocr-1.6`. Extra flags (`--limit`, `--concurrency`, `--timeout`, `--output`) go to `eval_ocr.py`.

To run the steps yourself:

```bash
./serve_models.sh start <key>     # waits until the model is healthy
uv run eval_ocr.py --model <key> --limit 50
uv run plot_results.py
./serve_models.sh stop <key>
```

`./serve_models.sh status` lists running containers. On a shared machine, `OCR_GPU=1` puts the container on GPU 1 instead of all GPUs, and `VLLM_MEMORY=0.4` caps the share of GPU memory vLLM takes. Each model runs in its own vLLM container on its own port and mounts `~/.cache/huggingface` for weights. One GPU usually holds one model at a time.

## Output

- `ocr_evaluation_results.csv`: one row per run, append-only.
- `ocr_evaluation_results.png`: bar chart from `plot_results.py`. It shows the latest run per model. `--all-runs` shows every run.
- `results/sanity_checks/`: for a fixed set of evenly spaced samples, the input image and a text file with ground truth next to prediction. Use it to check by eye that a model reads the images at all.

All of these are git-ignored.

## Models tested in October 2026

We added three models to the suite and compared them with GLM-OCR, which we already use. We publish the weaknesses we found, not the scores. None of the three replaces GLM-OCR.

- **LightOnOCR-3-4B** (`lighton-ocr-3`, Apache 2.0). Reads most lines well. It sometimes returns nothing for a line, and it cuts off the end of other lines. In our document benchmark it converts whole pages, which makes it much slower than GLM-OCR on long PDFs, and it recovers no table structure.
- **Nemotron-3.5-Parse** (`nemotron-parse`, NVIDIA Open Model License). The model is built to parse whole pages, not single lines. On short handwriting lines it often invents text, and it wraps many lines in an empty LaTeX table with the text in one cell. `eval_ocr.py` strips the region boxes but not this table markup, so that markup also counts as errors. On large plan drawings it runs out of output tokens and returns nothing.
- **PaddleOCR-VL-1.6** (`paddleocr-1.6`, Apache 2.0). Misreads umlauts and single letters ("willkörliche", "Zofallsauswahl"), and cuts off some lines.

We did not test two more candidates. Their weights use a modified OpenRAIL-M licence that is free only for small companies, so they fail our licence check.

- **Chandra-OCR-2** (Datalab): free only below USD 2 million in funding or revenue.
- **Indra-OCR** (Datalab): free only below USD 20 million.

The layout and pipeline candidates (Nemotron OCR v2, docling's Egret layout models, PP-OCRv6) run in the document benchmark linked under Related, not here.

## Add a model

1. Add an entry to `MODEL_CONFIGS` in `eval_ocr.py`: key, Hugging Face name, port and the prompt from the model card.
2. Add the same key, port and vLLM arguments to `serve_models.sh`.
3. Run `./run_benchmark.sh <key> --limit 50`.

## Limits

- One dataset, handwriting only.
- Prompts follow each model card. A different prompt can change a score, and we do not tune prompts per model.
- `nemotron-parse` answers with a box and a class around each text region. `eval_ocr.py` strips that markup before scoring. Its vLLM plugin ships inside the model repository and pins vLLM 0.26.0, so the container installs it at start. The model card's FlashAttention 3 flag is left out because it needs a Hopper GPU.
- Some models need a specific vLLM version. DeepSeek-OCR-2 uses a custom architecture that not every vLLM version supports. Check the model repository if its container exits at start.
- If a container exits at start, look at `docker logs <container>`. Usual causes are too little GPU memory, a missing Hugging Face token for a gated model, or an unsupported architecture.

## Related

[`dcc-docling-serve/benchmarks`](https://github.com/DCC-BS/dcc-docling-serve/tree/main/benchmarks) covers layout analysis and document conversion, which this suite does not.

## Layout

```
run_benchmark.sh   # start, evaluate, plot, stop
serve_models.sh    # start and stop vLLM containers
eval_ocr.py        # CER and WER on german_handwriting
plot_results.py    # comparison chart from the CSV
pyproject.toml
```
