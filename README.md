# AI model evaluation

The Data Competence Center Basel-Stadt uses this repository to test AI models before they go into use. The DCC AI team chooses every model that runs on the KDKP, the canton's AI platform. Here we publish our rules, our steps and the test code for speech-to-text, OCR and text-to-speech, so that others can check our work and reuse it. The code for anonymisation and decision models will follow later. Our test data, scores and rankings stay private.

A plain-language overview is on the [documentation site](https://dcc-bs.github.io/documentation/evaluation/).

## Our method

- [Selection standard](docs/selection-standard.md): the five basic checks, and what we look at for each kind of model
- [How we test](docs/eval-layers.md): the five rounds of testing, from public rankings to daily use
- [Model lifecycle](docs/lifecycle.md): how often we review, how we switch, how we go back, what we write down
- [Decision record template](docs/decision-record-template.md): what we write down for each model change
- [Data policy](docs/data-policy.md): which data we use, and how to run the tests on your own

## Test suites

| Suite | What it tests | Where |
|---|---|---|
| STT | Speech recognition in German and Swiss German (WER, CER, BLEU) | [`stt/`](stt/README.md) |
| OCR | Reading German handwriting with vision-language models (CER, WER) | [`ocr/`](ocr/README.md) |
| TTS | Synthetic voices, judged by listening | [`tts/`](tts/README.md) |
| PII/NER | Anonymisation models. Code not public yet | [`pii-ner/`](pii-ner/README.md) |
| Decision models | Language models on internal tasks. Code not public yet | [`decision-models/`](decision-models/README.md) |
| Document conversion | Layout and conversion tests | [`dcc-docling-serve/benchmarks`](https://github.com/DCC-BS/dcc-docling-serve/tree/main/benchmarks) |
| Speech API | Transcription, telling speakers apart, load tests | [`bentoml-faster-whisper`](https://github.com/DCC-BS/bentoml-faster-whisper) |
| LLM load tests | Load tests on the live services | [`vllm-benchmark`](https://github.com/DCC-BS/vllm-benchmark) |
| TextMate | Tests for the advice and simplification features | [`text-mate-backend/evals`](https://github.com/DCC-BS/text-mate-backend/tree/main/evals) |
| Transcription and translation apps | Route, service and integration tests | [`transcribo-backend`](https://github.com/DCC-BS/transcribo-backend), [`bs-translator-backend`](https://github.com/DCC-BS/bs-translator-backend) |
| Deep research | Tests for the research assistant | [`deep-research-evaluation`](https://github.com/DCC-BS/deep-research-evaluation) |

The documentation site lists the [models we run](https://dcc-bs.github.io/documentation/infrastructure/models) and describes our [hardware](https://dcc-bs.github.io/documentation/infrastructure/hardware).

## Quick start

Each suite is its own `uv` project.

You need:

- [uv](https://docs.astral.sh/uv/) for every suite.
- For STT and OCR, a running model server with an OpenAI-compatible API. Running a model needs Docker with the NVIDIA runtime and a GPU. For STT, see "Running a model server" in [`stt/README.md`](stt/README.md). For OCR, `ocr/run_benchmark.sh` starts the container for you, and `ocr/serve_models.sh` starts one on its own.
- For the TTS demo, a GPU and the vLLM install step described in [`tts/README.md`](tts/README.md).

The first STT sample run downloads the FLEURS dataset from Hugging Face.

```bash
# STT: download a small public sample, then test a model behind an OpenAI-compatible API
cd stt
uv sync --extra sample
uv run --extra sample python scripts/fetch_sample.py --out data/sample --n 10
uv run asr-eval openai_api --model <model-id> --data-dir data/sample

# OCR: start a model container, run the test, plot, stop (needs Docker and a CUDA GPU)
cd ocr
uv sync
./run_benchmark.sh <key> --limit 50

# TTS: install vLLM, then start the server and the listening demo
cd tts
uv sync
uv pip install vllm==0.14.0 --torch-backend=auto
./run_server.sh
```

## Who decides

The DCC AI team chooses and tests all models on the KDKP. We are setting up a governance board for these decisions.

## Contributing

Issues and pull requests are welcome. Please read the [data policy](docs/data-policy.md) first. Don't commit recordings, documents, results or model outputs from internal sources.

## Licence

MIT, see [LICENSE](LICENSE).

## Third-party code

`tts/main.py` is based on example code from Qwen3-TTS, which is licensed under Apache-2.0. The licence header stays in the file. Everything else in this repository is MIT.

## Development

A scan checks the repository for internal server names, secrets, personal names and result files before every commit. It reads a local list of names to block from `~/.config/ai-model-evaluation/denylist.txt`. Put one term per line, and never commit that file.

Turn on the pre-commit hook:

```
ln -s ../../scripts/pre-commit .git/hooks/pre-commit
```

Run the scan by hand:

```
uv run python scripts/check_public_safety.py .
```

If you pass a single file instead of a folder, the scan only checks its text. It skips the checks on file paths and result files.
