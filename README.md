# AI Model Evaluation

This repository holds the method and benchmark code behind model selection at the Data Competence Center Basel-Stadt. The DCC AI team selects every model that runs on the KDKP, the canton's AI platform. We publish the criteria, the process and the code for the STT, OCR and TTS suites so that others can check and reuse them. PII/NER and decision-model code follows later. We do not publish test data, scores or rankings.

Overview in plain language: <https://dcc-bs.github.io/documentation/evaluation/>

## Method

- [Selection standard](docs/selection-standard.md): hard gates and the investigation per task type
- [Evaluation layers](docs/eval-layers.md): how we test, from public benchmarks to production
- [Model lifecycle](docs/lifecycle.md): cadence, transition, rollback, records
- [Decision record template](docs/decision-record-template.md): one record per model change
- [Data policy](docs/data-policy.md): what data we use and how to plug in your own

## Evaluation suites

| Suite | What it tests | Where |
|---|---|---|
| STT | Speech recognition on German and Swiss German (WER, CER, BLEU) | [`stt/`](stt/README.md) |
| OCR | Vision-language models on German handwriting (CER, WER) | [`ocr/`](ocr/README.md) |
| TTS | Speech synthesis, judged by listening | [`tts/`](tts/README.md) |
| PII/NER | Anonymisation models. Code not public yet | [`pii-ner/`](pii-ner/README.md) |
| Decision models | Language models on internal tasks. Code not public yet | [`decision-models/`](decision-models/README.md) |
| Document conversion | Layout and conversion benchmarks | [`dcc-docling-serve/benchmarks`](https://github.com/DCC-BS/dcc-docling-serve/tree/main/benchmarks) |
| Speech API | Transcription and diarization tests, load tests | [`bentoml-faster-whisper`](https://github.com/DCC-BS/bentoml-faster-whisper) |
| LLM load tests | Load tests on live endpoints | [`vllm-benchmark`](https://github.com/DCC-BS/vllm-benchmark) |
| Text features | Advisor and simplify evals | [`text-mate-backend/evals`](https://github.com/DCC-BS/text-mate-backend/tree/main/evals) |
| Transcription and translation apps | Route, service and integration tests | [`transcribo-backend`](https://github.com/DCC-BS/transcribo-backend), [`bs-translator-backend`](https://github.com/DCC-BS/bs-translator-backend) |
| Deep research | Evaluation of the deep-research engine | [`deep-research-evaluation`](https://github.com/DCC-BS/deep-research-evaluation) |

The model catalog and hardware specs are on the documentation site: [models](https://dcc-bs.github.io/documentation/infrastructure/models), [hardware](https://dcc-bs.github.io/documentation/infrastructure/hardware).

## Quick start

Each suite is its own `uv` project.

Prerequisites:

- [uv](https://docs.astral.sh/uv/) for every suite.
- A running OpenAI-compatible model server for the STT and OCR suites. Serving a model needs Docker with the NVIDIA runtime and a GPU. For STT, see "Running a model server" in [`stt/README.md`](stt/README.md). For OCR, `ocr/run_benchmark.sh` starts the container for you, and `ocr/serve_models.sh` starts one on its own.
- A GPU and the vLLM install step for the TTS demo, see [`tts/README.md`](tts/README.md).

The first STT sample run downloads the FLEURS dataset from Hugging Face.

```bash
# STT: fetch a small public sample, then evaluate a model behind an OpenAI-compatible API
cd stt
uv sync --extra sample
uv run --extra sample python scripts/fetch_sample.py --out data/sample --n 10
uv run asr-eval openai_api --model <model-id> --data-dir data/sample

# OCR: start a model container, evaluate, plot, stop (needs Docker and a CUDA GPU)
cd ocr
uv sync
./run_benchmark.sh <key> --limit 50

# TTS: install vLLM, then start the server and the listening demo
cd tts
uv sync
uv pip install vllm==0.14.0 --torch-backend=auto
./run_server.sh
```

## Governance

The DCC AI team selects and evaluates all models on the KDKP. A governance board for model decisions is being established.

## Contributing

Issues and pull requests are welcome. Read the [data policy](docs/data-policy.md) first: do not commit recordings, documents, results or predictions from internal sources.

## Licence

MIT, see [LICENSE](LICENSE).

## Third-party code

`tts/main.py` is derived from example code of Qwen3-TTS and is licensed under Apache-2.0. The licence header stays in the file. The rest of the repository is MIT.

## Development

Before every commit the repository is scanned for internal hosts, secrets, personal names and result artefacts. The scan reads a local denylist from `~/.config/ai-model-evaluation/denylist.txt` (one term per line, never committed).

Enable the pre-commit hook:

```
ln -s ../../scripts/pre-commit .git/hooks/pre-commit
```

Run the scan manually:

```
uv run python scripts/check_public_safety.py .
```

If ROOT is a single file, the scan treats it as text only. It does not run the path and result-artefact checks.
