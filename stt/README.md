# ASR evaluation suite

This suite measures how well speech-to-text (ASR) models transcribe speech. It sends each audio clip in a folder to a model behind an HTTP API, compares the output with a reference transcript, and appends the scores to a CSV file.

## What we test

We want to know how a model does on German and Swiss German administrative speech: dictation, meetings, several speakers, domain terms and names. A model that scores well on clean read speech may do worse on this material, so we run our own recordings next to public benchmarks.

## Metrics

| Metric | Library | What it measures |
|---|---|---|
| WER | `jiwer` | Word error rate over the whole test set |
| CER | `jiwer` | Character error rate over the whole test set |
| BLEU | `sacrebleu` | Corpus BLEU of prediction against reference |

The suite does not normalise text. It does not lower-case or strip punctuation, and only trims surrounding whitespace. Case and punctuation differences count as errors. Some pipelines have a `post_process` step that removes model-specific markup, for example `qwen_asr` and `moss` in `asr_eval/pipelines/openai_api.py`. If your reference text is normalised differently from the model output, normalise one side before you compare scores.

A per-clip CSV (`results/predictions/<label>.csv`) keeps the reference, the prediction and the per-clip WER and CER, so you can inspect which clips a model got wrong.

## Dataset

Our own data is not published. It consists of internal recordings of structured and unstructured dictation and of meetings, with multiple speakers. It stays private because it contains real work conversations. The reasoning is in `docs/data-policy.md` at the repository root. The [test results page](https://dcc-bs.github.io/documentation/evaluation/results#speech-to-text) shows one total per model on this data.

You can run the suite on your own data or on the public sample below.

## Data layout

`--data-dir` (default `./data`) holds one subdirectory per test case. A case is used only if it has both files. Subdirectories missing either file are skipped without a message.

```
data/
├── case_001/
│   ├── RecordedAudio.wav
│   └── ground_truth_transcript.txt
└── case_002/
    └── ...
```

## Install

```bash
uv sync                # core install
uv sync --extra sample # adds the Hugging Face loader for the public sample
```

## Public sample (FLEURS)

`scripts/fetch_sample.py` downloads a few clips from [`google/fleurs`](https://huggingface.co/datasets/google/fleurs) (config `de_de`, split `test`, licence CC-BY-4.0) and writes them in the layout above.

```bash
uv run --extra sample python scripts/fetch_sample.py --out data/sample --n 10
uv run asr-eval openai_api --model <model-id> --data-dir data/sample
```

Limits of this sample:

- FLEURS is read speech in Standard German. It has no Swiss German, no dictation and no meetings.
- The FLEURS reference text is lower-case without punctuation. Models that output capitals and punctuation score worse for that reason alone, because the suite does not normalise.
- Ten clips are too few to compare models. The sample shows that the pipeline runs from audio to scores. It says nothing about how a model does on our use case.

## Running a model server

The evaluation calls an OpenAI-compatible API that you start separately, usually vLLM in Docker. These scripts in `servers/` bind port 8000 and start a named, detached container:

```bash
servers/start_voxtral.sh                  # RedHatAI/Voxtral-Small-24B-2507-FP8-dynamic
servers/start_granite.sh                  # ibm-granite/granite-4.0-1b-speech
servers/start_granite_plus.sh             # ibm-granite/granite-speech-4.1-2b-plus
servers/start_parakeet.sh                 # nvidia/parakeet-tdt-0.6b-v3
servers/start_nemotron_asr.sh             # nvidia/nemotron-3.5-asr-streaming-0.6b
servers/start_vibe_voice.sh               # microsoft/VibeVoice
servers/start_moss_transcribe_diarize.sh  # OpenMOSS-Team/MOSS-Transcribe-Diarize
servers/start_nemotron_transcribe.sh     # nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-FP8

# Swiss German fine-tunes (suite: benchmark.swiss-german.yaml)
servers/start_whisper_stt4sg.sh           # matildecs/whisper-large-stt4sg
servers/start_whisper_turbo_swiss.sh      # Flurin17/whisper-large-v3-turbo-swiss-german
servers/start_cohere_swiss.sh             # Flurin17/cohere-transcribe-swiss-german
servers/start_parakeet_swiss.sh           # Flurin17/parakeet-tdt-0.6b-v3-swiss-german
servers/start_nemotron_transcribe_swiss.sh  # Flurin17/nemotron-3.5-transcribe-swiss-german
```

The Whisper and Cohere fine-tunes run on vLLM's own transcription route, which cuts audio longer than 30 seconds. The Nemotron fine-tune is published in BF16 only (67 GB); its script has vLLM quantize it to FP8 while loading so it fits two 24 GB GPUs. It was trained on clips of at most 30 seconds, so run it with `--arg chunk_s=30`: the `nemotron_transcribe` pipeline then cuts each file at the quietest point before every 30-second mark. `--arg prompt=swiss_ft` selects its training prompt.

Remove a container with `docker rm -f <container>`. `asr-bench` can also start and stop the servers for you (see below).

The two NVIDIA models run on NeMo and have no native vLLM transcription. Their scripts start `servers/nemo_asr_server.py`, a small FastAPI shim that exposes `/v1/models` and `/v1/audio/transcriptions`. The standard `openai_api` pipeline then works unchanged.

`servers/start_nemotron_transcribe.sh` serves the FP8 checkpoint on two GPUs (tensor parallel) with vLLM 0.32.0 and the `nemo_speechlm` plugin, which it installs from a pinned NeMo Speech commit on start. The model only answers `/v1/chat/completions`, so use the `nemotron_transcribe` pipeline. The first start compiles FlashInfer kernels and takes several minutes; the kernels are cached in `~/.cache/flashinfer`.

`servers/start_vibe_voice.sh` clones the upstream repository <https://github.com/microsoft/VibeVoice> into `servers/VibeVoice/` (git-ignored) and mounts it into the container.

Any other endpoint that speaks the OpenAI `/v1/audio/transcriptions` API works too. Pass it with `--api-url`.

## Evaluate one model

```bash
uv run asr-eval openai_api --model "RedHatAI/Voxtral-Small-24B-2507-FP8-dynamic"

# Qwen ASR, stripping its <asr_text> wrapper tokens
uv run asr-eval openai_api --model "Qwen/Qwen3-ASR-1.7B" --arg post_process=qwen_asr

# VibeVoice (chat completions with base64 audio)
uv run asr-eval vibevoice

# MOSS-Transcribe-Diarize needs response_format and temperature set,
# and post_process=moss to strip its [start][Sxx]text[end] markers
uv run asr-eval openai_api --model "OpenMOSS-Team/MOSS-Transcribe-Diarize" \
  --arg response_format=json --arg temperature=0 --arg post_process=moss

# Nemotron 3.5 Transcribe (chat completions with base64 audio). --arg prompt
# picks a prompt preset: verbatim, clean (default), translate_de or swiss_de, or
# takes your own instruction text
uv run asr-eval nemotron_transcribe --arg prompt=clean
```

Common flags: `--api-url` (default `http://localhost:8000/v1`) and `--data-dir` (default `./data`). Pipeline constructor arguments go through repeatable `--arg KEY=VALUE`.

## Benchmark several models

`asr-bench` runs a YAML suite one model after another. Models share one GPU and port 8000, so runs are strictly serial. For a run with a `server:` block it starts the server, polls `{api_url}/models` until it answers, evaluates, then removes the container before the next run.

```bash
uv run asr-bench benchmark.example.yaml
uv run asr-bench benchmark.example.yaml --keep-going   # continue after a failed run
```

[`benchmark.example.yaml`](benchmark.example.yaml) shows the config shape. Leave out the `server:` block to use a server you started yourself. Values under `defaults:` merge into every run, and the run wins on conflict.

## Output

- `results/asr_evaluation_results.csv`: one row per run, append-only. Besides the scores it records the pipeline, the `--arg` values, the data folder, the predictions file and the commit of this repository. Files from older versions get the new columns, left empty, on the next run.
- `results/predictions/<label>.csv`: per-clip reference, prediction, WER and CER.
- `asr-plot` draws a WER/CER chart from the results CSV (`results/asr_metrics_plot.png`). Use `--top N` to limit it to N models.

Both `results/` and `data/` are git-ignored.

Set `AI_EVAL_RESULTS_DIR` to write to `$AI_EVAL_RESULTS_DIR/stt/` instead. We point it at a checkout of our private results repository, so that all runs from all machines end up in one place. `asr-plot` reads from the same folder.

## Add a model

Most models behind an OpenAI `/v1/audio/transcriptions` API need no code.

1. Add a start script in `servers/` that runs the model in a detached, named container on port 8000. Copy `start_granite.sh` and change the image and model id. The container name is what `asr-bench` uses for teardown.
2. Add a run to your benchmark YAML:

```yaml
runs:
  - name: my-model                  # label used for the predictions CSV
    pipeline: openai_api
    model: org/My-ASR-Model         # sent to the API as the model id
    args:                           # optional, passed to the pipeline as-is
      language: de
    server:
      script: servers/start_my_model.sh
      container: my-model-vllm      # must match --name in the script
      ready_timeout: 900            # seconds, default 900
```

If the model does not speak the OpenAI transcriptions API, add a pipeline:

1. Subclass `ASRPipeline` in `asr_eval/base.py` and implement `async transcribe(audio_path) -> str`.
2. Register the class in `PIPELINES` in `asr_eval/pipelines/__init__.py`.

Data loading, metrics and CSV output come from the base class. Constructor arguments need no CLI changes.

## Tests

```bash
uv run pytest tests -v
```

The tests cover the sample writer and the test-case loader. They do not need a model server or network access.

## Layout

```
asr_eval/
├── base.py            # ASRPipeline: data loading, metrics, persistence
├── evaluate.py        # asr-eval: single run
├── suite.py           # asr-bench: YAML suite
├── plot.py            # asr-plot: comparison chart
└── pipelines/         # openai_api, vibevoice, nemotron_transcribe
servers/               # Docker start scripts
scripts/fetch_sample.py
tests/                 # sample writer and loader tests
benchmark.example.yaml
```
