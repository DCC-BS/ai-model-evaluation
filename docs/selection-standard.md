# Selection standard

This page describes how the DCC AI team decides which models run on the KDKP, the AI platform of the Canton of Basel-Stadt. It has two parts: hard gates that drop a candidate, and an investigation that tells us how good the remaining candidates are for our work.

The public overview is on the [documentation site](https://dcc-bs.github.io/documentation/evaluation/). The current model catalog is on the [models page](https://dcc-bs.github.io/documentation/infrastructure/models). Hardware details are on the [hardware page](https://dcc-bs.github.io/documentation/infrastructure/hardware). We do not repeat either here.

The DCC AI team selects and evaluates all models on the KDKP. A governance board for model decisions is being established.

## Hard gates

A candidate that fails one gate is dropped. We do not trade a failed gate against a good benchmark score.

1. **Fits our GPUs.** Weights plus KV cache for the target sequence length and concurrency must fit on our hardware. Today that is the H100 80 GB. An H200 NVL 141 GB is on the roadmap.
2. **Fits one server.** We do not serve one model across several servers with vLLM, because cross-server parallelism costs latency and speed. Tensor parallelism across the GPUs inside one server is fine. Multi-server serving is not.
3. **Served by vLLM.** This gives us one way to serve models and an OpenAI-compatible API. There is one documented exception: speech-to-text runs on FasterWhisper with BentoML. STT candidates without vLLM support, for example NeMo models, can still be benchmarked. They need a separate serving decision before they can run on the platform.
4. **Permissive licence.** The licence must allow our usage scenarios.
5. **Hosted on Hugging Face.** We pull weights from there.

## Investigation per task type

Leaderboards are inputs, not decisions. The final call rests on the gates, our own evaluations, load tests and hands-on use. How we layer these is described in [eval-layers.md](eval-layers.md).

### Layout and OCR

We start with public benchmarks and leaderboards:

- [olmOCR-Bench](https://github.com/allenai/olmocr)
- [OCRBench v2 leaderboard](https://huggingface.co/spaces/ling99/OCRBench-v2-leaderboard)
- [ParseBench](https://huggingface.co/datasets/llamaindex/ParseBench)
- [OmniDocBench](https://github.com/opendatalab/OmniDocBench)
- [Hugging Face OCR leaderboards collection](https://huggingface.co/collections/davanstrien/ocr-leaderboards)
- [DocBank](https://huggingface.co/datasets/liminghao1630/DocBank)
- [dp-bench](https://huggingface.co/datasets/upstage/dp-bench)
- [Example layout-detection model (DETR)](https://huggingface.co/cmarkea/detr-layout-detection)

Then we run our own OCR benchmark ([`ocr/`](../ocr/README.md)) and the docling scenario and edge-case suite in [`dcc-docling-serve/benchmarks`](https://github.com/DCC-BS/dcc-docling-serve/tree/main/benchmarks).

### Speech-to-text

We run our own benchmark on German and Swiss German recordings and measure WER, CER and BLEU ([`stt/`](../stt/README.md)). Diarization and API behaviour are tested in [`bentoml-faster-whisper`](https://github.com/DCC-BS/bentoml-faster-whisper).

### Embeddings and reranking

We use the [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) as the starting point.

### Large language models

[Artificial Analysis](https://artificialanalysis.ai/) is our main external reference. It is a third-party service with its own methodology, which we do not control or reproduce, so we treat it as an input. We look at:

- cost and speed
- supported input types
- reasoning, and whether the reasoning effort can be steered
- size and architecture, including KV-cache use and VRAM per sequence length
- coding, reasoning, legal and other text tasks
- tool use
- verbosity
- openness
- bias

Then we check things the leaderboards do not tell us:

- **Memory.** We estimate it with `hf-mem` for the target sequence length and quantisation (see below).
- **Quantisation.** Which formats are fast and stable on H100 and H200, with no looping output. Also whether the KV cache can be quantised.
- **Context length.** What the model supports, and what we can serve.
- **Known vLLM issues** for that model.
- **Public reception.** Community reports and open issues.
- **Hands-on use** on our own development machines.

### Text-to-speech

No automated metric tells us whether a voice pronounces a Basel place name acceptably. The team listens, using a written protocol: [`tts/LISTENING_PROTOCOL.md`](../tts/LISTENING_PROTOCOL.md). Listeners check intelligibility, pronunciation of Swiss names and places, naturalness and voice consistency.

### PII/NER and decision models

We evaluate anonymisation (PII/NER) models and decision models on our own benchmarks. The code is not public yet. See [`pii-ner/`](../pii-ner/README.md) and [`decision-models/`](../decision-models/README.md).

## Technical depth

### Memory check with hf-mem

[`hf-mem`](https://pypi.org/project/hf-mem/) reads a model's metadata from the Hugging Face Hub and estimates the memory it needs, without downloading the weights. We run it through `uvx`. The relevant flags, checked against hf-mem 0.5.6 with `uvx hf-mem --help`:

| Flag | Purpose |
|---|---|
| `--model-id` | Model repository on the Hub (required) |
| `--revision` | Model revision |
| `--max-model-len` | Context length, prompt plus output. Derived from the model config if unset |
| `--batch-size` | Batch size, used to estimate KV-cache memory. Default 1 |
| `--kv-cache-dtype` | KV-cache data type, for example `auto`, `bfloat16` or `fp8` |
| `--experimental` | Enables KV-cache estimation. Applies to `...ForCausalLM` and `...ForConditionalGeneration` models |
| `--json-output` | Machine-readable output |

Example:

```bash
uvx hf-mem --model-id <org>/<model> --experimental --max-model-len 32768 --batch-size 8 --kv-cache-dtype fp8
```

Replace `<org>/<model>` with the repository you want to check. The KV-cache figure is an estimate. We treat it as a first filter for gate 1 and confirm with a real vLLM start on the target hardware.

### Serving

We serve with [vLLM](https://github.com/vllm-project/vllm). Gate 3 is about keeping one serving stack. Our [software page](https://dcc-bs.github.io/documentation/infrastructure/software) describes the stack, including the FasterWhisper and BentoML setup for speech-to-text.
