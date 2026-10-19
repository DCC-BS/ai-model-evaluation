# Selection standard

This page explains how the DCC AI team decides which models run on the KDKP, the AI platform of the Canton of Basel-Stadt. First, every model has to pass five basic checks. A model that fails one of them is out. Then we look closely at how well the remaining models do the work we need.

A shorter version for a general audience is on the [documentation site](https://dcc-bs.github.io/documentation/evaluation/). The models we run today are listed on the [models page](https://dcc-bs.github.io/documentation/infrastructure/models), and our servers are described on the [hardware page](https://dcc-bs.github.io/documentation/infrastructure/hardware).

The DCC AI team chooses and tests all models on the KDKP. We are setting up a governance board for these decisions.

## The five basic checks

A good score in a public ranking can't make up for a failed check.

1. **It fits on our graphics cards.** The model weights and the KV cache (the memory the model needs while it reads and writes text) must fit on our GPUs. We size this for the longest texts and the number of parallel users we expect. Today we run NVIDIA H100 NVL cards with 94 GB each. Cards with 141 GB (H200 NVL) are planned.
2. **It runs on a single server.** Spreading one model over several servers with vLLM makes it slower, so we don't do it. Splitting a model across the GPUs inside one server (tensor parallelism) is fine.
3. **It runs on vLLM.** We run all our language models with vLLM. That gives us one way to operate models and the same OpenAI-compatible API for every app. Speech-to-text is the one exception. It runs on FasterWhisper with BentoML. We can still test speech models that vLLM doesn't support, for example NeMo models. Before such a model could go live, we would have to decide separately how to run it.
4. **Its licence allows our use.** We read the licence and check that it covers what we want to do with the model.
5. **It is published on Hugging Face.** We download all model weights from there.

## What we look at for each kind of model

Public rankings help us decide which models deserve a closer look. The decision itself comes from the five checks, our own tests, load tests and our own experience with the model. [eval-layers.md](eval-layers.md) describes how these tests build on each other.

### Layout and OCR

We start with these public tests and rankings:

- [olmOCR-Bench](https://github.com/allenai/olmocr)
- [OCRBench v2 leaderboard](https://huggingface.co/spaces/ling99/OCRBench-v2-leaderboard)
- [ParseBench](https://huggingface.co/datasets/llamaindex/ParseBench)
- [OmniDocBench](https://github.com/opendatalab/OmniDocBench)
- [Hugging Face OCR leaderboards collection](https://huggingface.co/collections/davanstrien/ocr-leaderboards)
- [DocBank](https://huggingface.co/datasets/liminghao1630/DocBank)
- [dp-bench](https://huggingface.co/datasets/upstage/dp-bench)
- [Example layout-detection model (DETR)](https://huggingface.co/cmarkea/detr-layout-detection)

Then we run our own OCR test ([`ocr/`](../ocr/README.md)) and the docling tests for typical documents and difficult cases in [`dcc-docling-serve/benchmarks`](https://github.com/DCC-BS/dcc-docling-serve/tree/main/benchmarks).

### Speech-to-text

We run our own test with German and Swiss German recordings ([`stt/`](../stt/README.md)). It counts how many words and letters the model gets wrong (WER and CER) and how close the text is to the reference (BLEU). The tests in [`bentoml-faster-whisper`](https://github.com/DCC-BS/bentoml-faster-whisper) check how well the service tells speakers apart (diarization) and whether the API behaves as expected.

### Embeddings and reranking

We start with the [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard).

### Large language models

Our main outside source is [Artificial Analysis](https://artificialanalysis.ai/). It is an independent website with its own way of measuring, and we have no say in how it works. We use it to compare:

- price and speed
- which kinds of input a model accepts
- how well it reasons, and whether we can control how much it thinks
- size and design, including KV cache use and GPU memory per text length
- coding, reasoning, legal and other text tasks
- tool use
- how wordy the answers are
- how open the makers are about the model
- bias

Then we check what the rankings can't tell us:

- **Memory.** We estimate it with `hf-mem` for our text length and compression level (see below).
- **Compression (quantisation).** We look for formats that run fast and reliably on H100 and H200 cards. A reliable format doesn't get stuck repeating itself. We also check whether the KV cache can be compressed.
- **Text length.** How long a text the model supports, and how much of that we can serve.
- **Known vLLM problems** with that model.
- **What other users report**, in forums and open issues.
- **Our own experience** using it on our development machines.

### Text-to-speech

We know of no automatic test that tells us whether a voice pronounces a Basel place name properly. So the team listens and follows a written checklist: [`tts/LISTENING_PROTOCOL.md`](../tts/LISTENING_PROTOCOL.md). Listeners rate how easy the voice is to understand, how it pronounces Swiss names and places, how natural it sounds, and whether it stays the same voice from sentence to sentence.

### PII/NER and decision models

We test anonymisation models (PII/NER) and decision models with our own tests. That code isn't public yet. [`pii-ner/`](../pii-ner/README.md) and [`decision-models/`](../decision-models/README.md) describe what we test.

## Technical details

### Memory check with hf-mem

[`hf-mem`](https://pypi.org/project/hf-mem/) reads a model's settings from the Hugging Face Hub and estimates how much memory it needs. It doesn't download the weights. We run it with `uvx`. These are the useful flags, checked against hf-mem 0.5.6 with `uvx hf-mem --help`:

| Flag | What it does |
|---|---|
| `--model-id` | Model repository on the Hub (required) |
| `--revision` | Model revision |
| `--max-model-len` | Maximum text length, prompt plus answer. Taken from the model settings if you leave it out |
| `--batch-size` | Number of parallel requests, used for the KV cache estimate. Default 1 |
| `--kv-cache-dtype` | Number format of the KV cache, for example `auto`, `bfloat16` or `fp8` |
| `--experimental` | Turns on the KV cache estimate. Works for `...ForCausalLM` and `...ForConditionalGeneration` models |
| `--json-output` | Output for scripts |

Example:

```bash
uvx hf-mem --model-id <org>/<model> --experimental --max-model-len 32768 --batch-size 8 --kv-cache-dtype fp8
```

Replace `<org>/<model>` with the repository you want to check. The KV cache figure is an estimate. We use it to sort out models that clearly won't fit, and then start the model in vLLM on the real hardware to confirm.

### Serving

We serve models with [vLLM](https://github.com/vllm-project/vllm). Check 3 keeps us on one way of running them. The [software page](https://dcc-bs.github.io/documentation/infrastructure/software) describes our setup, including FasterWhisper and BentoML for speech-to-text.
