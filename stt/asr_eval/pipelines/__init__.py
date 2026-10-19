from asr_eval.pipelines.openai_api import OpenAIAPIPipeline
from asr_eval.pipelines.vibevoice import VibeVoicePipeline

PIPELINES: dict[str, type] = {
    "openai_api": OpenAIAPIPipeline,
    "vibevoice": VibeVoicePipeline,
}
