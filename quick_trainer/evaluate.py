"""Optional post-training smoke evaluation."""

from __future__ import annotations

import logging
from pathlib import Path

from quick_trainer.config import QuickTrainerConfig
from quick_trainer.utils import cuda_available, resolve_hf_token

logger = logging.getLogger(__name__)


def run_smoke_evaluation(model_dir: Path, config: QuickTrainerConfig) -> str:
    """Generate a short completion to verify the exported model loads."""
    if not config.evaluation.enabled:
        logger.info("Smoke evaluation disabled; skipping.")
        return ""

    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    token = resolve_hf_token(config)
    logger.info("Running smoke evaluation from %s", model_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        token=token,
        trust_remote_code=config.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        token=token,
        trust_remote_code=config.trust_remote_code,
        device_map="auto" if cuda_available() else None,
    )
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    outputs = generator(
        config.evaluation.prompt,
        max_new_tokens=config.evaluation.max_new_tokens,
        do_sample=False,
        return_full_text=False,
    )
    text = outputs[0]["generated_text"] if outputs else ""
    logger.info("Smoke evaluation output: %s", text[:500])
    return text
