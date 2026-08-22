FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/.cache/huggingface \
    HOME=/workspace

WORKDIR /app

RUN useradd --create-home --uid 1000 --shell /bin/bash trainer \
    && mkdir -p /workspace/.cache/huggingface \
    && chown -R trainer:trainer /workspace

COPY pyproject.toml README.md LICENSE ./
COPY quick_trainer ./quick_trainer

# Install into system site-packages (base image already provides torch).
RUN pip install --no-cache-dir --no-deps . \
    && pip install --no-cache-dir \
      "accelerate>=0.27.0" \
      "datasets>=2.18.0" \
      "huggingface-hub>=0.21.0" \
      "peft>=0.10.0" \
      "pyyaml>=6.0" \
      "transformers>=4.38.0" \
      "trl>=0.12.0" \
      "typer>=0.9.0" \
      "pydantic>=2.5.0" \
      "bitsandbytes>=0.42.0"

USER trainer

WORKDIR /workspace
ENTRYPOINT ["quick-trainer"]
CMD ["--help"]
