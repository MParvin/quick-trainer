FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY quick_trainer ./quick_trainer

RUN pip install --no-cache-dir \
      accelerate datasets huggingface-hub peft pyyaml \
      transformers trl typer pydantic pydantic-settings bitsandbytes \
    && pip install --no-cache-dir --no-deps .

WORKDIR /workspace
ENTRYPOINT ["quick-trainer"]
CMD ["--help"]
