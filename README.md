# Quick Trainer

Fine-tune Hugging Face language models from a YAML config or CLI flags, then publish the result to the Hugging Face Hub and/or Ollama.

## Features

- **Config-driven or CLI** — YAML file, command-line overrides, or both
- **Multiple datasets** — Hugging Face Hub ids or local JSON/JSONL files
- **Efficient training** — LoRA + optional 4-bit quantization via PEFT and TRL
- **Publish** — Upload merged weights or adapters to Hugging Face; create (and optionally push) an Ollama model
- **Safety defaults** — `trust_remote_code` off by default; Modelfile/path allowlists; token redaction in `validate`
- **CI/CD** — PR quality gates (`pytest`/`ruff`) and a manual GPU train workflow that builds from the git SHA

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,cpu]"          # CPU / local tools
# pip install -e ".[dev,cuda]"       # add bitsandbytes on Linux/Windows CUDA hosts

# Validate example config (secrets redacted in output)
quick-trainer validate configs/example.yaml

# Train from config
export HF_TOKEN=hf_...
quick-trainer train --config configs/example.yaml

# Or train from CLI only
quick-trainer train \
  --base-model meta-llama/Llama-3.2-1B-Instruct \
  --dataset databricks/databricks-dolly-15k \
  --epochs 1 \
  --hf-repo your-username/my-model \
  --no-ollama
```

## Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MParvin/quick-trainer/blob/master/notebooks/quick_trainer_colab.ipynb)

1. Click **Open in Colab** and set **Runtime → Change runtime type → GPU**.
2. Add a Colab secret named `HF_TOKEN` with your [Hugging Face write token](https://huggingface.co/settings/tokens).
3. In the notebook configuration cell, set `BASE_MODEL`, `HF_REPO_ID`, `COMMIT_MESSAGE`, and `QUICK_TRAINER_REF` (prefer a commit SHA).
4. Run all cells to install, validate, train, and upload to the Hub.

Defaults use ungated `TinyLlama`, 4-bit LoRA, and Ollama disabled. Gated models require accepting the Hub license first.

## Configuration

Copy `configs/example.yaml` and edit:

| Section | Purpose |
|---------|---------|
| `base_model` | Hugging Face model id or local path |
| `datasets` | One or more dataset sources |
| `training` | Epochs, batch size, LoRA, output dir, `merge_adapter` |
| `huggingface` | Upload target repo, visibility, `upload_adapter_only` |
| `ollama` | Local Ollama model name and optional push |
| `evaluation` | Optional post-train smoke generation |
| `trust_remote_code` | Opt-in Hub remote code execution (default `false`) |

### Dataset formats

**Instruction tuning** — set `instruction_field` and `response_field`:

```yaml
datasets:
  - path: databricks/databricks-dolly-15k
    split: train
    instruction_field: instruction
    response_field: response
```

**Plain text** — set `text_field`:

```yaml
datasets:
  - path: ./data/examples.jsonl
    text_field: text
```

**Code + docstring** — set `code_field`, `docstring_field`, and optional `code_language`:

```yaml
datasets:
  - path: google/code_x_glue_ct_code_to_text
    dataset_config: go
    code_field: code
    docstring_field: docstring
    code_language: go
```

**Multiple datasets** — list several entries; samples are concatenated.

## CLI reference

```text
quick-trainer train [OPTIONS]

  -c, --config PATH          YAML config file
  -m, --base-model TEXT      Base model id
  -d, --dataset TEXT         Dataset (repeatable)
  -o, --output-dir PATH      Output directory
  --epochs INT
  --batch-size INT
  --learning-rate FLOAT
  --max-seq-length INT
  --hf-repo TEXT             Hugging Face repo id
  --hf-private               Private HF repo
  --no-hf-upload             Skip HF upload
  --ollama-model TEXT        Ollama model name
  --ollama-push              Push to Ollama registry
  --no-ollama                Skip Ollama
  --trust-remote-code        Allow Hub remote code (unsafe)
  --skip-download            Skip pre-download step

quick-trainer validate CONFIG [--require-repo-path]
```

Prefer `HF_TOKEN` in the environment. Do not commit tokens in YAML.

CLI flags override values from the config file when both are provided.

## Hugging Face upload

1. Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) with **write** access.
2. Set `HF_TOKEN` in your environment or GitHub repository / `model-publish` environment secrets.
3. Set `huggingface.repo_id` in config (e.g. `username/my-model`).

By default the pipeline merges LoRA adapters into full weights before upload. Set
`huggingface.upload_adapter_only: true` or `training.merge_adapter: false` to publish adapters/checkpoints instead.

## Ollama publish

```yaml
ollama:
  enabled: true
  model_name: my-finetuned-model
  system_prompt: "You are a helpful assistant."
  push: false
  # gguf_path: ./models/my-model.gguf   # optional; must stay under the workspace
```

GGUF conversion is **not** performed by Quick Trainer. Convert merged weights with
[llama.cpp](https://github.com/ggerganov/llama.cpp) (or similar), then set `gguf_path`.

Requirements:

- [Ollama](https://ollama.com) installed and on `PATH`

## GitHub Actions

### Quality gate — `.github/workflows/ci.yml`

Runs on pull requests and pushes to `main`/`master`:

- `ruff check`
- `pytest`

### Train — `.github/workflows/train-and-push.yml`

Manual only (`workflow_dispatch`). Builds `mparvin/quick-trainer:<git-sha>` from the checkout, validates a path under `configs/`, then trains on a self-hosted GPU runner.

| Secret / env | Required | Description |
|--------------|----------|-------------|
| `HF_TOKEN` | Yes (for upload) | Hugging Face write token |
| `OLLAMA_HOST` | No | Remote Ollama host if applicable |

Create a GitHub Environment named **`model-publish`** (optional reviewers recommended). See [`docs/runbook-runner.md`](docs/runbook-runner.md).

Default dispatch config: `configs/ci-smoke.yaml`. Use `configs/golang-dev.yaml` only for intentional full runs.

## Project layout

```text
quick-trainer/
├── quick_trainer/
│   ├── cli.py          # Typer CLI
│   ├── config.py       # Pydantic config models
│   ├── downloader.py   # HF model/dataset loading
│   ├── trainer.py      # LoRA SFT training
│   ├── export.py       # Merge adapters for export
│   ├── uploader.py     # Hugging Face Hub upload
│   ├── ollama.py       # Ollama Modelfile + create/push
│   ├── evaluate.py     # Optional smoke generation
│   ├── formatting.py   # Dataset row formatting
│   ├── safety.py       # Path / Modelfile validation
│   ├── utils.py        # Token + CUDA helpers
│   └── pipeline.py     # End-to-end orchestration
├── configs/
│   ├── example.yaml
│   ├── colab.yaml
│   ├── ci-smoke.yaml
│   └── golang-dev.yaml
├── notebooks/
│   └── quick_trainer_colab.ipynb
├── docs/
│   └── runbook-runner.md
└── .github/workflows/
    ├── ci.yml
    └── train-and-push.yml
```

## Development

```bash
pip install -e ".[dev,cpu]"
pytest
ruff check quick_trainer tests
```

## Security

See [`SECURITY.md`](SECURITY.md). Highlights:

- `trust_remote_code` defaults to `false`
- `validate` redacts `hf_token`
- CI config paths are allowlisted under `configs/`

## License

MIT
