# Quick Trainer

Fine-tune Hugging Face language models from a YAML config or CLI flags, then publish the result to the Hugging Face Hub and/or Ollama.

## Features

- **Config-driven or CLI** — YAML file, command-line overrides, or both
- **Multiple datasets** — Hugging Face Hub ids or local JSON/JSONL files
- **Efficient training** — LoRA + optional 4-bit quantization via PEFT and TRL
- **Publish** — Upload merged weights to Hugging Face; create (and optionally push) an Ollama model
- **CI/CD** — GitHub Actions workflow to train and push from GitHub runners

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Validate example config
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

Run Quick Trainer in the browser with a free or paid Colab GPU:

1. Click **Open in Colab** above and set **Runtime → Change runtime type → GPU**.
2. Add a Colab secret named `HF_TOKEN` with your [Hugging Face write token](https://huggingface.co/settings/tokens).
3. In the notebook, set `huggingface.repo_id` to your target repo (e.g. `your-username/my-colab-model`).
4. Run all cells to install, validate, train, and upload to the Hub.

The notebook installs from GitHub, uses [`configs/colab.yaml`](configs/colab.yaml) defaults (4-bit LoRA, Ollama disabled), and optionally mounts Google Drive to persist outputs.

## Configuration

Copy `configs/example.yaml` and edit:

| Section | Purpose |
|---------|---------|
| `base_model` | Hugging Face model id or local path |
| `datasets` | One or more dataset sources |
| `training` | Epochs, batch size, LoRA, output dir |
| `huggingface` | Upload target repo and visibility |
| `ollama` | Local Ollama model name and optional push |

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
  --hf-token TEXT            HF token (prefer HF_TOKEN env)
  --skip-download            Skip pre-download step
```

CLI flags override values from the config file when both are provided.

## Hugging Face upload

1. Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) with **write** access.
2. Set `HF_TOKEN` in your environment or GitHub repository secrets.
3. Set `huggingface.repo_id` in config (e.g. `username/my-model`).

The pipeline merges LoRA adapters into full weights before upload.

## Ollama publish

Enable in config:

```yaml
ollama:
  enabled: true
  model_name: my-finetuned-model
  system_prompt: "You are a helpful assistant."
  push: false   # set true to run `ollama push`
```

Requirements:

- [Ollama](https://ollama.com) installed and on `PATH`
- For best results, provide a GGUF file via `ollama.gguf_path` (convert merged weights with [llama.cpp](https://github.com/ggerganov/llama.cpp))

## GitHub Actions

Workflow: `.github/workflows/train-and-push.yml`

### Secrets

| Secret | Required | Description |
|--------|----------|-------------|
| `HF_TOKEN` | Yes (for upload) | Hugging Face write token |
| `OLLAMA_HOST` | No | Remote Ollama host if applicable |

### Manual run

1. Go to **Actions → Train and Push Model → Run workflow**
2. Choose the config path (default: `configs/example.yaml`)

### GPU runners

Training on `ubuntu-latest` uses CPU PyTorch and suits smoke tests only. For real fine-tuning:

1. Add a self-hosted runner with a GPU
2. Set `runs-on: [self-hosted, gpu]` in the workflow
3. Enable the `train-and-push-gpu` job template

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
│   └── pipeline.py     # End-to-end orchestration
├── configs/
│   ├── example.yaml
│   └── colab.yaml
├── notebooks/
│   └── quick_trainer_colab.ipynb
└── .github/workflows/train-and-push.yml
```

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check quick_trainer
```

## License

MIT
