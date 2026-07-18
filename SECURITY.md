# Security Policy

## Reporting a vulnerability

Please report security issues privately via GitHub Security Advisories for this repository,
or contact the maintainers directly. Do not open a public issue for credential leaks or
remote code execution reports.

## Trust boundaries

Quick Trainer downloads models and datasets from the Hugging Face Hub and may execute
training code on the local machine or a self-hosted GitHub Actions runner.

- Prefer the `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` environment variables. Do not put write
  tokens in YAML committed to git.
- `trust_remote_code` defaults to `false`. Only enable it for models you trust.
- Ollama `system_prompt`, `base_template`, and `gguf_path` are validated to reduce
  Modelfile injection risk.
- Training configs used in CI must live under `configs/` and match a strict path allowlist.

## Self-hosted runners

Treat GPU runners as high-value hosts. See [`docs/runbook-runner.md`](docs/runbook-runner.md).
