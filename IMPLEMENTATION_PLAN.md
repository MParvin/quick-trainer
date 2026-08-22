# Implementation Plan

Ordered for risk reduction. Prefer small PRs when splitting work.

## Phase 1

- [x] Sanitize GitHub Actions inputs (env + allowlist)
  Priority: P0
  Files: `.github/workflows/train-and-push.yml`
  Estimated Time: 1–2h
  Risk: Low
  Dependencies: None

- [x] Manual-only train workflow; default `configs/ci-smoke.yaml`
  Priority: P0
  Files: `.github/workflows/train-and-push.yml`, `README.md`
  Estimated Time: 1h
  Risk: Low
  Dependencies: None

- [x] Build image from commit SHA in CI
  Priority: P0
  Files: `.github/workflows/train-and-push.yml`, `Dockerfile`, `Taskfile.yml`
  Estimated Time: 3–5h
  Risk: Medium
  Dependencies: Runner Docker build

- [x] Harden Modelfile generation + `gguf_path` allowlist
  Priority: P0
  Files: `quick_trainer/ollama.py`, `quick_trainer/safety.py`, `tests/`
  Estimated Time: 2–3h
  Risk: Low
  Dependencies: None

- [x] `trust_remote_code` default false
  Priority: P0
  Files: `quick_trainer/config.py`, `downloader.py`, `export.py`, configs, README, tests
  Estimated Time: 2h
  Risk: Low–Med
  Dependencies: None

## Phase 2

- [x] Redact tokens in `validate`; deprecate `--hf-token`
  Priority: P1
  Files: `quick_trainer/cli.py`, `config.py`, tests
  Estimated Time: 1–2h
  Risk: Low
  Dependencies: None

- [x] Dockerfile installs from project metadata; pin versions; non-root user
  Priority: P1
  Files: `Dockerfile`
  Estimated Time: 2–4h
  Risk: Medium
  Dependencies: Phase 1 image build

- [x] Colab install pin via `QUICK_TRAINER_REF`
  Priority: P1
  Files: `notebooks/quick_trainer_colab.ipynb`, README
  Estimated Time: 1h
  Risk: Low
  Dependencies: None

## Phase 3

- [x] Add `.github/workflows/ci.yml` (`pytest` + `ruff`)
  Priority: P1
  Files: `.github/workflows/ci.yml`, `pyproject.toml`
  Estimated Time: 2–3h
  Risk: Low
  Dependencies: None

- [x] Fix LoRA target-module fallback
  Priority: P1
  Files: `quick_trainer/trainer.py`, tests
  Estimated Time: 1–2h
  Risk: Low
  Dependencies: None

- [x] Expand tests (Modelfile, redaction, CLI, targets)
  Priority: P1
  Files: `tests/`
  Estimated Time: 4–6h
  Risk: Low
  Dependencies: Phase 1–2

- [x] Align README with real workflows
  Priority: P1
  Files: `README.md`
  Estimated Time: 1h
  Risk: Low
  Dependencies: Phase 1

## Phase 4

- [x] Non-root Docker user + cache path updates
  Priority: P2
  Files: `Dockerfile`, `Taskfile.yml`, workflow
  Estimated Time: 2–3h
  Risk: Medium
  Dependencies: Phase 1

- [x] Workflow concurrency + `model-publish` environment
  Priority: P2
  Files: `.github/workflows/train-and-push.yml`
  Estimated Time: 1–2h
  Risk: Low
  Dependencies: Phase 1

- [x] SECURITY.md + runner runbook
  Priority: P2
  Files: `SECURITY.md`, `docs/runbook-runner.md`
  Estimated Time: 2h
  Risk: Low
  Dependencies: Phase 1

## Phase 5

- [x] Remove unused `quantize`; document external GGUF conversion
  Priority: P2
  Files: `config.py`, `ollama.py`, README
  Estimated Time: 1h
  Risk: Low
  Dependencies: None

- [x] Optional smoke evaluation hook
  Priority: P3
  Files: `evaluate.py`, `pipeline.py`, `config.py`
  Estimated Time: 1–2d
  Risk: Med
  Dependencies: Phase 3

- [x] Remove unused deps/helpers; generalize code-language formatter
  Priority: P3
  Files: `pyproject.toml`, `formatting.py`, `config.py`
  Estimated Time: 2–4h
  Risk: Low
  Dependencies: None

## Phase 6

- [x] Split CI vs train workflows; adapter-only publish; `[cpu]`/`[cuda]` extras
  Priority: P3
  Files: workflows, `config.py`, `pipeline.py`, `pyproject.toml`
  Estimated Time: 2–5d
  Risk: Med
  Dependencies: Phases 1–4
