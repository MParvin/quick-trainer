# Self-hosted GPU runner runbook

## Purpose

The `Train and Push Model` workflow runs only on `workflow_dispatch` against runners labeled
`self-hosted` and `gpu`. It builds a Docker image from the checked-out commit, validates a
config under `configs/`, then trains and optionally publishes.

## One-time setup

1. Install Docker with NVIDIA Container Toolkit on the runner host.
2. Register a GitHub Actions runner with labels `self-hosted` and `gpu`.
3. In the GitHub repo, create an Environment named `model-publish`.
   - Optionally require reviewers before publish jobs proceed.
   - Attach `HF_TOKEN` (and optional `OLLAMA_HOST`) as environment or repository secrets.
4. Ensure the runner user can run `docker build` and `docker run --gpus all`.

## Safe operating procedure

1. Prefer `configs/ci-smoke.yaml` for pipeline verification.
2. Use `configs/golang-dev.yaml` (or a custom config) only for intentional full trains.
3. Confirm `huggingface.repo_id` and upload flags before dispatching.
4. Pin/publish images with the git SHA tag produced by the workflow when pushing to a registry.

## Incident response

- Revoke compromised `HF_TOKEN` values immediately on Hugging Face.
- Drain or reinstall the self-hosted runner if untrusted workflow code may have executed.
- Delete unexpected Hub model revisions and Ollama tags created during the incident window.

## Rollback

Hub uploads are immutable commits. To roll back a bad publish:

1. Re-upload a known-good artifact, or
2. Point consumers at a previous Hub revision / delete the bad repo revision if policy allows.
